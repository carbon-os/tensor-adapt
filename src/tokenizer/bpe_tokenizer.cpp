#include <tensor/tokenizer/bpe_tokenizer.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace tensor::tokenizer {

// ── UTF-8 encoder ─────────────────────────────────────────────────────────────

std::string BPETokenizer::codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// ── Byte encoder (mirrors HuggingFace bytes_to_unicode()) ────────────────────
//
//  Maps each of the 256 byte values to a unique printable unicode character
//  so the BPE vocabulary never contains raw control bytes.

void BPETokenizer::build_byte_encoder() {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; i++) bs.push_back(i);    // 33–126
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i); // 161–172
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i); // 174–255

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    for (std::size_t i = 0; i < bs.size(); i++)
        byte_encoder_[static_cast<uint8_t>(bs[i])] = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
}

// ── Pre-tokenizer ─────────────────────────────────────────────────────────────
//
//  Simplified GPT-2 style split. Covers English text datasets correctly.
//  \p{L}/\p{N} Unicode properties (requiring PCRE2) are approximated as
//  [a-zA-Z] and [0-9] — sufficient for openwebtext, TinyStories, etc.

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    static const std::regex PAT(
        // contractions (case variations)
        "(?:'[sStTmMdD]|'[rR][eE]|'[vV][eE]|'[lL][lL])"
        // optional-space + word
        "| ?[a-zA-Z]+"
        // optional-space + number run
        "| ?[0-9]+"
        // optional-space + punctuation / symbols (non-whitespace, non-word)
        "| ?[^\\s\\w]+"
        // trailing whitespace (spaces that follow non-whitespace)
        "|\\s+",
        std::regex::optimize
    );

    std::vector<std::string> out;
    auto it  = std::sregex_iterator(text.begin(), text.end(), PAT);
    auto end = std::sregex_iterator();
    for (; it != end; ++it)
        out.push_back(it->str());
    return out;
}

// ── BPE core ──────────────────────────────────────────────────────────────────

std::vector<std::string> BPETokenizer::apply_bpe(const std::string& word) const {
    // Start: one symbol per byte (byte-encoded)
    std::vector<std::string> syms;
    syms.reserve(word.size());
    for (unsigned char c : word)
        syms.push_back(byte_encoder_[c]);

    // Iteratively merge the highest-priority (lowest rank) adjacent pair
    while (syms.size() > 1) {
        int  best_rank = INT_MAX;
        int  best_idx  = -1;

        for (int i = 0; i < static_cast<int>(syms.size()) - 1; i++) {
            auto it = merge_ranks_.find({syms[i], syms[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = i;
            }
        }

        if (best_idx == -1) break;

        syms[best_idx] += syms[best_idx + 1];
        syms.erase(syms.begin() + best_idx + 1);
    }

    return syms;
}

// ── Encode a single pre-tokenized word → IDs ─────────────────────────────────

void BPETokenizer::encode_word(const std::string& word,
                                std::vector<uint32_t>& out) const {
    for (auto& sym : apply_bpe(word)) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end())
            out.push_back(it->second);
        // In byte-level BPE every byte has a vocab entry, so unknowns
        // should not occur. Silently skip if they somehow do.
    }
}

// ── Load ──────────────────────────────────────────────────────────────────────

BPETokenizer BPETokenizer::load(const std::string& model_dir) {
    BPETokenizer tok;
    tok.build_byte_encoder();

    fs::path dir       = model_dir;
    fs::path tok_json  = dir / "tokenizer.json";
    fs::path vocab_f   = dir / "vocab.json";
    fs::path merges_f  = dir / "merges.txt";

    if (fs::exists(tok_json)) {
        std::ifstream f(tok_json);
        if (!f) throw TokenizerError("cannot open " + tok_json.string());
        json j = json::parse(f);

        auto& model = j.at("model");

        // Vocab
        for (auto& [k, v] : model.at("vocab").items())
            tok.token_to_id_[k] = v.get<uint32_t>();

        // Merges: array of "a b" strings
        int rank = 0;
        for (auto& m : model.at("merges")) {
            std::string s  = m.get<std::string>();
            auto         sp = s.find(' ');
            if (sp == std::string::npos) { rank++; continue; }
            tok.merge_ranks_[{s.substr(0, sp), s.substr(sp + 1)}] = rank++;
        }

        // Added / special tokens
        if (j.contains("added_tokens")) {
            for (auto& at : j["added_tokens"]) {
                std::string content = at.value("content", "");
                uint32_t    id      = at.value("id", 0u);
                if (!content.empty())
                    tok.added_tokens_[content] = id;
            }
        }

    } else if (fs::exists(vocab_f) && fs::exists(merges_f)) {
        // Fallback: separate vocab.json + merges.txt
        {
            std::ifstream f(vocab_f);
            if (!f) throw TokenizerError("cannot open " + vocab_f.string());
            json j = json::parse(f);
            for (auto& [k, v] : j.items())
                tok.token_to_id_[k] = v.get<uint32_t>();
        }
        {
            std::ifstream f(merges_f);
            if (!f) throw TokenizerError("cannot open " + merges_f.string());
            std::string line;
            int rank = 0;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#') continue;
                auto sp = line.find(' ');
                if (sp == std::string::npos) continue;
                tok.merge_ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
            }
        }
    } else {
        throw TokenizerError(
            "no tokenizer.json or vocab.json+merges.txt found in: " + model_dir);
    }

    // Read EOS/BOS from tokenizer_config.json if present
    fs::path cfg_f = dir / "tokenizer_config.json";
    if (fs::exists(cfg_f)) {
        std::ifstream f(cfg_f);
        json j = json::parse(f);

        auto resolve_token = [&](const std::string& key) -> uint32_t {
            if (!j.contains(key)) return UINT32_MAX;
            std::string s;
            if (j[key].is_string())                                    s = j[key].get<std::string>();
            else if (j[key].is_object() && j[key].contains("content")) s = j[key]["content"].get<std::string>();
            else return UINT32_MAX;

            auto it = tok.added_tokens_.find(s);
            if (it != tok.added_tokens_.end()) return it->second;
            auto it2 = tok.token_to_id_.find(s);
            if (it2 != tok.token_to_id_.end()) return it2->second;
            return UINT32_MAX;
        };

        uint32_t eos = resolve_token("eos_token");
        uint32_t bos = resolve_token("bos_token");
        if (eos != UINT32_MAX) tok.eos_id_ = eos;
        if (bos != UINT32_MAX) tok.bos_id_ = bos;
    }

    std::cerr << "[tokenizer] vocab=" << tok.token_to_id_.size()
              << " merges="           << tok.merge_ranks_.size()
              << " added="            << tok.added_tokens_.size()
              << " eos="              << tok.eos_id_  << "\n";

    return tok;
}

// ── Encode ────────────────────────────────────────────────────────────────────

std::vector<uint32_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> ids;

    if (added_tokens_.empty()) {
        // Fast path: no special tokens to scan for
        for (auto& word : pre_tokenize(text))
            encode_word(word, ids);
        return ids;
    }

    // Scan for special tokens first, encode normal text between them.
    // O(n * |added_tokens|) but dataset prep is offline so this is fine.
    std::string_view remaining(text);

    while (!remaining.empty()) {
        // Find the earliest-occurring special token
        std::size_t best_pos = std::string_view::npos;
        std::string best_tok;
        uint32_t    best_id  = 0;

        for (auto& [tok_str, tok_id] : added_tokens_) {
            auto pos = remaining.find(tok_str);
            if (pos != std::string_view::npos &&
                (best_pos == std::string_view::npos || pos < best_pos)) {
                best_pos = pos;
                best_tok = tok_str;
                best_id  = tok_id;
            }
        }

        if (best_pos == std::string_view::npos) {
            // No more special tokens — encode the rest normally
            for (auto& word : pre_tokenize(std::string(remaining)))
                encode_word(word, ids);
            break;
        }

        // Encode text before the special token
        if (best_pos > 0) {
            for (auto& word : pre_tokenize(std::string(remaining.substr(0, best_pos))))
                encode_word(word, ids);
        }

        ids.push_back(best_id);
        remaining.remove_prefix(best_pos + best_tok.size());
    }

    return ids;
}

} // namespace tensor::tokenizer