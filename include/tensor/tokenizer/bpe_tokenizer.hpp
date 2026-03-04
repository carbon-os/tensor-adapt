#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensor::tokenizer {

struct TokenizerError : std::runtime_error {
    explicit TokenizerError(const std::string& m)
        : std::runtime_error("TokenizerError: " + m) {}
};

// ─────────────────────────────────────────────────────────────
//  BPETokenizer — byte-level BPE compatible with HuggingFace
//  tokenizer.json (or vocab.json + merges.txt fallback).
//
//  Encodes UTF-8 text → flat vector of uint32_t token IDs.
//  Pre-tokenization uses a simplified GPT-2 style regex that
//  handles English text correctly. Full Unicode \p{L}/\p{N}
//  support would require PCRE2; ASCII coverage is sufficient
//  for standard LLM pre-training datasets.
// ─────────────────────────────────────────────────────────────

class BPETokenizer {
public:
    // Load from a model directory. Looks for tokenizer.json first,
    // then falls back to vocab.json + merges.txt. Reads eos/bos
    // from tokenizer_config.json if present.
    static BPETokenizer load(const std::string& model_dir);

    // Encode text → token ID sequence. Special/added tokens
    // (e.g. <|endoftext|>) are matched before BPE runs.
    std::vector<uint32_t> encode(const std::string& text) const;

    uint32_t    eos_id()     const { return eos_id_; }
    uint32_t    bos_id()     const { return bos_id_; }
    std::size_t vocab_size() const { return token_to_id_.size(); }

private:
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::unordered_map<std::string, uint32_t> added_tokens_;

    struct PairHash {
        std::size_t operator()(const std::pair<std::string, std::string>& p) const {
            std::size_t h1 = std::hash<std::string>{}(p.first);
            std::size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 * 2654435761ULL);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;

    std::string byte_encoder_[256]; // byte value → UTF-8 encoded unicode char

    uint32_t eos_id_ = 151643; // Qwen2 default; overridden from tokenizer_config.json
    uint32_t bos_id_ = 151643;

    void                     build_byte_encoder();
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    std::vector<std::string> apply_bpe(const std::string& word) const;
    void                     encode_word(const std::string& word,
                                         std::vector<uint32_t>& out) const;

    static std::string codepoint_to_utf8(uint32_t cp);
};

} // namespace tensor::tokenizer