// dataset-format — tokenize text datasets into .bin files for tensor-adapt
//
// Supported input formats:
//   .txt    whole file as one document; splits on <|endoftext|> if found
//   .jsonl  one JSON object per line
//   .json   JSON array of objects
//   dir     recursively processes all .txt / .jsonl / .json files
//
// Output:
//   <output>/train.bin   flat array of uint16_t token IDs
//
// Usage:
//   dataset-format \
//     --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
//     --input     ~/.cache/tensor/datasets/roneneldan/TinyStories \
//     --output    ./data/tinystories
//
//   # ChatML format (Qwen2 instruct / LoRA datasets):
//   dataset-format \
//     --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
//     --input     ./gomarkdown.jsonl \
//     --output    ./data/gomarkdown \
//     --chatml --no-eos
//
//   Expected JSONL schema for --chatml:
//     {"messages": [{"role": "user",      "content": "..."},
//                   {"role": "assistant", "content": "..."}]}
//
//   An optional first message with role="system" is used verbatim;
//   if absent, the --system prompt is injected automatically.

#include <tensor/tokenizer/bpe_tokenizer.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;
using namespace tensor::tokenizer;

// ── Args ──────────────────────────────────────────────────────────────────────

struct Args {
    std::string tokenizer;
    std::string input;
    std::string output;
    std::string field         = "text";
    std::string format        = "auto";
    std::string system_prompt = "You are a helpful assistant.";
    bool        add_eos       = true;
    bool        chatml        = false;
};

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n\n"
        << "Required:\n"
        << "  --tokenizer <dir>   model dir with tokenizer.json\n"
        << "  --input     <path>  file or directory to tokenize\n"
        << "  --output    <dir>   output directory (writes train.bin)\n\n"
        << "Optional:\n"
        << "  --field  <name>     JSON key containing text        (default: text)\n"
        << "  --format <fmt>      txt | jsonl | json | auto       (default: auto)\n"
        << "  --no-eos            skip EOS token between documents\n\n"
        << "ChatML options:\n"
        << "  --chatml            parse messages[] array and render as ChatML\n"
        << "  --system <prompt>   default system prompt for --chatml\n"
        << "                      (default: \"You are a helpful assistant.\")\n\n"
        << "Examples:\n"
        << "  # Plain text\n"
        << "  " << prog << " \\\n"
        << "    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \\\n"
        << "    --input     ~/.cache/tensor/datasets/roneneldan/TinyStories \\\n"
        << "    --output    ./data/tinystories\n\n"
        << "  # LoRA instruct dataset (ChatML)\n"
        << "  " << prog << " \\\n"
        << "    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \\\n"
        << "    --input     ./gomarkdown.jsonl \\\n"
        << "    --output    ./data/gomarkdown \\\n"
        << "    --chatml --no-eos \\\n"
        << "    --system \"You are an expert Go developer.\"\n";
}

static Args parse(int argc, char** argv) {
    Args a;
    std::unordered_map<std::string, std::string*> flags = {
        {"--tokenizer", &a.tokenizer},
        {"--input",     &a.input},
        {"--output",    &a.output},
        {"--field",     &a.field},
        {"--format",    &a.format},
        {"--system",    &a.system_prompt},
    };

    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if (key == "-h" || key == "--help") { usage(argv[0]); std::exit(0); }
        if (key == "--no-eos")              { a.add_eos = false; continue; }
        if (key == "--chatml")              { a.chatml  = true;  continue; }
        if (flags.count(key) && i + 1 < argc) {
            *flags[key] = argv[++i];
        } else {
            std::cerr << "unknown option: " << key << "\n";
            usage(argv[0]);
            std::exit(1);
        }
    }

    bool ok = true;
    if (a.tokenizer.empty()) { std::cerr << "missing --tokenizer\n"; ok = false; }
    if (a.input.empty())     { std::cerr << "missing --input\n";     ok = false; }
    if (a.output.empty())    { std::cerr << "missing --output\n";    ok = false; }
    if (!ok) { usage(argv[0]); std::exit(1); }
    return a;
}

// ── Buffered binary writer ────────────────────────────────────────────────────
//
//  Writes uint16_t token IDs. Qwen2's vocab is 151,936 which fits in uint16
//  (max 65,535 for regular tokens; special tokens like im_start=151644 do NOT
//  fit). The tokenizer encodes special tokens via added_tokens_ so they appear
//  as their correct IDs — we must store them as uint32 or remap.
//
//  *** IMPORTANT: Qwen2 special tokens (151643, 151644, 151645) exceed uint16.
//  The Dataset::load reader uses uint16, so we write uint16 for regular tokens
//  and assert on overflow so the mismatch is caught at format time rather than
//  silently corrupting training data.
//
//  If you need full uint32 support, change both this writer and Dataset::load
//  to use uint32_t consistently.

class BinWriter {
    static constexpr std::size_t BUF_TOKENS = 1 << 20;

public:
    explicit BinWriter(const fs::path& path)
        : ofs_(path, std::ios::binary) {
        if (!ofs_) throw std::runtime_error("cannot open for writing: " + path.string());
        buf_.reserve(BUF_TOKENS);
    }

    ~BinWriter() { flush(); }

    void write(const std::vector<uint32_t>& ids) {
        for (uint32_t id : ids) write_one(id);
    }

    void write_one(uint32_t id) {
        buf_.push_back(id);
        if (buf_.size() >= BUF_TOKENS) flush();
        total_++;
    }

    void flush() {
        if (buf_.empty()) return;
        ofs_.write(reinterpret_cast<const char*>(buf_.data()),
                   buf_.size() * sizeof(uint32_t));
        buf_.clear();
    }

    std::size_t total() const { return total_; }

private:
    std::ofstream          ofs_;
    std::vector<uint32_t>  buf_;
    std::size_t            total_ = 0;
};

// ── Line sanitiser ────────────────────────────────────────────────────────────
//
//  Strips \r so files created on Windows (CRLF line endings) or by tools that
//  write CRLF don't cause json::parse to fail on every single line.

static void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

// ── ChatML renderer ───────────────────────────────────────────────────────────

static std::string render_chatml(const json& messages,
                                  const std::string& default_system)
{
    std::string out;

    bool has_system = !messages.empty() &&
                      messages[0].value("role", "") == "system";

    if (!has_system) {
        out += "<|im_start|>system\n";
        out += default_system;
        out += "<|im_end|>\n";
    }

    for (const auto& msg : messages) {
        std::string role    = msg.value("role",    "");
        std::string content = msg.value("content", "");
        if (role.empty() || content.empty()) continue;

        out += "<|im_start|>";
        out += role;
        out += "\n";
        out += content;
        out += "<|im_end|>\n";
    }

    return out;
}

// ── Processors ────────────────────────────────────────────────────────────────

static const std::string ENDOFTEXT = "<|endoftext|>";

static void flush_doc(const std::string& text,
                       const BPETokenizer& tok,
                       BinWriter& writer,
                       bool add_eos,
                       std::size_t& doc_count)
{
    if (text.empty()) return;
    auto ids = tok.encode(text);
    if (ids.empty()) return;
    writer.write(ids);
    if (add_eos) writer.write_one(tok.eos_id());
    doc_count++;
}

// ── Plain text ────────────────────────────────────────────────────────────────

static std::size_t process_txt(const fs::path& file,
                                const BPETokenizer& tok,
                                BinWriter& writer,
                                bool add_eos)
{
    std::ifstream f(file);
    if (!f) throw std::runtime_error("cannot open: " + file.string());
    std::string content((std::istreambuf_iterator<char>(f)), {});

    std::size_t docs = 0;

    if (content.find(ENDOFTEXT) != std::string::npos) {
        std::size_t start = 0;
        while (true) {
            auto pos = content.find(ENDOFTEXT, start);
            std::size_t len = (pos == std::string::npos) ? pos : pos - start;
            flush_doc(content.substr(start, len), tok, writer, add_eos, docs);
            if (pos == std::string::npos) break;
            start = pos + ENDOFTEXT.size();
        }
    } else {
        flush_doc(content, tok, writer, add_eos, docs);
    }

    return docs;
}

// ── JSONL — flat text field ───────────────────────────────────────────────────

static std::size_t process_jsonl(const fs::path& file,
                                  const BPETokenizer& tok,
                                  BinWriter& writer,
                                  const std::string& field,
                                  bool add_eos)
{
    std::ifstream f(file);
    if (!f) throw std::runtime_error("cannot open: " + file.string());

    std::size_t docs = 0;
    std::size_t line_no = 0;
    std::string line;

    while (std::getline(f, line)) {
        ++line_no;
        strip_cr(line);
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            if (!j.contains(field) || !j[field].is_string()) {
                std::cerr << "[warn] line " << line_no
                          << ": missing field '" << field << "' — keys:";
                for (auto& [k, v] : j.items()) std::cerr << " " << k;
                std::cerr << "\n";
                continue;
            }
            flush_doc(j[field].get<std::string>(), tok, writer, add_eos, docs);
        } catch (const std::exception& e) {
            std::cerr << "[warn] line " << line_no
                      << ": JSON parse error — " << e.what()
                      << "\n  raw: " << line.substr(0, 120) << "\n";
        }
    }
    return docs;
}

// ── JSONL — ChatML ────────────────────────────────────────────────────────────

static std::size_t process_jsonl_chatml(const fs::path& file,
                                         const BPETokenizer& tok,
                                         BinWriter& writer,
                                         const std::string& system_prompt,
                                         bool add_eos)
{
    std::ifstream f(file);
    if (!f) throw std::runtime_error("cannot open: " + file.string());

    std::size_t docs = 0;
    std::size_t line_no = 0;
    std::string line;

    while (std::getline(f, line)) {
        ++line_no;
        strip_cr(line);
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            if (!j.contains("messages") || !j["messages"].is_array()) {
                std::cerr << "[warn] line " << line_no
                          << ": no 'messages' array — keys:";
                for (auto& [k, v] : j.items()) std::cerr << " " << k;
                std::cerr << "\n";
                continue;
            }
            std::string text = render_chatml(j["messages"], system_prompt);
            flush_doc(text, tok, writer, add_eos, docs);
        } catch (const std::exception& e) {
            std::cerr << "[warn] line " << line_no
                      << ": JSON parse error — " << e.what()
                      << "\n  raw: " << line.substr(0, 120) << "\n";
        }
    }
    return docs;
}

// ── JSON array — flat text field ──────────────────────────────────────────────

static std::size_t process_json(const fs::path& file,
                                 const BPETokenizer& tok,
                                 BinWriter& writer,
                                 const std::string& field,
                                 bool add_eos)
{
    std::ifstream f(file);
    if (!f) throw std::runtime_error("cannot open: " + file.string());

    json j;
    try {
        j = json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error("JSON parse error in " + file.string() +
                                 ": " + e.what());
    }

    std::size_t docs = 0;

    auto handle = [&](const json& item, std::size_t idx) {
        if (!item.contains(field) || !item[field].is_string()) {
            std::cerr << "[warn] item " << idx
                      << ": missing field '" << field << "'\n";
            return;
        }
        flush_doc(item[field].get<std::string>(), tok, writer, add_eos, docs);
    };

    if (j.is_array()) {
        for (std::size_t i = 0; i < j.size(); i++) handle(j[i], i);
    } else {
        handle(j, 0);
    }
    return docs;
}

// ── JSON array — ChatML ───────────────────────────────────────────────────────

static std::size_t process_json_chatml(const fs::path& file,
                                        const BPETokenizer& tok,
                                        BinWriter& writer,
                                        const std::string& system_prompt,
                                        bool add_eos)
{
    std::ifstream f(file);
    if (!f) throw std::runtime_error("cannot open: " + file.string());

    json j;
    try {
        j = json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error("JSON parse error in " + file.string() +
                                 ": " + e.what());
    }

    std::size_t docs = 0;

    auto handle = [&](const json& item, std::size_t idx) {
        if (!item.contains("messages") || !item["messages"].is_array()) {
            std::cerr << "[warn] item " << idx << ": no 'messages' array\n";
            return;
        }
        std::string text = render_chatml(item["messages"], system_prompt);
        flush_doc(text, tok, writer, add_eos, docs);
    };

    if (j.is_array()) {
        for (std::size_t i = 0; i < j.size(); i++) handle(j[i], i);
    } else {
        handle(j, 0);
    }
    return docs;
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

static std::size_t process_file(const fs::path& file,
                                 const Args& args,
                                 const BPETokenizer& tok,
                                 BinWriter& writer)
{
    std::string fmt = args.format;
    if (fmt == "auto") {
        auto ext = file.extension().string();
        if      (ext == ".jsonl") fmt = "jsonl";
        else if (ext == ".json")  fmt = "json";
        else                      fmt = "txt";
    }

    if (args.chatml) {
        if (fmt == "jsonl") return process_jsonl_chatml(file, tok, writer,
                                                        args.system_prompt, args.add_eos);
        if (fmt == "json")  return process_json_chatml (file, tok, writer,
                                                        args.system_prompt, args.add_eos);
        throw std::runtime_error(
            "--chatml requires .jsonl or .json input, got: " + file.string());
    }

    if (fmt == "jsonl") return process_jsonl(file, tok, writer, args.field, args.add_eos);
    if (fmt == "json")  return process_json (file, tok, writer, args.field, args.add_eos);
    return                     process_txt  (file, tok, writer, args.add_eos);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args = parse(argc, argv);

    try {
        // 1. Tokenizer
        std::cerr << "[dataset-format] tokenizer: " << args.tokenizer << "\n";
        auto tok = BPETokenizer::load(args.tokenizer);

        // 2. Output
        fs::create_directories(args.output);
        fs::path out_path = fs::path(args.output) / "train.bin";
        BinWriter writer(out_path);

        // 3. Collect input files
        fs::path inp(args.input);
        std::vector<fs::path> files;

        if (fs::is_regular_file(inp)) {
            files.push_back(inp);
        } else if (fs::is_directory(inp)) {
            for (auto& e : fs::recursive_directory_iterator(inp)) {
                if (!e.is_regular_file()) continue;
                auto ext = e.path().extension().string();
                if (ext == ".txt" || ext == ".jsonl" || ext == ".json")
                    files.push_back(e.path());
            }
            std::sort(files.begin(), files.end());
        } else {
            throw std::runtime_error("input path not found: " + args.input);
        }

        if (files.empty())
            throw std::runtime_error(
                "no .txt / .jsonl / .json files found in: " + args.input);

        std::cerr << "[dataset-format] " << files.size() << " file(s) to process\n";
        if (args.chatml)
            std::cerr << "[dataset-format] mode: ChatML  system=\""
                      << args.system_prompt << "\"\n";
        if (args.add_eos)
            std::cerr << "[dataset-format] appending eos=" << tok.eos_id()
                      << " between documents\n";

        // 4. Process
        std::size_t total_docs = 0;

        for (auto& f : files) {
            std::size_t before = writer.total();
            std::size_t docs   = process_file(f, args, tok, writer);
            std::cerr << "[dataset-format] " << f.filename().string()
                      << "  docs=" << docs
                      << "  tokens=" << (writer.total() - before) << "\n";
            total_docs += docs;
        }

        writer.flush();

        std::size_t total_tokens = writer.total();
        std::cerr << "\n[dataset-format] done\n"
                  << "  documents : " << total_docs                              << "\n"
                  << "  tokens    : " << total_tokens                             << "\n"
                  << "  output    : " << out_path.string()                        << "\n"
                  << "  size      : " << (total_tokens * 4 / (1024 * 1024)) << " MB\n";

    } catch (const std::exception& e) {
        std::cerr << "[dataset-format] error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}