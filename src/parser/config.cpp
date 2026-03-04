#include <tensor/parser/config.hpp>
#include <tensor/parser/errors.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace tensor::parser {

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────

static json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw ParseError("cannot open: " + path);
    }
    try {
        return json::parse(f);
    } catch (const json::exception& e) {
        throw ParseError(std::string("JSON parse error in ") + path + ": " + e.what());
    }
}

// ─────────────────────────────────────────────────────────────
//  ModelConfig::Impl
// ─────────────────────────────────────────────────────────────

struct ModelConfig::Impl {
    json root;
};

ModelConfig::ModelConfig(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ModelConfig::~ModelConfig() = default;

ModelConfig ModelConfig::from_file(const std::string& path) {
    auto impl = std::make_unique<Impl>();
    impl->root = load_json(path);
    std::cerr << "[DEBUG config] " << path << ":\n"
              << impl->root.dump(2) << "\n";   // ← add this
    return ModelConfig(std::move(impl));
}

ModelConfig ModelConfig::from_dir(const std::string& dir) {
    return from_file((fs::path(dir) / "config.json").string());
}

// ── accessors ────────────────────────────────────────────────

std::string ModelConfig::get_string(const std::string& k, const std::string& def) const {
    if (!impl_->root.contains(k)) return def;
    const auto& v = impl_->root[k];
    if (v.is_string()) return v.get<std::string>();
    return def;
}

std::size_t ModelConfig::get_size(const std::string& k, std::size_t def) const {
    if (!impl_->root.contains(k)) return def;
    const auto& v = impl_->root[k];
    if (v.is_number()) return v.get<std::size_t>();
    return def;
}

float ModelConfig::get_float(const std::string& k, float def) const {
    if (!impl_->root.contains(k)) return def;
    const auto& v = impl_->root[k];
    if (v.is_number()) return v.get<float>();
    return def;
}

int64_t ModelConfig::get_int(const std::string& k, int64_t def) const {
    if (!impl_->root.contains(k)) return def;
    const auto& v = impl_->root[k];
    if (v.is_number()) return v.get<int64_t>();
    return def;
}

bool ModelConfig::get_bool(const std::string& k, bool def) const {
    if (!impl_->root.contains(k)) return def;
    const auto& v = impl_->root[k];
    if (v.is_boolean()) return v.get<bool>();
    return def;
}

std::string ModelConfig::architecture() const {
    if (impl_->root.contains("architectures") && impl_->root["architectures"].is_array()
        && !impl_->root["architectures"].empty()) {
        const auto& first = impl_->root["architectures"][0];
        if (first.is_string()) return first.get<std::string>();
    }
    return get_string("model_type");
}

std::string ModelConfig::model_type()              const { return get_string("model_type"); }
std::string ModelConfig::torch_dtype()             const { return get_string("torch_dtype", "float32"); }
std::size_t ModelConfig::vocab_size()              const { return get_size("vocab_size", 32000); }
std::size_t ModelConfig::hidden_size()             const { return get_size("hidden_size", 4096); }
std::size_t ModelConfig::num_hidden_layers()       const { return get_size("num_hidden_layers", 32); }
std::size_t ModelConfig::num_attention_heads()     const { return get_size("num_attention_heads", 32); }
std::size_t ModelConfig::intermediate_size()       const { return get_size("intermediate_size", 11008); }
std::size_t ModelConfig::max_position_embeddings() const { return get_size("max_position_embeddings", 4096); }
float       ModelConfig::rms_norm_eps()            const { return get_float("rms_norm_eps", 1e-5f); }
float       ModelConfig::rope_theta()              const { return get_float("rope_theta", 10000.0f); }

// Llama-3.x ships eos_token_id as an array e.g. [128001, 128008, 128009].
// Return the first element in that case; fall back to the scalar default otherwise.
int32_t ModelConfig::bos_token_id() const {
    if (!impl_->root.contains("bos_token_id")) return 1;
    const auto& v = impl_->root["bos_token_id"];
    if (v.is_number())                         return v.get<int32_t>();
    if (v.is_array() && !v.empty() && v[0].is_number())
                                               return v[0].get<int32_t>();
    return 1;
}

int32_t ModelConfig::eos_token_id() const {
    if (!impl_->root.contains("eos_token_id")) return 2;
    const auto& v = impl_->root["eos_token_id"];
    if (v.is_number())                         return v.get<int32_t>();
    if (v.is_array() && !v.empty() && v[0].is_number())
                                               return v[0].get<int32_t>();
    return 2;
}

std::size_t ModelConfig::num_key_value_heads() const {
    return get_size("num_key_value_heads", num_attention_heads());
}

// ── rope_scaling nested block ─────────────────────────────────
//
// config.json encodes this as:
//   "rope_scaling": { "rope_type": "llama3", "factor": 32.0, ... }
//
// All helpers return the given default if the key or sub-key is absent.

std::string ModelConfig::rope_scaling_type() const {
    if (!impl_->root.contains("rope_scaling")) return "";
    const auto& rs = impl_->root["rope_scaling"];
    if (rs.contains("rope_type") && rs["rope_type"].is_string())
        return rs["rope_type"].get<std::string>();
    return "";
}

float ModelConfig::rope_scaling_factor() const {
    if (!impl_->root.contains("rope_scaling")) return 1.0f;
    const auto& rs = impl_->root["rope_scaling"];
    if (rs.contains("factor") && rs["factor"].is_number())
        return rs["factor"].get<float>();
    return 1.0f;
}

float ModelConfig::rope_scaling_low_freq_factor() const {
    if (!impl_->root.contains("rope_scaling")) return 1.0f;
    const auto& rs = impl_->root["rope_scaling"];
    if (rs.contains("low_freq_factor") && rs["low_freq_factor"].is_number())
        return rs["low_freq_factor"].get<float>();
    return 1.0f;
}

float ModelConfig::rope_scaling_high_freq_factor() const {
    if (!impl_->root.contains("rope_scaling")) return 4.0f;
    const auto& rs = impl_->root["rope_scaling"];
    if (rs.contains("high_freq_factor") && rs["high_freq_factor"].is_number())
        return rs["high_freq_factor"].get<float>();
    return 4.0f;
}

int ModelConfig::rope_scaling_original_max_position_embeddings() const {
    if (!impl_->root.contains("rope_scaling")) return 8192;
    const auto& rs = impl_->root["rope_scaling"];
    if (rs.contains("original_max_position_embeddings") &&
        rs["original_max_position_embeddings"].is_number())
        return rs["original_max_position_embeddings"].get<int>();
    return 8192;
}

// ─────────────────────────────────────────────────────────────
//  TokenizerConfig::Impl
// ─────────────────────────────────────────────────────────────

struct TokenizerConfig::Impl { json root; };

TokenizerConfig::TokenizerConfig(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TokenizerConfig::~TokenizerConfig() = default;

TokenizerConfig TokenizerConfig::from_file(const std::string& path) {
    auto impl = std::make_unique<Impl>();
    impl->root = load_json(path);
    return TokenizerConfig(std::move(impl));
}

TokenizerConfig TokenizerConfig::from_dir(const std::string& dir) {
    return from_file((fs::path(dir) / "tokenizer_config.json").string());
}

static int32_t tok_id_field(const json& root, const std::string& key) {
    if (!root.contains(key)) return -1;
    const auto& v = root[key];
    if (v.is_number())  return v.get<int32_t>();
    if (v.is_object() && v.contains("content")) {
        // HF sometimes wraps special tokens as {"content": "<|...|>", ...}
        return -1; // id not directly available — resolved by tokenizer
    }
    return -1;
}

std::string TokenizerConfig::tokenizer_class() const {
    if (!impl_->root.contains("tokenizer_class")) return "";
    const auto& v = impl_->root["tokenizer_class"];
    if (v.is_string()) return v.get<std::string>();
    return "";
}

std::string TokenizerConfig::chat_template() const {
    if (!impl_->root.contains("chat_template")) return "";
    const auto& v = impl_->root["chat_template"];

    // Old format: single Jinja string
    if (v.is_string()) return v.get<std::string>();

    // New HF format: array of { "name": "...", "template": "..." }
    if (v.is_array()) {
        // 1. Prefer the entry named "default"
        for (const auto& entry : v) {
            if (entry.is_object() &&
                entry.value("name", std::string{}) == "default")
            {
                return entry.value("template", std::string{});
            }
        }
        // 2. Fall back to the first entry with a template key
        if (!v.empty() && v[0].is_object()) {
            return v[0].value("template", std::string{});
        }
    }

    return "";
}

int32_t TokenizerConfig::bos_token_id() const { return tok_id_field(impl_->root, "bos_token_id"); }
int32_t TokenizerConfig::eos_token_id() const { return tok_id_field(impl_->root, "eos_token_id"); }
int32_t TokenizerConfig::pad_token_id() const { return tok_id_field(impl_->root, "pad_token_id"); }

bool TokenizerConfig::add_bos_token() const {
    if (!impl_->root.contains("add_bos_token")) return true;
    const auto& v = impl_->root["add_bos_token"];
    if (v.is_boolean()) return v.get<bool>();
    return true;
}

bool TokenizerConfig::add_eos_token() const {
    if (!impl_->root.contains("add_eos_token")) return false;
    const auto& v = impl_->root["add_eos_token"];
    if (v.is_boolean()) return v.get<bool>();
    return false;
}

} // namespace tensor::parser