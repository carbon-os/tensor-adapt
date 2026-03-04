#pragma once

#include <tensor/parser/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tensor::parser {

// ─────────────────────────────────────────────────────────────
//  ModelConfig — typed view over config.json
// ─────────────────────────────────────────────────────────────

class ModelConfig {
public:
    static ModelConfig from_file(const std::string& path);
    static ModelConfig from_dir (const std::string& dir);   // reads <dir>/config.json

    // Common fields — all models
    std::string architecture()             const;  // e.g. "LlamaForCausalLM"
    std::string model_type()               const;  // e.g. "llama"
    std::string torch_dtype()              const;  // e.g. "bfloat16"
    std::size_t vocab_size()               const;
    std::size_t hidden_size()              const;
    std::size_t num_hidden_layers()        const;
    std::size_t num_attention_heads()      const;
    std::size_t num_key_value_heads()      const;
    std::size_t intermediate_size()        const;
    std::size_t max_position_embeddings()  const;
    float       rms_norm_eps()             const;
    float       rope_theta()               const;
    int32_t     bos_token_id()             const;
    int32_t     eos_token_id()             const;

    // RoPE scaling fields
    std::string rope_scaling_type()                             const;
    float       rope_scaling_factor()                           const;
    float       rope_scaling_low_freq_factor()                  const;
    float       rope_scaling_high_freq_factor()                 const;
    int         rope_scaling_original_max_position_embeddings() const;

    // Raw JSON field access for architecture-specific keys.
    std::string get_string(const std::string& key, const std::string& def = "") const;
    std::size_t get_size  (const std::string& key, std::size_t        def = 0)  const;
    float       get_float (const std::string& key, float              def = 0)  const;
    int64_t     get_int   (const std::string& key, int64_t            def = 0)  const;
    bool        get_bool  (const std::string& key, bool               def = false) const;

    ModelConfig(ModelConfig&&) noexcept            = default;
    ModelConfig& operator=(ModelConfig&&) noexcept = default;
    ModelConfig(const ModelConfig&)                = delete;
    ModelConfig& operator=(const ModelConfig&)     = delete;
    ~ModelConfig();

private:
    struct Impl;
    explicit ModelConfig(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

// ─────────────────────────────────────────────────────────────
//  TokenizerConfig — typed view over tokenizer_config.json
// ─────────────────────────────────────────────────────────────

class TokenizerConfig {
public:
    static TokenizerConfig from_file(const std::string& path);
    static TokenizerConfig from_dir (const std::string& dir);

    std::string tokenizer_class()  const;
    std::string chat_template()    const;
    int32_t     bos_token_id()     const;
    int32_t     eos_token_id()     const;
    int32_t     pad_token_id()     const;
    bool        add_bos_token()    const;
    bool        add_eos_token()    const;

    TokenizerConfig(TokenizerConfig&&) noexcept            = default;
    TokenizerConfig& operator=(TokenizerConfig&&) noexcept = default;
    TokenizerConfig(const TokenizerConfig&)                = delete;
    TokenizerConfig& operator=(const TokenizerConfig&)     = delete;
    ~TokenizerConfig();

private:
    struct Impl;
    explicit TokenizerConfig(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

} // namespace tensor::parser