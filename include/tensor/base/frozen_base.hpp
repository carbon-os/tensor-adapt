#pragma once

#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/core/dtype.hpp>
#include <tensor/core/shape.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace tensor::base {

using backend::cuda::Tensor;
using backend::cuda::Device;

// ─────────────────────────────────────────────────────────────
//  Architecture type — detected from config.json
// ─────────────────────────────────────────────────────────────

enum class ArchType {
    Qwen2,
    LLaMA,
    Unknown,
};

// ─────────────────────────────────────────────────────────────
//  BaseConfig — typed fields needed by the trainer and adapter
// ─────────────────────────────────────────────────────────────

struct BaseConfig {
    ArchType    arch            = ArchType::Unknown;
    std::string arch_name;          // "qwen2", "llama", etc.
    std::string model_id;           // e.g. "Qwen/Qwen2.5-0.5B"
    std::string base_sha;           // hex digest of weight files

    std::size_t vocab_size         = 0;
    std::size_t hidden_size        = 0;
    std::size_t num_layers         = 0;
    std::size_t num_q_heads        = 0;
    std::size_t num_kv_heads       = 0;
    std::size_t head_dim           = 0;     // hidden_size / num_q_heads
    std::size_t intermediate_size  = 0;
    float       rms_norm_eps       = 1e-6f;
    float       rope_theta         = 1e6f;
    bool        tie_embeddings     = true;  // lm_head shares embed_tokens.weight
};

// ─────────────────────────────────────────────────────────────
//  LayerWeights — GPU tensors for one transformer layer.
//  All weights are BF16, read-only after load.
// ─────────────────────────────────────────────────────────────

struct LayerWeights {
    // Attention norms + projections
    Tensor input_norm_w;   // [hidden]       RMSNorm weight
    Tensor q_proj_w;       // [hidden, hidden]
    Tensor k_proj_w;       // [kv_dim, hidden]
    Tensor v_proj_w;       // [kv_dim, hidden]
    Tensor o_proj_w;       // [hidden, hidden]

    // FFN norm + projections
    Tensor post_norm_w;    // [hidden]
    Tensor gate_proj_w;    // [intermediate, hidden]
    Tensor up_proj_w;      // [intermediate, hidden]
    Tensor down_proj_w;    // [hidden, intermediate]
};

// ─────────────────────────────────────────────────────────────
//  FrozenBase — read-only base model on GPU
// ─────────────────────────────────────────────────────────────

struct FrozenBase {
    BaseConfig               config;
    Tensor                   embed_w;      // [vocab, hidden]
    std::vector<LayerWeights> layers;
    Tensor                   final_norm_w; // [hidden]
    Tensor                   lm_head_w;   // [vocab, hidden] — may alias embed_w

    // Not copyable — GPU memory is exclusive.
    FrozenBase(FrozenBase&&)            = default;
    FrozenBase& operator=(FrozenBase&&) = default;
    FrozenBase(const FrozenBase&)       = delete;
    FrozenBase& operator=(const FrozenBase&) = delete;
    FrozenBase() = default;
};

} // namespace tensor::base