// frozen_base.hpp
#pragma once

#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/core/dtype.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace tensor::base {

enum class ArchType { Unknown, Qwen2, LLaMA };

struct BaseConfig {
    std::string arch_name;
    std::string base_sha;
    ArchType    arch             = ArchType::Unknown;
    std::size_t vocab_size       = 0;
    std::size_t hidden_size      = 0;
    std::size_t num_layers       = 0;
    std::size_t num_q_heads      = 0;
    std::size_t num_kv_heads     = 0;
    std::size_t head_dim         = 0;
    std::size_t intermediate_size = 0;
    float       rms_norm_eps     = 1e-6f;
    float       rope_theta       = 10000.f;
    bool        tie_embeddings   = true;
};

struct LayerWeights {
    backend::cuda::Tensor input_norm_w;

    // Attention projections.
    // Q/K/V carry a learned bias (Qwen2 architecture feature for RoPE extrapolation).
    // O projection and all MLP projections are bias-free.
    backend::cuda::Tensor q_proj_w, q_proj_b;  // [Hq*hd, H], [Hq*hd]
    backend::cuda::Tensor k_proj_w, k_proj_b;  // [Hkv*hd, H], [Hkv*hd]
    backend::cuda::Tensor v_proj_w, v_proj_b;  // [Hkv*hd, H], [Hkv*hd]
    backend::cuda::Tensor o_proj_w;             // [H, Hq*hd]

    backend::cuda::Tensor post_norm_w;

    // MLP projections — no bias.
    backend::cuda::Tensor gate_proj_w;          // [I, H]
    backend::cuda::Tensor up_proj_w;            // [I, H]
    backend::cuda::Tensor down_proj_w;          // [H, I]
};

struct FrozenBase {
    BaseConfig                   config;
    backend::cuda::Tensor        embed_w;
    std::vector<LayerWeights>    layers;
    backend::cuda::Tensor        final_norm_w;
    backend::cuda::Tensor        lm_head_w;    // empty when tie_embeddings=true
};

struct BaseError : std::runtime_error {
    explicit BaseError(const std::string& msg)
        : std::runtime_error("BaseError: " + msg) {}
};

} // namespace tensor::base