#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/device.hpp>
#include <tensor/backend/cuda/tensor.hpp>

#include <cstddef>
#include <vector>

namespace tensor::base::arch {

using backend::cuda::Device;
using backend::cuda::Tensor;

// ─────────────────────────────────────────────────────────────
//  Per-layer forward cache — saved activations for backward.
// ─────────────────────────────────────────────────────────────

struct Qwen2LayerCache {
    // Attention block
    Tensor in_norm_rms;   // [B*T] — RMS values from input layernorm
    Tensor x_normed;      // [B, T, H]     — input after layernorm
    Tensor Q, K, V;       // [B, T, H, hd] — after projection + RoPE (Q/K)
    Tensor attn_out;      // [B, T, H, hd] — attention output (pre o_proj)
    Tensor attn_w;        // [B, Hq, T, T] — saved attention weights
    Tensor h_mid;         // [B*T, H]      — after o_proj, before residual

    // FFN block
    Tensor post_norm_rms; // [B*T]
    Tensor x_normed2;     // [B*T, H]
    Tensor gate_out;      // [B*T, I]
    Tensor up_out;        // [B*T, I]
    Tensor act_out;       // [B*T, I]  — silu(gate) * up
    Tensor ffn_out;       // [B*T, H]  — after down_proj

    // LoRA injection points (inputs to Q/K/V/O projections)
    // These are slices or aliases into x_normed / attn_out —
    // stored separately for clarity in the backward pass.
};

struct Qwen2ForwardResult {
    Tensor                      logits;       // [B*T, V]
    Tensor                      loss_d;       // scalar F32 (mean loss)
    Tensor                      dlogits;      // [B*T, V]  — gradient of loss w.r.t. logits
    Tensor                      final_rms;    // [B*T]     — saved from final norm
    std::vector<Qwen2LayerCache> layer_cache;
    Tensor                      embed_in;     // [B*T, H]  — embedded tokens
};

// ─────────────────────────────────────────────────────────────
//  Qwen2Base — forward and backward for Qwen2.5
// ─────────────────────────────────────────────────────────────

class Qwen2Base {
public:
    // Forward pass. tokens: [B, T] int32 on device.
    // Stores all activations needed for backward in result.layer_cache.
    static Qwen2ForwardResult forward(
        const FrozenBase& base,
        const Tensor&     tokens,     // [B*T] int32
        int B, int T,
        const Device&     dev);

    // Backward pass given dlogits (from cross-entropy).
    // Returns gradients for LoRA injection points in each layer.
    // grad_in[l] = {dQ_in, dK_in, dV_in, dO_in, dFFN_gate_in, dFFN_up_in}
    // which are the gradients w.r.t. the inputs to each LoRA-injected projection.
    struct LayerGrads {
        Tensor dx_attn_in;  // grad w.r.t. input of q/k/v projections [B*T, H]
        Tensor dx_o_in;     // grad w.r.t. input of o_proj [B*T, H]
        Tensor dx_ffn_in;   // grad w.r.t. input of gate/up projections [B*T, H]
    };

    static std::vector<LayerGrads> backward(
        const FrozenBase&         base,
        const Qwen2ForwardResult& fwd,
        int B, int T,
        const Device&             dev);
};

} // namespace tensor::base::arch