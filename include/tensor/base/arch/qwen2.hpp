// qwen2.hpp
#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <vector>

namespace tensor::base::arch {

using backend::cuda::Tensor;
using backend::cuda::Device;

// ─────────────────────────────────────────────────────────────
//  Per-layer activations saved for the backward pass.
// ─────────────────────────────────────────────────────────────

struct Qwen2LayerCache {
    // ── Attention block ───────────────────────────────────────
    Tensor pre_attn_res;   // BF16 [BT, H]      residual before input layernorm
    Tensor in_norm_rms;    // F32  [BT]          rsqrt(mean(x²)+ε) for rms_norm_bwd
    Tensor x_normed;       // BF16 [BT, H]       post-norm input to Q/K/V projections

    Tensor Q, K, V;        // BF16               post-projection, post-bias, pre-RoPE
    Tensor attn_out;       // BF16 [BT, Hq*hd]   output of SDPA
    Tensor attn_w;         // F32  [B, Hq, T, T]  saved weights for sdpa_bwd
    Tensor h_mid;          // BF16 [BT, H]        output of o_proj

    // ── FFN block ─────────────────────────────────────────────
    Tensor pre_ffn_res;    // BF16 [BT, H]        residual before post-attn layernorm
    Tensor post_norm_rms;  // F32  [BT]
    Tensor x_normed2;      // BF16 [BT, H]        post-norm input to gate/up projections

    Tensor gate_out;       // BF16 [BT, I]
    Tensor up_out;         // BF16 [BT, I]
    Tensor act_out;        // BF16 [BT, I]        silu(gate) * up
    Tensor ffn_out;        // BF16 [BT, H]        output of down_proj
};

struct Qwen2ForwardResult {
    Tensor embed_in;                      // BF16 [BT, H]
    std::vector<Qwen2LayerCache> layer_cache;

    Tensor x_final;                       // BF16 [BT, H]  residual before final norm
    Tensor final_rms;                     // F32  [BT]

    Tensor logits;                        // BF16 [BT, V]
    Tensor dlogits;                       // BF16 [BT, V]  gradient buffer for CE
};

// ─────────────────────────────────────────────────────────────
//  Qwen2Base — stateless forward/backward
// ─────────────────────────────────────────────────────────────

class Qwen2Base {
public:
    // ─────────────────────────────────────────────────────────
    //  LayerGrads — gradients needed by the LoRA trainer.
    //
    //  For correct LoRA gradient computation, apply_lora_bwd needs
    //  the gradient w.r.t. each projection's OUTPUT, not its input.
    //
    //  Projection input grads (dx_attn_in, dx_ffn_in) flow back through
    //  the normalization layers and are used to build the residual gradient
    //  chain inside Qwen2Base::backward — they are not the right signal for
    //  LoRA weight updates.
    //
    //  Correct mapping:
    //    lora_q backward:  x = x_normed,  grad_out = dQ
    //    lora_k backward:  x = x_normed,  grad_out = dK
    //    lora_v backward:  x = x_normed,  grad_out = dV
    //    lora_o backward:  x = attn_out,  grad_out = do_proj
    // ─────────────────────────────────────────────────────────
    struct LayerGrads {
        // Gradients w.r.t. projection OUTPUTS — correct LoRA upstream grads.
        // dQ and dK are post-RoPE-backward (i.e., they correspond to the
        // pre-RoPE Q/K tensors, matching what A_q and A_k were applied to).
        Tensor dQ;          // BF16 [BT, Hq*hd]
        Tensor dK;          // BF16 [BT, Hkv*hd]
        Tensor dV;          // BF16 [BT, Hkv*hd]
        Tensor do_proj;     // BF16 [BT, H]        grad w.r.t. h_mid (o_proj output)

        // Gradients w.r.t. normed inputs — used by centroid accumulator.
        Tensor dx_attn_in;  // BF16 [BT, H]        grad w.r.t. x_normed
        Tensor dx_ffn_in;   // BF16 [BT, H]        grad w.r.t. x_normed2
    };

    static Qwen2ForwardResult forward(
        const FrozenBase& base,
        const Tensor&     tokens,
        int B, int T,
        const Device&     dev);

    static std::vector<LayerGrads> backward(
        const FrozenBase&         base,
        const Qwen2ForwardResult& fwd,
        int B, int T,
        const Device&             dev);
};

} // namespace tensor::base::arch