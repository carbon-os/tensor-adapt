#pragma once

#include <tensor/adapter/adapter_config.hpp>
#include <tensor/base/frozen_base.hpp>
#include <tensor/base/arch/qwen2.hpp>
#include <tensor/backend/cuda/device.hpp>
#include <tensor/backend/cuda/tensor.hpp>

#include <vector>
#include <string>

namespace tensor::adapter {

using backend::cuda::Tensor;
using backend::cuda::Device;

// ─────────────────────────────────────────────────────────────
//  LoRA parameter pair — A and B matrices for one projection.
//
//  Working copy: BF16 (used in forward/backward).
//  Master copy:  F32  (used in AdamW update).
//  Moments:      F32  (m and v for AdamW).
//  Grad:         F32  (accumulated gradient).
// ─────────────────────────────────────────────────────────────

struct LoraPair {
    Tensor A_bf16, B_bf16;   // BF16 working weights
    Tensor A_f32,  B_f32;    // F32 master weights
    Tensor mA, vA;           // AdamW moments for A
    Tensor mB, vB;           // AdamW moments for B
    Tensor gA, gB;           // F32 accumulated gradients

    int in_dim  = 0;
    int out_dim = 0;
    int rank    = 0;
};

// ─────────────────────────────────────────────────────────────
//  Per-layer adapter state
// ─────────────────────────────────────────────────────────────

struct LayerAdapter {
    // Q, K, V, O always injected (for 0.5B range).
    LoraPair lora_q;
    LoraPair lora_k;
    LoraPair lora_v;
    LoraPair lora_o;
};

// ─────────────────────────────────────────────────────────────
//  AdapterModel
// ─────────────────────────────────────────────────────────────

class AdapterModel {
public:
    static AdapterModel create(
        const base::FrozenBase& base,
        const AdapterConfig&    cfg,
        const Device&           dev);

    // Forward delta for one layer's LoRA injection.
    // Adds scale * B*(A*x) into existing projection output.
    void apply_lora_fwd(
        std::size_t  layer,
        LoraPair&    lp,
        const Tensor& x,        // [BT, in_dim]
        Tensor&       out,      // [BT, out_dim] — modified in-place
        int BT,
        const Device& dev);

    // Backward through LoRA: accumulate gradients into lp.gA, lp.gB.
    // Returns grad w.r.t. x (to pass backward through previous ops).
    void apply_lora_bwd(
        LoraPair&     lp,
        const Tensor& x,        // [BT, in_dim]  — saved from forward
        const Tensor& grad_out, // [BT, out_dim] — upstream gradient
        Tensor&       grad_x,   // [BT, in_dim]  — accumulated into (beta=1)
        int BT,
        const Device& dev);

    // Zero all gradient accumulators.
    void zero_grad(const Device& dev);

    // Total trainable parameter count.
    std::size_t param_count() const;

    const AdapterConfig& config() const { return cfg_; }

    std::vector<LayerAdapter> layers;

private:
    AdapterConfig cfg_;

    static LoraPair make_pair(int in_dim, int out_dim, int rank, const Device& dev);
};

} // namespace tensor::adapter