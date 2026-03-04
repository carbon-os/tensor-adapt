#pragma once

#include <tensor/adapter/adapter_model.hpp>
#include <tensor/backend/cuda/device.hpp>

namespace tensor::trainer {

// ─────────────────────────────────────────────────────────────
//  AdamW — adapter-specific settings (no weight decay,
//  tighter beta2 for narrow adapter signal).
// ─────────────────────────────────────────────────────────────

struct AdamW {
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float weight_decay = 0.f;

    explicit AdamW(const adapter::AdapterConfig& cfg)
        : beta1(cfg.adam_beta1)
        , beta2(cfg.adam_beta2)
        , weight_decay(cfg.weight_decay)
    {}

    // Update one LoraPair using its accumulated F32 gradients.
    // step: global step count (1-based, for bias correction).
    void step(
        adapter::LoraPair&            lp,
        float                         lr,
        int                           step,
        const backend::cuda::Device&  dev) const;

    // Step all LoRA parameters across all layers.
    void step_all(
        adapter::AdapterModel&       model,
        float                        lr,
        int                          step,
        const backend::cuda::Device& dev) const;
};

} // namespace tensor::trainer