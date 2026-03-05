// adapter_config.hpp
#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/device.hpp>
#include <cstddef>
#include <string>

namespace tensor::adapter {

// ─────────────────────────────────────────────────────────────
//  AdapterConfig
//
//  All fields have sane defaults. The normal construction path is:
//
//    auto cfg = AdapterConfig::for_base(base, dev);   // auto-heuristics
//    apply_overrides(cfg, cli_args);                  // CLI wins
//
//  Fields marked [auto] are set by for_base() based on model parameter
//  count and available VRAM. Any field can be overridden after the fact
//  — the training loop reads only from the final struct, so order of
//  assignment doesn't matter.
// ─────────────────────────────────────────────────────────────

struct AdapterConfig {

    // ── LoRA capacity ─────────────────────────────────────────
    // [auto] rank ladder: <200M→2, <800M→4, <2B→8, <8B→16, <20B→32, else→64
    int         rank        = 16;
    float       alpha       = 16.f;   // effective scale = alpha / rank
    float scale() const { return alpha / (float)rank; }

    // Which attention projections to inject LoRA into.
    // [auto] Q/K/V/O always on; up/down only for models ≥ 2B.
    bool inject_q    = true;
    bool inject_k    = true;
    bool inject_v    = true;
    bool inject_o    = true;
    bool inject_up   = false;
    bool inject_down = false;

    // ── Optimiser ─────────────────────────────────────────────
    // [auto] lr=2e-4; warmup scales with model size (100/200/500 steps)
    float       lr           = 2e-4f;
    float       grad_clip    = 1.f;
    std::size_t warmup_steps = 200;

    // AdamW moment / regularisation — rarely need changing, but exposed
    // so experiments can pin them from the CLI if needed.
    float       adam_beta1   = 0.9f;
    float       adam_beta2   = 0.999f;
    float       adam_eps     = 1e-8f;
    float       weight_decay = 0.f;

    // ── Batch / sequence ──────────────────────────────────────
    // [auto] VRAM-tiered: see adapter_config.cpp for tier table.
    std::size_t batch_size   = 4;
    std::size_t seq_len      = 2048;

    // ── Metadata (filled by for_base, not overrideable) ───────
    std::string architecture;
    std::size_t base_parameters = 0;

    // ── Factory ───────────────────────────────────────────────
    // Queries free VRAM *after* the base model is loaded so the
    // memory budget reflects activations, not weights.
    static AdapterConfig for_base(
        const base::FrozenBase&       base,
        const backend::cuda::Device&  dev);
};

} // namespace tensor::adapter