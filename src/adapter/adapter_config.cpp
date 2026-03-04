// adapter_config.cpp
#include <tensor/adapter/adapter_config.hpp>
#include <tensor/base/frozen_base.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <cstddef>

namespace tensor::adapter {

// ── VRAM tier selection ───────────────────────────────────────
//
// Called after the base model is already loaded on the device, so
// cudaMemGetInfo returns the budget genuinely available for activations.
//
// The dominant allocation per training step is the attention weight
// cache stored for the backward pass:
//   attn_w per layer = B * Hq * T * T * sizeof(float)
//   total            = num_layers * B * Hq * T * T * 4 bytes
//
// Other per-step activations (normed x, Q/K/V, FFN intermediates) are
// significant but roughly linear in B*T, so T² dominates.
//
// Tier table (free VRAM after model load):
//
//   < 1 GB  → batch=1  seq=128   attn_cache ≈  0.02 GB (24 layers, Hq=14)
//   1–2 GB  → batch=1  seq=256   attn_cache ≈  0.09 GB
//   2–4 GB  → batch=1  seq=512   attn_cache ≈  0.36 GB
//   4–8 GB  → batch=2  seq=1024  attn_cache ≈  2.9  GB
//   8–16 GB → batch=4  seq=1024  attn_cache ≈  5.7  GB
//   16+ GB  → batch=8  seq=2048  attn_cache ≈ 45    GB  (A100/H100)
//
// Estimates above assume Qwen2.5-0.5B (Hq=14, 24 layers).
// Larger models with more heads will hit the ceiling at lower seq_len —
// the tiers are conservative enough that they hold across model sizes
// up to ~7B within each VRAM band.

struct VramTier {
    std::size_t min_free_bytes;
    std::size_t batch;
    std::size_t seq;
};

static const VramTier TIERS[] = {
    { 16ULL * 1024 * 1024 * 1024,  8, 2048 },  // 16 GB+
    {  8ULL * 1024 * 1024 * 1024,  4, 1024 },  //  8 GB
    {  4ULL * 1024 * 1024 * 1024,  2, 1024 },  //  4 GB
    {  2ULL * 1024 * 1024 * 1024,  1,  512 },  //  2 GB
    {  1ULL * 1024 * 1024 * 1024,  1,  256 },  //  1 GB
    {                           0,  1,  128 },  // fallback
};

AdapterConfig AdapterConfig::for_base(
    const base::FrozenBase&      b,
    const backend::cuda::Device& dev)
{
    const base::BaseConfig& bc = b.config;

    // ── Base architecture config (rank ladder) ────────────────

    std::size_t params =
        bc.vocab_size * bc.hidden_size +
        bc.num_layers * (
            4 * bc.hidden_size * bc.hidden_size +
            2 * bc.hidden_size * bc.num_kv_heads * bc.head_dim +
            3 * bc.hidden_size * bc.intermediate_size
        );

    AdapterConfig cfg;
    cfg.architecture    = bc.arch_name;
    cfg.base_parameters = params;

    if (params < 200'000'000ULL) {
        cfg.rank = 2; cfg.alpha = 2.f;
        cfg.inject_up = false; cfg.inject_down = false;
    } else if (params < 800'000'000ULL) {
        cfg.rank = 4; cfg.alpha = 4.f;
        cfg.inject_up = false; cfg.inject_down = false;
    } else if (params < 2'000'000'000ULL) {
        cfg.rank = 8;  cfg.alpha = 8.f;
        cfg.inject_up = true; cfg.inject_down = true;
    } else if (params < 8'000'000'000ULL) {
        cfg.rank = 16; cfg.alpha = 16.f;
        cfg.inject_up = true; cfg.inject_down = true;
    } else if (params < 20'000'000'000ULL) {
        cfg.rank = 32; cfg.alpha = 32.f;
        cfg.inject_up = true; cfg.inject_down = true;
    } else {
        cfg.rank = 64; cfg.alpha = 64.f;
        cfg.inject_up = true; cfg.inject_down = true;
    }

    // ── VRAM-based batch / seq scaling ───────────────────────

    cudaSetDevice(dev.id());
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    // Leave a 256 MB headroom for temp GEMM buffers and the emulated
    // BF16 cast allocations on pre-Ampere hardware.
    const std::size_t HEADROOM = 256ULL * 1024 * 1024;
    std::size_t budget = (free_bytes > HEADROOM) ? (free_bytes - HEADROOM) : 0;

    for (const auto& t : TIERS) {
        if (budget >= t.min_free_bytes) {
            cfg.batch_size = t.batch;
            cfg.seq_len    = t.seq;
            break;
        }
    }

    // Warmup scales with model size, not VRAM.
    if (params < 1'000'000'000ULL) {
        cfg.warmup_steps = 100;
    } else if (params < 8'000'000'000ULL) {
        cfg.warmup_steps = 200;
    } else {
        cfg.warmup_steps = 500;
    }

    std::cerr << "[adapter] VRAM free=" << (free_bytes  / 1024 / 1024) << " MB"
              << " total=" << (total_bytes / 1024 / 1024) << " MB"
              << " → batch=" << cfg.batch_size
              << " seq="     << cfg.seq_len << "\n";

    return cfg;
}

} // namespace tensor::adapter