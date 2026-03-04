#pragma once

#include <tensor/base/frozen_base.hpp>
#include <cstddef>
#include <string>

namespace tensor::adapter {

// ─────────────────────────────────────────────────────────────
//  AdapterConfig — derived from base, locked per model class.
//  See README config ladder.
// ─────────────────────────────────────────────────────────────

struct AdapterConfig {
    int         rank         = 16;
    float       alpha        = 16.f;
    bool        inject_q     = true;
    bool        inject_k     = true;
    bool        inject_v     = true;
    bool        inject_o     = true;
    bool        inject_up    = false;
    bool        inject_down  = false;

    float       lr           = 2e-4f;
    float       adam_beta1   = 0.9f;
    float       adam_beta2   = 0.999f;
    float       weight_decay = 0.f;
    float       grad_clip    = 1.f;
    std::size_t batch_size   = 4;
    std::size_t seq_len      = 2048;
    std::size_t warmup_steps = 200;

    std::string architecture;       // "qwen2", "llama", ...
    std::size_t base_parameters = 0;

    static AdapterConfig for_base(const base::FrozenBase& base);

    float scale() const { return alpha / (float)rank; }
};

} // namespace tensor::adapter