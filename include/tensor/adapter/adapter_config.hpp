// adapter_config.hpp
#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/device.hpp>
#include <cstddef>
#include <string>

namespace tensor::adapter {

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

    std::string architecture;
    std::size_t base_parameters = 0;

    // Derive config from base architecture, then scale batch/seq to fit device VRAM.
    // Device is queried for free memory after the base model is already loaded —
    // so the budget reflects what is actually available for training activations.
    static AdapterConfig for_base(
        const base::FrozenBase&          base,
        const backend::cuda::Device&     dev);

    float scale() const { return alpha / (float)rank; }
};

} // namespace tensor::adapter