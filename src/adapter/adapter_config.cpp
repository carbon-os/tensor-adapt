#include <tensor/adapter/adapter_config.hpp>
#include <tensor/base/frozen_base.hpp>

namespace tensor::adapter {

AdapterConfig AdapterConfig::for_base(const base::FrozenBase& b) {
    const base::BaseConfig& bc = b.config;

    // Estimate parameter count from architecture dims.
    std::size_t params =
        bc.vocab_size * bc.hidden_size +            // embedding
        bc.num_layers * (
            4 * bc.hidden_size * bc.hidden_size +   // Q/O projections
            2 * bc.hidden_size * bc.num_kv_heads * bc.head_dim + // K/V
            3 * bc.hidden_size * bc.intermediate_size            // FFN
        );

    AdapterConfig cfg;
    cfg.architecture    = bc.arch_name;
    cfg.base_parameters = params;

    // Config ladder (README table).
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

    // Scale warmup + batch with model size.
    if (params < 1'000'000'000ULL) {
        cfg.batch_size   = 8;
        cfg.warmup_steps = 100;
    } else if (params < 8'000'000'000ULL) {
        cfg.batch_size   = 4;
        cfg.warmup_steps = 200;
    } else {
        cfg.batch_size   = 2;
        cfg.warmup_steps = 500;
    }

    return cfg;
}

} // namespace tensor::adapter