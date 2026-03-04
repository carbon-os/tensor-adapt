#pragma once

#include <tensor/adapter/adapter_config.hpp>
#include <tensor/adapter/adapter_model.hpp>
#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/device.hpp>
#include <tensor/data/dataset.hpp>

#include <cstddef>
#include <string>

namespace tensor::trainer {

// ─────────────────────────────────────────────────────────────
//  StepMetrics — returned from each training step
// ─────────────────────────────────────────────────────────────

struct StepMetrics {
    std::size_t step;
    std::size_t tokens_consumed;
    float       loss;
    float       learning_rate;
    float       grad_norm;
    double      step_ms;        // wall-clock time for this step (ms)
    double      ema_ms;         // exponential moving average ms/step
    double      tokens_per_sec; // derived from ema_ms
};

// ─────────────────────────────────────────────────────────────
//  AdaptOptions — operational settings only
// ─────────────────────────────────────────────────────────────

struct AdaptOptions {
    std::string  domain;           // e.g. "golang/gin"
    std::string  output_dir;
    std::size_t  checkpoint_every = 1000;
    std::string  resume_from;      // "" = start fresh
    std::string  device            = "cuda:0";
    uint64_t     seed              = 42;
    bool         log_to_stdout     = true;
    std::size_t  tokens            = 50'000'000;
};

// ─────────────────────────────────────────────────────────────
//  AdaptTrainer
// ─────────────────────────────────────────────────────────────

class AdaptTrainer {
public:
    static AdaptTrainer create(
        const base::FrozenBase&       base,
        const adapter::AdapterConfig& cfg,
        data::Dataset&                dataset,
        const AdaptOptions&           opts,
        const backend::cuda::Device&  dev);

    void run();

    bool         done() const;
    StepMetrics  step();

    void save_checkpoint(const std::string& dir);
    void save_adapter   (const std::string& dir);

    // Destructor declared here, defined in adapt_trainer.cu where Impl is complete.
    ~AdaptTrainer();

    // Move only — unique_ptr<Impl> is not copyable.
    AdaptTrainer(AdaptTrainer&&) noexcept;
    AdaptTrainer& operator=(AdaptTrainer&&) noexcept;
    AdaptTrainer(const AdaptTrainer&)            = delete;
    AdaptTrainer& operator=(const AdaptTrainer&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit AdaptTrainer(std::unique_ptr<Impl>);
};

} // namespace tensor::trainer