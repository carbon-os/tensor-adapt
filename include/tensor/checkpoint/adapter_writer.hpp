#pragma once

#include <tensor/adapter/adapter_model.hpp>
#include <tensor/base/frozen_base.hpp>

#include <cstddef>
#include <string>

namespace tensor::checkpoint {

// ─────────────────────────────────────────────────────────────
//  AdapterWriter — writes adapter artifacts to disk
// ─────────────────────────────────────────────────────────────

class AdapterWriter {
public:
    explicit AdapterWriter(const std::string& base_output_dir);

    // Write mid-run checkpoint (includes optimizer state).
    void save_checkpoint(
        const adapter::AdapterModel& model,
        const base::FrozenBase&      base,
        const std::string&           domain,
        std::size_t                  step,
        std::size_t                  tokens,
        const std::string&           ckpt_dir);

    // Write final adapter (no optimizer state).
    void save_final(
        const adapter::AdapterModel& model,
        const base::FrozenBase&      base,
        const std::string&           domain,
        std::size_t                  step,
        std::size_t                  tokens,
        const std::string&           out_dir);

private:
    std::string base_dir_;

    void write_safetensors(
        const adapter::AdapterModel& model,
        const std::string& path);

    void write_adapter_json(
        const adapter::AdapterModel& model,
        const base::FrozenBase&      base,
        const std::string&           domain,
        std::size_t                  step,
        std::size_t                  tokens,
        const std::string&           path);
};

} // namespace tensor::checkpoint