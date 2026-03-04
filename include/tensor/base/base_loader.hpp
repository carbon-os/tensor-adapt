#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <string>

namespace tensor::base {

struct BaseError : std::runtime_error {
    explicit BaseError(const std::string& m)
        : std::runtime_error("BaseError: " + m) {}
};

// ─────────────────────────────────────────────────────────────
//  BaseLoader
//
//  Reads config.json from dir, detects architecture,
//  uploads all weights to device.
//
//  arch_hint:  pass "qwen2" to bypass detection (useful for
//              testing with non-standard config.json keys).
//              Pass "" to detect automatically.
// ─────────────────────────────────────────────────────────────

class BaseLoader {
public:
    static FrozenBase load(
        const std::string&           dir,
        const backend::cuda::Device& dev,
        const std::string&           arch_hint = "");
};

} // namespace tensor::base