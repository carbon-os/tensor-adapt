#pragma once

#include <tensor/core/tensor_view.hpp>
#include <tensor/parser/safetensors.hpp>
#include <tensor/parser/errors.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensor::parser {

using core::TensorView;

// ─────────────────────────────────────────────────────────────
//  WeightMap — unified weight access, format-agnostic.
//
//  Everything above the parser layer (models, inference) talks
//  only to WeightMap. Format is decided once at load time and
//  never leaks upward.
//
//  Currently wraps SafeTensors. GGUF support comes next.
// ─────────────────────────────────────────────────────────────

class WeightMap {
public:
    // ── factory ──────────────────────────────────────────────

    // Auto-detect format from directory.
    // Checks for model.safetensors, sharded .safetensors, or .gguf.
    static WeightMap open(const std::string& dir);

    // Explicit safetensors: single file, shard list, or directory.
    static WeightMap from_safetensors(const std::string& path);
    static WeightMap from_safetensors(const std::vector<std::string>& paths);

    // ── tensor access ─────────────────────────────────────────

    bool        contains(const std::string& name) const noexcept;
    std::size_t size()        const noexcept;
    std::size_t total_bytes() const noexcept;

    std::vector<std::string> names() const;

    // Zero-copy view into mmap region.
    // Throws TensorNotFound if name is absent.
    TensorView tensor(const std::string& name) const;

    std::unordered_map<std::string, TensorView> tensors() const;

    // Move-only
    WeightMap(WeightMap&&) noexcept            = default;
    WeightMap& operator=(WeightMap&&) noexcept = default;
    WeightMap(const WeightMap&)                = delete;
    WeightMap& operator=(const WeightMap&)     = delete;
    ~WeightMap();

private:
    struct Impl;
    explicit WeightMap(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

} // namespace tensor::parser