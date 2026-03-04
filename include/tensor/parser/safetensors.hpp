#pragma once

#include <tensor/core/dtype.hpp>
#include <tensor/core/shape.hpp>
#include <tensor/core/tensor_view.hpp>
#include <tensor/parser/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensor::parser {

using core::DType;
using core::Shape;
using core::TensorView;

// ─────────────────────────────────────────────────────────────
//  TensorInfo — header-only descriptor, no data pointer.
//
//  Returned by SafeTensors::info(). Useful when you need
//  shape/dtype/offset metadata without touching any bytes.
// ─────────────────────────────────────────────────────────────

struct TensorInfo {
    DType         dtype;
    Shape         shape;
    std::uint64_t offset_begin; // byte offset relative to start of data buffer
    std::uint64_t offset_end;   // exclusive

    std::size_t rank()  const noexcept { return shape.rank();  }
    std::size_t numel() const noexcept { return shape.numel(); }

    std::size_t nbytes() const noexcept {
        return static_cast<std::size_t>(offset_end - offset_begin);
    }
};

// ─────────────────────────────────────────────────────────────
//  SafeTensors — memory-mapped .safetensors reader
//
//  File layout (little-endian):
//    [8 bytes : uint64  header_len]
//    [header_len bytes : UTF-8 JSON]
//    [remainder : raw weight data]
//
//  JSON structure:
//    {
//      "__metadata__": { "key": "value", ... },
//      "tensor_name":  {
//        "dtype": "BF16",
//        "shape": [151936, 1024],
//        "data_offsets": [begin, end]   <- relative to data buffer start
//      },
//      ...
//    }
//
//  TensorView.data points directly into the mmap region.
//  No weight bytes are ever copied.
//  Supports single files and sharded multi-file models.
// ─────────────────────────────────────────────────────────────

class SafeTensors {
public:
    // ── construction (factory) ───────────────────────────────

    // Open a single .safetensors file.
    static SafeTensors open(const std::string& path);

    // Open a sharded model from an explicit ordered file list.
    // Tensor names must be unique across shards.
    static SafeTensors open(const std::vector<std::string>& paths);

    // Open all .safetensors files in a directory, sorted by filename.
    // Skips any non-.safetensors files.
    static SafeTensors open_dir(const std::string& dir);

    // ── move-only (mmap resources inside) ───────────────────
    //
    // Defined in safetensors.cpp where Impl is complete.
    // = default in the header would instantiate the move at the
    // point of declaration where Impl is still an incomplete type,
    // triggering a sizeof-on-incomplete-type error.

    SafeTensors(SafeTensors&&) noexcept;
    SafeTensors& operator=(SafeTensors&&) noexcept;

    SafeTensors(const SafeTensors&)            = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    // Defined in .cpp where Impl is complete.
    ~SafeTensors();

    // ── metadata ─────────────────────────────────────────────

    // Full __metadata__ map, string -> string.
    const std::unordered_map<std::string, std::string>& metadata() const noexcept;

    // Lookup a single metadata key.
    // Throws MetaNotFound if the key does not exist.
    const std::string& metadata(const std::string& key) const;

    // Lookup with a default — never throws.
    std::string metadata(const std::string& key,
                         const std::string& default_val) const noexcept;

    bool has_metadata(const std::string& key) const noexcept;

    // ── tensor inventory ─────────────────────────────────────

    std::vector<std::string> names() const;
    std::size_t              size()  const noexcept;
    bool                     contains(const std::string& name) const noexcept;

    // Total bytes of all weight data across all shards.
    std::size_t total_bytes() const noexcept;

    // ── tensor access ────────────────────────────────────────

    // Header descriptor only — no data pointer, no mmap touch.
    // Throws TensorNotFound if name is absent.
    const TensorInfo& info(const std::string& name) const;

    // Zero-copy view into mmap region.
    // Throws TensorNotFound if name is absent.
    TensorView tensor(const std::string& name) const;

    // Iterate all tensors as (name, TensorView) pairs.
    std::unordered_map<std::string, TensorView> tensors() const;

private:
    struct Impl;

    // parse_shard is a private static method so it can name Impl freely.
    static void parse_shard(const std::string& path, Impl& impl);

    explicit SafeTensors(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace tensor::parser