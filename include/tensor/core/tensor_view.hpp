#pragma once

#include <tensor/core/dtype.hpp>
#include <tensor/core/shape.hpp>

#include <cstddef>
#include <stdexcept>

namespace tensor::core {

// ─────────────────────────────────────────────────────────────
//  Span — minimal C++17-compatible non-owning view.
//  Replaces std::span which requires C++20 and is unavailable
//  in CUDA device-code translation units.
// ─────────────────────────────────────────────────────────────

template<typename T>
struct Span {
    const T*    ptr = nullptr;
    std::size_t len = 0;

    const T* begin() const noexcept { return ptr; }
    const T* end()   const noexcept { return ptr + len; }
    const T& operator[](std::size_t i) const noexcept { return ptr[i]; }
    std::size_t size()       const noexcept { return len; }
    std::size_t size_bytes() const noexcept { return len * sizeof(T); }
    bool empty()             const noexcept { return len == 0; }
};

// ─────────────────────────────────────────────────────────────
//  TensorView — non-owning view over a contiguous buffer.
//
//  No allocation. No lifetime management. The memory is owned
//  by whoever produced it (mmap region, device buffer, test
//  array). TensorView just describes it.
// ─────────────────────────────────────────────────────────────

struct TensorView {
    const void* data  = nullptr; // pointer into the buffer — never owned
    DType       dtype = DType::F32;
    Shape       shape = {};

    // ── dimension helpers ────────────────────────────────────

    std::size_t rank()  const noexcept { return shape.rank();  }
    std::size_t numel() const noexcept { return shape.numel(); }

    std::size_t nbytes() const noexcept {
        return numel() * dtype_size(dtype);
    }

    // ── typed span ───────────────────────────────────────────
    //
    // Returns a Span<const T> over the raw bytes.
    //
    // Contract: sizeof(T) must equal dtype_size(dtype).
    // This intentionally allows, e.g., as<uint16_t>() for both
    // BF16 and F16, since both are 2-byte raw values. The caller
    // decides how to interpret the bits.
    //
    // Throws std::invalid_argument if sizes do not match.

    template<typename T>
    Span<T> as() const {
        if (sizeof(T) != dtype_size(dtype)) {
            throw std::invalid_argument(
                "TensorView::as<T>(): sizeof(T) (" +
                std::to_string(sizeof(T)) +
                ") does not match dtype_size(" +
                to_string(dtype) + ") (" +
                std::to_string(dtype_size(dtype)) + ")"
            );
        }
        return Span<T>{ static_cast<const T*>(data), numel() };
    }
};

} // namespace tensor::core