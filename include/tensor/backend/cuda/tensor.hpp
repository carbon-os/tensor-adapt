#pragma once

#include <tensor/backend/cuda/device.hpp>
#include <tensor/core/dtype.hpp>
#include <tensor/core/shape.hpp>
#include <tensor/core/tensor_view.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace tensor::backend::cuda {

using core::DType;
using core::Shape;
using core::TensorView;

// ─────────────────────────────────────────────────────────────
//  Tensor — owning GPU buffer with shape and dtype
//
//  Data lives on the device that created it.
//  The Tensor owns its memory and frees it on destruction.
//
//  No implicit copies — move only.
//  Use to_device() / to_host() for explicit transfers.
// ─────────────────────────────────────────────────────────────

class Tensor {
public:
    // Default constructs a null/empty tensor — no allocation.
    // Allows structs containing Tensor members to be default-constructed.
    Tensor() = default;

    // Allocate uninitialised GPU memory.
    static Tensor empty(const Shape& shape, DType dtype, const Device& dev);
    static Tensor zeros(const Shape& shape, DType dtype, const Device& dev);

    // Upload a host TensorView to device.
    // Caller must ensure dtype and numel match.
    static Tensor from_host(const TensorView& src, const Device& dev);

    // Named convenience constructors.
    static Tensor empty_bf16(const Shape& shape, const Device& dev);
    static Tensor empty_f32 (const Shape& shape, const Device& dev);
    static Tensor zeros_f32 (const Shape& shape, const Device& dev);

    // ── accessors ────────────────────────────────────────────

    const Shape& shape()     const noexcept { return shape_;     }
    DType        dtype()     const noexcept { return dtype_;     }
    int          device_id() const noexcept { return device_id_; }

    std::size_t rank()   const noexcept { return shape_.rank();  }
    std::size_t numel()  const noexcept { return shape_.numel(); }
    std::size_t nbytes() const noexcept { return numel() * core::dtype_size(dtype_); }

    // True if this tensor holds no allocation (default-constructed or moved-from).
    bool empty() const noexcept { return ptr_ == nullptr; }

    // Raw device pointer — use with kernel launches or cuBLAS calls.
    void*       data()       noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }

    // Typed pointer access — no bounds or dtype check.
    __nv_bfloat16*       bf16()       noexcept { return static_cast<__nv_bfloat16*>(ptr_); }
    const __nv_bfloat16* bf16() const noexcept { return static_cast<const __nv_bfloat16*>(ptr_); }
    float*               f32()        noexcept { return static_cast<float*>(ptr_); }
    const float*         f32()  const noexcept { return static_cast<const float*>(ptr_); }
    int*                 i32()        noexcept { return static_cast<int*>(ptr_); }
    const int*           i32()  const noexcept { return static_cast<const int*>(ptr_); }

    // Copy device memory to a host vector.
    std::vector<float>    to_host_f32() const;
    std::vector<uint32_t> to_host_u32() const;

    // Zero-fill.
    void zero_(const Device& dev);

    // Move-only.
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor();

private:
    void*  ptr_       = nullptr;
    Shape  shape_;
    DType  dtype_     = DType::F32;
    int    device_id_ = 0;
};

// ─────────────────────────────────────────────────────────────
//  GemmParams — typed matmul descriptor passed to cuBLAS helpers
// ─────────────────────────────────────────────────────────────

struct GemmParams {
    // C = alpha * op(A) * op(B) + beta * C
    // op = transpose if trans == true
    bool  transA = false;
    bool  transB = false;
    int   M = 0, N = 0, K = 0;
    float alpha = 1.0f;
    float beta  = 0.0f;
};

// cuBLAS BF16 GEMM: C[M,N] = alpha * op(A) * op(B) + beta * C
// A, B, C are __nv_bfloat16 device pointers.
// Accumulation is in F32.
void gemm_bf16(
    const Device&        dev,
    const GemmParams&    p,
    const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* B, int ldb,
    __nv_bfloat16*       C, int ldc
);

// F32 GEMM — used for centroid accumulation and small ops.
void gemm_f32(
    const Device&     dev,
    const GemmParams& p,
    const float*      A, int lda,
    const float*      B, int ldb,
    float*            C, int ldc
);

} // namespace tensor::backend::cuda