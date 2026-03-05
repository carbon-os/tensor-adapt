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
// ─────────────────────────────────────────────────────────────

class Tensor {
public:
    Tensor() = default;

    static Tensor empty(const Shape& shape, DType dtype, const Device& dev);
    static Tensor zeros(const Shape& shape, DType dtype, const Device& dev);
    static Tensor from_host(const TensorView& src, const Device& dev);

    static Tensor empty_bf16(const Shape& shape, const Device& dev);
    static Tensor empty_f32 (const Shape& shape, const Device& dev);
    static Tensor zeros_f32 (const Shape& shape, const Device& dev);

    const Shape& shape()     const noexcept { return shape_;     }
    DType        dtype()     const noexcept { return dtype_;     }
    int          device_id() const noexcept { return device_id_; }

    std::size_t rank()   const noexcept { return shape_.rank();  }
    std::size_t numel()  const noexcept { return shape_.numel(); }
    std::size_t nbytes() const noexcept { return numel() * core::dtype_size(dtype_); }

    bool empty() const noexcept { return ptr_ == nullptr; }

    void* data()       noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }

    __nv_bfloat16* bf16()       noexcept { return static_cast<__nv_bfloat16*>(ptr_); }
    const __nv_bfloat16* bf16() const noexcept { return static_cast<const __nv_bfloat16*>(ptr_); }
    float* f32()        noexcept { return static_cast<float*>(ptr_); }
    const float* f32()  const noexcept { return static_cast<const float*>(ptr_); }
    int* i32()        noexcept { return static_cast<int*>(ptr_); }
    const int* i32()  const noexcept { return static_cast<const int*>(ptr_); }

    // ── ADDED THIS METHOD ────────────────────────────────────
    std::vector<__nv_bfloat16> to_host_bf16() const;
    // ─────────────────────────────────────────────────────────

    std::vector<float>    to_host_f32() const;
    std::vector<uint32_t> to_host_u32() const;

    void zero_(const Device& dev);

    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor();

private:
    void* ptr_       = nullptr;
    Shape  shape_;
    DType  dtype_     = DType::F32;
    int    device_id_ = 0;
};

struct GemmParams {
    bool  transA = false;
    bool  transB = false;
    int   M = 0, N = 0, K = 0;
    float alpha = 1.0f;
    float beta  = 0.0f;
};

void gemm_bf16(
    const Device&        dev,
    const GemmParams&    p,
    const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* B, int ldb,
    __nv_bfloat16* C, int ldc
);

void gemm_f32(
    const Device&        dev,
    const GemmParams&    p,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
);

} // namespace tensor::backend::cuda