#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/backend/cuda/device.hpp>
#include <tensor/backend/cuda/ops.hpp>
#include <tensor/core/dtype.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor::backend::cuda {

// ── allocation ───────────────────────────────────────────────

Tensor Tensor::empty(const Shape& shape, DType dtype, const Device& dev) {
    Tensor t;
    t.shape_     = shape;
    t.dtype_     = dtype;
    t.device_id_ = dev.id();
    std::size_t bytes = shape.numel() * core::dtype_size(dtype);
    if (bytes > 0) {
        CUDA_CHECK(cudaSetDevice(dev.id()));
        CUDA_CHECK(cudaMalloc(&t.ptr_, bytes));
    }
    return t;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype, const Device& dev) {
    Tensor t = empty(shape, dtype, dev);
    if (t.ptr_) {
        CUDA_CHECK(cudaMemsetAsync(t.ptr_, 0, t.nbytes(), dev.stream()));
    }
    return t;
}

Tensor Tensor::from_host(const TensorView& src, const Device& dev) {
    Tensor t = empty(src.shape, src.dtype, dev);
    if (src.data && t.ptr_) {
        CUDA_CHECK(cudaMemcpyAsync(
            t.ptr_, src.data, t.nbytes(),
            cudaMemcpyHostToDevice, dev.stream()));
    }
    return t;
}

Tensor Tensor::empty_bf16(const Shape& shape, const Device& dev) {
    return empty(shape, DType::BF16, dev);
}
Tensor Tensor::empty_f32(const Shape& shape, const Device& dev) {
    return empty(shape, DType::F32, dev);
}
Tensor Tensor::zeros_f32(const Shape& shape, const Device& dev) {
    return zeros(shape, DType::F32, dev);
}

// ── host readback ─────────────────────────────────────────────

std::vector<float> Tensor::to_host_f32() const {
    std::vector<float> out(numel());
    CUDA_CHECK(cudaMemcpy(out.data(), ptr_, nbytes(), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<uint32_t> Tensor::to_host_u32() const {
    std::vector<uint32_t> out(numel());
    CUDA_CHECK(cudaMemcpy(out.data(), ptr_, nbytes(), cudaMemcpyDeviceToHost));
    return out;
}

// ── zero ─────────────────────────────────────────────────────

void Tensor::zero_(const Device& dev) {
    if (ptr_) {
        CUDA_CHECK(cudaMemsetAsync(ptr_, 0, nbytes(), dev.stream()));
    }
}

// ── move / destroy ────────────────────────────────────────────

Tensor::Tensor(Tensor&& o) noexcept
    : ptr_(o.ptr_), shape_(std::move(o.shape_)), dtype_(o.dtype_), device_id_(o.device_id_)
{
    o.ptr_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        if (ptr_) cudaFree(ptr_);
        ptr_       = o.ptr_;
        shape_     = std::move(o.shape_);
        dtype_     = o.dtype_;
        device_id_ = o.device_id_;
        o.ptr_     = nullptr;
    }
    return *this;
}

Tensor::~Tensor() {
    if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; }
}

// ── GEMM parameter validation ────────────────────────────────
//
// cuBLAS column-major leading dimension rules:
//   opA=N → lda >= M,  opA=T → lda >= K
//   opB=N → ldb >= K,  opB=T → ldb >= N
//   ldc >= M
//
// For row-major inputs passed with lda=col_count, these become:
//   opA=N (rm[M,K]) → lda=K >= M? No — lda=K, required lda>=M.
//   The actual cuBLAS requirement is on the col-major view:
//   lda is the number of rows in col-major = col-count in row-major.
//   So the real checks are just lda/ldb/ldc > 0 and M,N,K > 0.

static void validate_gemm(const char* name,
                           const GemmParams& p,
                           int lda, int ldb, int ldc)
{
    if (p.M <= 0 || p.N <= 0 || p.K <= 0)
        throw CudaError(std::string(name) + ": M/N/K must be > 0, got M=" +
                        std::to_string(p.M) + " N=" + std::to_string(p.N) +
                        " K=" + std::to_string(p.K));
    if (lda <= 0 || ldb <= 0 || ldc <= 0)
        throw CudaError(std::string(name) + ": leading dims must be > 0, got lda=" +
                        std::to_string(lda) + " ldb=" + std::to_string(ldb) +
                        " ldc=" + std::to_string(ldc));
}

// ── GEMM helpers ─────────────────────────────────────────────

void gemm_bf16(
    const Device&        dev,
    const GemmParams&    p,
    const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* B, int ldb,
    __nv_bfloat16*       C, int ldc)
{
    validate_gemm("gemm_bf16", p, lda, ldb, ldc);

    if (dev.supports_bf16_gemm()) {
        // Native BF16 GEMM — Ampere (sm_80) and newer.
        cublasOperation_t opA = p.transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = p.transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_CHECK(cublasGemmEx(
            dev.cublas(),
            opA, opB,
            p.M, p.N, p.K,
            &p.alpha,
            A, CUDA_R_16BF, lda,
            B, CUDA_R_16BF, ldb,
            &p.beta,
            C, CUDA_R_16BF, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    } else {
        // Emulated BF16 GEMM — Volta / Turing fallback.
        // Cast inputs to F32, run F32 GEMM, cast output back to BF16.
        std::size_t nA = (std::size_t)(p.transA ? p.K * p.M : p.M * p.K);
        std::size_t nB = (std::size_t)(p.transB ? p.N * p.K : p.K * p.N);
        std::size_t nC = (std::size_t)(p.M * p.N);

        float *Af = nullptr, *Bf = nullptr, *Cf = nullptr;
        CUDA_CHECK(cudaSetDevice(dev.id()));
        CUDA_CHECK(cudaMalloc(&Af, nA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&Bf, nB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&Cf, nC * sizeof(float)));

        ops::cast_bf16_to_f32(A, Af, (int)nA, dev.stream());
        ops::cast_bf16_to_f32(B, Bf, (int)nB, dev.stream());

        if (p.beta != 0.f)
            ops::cast_bf16_to_f32(C, Cf, (int)nC, dev.stream());

        GemmParams fp = p;
        gemm_f32(dev, fp, Af, lda, Bf, ldb, Cf, ldc);

        ops::cast_f32_to_bf16(Cf, C, (int)nC, dev.stream());

        CUDA_CHECK(cudaStreamSynchronize(dev.stream()));
        cudaFree(Af);
        cudaFree(Bf);
        cudaFree(Cf);
    }
}

void gemm_f32(
    const Device&     dev,
    const GemmParams& p,
    const float*      A, int lda,
    const float*      B, int ldb,
    float*            C, int ldc)
{
    validate_gemm("gemm_f32", p, lda, ldb, ldc);

    cublasOperation_t opA = p.transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = p.transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(
        dev.cublas(),
        opA, opB,
        p.M, p.N, p.K,
        &p.alpha,
        A, lda,
        B, ldb,
        &p.beta,
        C, ldc
    ));
}

} // namespace tensor::backend::cuda