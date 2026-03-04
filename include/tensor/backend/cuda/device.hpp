#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace tensor::backend::cuda {

// ─────────────────────────────────────────────────────────────
//  CudaError
// ─────────────────────────────────────────────────────────────

struct CudaError : std::runtime_error {
    explicit CudaError(const std::string& msg)
        : std::runtime_error("CudaError: " + msg) {}
};

// ─────────────────────────────────────────────────────────────
//  CUDA / cuBLAS check macros
// ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            throw ::tensor::backend::cuda::CudaError(                      \
                std::string(cudaGetErrorString(_e)) +                       \
                " at " __FILE__ ":" + std::to_string(__LINE__));            \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            throw ::tensor::backend::cuda::CudaError(                      \
                "cuBLAS status " + std::to_string(static_cast<int>(_s)) +  \
                " at " __FILE__ ":" + std::to_string(__LINE__));            \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────────────────────
//  Device — RAII CUDA device context
//
//  Holds a per-device cuBLAS handle and a dedicated stream.
//  All operations on this device use stream() so kernel
//  launches and cuBLAS calls are serialised per device.
// ─────────────────────────────────────────────────────────────

class Device {
public:
    // Parse "cuda:0" → open device 0.
    static Device open(const std::string& spec);

    int            id()     const noexcept { return id_;     }
    cublasHandle_t cublas() const noexcept { return cublas_; }
    cudaStream_t   stream() const noexcept { return stream_; }

    // Synchronise the device stream — call before reading host results.
    void sync() const;

    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;

    ~Device();

private:
    Device() = default;

    int            id_     = 0;
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t   stream_ = nullptr;
};

} // namespace tensor::backend::cuda