#include <tensor/backend/cuda/device.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <string>
#include <stdexcept>

namespace tensor::backend::cuda {

static int parse_spec(const std::string& spec) {
    // "cuda:0" → 0, "cuda:1" → 1, etc.
    if (spec.rfind("cuda:", 0) != 0) {
        throw CudaError("invalid device spec \"" + spec +
                        "\"; expected \"cuda:N\"");
    }
    try {
        return std::stoi(spec.substr(5));
    } catch (...) {
        throw CudaError("invalid device index in \"" + spec + "\"");
    }
}

Device Device::open(const std::string& spec) {
    Device d;
    d.id_ = parse_spec(spec);

    CUDA_CHECK(cudaSetDevice(d.id_));
    CUDA_CHECK(cudaStreamCreate(&d.stream_));

    CUBLAS_CHECK(cublasCreate(&d.cublas_));
    CUBLAS_CHECK(cublasSetStream(d.cublas_, d.stream_));
    // Use tensor cores where available.
    CUBLAS_CHECK(cublasSetMathMode(d.cublas_, CUBLAS_DEFAULT_MATH));

    return d;
}

void Device::sync() const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Device::~Device() {
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

Device::Device(Device&& o) noexcept
    : id_(o.id_), cublas_(o.cublas_), stream_(o.stream_)
{
    o.cublas_ = nullptr;
    o.stream_ = nullptr;
}

Device& Device::operator=(Device&& o) noexcept {
    if (this != &o) {
        if (cublas_) cublasDestroy(cublas_);
        if (stream_) cudaStreamDestroy(stream_);
        id_     = o.id_;
        cublas_ = o.cublas_;
        stream_ = o.stream_;
        o.cublas_ = nullptr;
        o.stream_ = nullptr;
    }
    return *this;
}

} // namespace tensor::backend::cuda