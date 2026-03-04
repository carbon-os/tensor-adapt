// base_loader.hpp
#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <string>

namespace tensor::base {

struct BaseLoader {
    static FrozenBase load(
        const std::string& dir,
        const backend::cuda::Device& dev,
        const std::string& arch_hint = "");
};

} // namespace tensor::base