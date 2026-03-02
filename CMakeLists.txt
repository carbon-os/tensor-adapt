cmake_minimum_required(VERSION 3.21)
project(tensor-adapt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────────────────────────────────────
#  Build options
# ─────────────────────────────────────────────────────────────────────────────

option(TENSOR_BACKEND_CUDA   "Enable CUDA backend"    ON)
option(TENSOR_BACKEND_CPU    "Enable CPU fallback"     ON)
option(TENSOR_BUILD_TOOLS    "Build CLI tools"         ON)
option(TENSOR_BUILD_TESTS    "Build tests"             OFF)

# ─────────────────────────────────────────────────────────────────────────────
#  CUDA — optional, only enabled when present and TENSOR_BACKEND_CUDA=ON
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_BACKEND_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(TENSOR_CUDA_FOUND TRUE)
        message(STATUS "tensor-adapt: CUDA backend enabled")
    else()
        message(STATUS "tensor-adapt: CUDA not found, CPU only")
        set(TENSOR_CUDA_FOUND FALSE)
    endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Dependencies
# ─────────────────────────────────────────────────────────────────────────────

find_package(CURL           CONFIG REQUIRED)
find_package(nlohmann_json  CONFIG REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::core
#  Zero-dependency header-only primitives: DType, Shape, TensorView.
# ─────────────────────────────────────────────────────────────────────────────

add_library(tensor_core INTERFACE)
add_library(tensor::core ALIAS tensor_core)
target_include_directories(tensor_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::resolve
#  General-purpose HuggingFace resolver.
#  Handles model repos (safetensors/GGUF) and dataset repos (parquet/arrow).
# ─────────────────────────────────────────────────────────────────────────────

add_library(tensor_resolve STATIC
    src/resolve/resolver.cpp
)
add_library(tensor::resolve ALIAS tensor_resolve)
target_include_directories(tensor_resolve
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_resolve
    PUBLIC  tensor::core
    PRIVATE CURL::libcurl
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────────────────────
#  CLI tools
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_BUILD_TOOLS)

    # hf-fetch: download models and datasets from HuggingFace
    add_executable(hf-fetch tools/hf-fetch.cpp)
    target_link_libraries(hf-fetch PRIVATE tensor::resolve)

endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_BUILD_TESTS)
    enable_testing()
    # tests go here as the project grows
endif()