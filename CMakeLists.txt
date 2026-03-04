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
#  CUDA
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
        # sm_70 = Volta, sm_75 = Turing (GTX 1650 etc.), sm_80+ = Ampere and newer
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA archs")
        message(STATUS "tensor-adapt: CUDA backend enabled (archs=${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "tensor-adapt: CUDA not found — CPU only")
        set(TENSOR_CUDA_FOUND FALSE)
    endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Dependencies
# ─────────────────────────────────────────────────────────────────────────────

find_package(CURL          CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenSSL       REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::core  (header-only)
# ─────────────────────────────────────────────────────────────────────────────

add_library(tensor_core INTERFACE)
add_library(tensor::core ALIAS tensor_core)
target_include_directories(tensor_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::parser
# ─────────────────────────────────────────────────────────────────────────────

add_library(tensor_parser STATIC
    src/parser/safetensors.cpp
    src/parser/config.cpp
    src/parser/weight_map.cpp
)
add_library(tensor::parser ALIAS tensor_parser)
target_include_directories(tensor_parser
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/parser/detail
)
target_link_libraries(tensor_parser
    PUBLIC  tensor::core
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::resolve
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
#  tensor::data
# ─────────────────────────────────────────────────────────────────────────────

add_library(tensor_data STATIC
    src/data/dataset.cpp
)
add_library(tensor::data ALIAS tensor_data)
target_include_directories(tensor_data
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_data
    PUBLIC tensor::core
)

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::tokenizer
# ─────────────────────────────────────────────────────────────────────────────

add_library(tensor_tokenizer STATIC
    src/tokenizer/bpe_tokenizer.cpp
)
add_library(tensor::tokenizer ALIAS tensor_tokenizer)
target_include_directories(tensor_tokenizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_tokenizer
    PUBLIC  tensor::core
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::backend_cuda
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_CUDA_FOUND)
    add_library(tensor_backend_cuda STATIC
        src/backend/cuda/device.cu
        src/backend/cuda/tensor.cu
        src/backend/cuda/ops.cu
    )
    add_library(tensor::backend_cuda ALIAS tensor_backend_cuda)
    target_include_directories(tensor_backend_cuda
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_backend_cuda
        PUBLIC tensor::core
        PUBLIC CUDA::cudart
        PUBLIC CUDA::cublas
    )
    set_target_properties(tensor_backend_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::base
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_CUDA_FOUND)
    add_library(tensor_base STATIC
        src/base/base_loader.cpp
        src/base/arch/qwen2.cu
    )
    add_library(tensor::base ALIAS tensor_base)
    target_include_directories(tensor_base
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_base
        PUBLIC  tensor::core
        PUBLIC  tensor::parser
        PUBLIC  tensor::backend_cuda
        PRIVATE OpenSSL::Crypto
    )
    set_target_properties(tensor_base PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::adapter
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_CUDA_FOUND)
    add_library(tensor_adapter STATIC
        src/adapter/adapter_config.cpp
        src/adapter/adapter_model.cu
    )
    add_library(tensor::adapter ALIAS tensor_adapter)
    target_include_directories(tensor_adapter
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_adapter
        PUBLIC tensor::core
        PUBLIC tensor::base
        PUBLIC tensor::backend_cuda
    )
    set_target_properties(tensor_adapter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::centroid
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_CUDA_FOUND)
    add_library(tensor_centroid STATIC
        src/centroid/centroid_accumulator.cpp
        src/centroid/centroid_merge.cpp
    )
    add_library(tensor::centroid ALIAS tensor_centroid)
    target_include_directories(tensor_centroid
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_centroid
        PUBLIC tensor::core
        PUBLIC tensor::backend_cuda
    )
    set_target_properties(tensor_centroid PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::checkpoint
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_CUDA_FOUND)
    add_library(tensor_checkpoint STATIC
        src/checkpoint/adapter_writer.cpp
    )
    add_library(tensor::checkpoint ALIAS tensor_checkpoint)
    target_include_directories(tensor_checkpoint
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_checkpoint
        PUBLIC  tensor::core
        PUBLIC  tensor::adapter
        PRIVATE nlohmann_json::nlohmann_json
    )
    set_target_properties(tensor_checkpoint PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  tensor::trainer
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_CUDA_FOUND)
    add_library(tensor_trainer STATIC
        src/trainer/adamw.cu
        src/trainer/adapt_trainer.cu
    )
    add_library(tensor::trainer ALIAS tensor_trainer)
    target_include_directories(tensor_trainer
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_trainer
        PUBLIC tensor::core
        PUBLIC tensor::adapter
        PUBLIC tensor::base
        PUBLIC tensor::centroid
        PUBLIC tensor::checkpoint
        PUBLIC tensor::data
        PUBLIC tensor::backend_cuda
    )
    set_target_properties(tensor_trainer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  CLI tools
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_BUILD_TOOLS)

    # hf: download models and datasets from HuggingFace
    add_executable(hf tools/hf.cpp)
    target_link_libraries(hf PRIVATE tensor::resolve)

    # dataset-format: tokenize text datasets to .bin for tensor-adapt
    # intentionally outside the CUDA guard — no GPU needed
    add_executable(dataset-format tools/dataset-format.cpp)
    target_link_libraries(dataset-format PRIVATE tensor::tokenizer)

    if(TENSOR_CUDA_FOUND)

        # tensor-adapt: train a LoRA adapter
        add_executable(tensor-adapt tools/tensor-adapt.cpp)
        target_link_libraries(tensor-adapt PRIVATE tensor::trainer)
        set_target_properties(tensor-adapt PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        # centroid-merge: standalone centroid merge after training
        add_executable(centroid-merge tools/centroid-merge.cpp)
        target_link_libraries(centroid-merge PRIVATE tensor::centroid)
        set_target_properties(centroid-merge PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

    endif()

endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

if(TENSOR_BUILD_TESTS)
    enable_testing()
    # tests go here as the project grows
endif()