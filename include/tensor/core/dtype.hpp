#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace tensor::core {

// ─────────────────────────────────────────────────────────────
//  DType — element type descriptor, zero-dependency
// ─────────────────────────────────────────────────────────────

enum class DType {
    F64,
    F32,
    BF16,
    F16,
    F8_E5M2,
    F8_E4M3,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL,
};

inline std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::F64:     return 8;
        case DType::F32:     return 4;
        case DType::BF16:    return 2;
        case DType::F16:     return 2;
        case DType::F8_E5M2: return 1;
        case DType::F8_E4M3: return 1;
        case DType::I64:     return 8;
        case DType::I32:     return 4;
        case DType::I16:     return 2;
        case DType::I8:      return 1;
        case DType::U8:      return 1;
        case DType::BOOL:    return 1;
    }
    throw std::invalid_argument("dtype_size: unknown DType");
}

inline std::string to_string(DType d) {
    switch (d) {
        case DType::F64:     return "F64";
        case DType::F32:     return "F32";
        case DType::BF16:    return "BF16";
        case DType::F16:     return "F16";
        case DType::F8_E5M2: return "F8_E5M2";
        case DType::F8_E4M3: return "F8_E4M3";
        case DType::I64:     return "I64";
        case DType::I32:     return "I32";
        case DType::I16:     return "I16";
        case DType::I8:      return "I8";
        case DType::U8:      return "U8";
        case DType::BOOL:    return "BOOL";
    }
    throw std::invalid_argument("to_string(DType): unknown DType");
}

} // namespace tensor::core