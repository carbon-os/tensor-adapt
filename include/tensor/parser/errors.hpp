#pragma once

#include <stdexcept>
#include <string>

namespace tensor::parser {

// ─────────────────────────────────────────────────────────────
//  Parser exception hierarchy
// ─────────────────────────────────────────────────────────────

// Malformed file: truncated header, bad magic, corrupt JSON, etc.
struct ParseError : std::runtime_error {
    explicit ParseError(const std::string& msg)
        : std::runtime_error("ParseError: " + msg) {}
};

// Requested tensor name not found in the weight map.
struct TensorNotFound : std::runtime_error {
    explicit TensorNotFound(const std::string& name)
        : std::runtime_error("TensorNotFound: \"" + name + "\"") {}
};

// Requested metadata key not found in __metadata__ block.
struct MetaNotFound : std::runtime_error {
    explicit MetaNotFound(const std::string& key)
        : std::runtime_error("MetaNotFound: \"" + key + "\"") {}
};

// Wrong file format or unsupported version.
struct FormatError : std::runtime_error {
    explicit FormatError(const std::string& msg)
        : std::runtime_error("FormatError: " + msg) {}
};

} // namespace tensor::parser