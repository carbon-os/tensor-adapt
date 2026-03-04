#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace tensor::data {

struct DataError : std::runtime_error {
    explicit DataError(const std::string& m)
        : std::runtime_error("DataError: " + m) {}
};

// ─────────────────────────────────────────────────────────────
//  Dataset — streaming binary token reader.
//
//  Data format: one or more .bin files, each containing a flat
//  array of uint32_t token IDs (little-endian).
//
//  Training example: a window of seq_len+1 tokens.
//    input  = tokens[0 .. seq_len-1]
//    target = tokens[1 .. seq_len]
//
//  Reads are sequential and wrap around at end of data.
// ─────────────────────────────────────────────────────────────

class Dataset {
public:
    // Load all .bin files from dir, sorted by name.
    static Dataset load(const std::string& dir);

    // Fetch the next batch. Returns:
    //   input_ids:  [B * T] int32 (as signed for CUDA compatibility)
    //   target_ids: [B * T] int32
    std::pair<std::vector<int32_t>, std::vector<int32_t>>
    next_batch(int B, int T);

    std::size_t total_tokens() const { return total_tokens_; }
    std::size_t tokens_consumed() const { return tokens_consumed_; }

private:
    std::vector<uint32_t> tokens_;   // all tokens concatenated in memory
    std::size_t pos_       = 0;
    std::size_t total_tokens_ = 0;
    std::size_t tokens_consumed_ = 0;
};

} // namespace tensor::data