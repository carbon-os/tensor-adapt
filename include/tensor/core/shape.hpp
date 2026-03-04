#pragma once

#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace tensor::core {

// ─────────────────────────────────────────────────────────────
//  Shape — dimension vector with arithmetic helpers
//  Inherits std::vector so all range-based operations work.
// ─────────────────────────────────────────────────────────────

class Shape : public std::vector<std::size_t> {
public:
    using std::vector<std::size_t>::vector;

    // Number of dimensions.
    std::size_t rank() const noexcept {
        return size();
    }

    // Total element count — product of all dimensions.
    // Returns 1 for a rank-0 scalar (empty shape).
    std::size_t numel() const noexcept {
        if (empty()) return 1;
        return std::accumulate(
            begin(), end(),
            std::size_t{1},
            std::multiplies<std::size_t>{}
        );
    }
};

} // namespace tensor::core