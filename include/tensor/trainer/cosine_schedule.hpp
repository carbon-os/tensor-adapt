#pragma once

#include <cmath>
#include <cstddef>

namespace tensor::trainer {

// ─────────────────────────────────────────────────────────────
//  CosineSchedule — warmup + cosine decay to lr * floor_frac.
// ─────────────────────────────────────────────────────────────

struct CosineSchedule {
    float       base_lr;
    std::size_t warmup_steps;
    std::size_t total_steps;
    float       floor_frac = 0.1f;  // minimum LR = base_lr * floor_frac

    float lr_at(std::size_t step) const {
        if (step < warmup_steps) {
            // Linear warmup.
            return base_lr * (float)(step + 1) / (float)warmup_steps;
        }
        float progress = (float)(step - warmup_steps) /
                         (float)(total_steps - warmup_steps);
        progress = std::min(progress, 1.f);
        float cos_val = 0.5f * (1.f + std::cos(M_PIf * progress));
        float floor   = base_lr * floor_frac;
        return floor + (base_lr - floor) * cos_val;
    }
};

} // namespace tensor::trainer