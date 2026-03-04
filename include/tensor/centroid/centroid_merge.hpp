#pragma once

#include <string>

namespace tensor::centroid {

// ─────────────────────────────────────────────────────────────
//  CentroidMerge — standalone tool to merge snapshots.
//  Callable from centroid-merge CLI or after training.
// ─────────────────────────────────────────────────────────────

struct CentroidMerge {
    // Merge all .centroid files in snapshot_dir.
    // Writes output to out_path.
    // k: number of clusters.
    static void run(
        const std::string& snapshot_dir,
        const std::string& out_path,
        int k);
};

} // namespace tensor::centroid