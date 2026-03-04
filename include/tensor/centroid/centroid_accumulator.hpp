#pragma once

#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace tensor::centroid {

// ─────────────────────────────────────────────────────────────
//  CentroidAccumulator
//
//  Runs as a side-channel during training.
//  For every batch, accumulates gradient-weighted hidden states:
//
//      C += h_t * ||∇L_t||
//
//  across all tokens t. Writes per-checkpoint snapshots to disk.
//  These snapshots feed CentroidMerge after training.
// ─────────────────────────────────────────────────────────────

class CentroidAccumulator {
public:
    CentroidAccumulator(int hidden_dim, const std::string& snapshot_dir);

    // Called once per batch per layer.
    // hidden:     [BT, D] BF16 — hidden states
    // grad_signal:[BT, D] BF16 — gradient w.r.t. hidden (proxy for ∇L)
    void accumulate(
        const backend::cuda::Tensor& hidden,
        const backend::cuda::Tensor& grad_signal,
        const backend::cuda::Device& dev);

    // Write current accumulated centroid to disk as step-N.centroid.
    void write_snapshot(std::size_t step);

    // After training: run k-means over all snapshots → adapter.centroid.
    void merge_and_write(int k, const std::string& out_path);

private:
    int         D_;
    std::string dir_;
    std::size_t samples_ = 0;

    // Running F32 accumulator on host (small enough: 896 floats for 0.5B).
    std::vector<float> acc_;
    float              acc_weight_ = 0.f;

    // All snapshot centroids for merge (loaded from disk).
    std::vector<std::vector<float>> snapshots_;
};

} // namespace tensor::centroid