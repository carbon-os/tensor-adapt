#include <tensor/centroid/centroid_accumulator.hpp>
#include <tensor/backend/cuda/ops.hpp>

#include <cuda_runtime.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace tensor::centroid {

using namespace backend::cuda;
namespace ops = backend::cuda::ops;

CentroidAccumulator::CentroidAccumulator(int D, const std::string& dir)
    : D_(D), dir_(dir), acc_(D, 0.f)
{
    fs::create_directories(dir);
}

void CentroidAccumulator::accumulate(
    const Tensor& hidden,
    const Tensor& grad_signal,
    const Device& dev)
{
    int BT = hidden.shape()[0];
    int D  = hidden.shape()[1];
    if (D != D_) return;

    // Compute per-token gradient norm: ||grad_signal[t,:]||_2
    // Then weight hidden[t,:] by that norm and add to accumulator.
    // Done on CPU after pulling small tensors.
    // For 0.5B: D=896, BT=batch*seq=4*2048=8192 → 8192*896*2 bytes ≈ 14 MB.
    // Pull to CPU once per batch for accumulation (not on the critical path).

    dev.sync();
    auto h_host = hidden.to_host_f32();      // after cast — but hidden is BF16
    auto g_host = grad_signal.to_host_f32();

    // h_host/g_host are from BF16 tensors — we need F32 read.
    // to_host_f32() assumes F32 layout; we need cast first.
    // Use temporary F32 tensors.
    Tensor h_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)D}, dev);
    Tensor g_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)D}, dev);
    ops::cast_bf16_to_f32(hidden.bf16(), h_f32.f32(), BT * D, dev.stream());
    ops::cast_bf16_to_f32(grad_signal.bf16(), g_f32.f32(), BT * D, dev.stream());
    dev.sync();

    auto hh = h_f32.to_host_f32();
    auto gh = g_f32.to_host_f32();

    for (int t = 0; t < BT; t++) {
        // Gradient norm for token t.
        float gnorm = 0.f;
        for (int d = 0; d < D; d++) {
            float v = gh[t * D + d];
            gnorm += v * v;
        }
        gnorm = std::sqrt(gnorm);

        // Weighted accumulation.
        for (int d = 0; d < D; d++) {
            acc_[d] += hh[t * D + d] * gnorm;
        }
        acc_weight_ += gnorm;
        samples_++;
    }
}

void CentroidAccumulator::write_snapshot(std::size_t step) {
    if (acc_weight_ == 0.f) return;

    // Normalise.
    std::vector<float> centroid(D_);
    for (int d = 0; d < D_; d++) {
        centroid[d] = acc_[d] / acc_weight_;
    }

    std::string path = dir_ + "/step-" + std::to_string(step) + ".centroid";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&D_), sizeof(int));
    f.write(reinterpret_cast<const char*>(centroid.data()), D_ * sizeof(float));
    f.write(reinterpret_cast<const char*>(&acc_weight_), sizeof(float));
    f.close();

    snapshots_.push_back(centroid);
    std::cerr << "[centroid] snapshot → " << path
              << " (weight=" << acc_weight_ << ")\n";
}

void CentroidAccumulator::merge_and_write(int k, const std::string& out_path) {
    // Load all snapshot files from dir_ if snapshots_ is empty.
    if (snapshots_.empty()) {
        for (const auto& e : fs::directory_iterator(dir_)) {
            if (e.path().extension() == ".centroid") {
                std::ifstream f(e.path(), std::ios::binary);
                int d = 0;
                f.read(reinterpret_cast<char*>(&d), sizeof(int));
                if (d != D_) continue;
                std::vector<float> c(d);
                f.read(reinterpret_cast<char*>(c.data()), d * sizeof(float));
                snapshots_.push_back(c);
            }
        }
    }

    if (snapshots_.empty()) {
        std::cerr << "[centroid] no snapshots to merge\n";
        return;
    }

    int N = snapshots_.size();
    k = std::min(k, N);

    // Simple k-means over snapshot centroids.
    std::vector<std::vector<float>> centers(k, std::vector<float>(D_, 0.f));

    // Init: spread evenly.
    for (int i = 0; i < k; i++) {
        centers[i] = snapshots_[(i * N) / k];
    }

    std::vector<int> assignments(N, 0);
    for (int iter = 0; iter < 50; iter++) {
        // Assign.
        bool changed = false;
        for (int n = 0; n < N; n++) {
            float best_d = 1e38f;
            int   best_k = 0;
            for (int ki = 0; ki < k; ki++) {
                float d = 0.f;
                for (int i = 0; i < D_; i++) {
                    float v = snapshots_[n][i] - centers[ki][i];
                    d += v * v;
                }
                if (d < best_d) { best_d = d; best_k = ki; }
            }
            if (assignments[n] != best_k) { assignments[n] = best_k; changed = true; }
        }
        if (!changed) break;

        // Update.
        std::vector<int> counts(k, 0);
        for (auto& c : centers) std::fill(c.begin(), c.end(), 0.f);
        for (int n = 0; n < N; n++) {
            int ki = assignments[n];
            for (int i = 0; i < D_; i++) centers[ki][i] += snapshots_[n][i];
            counts[ki]++;
        }
        for (int ki = 0; ki < k; ki++) {
            if (counts[ki] > 0) {
                for (auto& v : centers[ki]) v /= counts[ki];
            }
        }
    }

    // Write: [int k][int D][k * D floats]
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&k), sizeof(int));
    f.write(reinterpret_cast<const char*>(&D_), sizeof(int));
    for (const auto& c : centers) {
        f.write(reinterpret_cast<const char*>(c.data()), D_ * sizeof(float));
    }
    std::cerr << "[centroid] merged " << N << " snapshots → k=" << k
              << " clusters, " << out_path << "\n";
}

} // namespace tensor::centroid