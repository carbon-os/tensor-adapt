#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <vector>

// Forward declaration
namespace tensor::adapter { class AdapterModel; }

namespace tensor::base::arch {

using backend::cuda::Tensor;
using backend::cuda::Device;

struct Qwen2LayerCache {
    Tensor pre_attn_res;
    Tensor in_norm_rms;
    Tensor x_normed;

    Tensor Q, K, V;
    Tensor attn_out;
    Tensor attn_w;
    Tensor h_mid;

    Tensor pre_ffn_res;
    Tensor post_norm_rms;
    Tensor x_normed2;

    Tensor gate_out;
    Tensor up_out;
    Tensor act_out;
    Tensor ffn_out;
};

struct Qwen2ForwardResult {
    Tensor embed_in;
    std::vector<Qwen2LayerCache> layer_cache;

    Tensor x_final;
    Tensor final_rms;

    Tensor logits;
    Tensor dlogits;
};

class Qwen2Base {
public:
    struct LayerGrads {
        Tensor dQ;
        Tensor dK;
        Tensor dV;
        Tensor do_proj;

        Tensor dx_attn_in;
        Tensor dx_ffn_in;
    };

    static Qwen2ForwardResult forward(
        const FrozenBase&          base,
        const Tensor&              tokens,
        int B, int T,
        const Device&              dev,
        adapter::AdapterModel* adapter = nullptr);

    // FIXED: Added adapter argument here
    static std::vector<LayerGrads> backward(
        const FrozenBase&         base,
        const Qwen2ForwardResult& fwd,
        int B, int T,
        const Device&             dev,
        adapter::AdapterModel* adapter = nullptr); 
};

} // namespace tensor::base::arch