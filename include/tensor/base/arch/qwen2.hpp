// qwen2.hpp
#pragma once

#include <tensor/base/frozen_base.hpp>
#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <vector>

// Forward declaration — avoids circular dependency with adapter_model.hpp
// (adapter_model.hpp includes qwen2.hpp transitively).
// The full type is only needed in qwen2.cu where apply_lora_fwd is called.
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

    // adapter is optional — pass nullptr for inference / base-only forward.
    // When non-null, LoRA deltas are injected after Q, K, V, and O projections
    // in every layer before the results are used downstream.
    static Qwen2ForwardResult forward(
        const FrozenBase&          base,
        const Tensor&              tokens,
        int B, int T,
        const Device&              dev,
        adapter::AdapterModel*     adapter = nullptr);

    static std::vector<LayerGrads> backward(
        const FrozenBase&         base,
        const Qwen2ForwardResult& fwd,
        int B, int T,
        const Device&             dev);
};

} // namespace tensor::base::arch