// qwen2.cu
#include <tensor/base/arch/qwen2.hpp>
#include <tensor/backend/cuda/device.hpp>
#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/backend/cuda/ops.hpp>

#include <cuda_bf16.h>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace tensor::base::arch {

using namespace backend::cuda;
namespace ops = backend::cuda::ops;

// ─────────────────────────────────────────────────────────────
//  Projection helpers — see original file for cuBLAS convention notes
// ─────────────────────────────────────────────────────────────

static void proj_fwd(const Device& dev,
                     const Tensor& W, const Tensor& x, Tensor& y,
                     int BT, int in_dim, int out_dim)
{
    GemmParams p;
    p.transA = true;   p.transB = false;
    p.M = out_dim;     p.N = BT;     p.K = in_dim;
    p.alpha = 1.f;     p.beta  = 0.f;
    gemm_bf16(dev, p, W.bf16(), in_dim, x.bf16(), in_dim, y.bf16(), out_dim);
}

static void proj_bwd_x(const Device& dev,
                       const Tensor& W, const Tensor& dy, Tensor& dx,
                       int BT, int out_dim, int in_dim, float beta = 0.f)
{
    GemmParams p;
    p.transA = false;  p.transB = false;
    p.M = in_dim;      p.N = BT;     p.K = out_dim;
    p.alpha = 1.f;     p.beta  = beta;
    gemm_bf16(dev, p, W.bf16(), in_dim, dy.bf16(), out_dim, dx.bf16(), in_dim);
}

// ─────────────────────────────────────────────────────────────
//  Forward
// ─────────────────────────────────────────────────────────────

Qwen2ForwardResult Qwen2Base::forward(
    const FrozenBase& base, const Tensor& tokens,
    int B, int T, const Device& dev)
{
    const BaseConfig& bc = base.config;
    int BT  = B * T;
    int H   = bc.hidden_size;
    int I   = bc.intermediate_size;
    int Hq  = bc.num_q_heads;
    int Hkv = bc.num_kv_heads;
    int hd  = bc.head_dim;
    int V   = bc.vocab_size;
    float eps = bc.rms_norm_eps;

    Qwen2ForwardResult res;

    // ── 1. Embed ──────────────────────────────────────────────
    res.embed_in = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    ops::embed_fwd(base.embed_w.bf16(), tokens.i32(),
                   res.embed_in.bf16(), BT, H, dev.stream());

    Tensor residual = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    cudaMemcpyAsync(residual.data(), res.embed_in.data(),
                    residual.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

    res.layer_cache.resize(bc.num_layers);

    // ── 2. Transformer layers ──────────────────────────────────
    for (std::size_t l = 0; l < bc.num_layers; l++) {
        const LayerWeights& lw = base.layers[l];
        Qwen2LayerCache&    lc = res.layer_cache[l];

        // Save pre-input-norm residual for rms_norm_bwd.
        lc.pre_attn_res = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(lc.pre_attn_res.data(), residual.data(),
                        residual.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

        // Input layernorm.
        lc.in_norm_rms = Tensor::empty_f32({(std::size_t)BT}, dev);
        lc.x_normed    = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_fwd(residual.bf16(), lw.input_norm_w.bf16(),
                          lc.x_normed.bf16(), lc.in_norm_rms.f32(),
                          B, T, H, eps, dev.stream());

        // Q, K, V projections + bias.
        int kv_dim = Hkv * hd;
        lc.Q = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)(Hq * hd)}, dev);
        lc.K = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)kv_dim}, dev);
        lc.V = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)kv_dim}, dev);

        proj_fwd(dev, lw.q_proj_w, lc.x_normed, lc.Q, BT, H, Hq * hd);
        ops::add_bias(lc.Q.bf16(), lw.q_proj_b.bf16(), BT, Hq * hd, dev.stream());

        proj_fwd(dev, lw.k_proj_w, lc.x_normed, lc.K, BT, H, kv_dim);
        ops::add_bias(lc.K.bf16(), lw.k_proj_b.bf16(), BT, kv_dim, dev.stream());

        proj_fwd(dev, lw.v_proj_w, lc.x_normed, lc.V, BT, H, kv_dim);
        ops::add_bias(lc.V.bf16(), lw.v_proj_b.bf16(), BT, kv_dim, dev.stream());

        // RoPE (in-place, applied after bias).
        ops::rope_fwd(lc.Q.bf16(), B, T, Hq,  hd, bc.rope_theta, dev.stream());
        ops::rope_fwd(lc.K.bf16(), B, T, Hkv, hd, bc.rope_theta, dev.stream());

        // Attention — passes Device& so sdpa can dispatch to batched path on sm80+.
        lc.attn_out = Tensor::empty_bf16(
            {(std::size_t)BT, (std::size_t)(Hq * hd)}, dev);
        lc.attn_w = Tensor::empty_f32(
            {(std::size_t)B, (std::size_t)Hq,
             (std::size_t)T, (std::size_t)T}, dev);
        ops::sdpa_fwd(lc.Q.bf16(), lc.K.bf16(), lc.V.bf16(),
                      lc.attn_out.bf16(), lc.attn_w.f32(),
                      B, T, Hq, Hkv, hd, dev);

        // O projection (no bias).
        lc.h_mid = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        proj_fwd(dev, lw.o_proj_w, lc.attn_out, lc.h_mid, BT, Hq * hd, H);

        ops::add_inplace(residual.bf16(), lc.h_mid.bf16(), BT * H, dev.stream());

        // Save pre-post-attn-norm residual for rms_norm_bwd.
        lc.pre_ffn_res = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(lc.pre_ffn_res.data(), residual.data(),
                        residual.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

        // Post-attention layernorm.
        lc.post_norm_rms = Tensor::empty_f32({(std::size_t)BT}, dev);
        lc.x_normed2     = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_fwd(residual.bf16(), lw.post_norm_w.bf16(),
                          lc.x_normed2.bf16(), lc.post_norm_rms.f32(),
                          B, T, H, eps, dev.stream());

        // Gate + up projections, SiLU, down projection.
        lc.gate_out = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        lc.up_out   = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        proj_fwd(dev, lw.gate_proj_w, lc.x_normed2, lc.gate_out, BT, H, I);
        proj_fwd(dev, lw.up_proj_w,   lc.x_normed2, lc.up_out,   BT, H, I);

        lc.act_out = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        ops::silu_mul_fwd(lc.gate_out.bf16(), lc.up_out.bf16(),
                          lc.act_out.bf16(), BT * I, dev.stream());

        lc.ffn_out = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        proj_fwd(dev, lw.down_proj_w, lc.act_out, lc.ffn_out, BT, I, H);

        ops::add_inplace(residual.bf16(), lc.ffn_out.bf16(), BT * H, dev.stream());
    }

    // ── 3. Save pre-final-norm residual ───────────────────────
    res.x_final = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    cudaMemcpyAsync(res.x_final.data(), residual.data(),
                    residual.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

    // ── 4. Final norm ──────────────────────────────────────────
    res.final_rms = Tensor::empty_f32({(std::size_t)BT}, dev);
    Tensor x_final_normed = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    ops::rms_norm_fwd(residual.bf16(), base.final_norm_w.bf16(),
                      x_final_normed.bf16(), res.final_rms.f32(),
                      B, T, H, eps, dev.stream());

    // ── 5. LM head ────────────────────────────────────────────
    const Tensor& lm_w = bc.tie_embeddings ? base.embed_w : base.lm_head_w;
    res.logits  = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)V}, dev);
    res.dlogits = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)V}, dev);
    {
        GemmParams gp;
        gp.transA = true;  gp.transB = false;
        gp.M = V;          gp.N = BT;   gp.K = H;
        gp.alpha = 1.f;    gp.beta  = 0.f;
        gemm_bf16(dev, gp,
                  lm_w.bf16(),           H,
                  x_final_normed.bf16(), H,
                  res.logits.bf16(),     V);
    }

    return res;
}

// ─────────────────────────────────────────────────────────────
//  Backward
// ─────────────────────────────────────────────────────────────

std::vector<Qwen2Base::LayerGrads> Qwen2Base::backward(
    const FrozenBase&         base,
    const Qwen2ForwardResult& fwd,
    int B, int T,
    const Device&             dev)
{
    const BaseConfig& bc = base.config;
    int BT     = B * T;
    int H      = bc.hidden_size;
    int I      = bc.intermediate_size;
    int Hq     = bc.num_q_heads;
    int Hkv    = bc.num_kv_heads;
    int hd     = bc.head_dim;
    int kv_dim = Hkv * hd;
    float eps  = bc.rms_norm_eps;

    std::vector<LayerGrads> result(bc.num_layers);

    // Backward through LM head.
    const Tensor& lm_w = bc.tie_embeddings ? base.embed_w : base.lm_head_w;
    Tensor dres = Tensor::zeros({(std::size_t)BT, (std::size_t)H},
                                core::DType::BF16, dev);
    proj_bwd_x(dev, lm_w, fwd.dlogits, dres, BT, bc.vocab_size, H);

    // Backward through final RMSNorm.
    Tensor dfinal_norm_w = Tensor::zeros_f32({(std::size_t)H}, dev);
    Tensor dres_norm     = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    ops::rms_norm_bwd(
        dres.bf16(),
        fwd.x_final.bf16(),
        base.final_norm_w.bf16(),
        fwd.final_rms.f32(),
        dres_norm.bf16(), dfinal_norm_w.f32(),
        B, T, H, eps, dev.stream());

    Tensor grad = std::move(dres_norm);

    for (int l = (int)bc.num_layers - 1; l >= 0; l--) {
        const LayerWeights&    lw = base.layers[l];
        const Qwen2LayerCache& lc = fwd.layer_cache[l];
        LayerGrads&            lg = result[l];

        // ── FFN backward ───────────────────────────────────────

        Tensor dffn = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(dffn.data(), grad.data(), grad.nbytes(),
                        cudaMemcpyDeviceToDevice, dev.stream());

        Tensor dact = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        proj_bwd_x(dev, lw.down_proj_w, dffn, dact, BT, H, I);

        Tensor dgate = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        Tensor dup   = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        ops::silu_mul_bwd(dact.bf16(), lc.gate_out.bf16(), lc.up_out.bf16(),
                          dgate.bf16(), dup.bf16(), BT * I, dev.stream());

        Tensor dx_normed2 = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)H}, core::DType::BF16, dev);
        proj_bwd_x(dev, lw.gate_proj_w, dgate, dx_normed2, BT, I, H, 0.f);
        proj_bwd_x(dev, lw.up_proj_w,   dup,   dx_normed2, BT, I, H, 1.f);

        lg.dx_ffn_in = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(lg.dx_ffn_in.data(), dx_normed2.data(),
                        dx_normed2.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

        Tensor dpost_norm_w = Tensor::zeros_f32({(std::size_t)H}, dev);
        Tensor dffn_in      = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_bwd(
            dx_normed2.bf16(),
            lc.pre_ffn_res.bf16(),
            lw.post_norm_w.bf16(),
            lc.post_norm_rms.f32(),
            dffn_in.bf16(), dpost_norm_w.f32(),
            B, T, H, eps, dev.stream());

        ops::add_inplace(grad.bf16(), dffn_in.bf16(), BT * H, dev.stream());

        // ── Attention backward ─────────────────────────────────

        lg.do_proj = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(lg.do_proj.data(), grad.data(), grad.nbytes(),
                        cudaMemcpyDeviceToDevice, dev.stream());

        Tensor dx_attn_out = Tensor::empty_bf16(
            {(std::size_t)BT, (std::size_t)(Hq * hd)}, dev);
        proj_bwd_x(dev, lw.o_proj_w, grad, dx_attn_out, BT, H, Hq * hd);

        // SDPA backward — passes Device& for batched dispatch on sm80+.
        lg.dQ = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)(Hq * hd)}, core::DType::BF16, dev);
        lg.dK = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)kv_dim}, core::DType::BF16, dev);
        lg.dV = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)kv_dim}, core::DType::BF16, dev);
        ops::sdpa_bwd(dx_attn_out.bf16(),
                      lc.Q.bf16(), lc.K.bf16(), lc.V.bf16(),
                      lc.attn_w.f32(),
                      lg.dQ.bf16(), lg.dK.bf16(), lg.dV.bf16(),
                      B, T, Hq, Hkv, hd, dev);

        // RoPE backward — converts dQ/dK back to the pre-RoPE frame.
        ops::rope_bwd(lg.dQ.bf16(), B, T, Hq,  hd, bc.rope_theta, dev.stream());
        ops::rope_bwd(lg.dK.bf16(), B, T, Hkv, hd, bc.rope_theta, dev.stream());

        Tensor dx_normed = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)H}, core::DType::BF16, dev);
        proj_bwd_x(dev, lw.q_proj_w, lg.dQ, dx_normed, BT, Hq * hd, H, 0.f);
        proj_bwd_x(dev, lw.k_proj_w, lg.dK, dx_normed, BT, kv_dim,   H, 1.f);
        proj_bwd_x(dev, lw.v_proj_w, lg.dV, dx_normed, BT, kv_dim,   H, 1.f);

        lg.dx_attn_in = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(lg.dx_attn_in.data(), dx_normed.data(),
                        dx_normed.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

        Tensor din_norm_w = Tensor::zeros_f32({(std::size_t)H}, dev);
        Tensor dattn_in   = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_bwd(
            dx_normed.bf16(),
            lc.pre_attn_res.bf16(),
            lw.input_norm_w.bf16(),
            lc.in_norm_rms.f32(),
            dattn_in.bf16(), din_norm_w.f32(),
            B, T, H, eps, dev.stream());

        ops::add_inplace(grad.bf16(), dattn_in.bf16(), BT * H, dev.stream());
    }

    return result;
}

} // namespace tensor::base::arch