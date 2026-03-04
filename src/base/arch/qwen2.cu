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
//  Row-major → cuBLAS col-major convention
//
//  Row-major X[r,c] with lda=c appears to cuBLAS as col-major X^T[c,r].
//
//  To compute row-major C[m,n] = A[m,k] @ B[k,n]:
//    C^T[n,m] = B^T[k,n]→col = A^T... use the swap trick:
//    Pass (A_arg=B, transA=T) and (B_arg=A, transB=N) with M=n, N=m, K=k.
//
//  Concrete rules used here:
//
//    y[BT,od] = x[BT,id] @ W[od,id]^T
//      → transA=T(W), transB=N(x), M=od, N=BT, K=id
//      → A=W lda=id, B=x ldb=id, C=y ldc=od
//
//    dx[BT,id] = dy[BT,od] @ W[od,id]
//      → transA=N(W), transB=N(dy), M=id, N=BT, K=od
//      → A=W lda=id, B=dy ldb=od, C=dx ldc=id
// ─────────────────────────────────────────────────────────────

// W: [out_dim, in_dim],  x: [BT, in_dim]  →  y: [BT, out_dim]
static void proj_fwd(
    const Device& dev,
    const Tensor& W,
    const Tensor& x,
    Tensor&       y,
    int BT, int in_dim, int out_dim)
{
    GemmParams p;
    p.transA = true;   p.transB = false;
    p.M = out_dim;     p.N = BT;      p.K = in_dim;
    p.alpha = 1.f;     p.beta = 0.f;
    gemm_bf16(dev, p,
              W.bf16(), in_dim,
              x.bf16(), in_dim,
              y.bf16(), out_dim);
}

// W: [out_dim, in_dim],  dy: [BT, out_dim]  →  dx: [BT, in_dim]
static void proj_bwd_x(
    const Device& dev,
    const Tensor& W,
    const Tensor& dy,
    Tensor&       dx,
    int BT, int out_dim, int in_dim,
    float beta = 0.f)
{
    GemmParams p;
    p.transA = false;  p.transB = false;
    p.M = in_dim;      p.N = BT;       p.K = out_dim;
    p.alpha = 1.f;     p.beta = beta;
    gemm_bf16(dev, p,
              W.bf16(),  in_dim,
              dy.bf16(), out_dim,
              dx.bf16(), in_dim);
}

// ─────────────────────────────────────────────────────────────
//  Forward
// ─────────────────────────────────────────────────────────────

Qwen2ForwardResult Qwen2Base::forward(
    const FrozenBase& base,
    const Tensor&     tokens,
    int B, int T,
    const Device&     dev)
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

    // ── 1. Embed ─────────────────────────────────────────────
    res.embed_in = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    ops::embed_fwd(base.embed_w.bf16(), tokens.i32(),
                   res.embed_in.bf16(), BT, H, dev.stream());

    Tensor residual = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    cudaMemcpyAsync(residual.data(), res.embed_in.data(),
                    residual.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

    res.layer_cache.resize(bc.num_layers);

    // ── 2. Transformer layers ─────────────────────────────────
    for (std::size_t l = 0; l < bc.num_layers; l++) {
        const LayerWeights& lw = base.layers[l];
        Qwen2LayerCache&    lc = res.layer_cache[l];

        // Input layernorm
        lc.in_norm_rms = Tensor::empty_f32({(std::size_t)BT}, dev);
        lc.x_normed    = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_fwd(
            residual.bf16(), lw.input_norm_w.bf16(),
            lc.x_normed.bf16(), lc.in_norm_rms.f32(),
            B, T, H, eps, dev.stream());

        // Q, K, V projections
        int kv_dim = Hkv * hd;
        lc.Q = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)(Hq * hd)}, dev);
        lc.K = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)kv_dim}, dev);
        lc.V = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)kv_dim}, dev);

        proj_fwd(dev, lw.q_proj_w, lc.x_normed, lc.Q, BT, H, Hq * hd);
        proj_fwd(dev, lw.k_proj_w, lc.x_normed, lc.K, BT, H, kv_dim);
        proj_fwd(dev, lw.v_proj_w, lc.x_normed, lc.V, BT, H, kv_dim);

        // RoPE (in-place)
        ops::rope_fwd(lc.Q.bf16(), B, T, Hq,  hd, bc.rope_theta, dev.stream());
        ops::rope_fwd(lc.K.bf16(), B, T, Hkv, hd, bc.rope_theta, dev.stream());

        // Attention
        lc.attn_out = Tensor::empty_bf16(
            {(std::size_t)BT, (std::size_t)(Hq * hd)}, dev);
        lc.attn_w = Tensor::empty_f32(
            {(std::size_t)B, (std::size_t)Hq,
             (std::size_t)T, (std::size_t)T}, dev);
        ops::sdpa_fwd(lc.Q.bf16(), lc.K.bf16(), lc.V.bf16(),
                      lc.attn_out.bf16(), lc.attn_w.f32(),
                      B, T, Hq, Hkv, hd, dev.stream());

        // O projection
        lc.h_mid = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        proj_fwd(dev, lw.o_proj_w, lc.attn_out, lc.h_mid, BT, Hq * hd, H);

        // Residual add
        ops::add_inplace(residual.bf16(), lc.h_mid.bf16(), BT * H, dev.stream());

        // Post-attention layernorm
        lc.post_norm_rms = Tensor::empty_f32({(std::size_t)BT}, dev);
        lc.x_normed2     = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_fwd(
            residual.bf16(), lw.post_norm_w.bf16(),
            lc.x_normed2.bf16(), lc.post_norm_rms.f32(),
            B, T, H, eps, dev.stream());

        // Gate + up projections
        lc.gate_out = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        lc.up_out   = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        proj_fwd(dev, lw.gate_proj_w, lc.x_normed2, lc.gate_out, BT, H, I);
        proj_fwd(dev, lw.up_proj_w,   lc.x_normed2, lc.up_out,   BT, H, I);

        // SiLU gated activation
        lc.act_out = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        ops::silu_mul_fwd(lc.gate_out.bf16(), lc.up_out.bf16(),
                          lc.act_out.bf16(), BT * I, dev.stream());

        // Down projection
        lc.ffn_out = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        proj_fwd(dev, lw.down_proj_w, lc.act_out, lc.ffn_out, BT, I, H);

        // Residual add
        ops::add_inplace(residual.bf16(), lc.ffn_out.bf16(), BT * H, dev.stream());
    }

    // ── 3. Final norm ─────────────────────────────────────────
    res.final_rms = Tensor::empty_f32({(std::size_t)BT}, dev);
    Tensor x_final = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    ops::rms_norm_fwd(
        residual.bf16(), base.final_norm_w.bf16(),
        x_final.bf16(), res.final_rms.f32(),
        B, T, H, eps, dev.stream());

    // ── 4. LM head — logits[BT, V] = x_final[BT, H] @ lm_w[V, H]^T ──
    // Same shape as proj_fwd: out_dim=V, in_dim=H
    //   transA=T(lm_w), transB=N(x_final), M=V, N=BT, K=H
    //   lda=H, ldb=H, ldc=V
    const Tensor& lm_w = bc.tie_embeddings ? base.embed_w : base.lm_head_w;
    res.logits  = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)V}, dev);
    res.dlogits = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)V}, dev);
    {
        GemmParams gp;
        gp.transA = true;  gp.transB = false;
        gp.M = V;          gp.N = BT;   gp.K = H;
        gp.alpha = 1.f;    gp.beta = 0.f;
        gemm_bf16(dev, gp,
                  lm_w.bf16(),    H,
                  x_final.bf16(), H,
                  res.logits.bf16(), V);
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

    // Backward through LM head: dres[BT,H] = dlogits[BT,V] @ lm_w[V,H]
    //   proj_bwd_x: W=[V,H], dy=[BT,V], dx=[BT,H]
    //   out_dim=V, in_dim=H
    const Tensor& lm_w = bc.tie_embeddings ? base.embed_w : base.lm_head_w;
    Tensor dres = Tensor::zeros({(std::size_t)BT, (std::size_t)H},
                                core::DType::BF16, dev);
    proj_bwd_x(dev, lm_w, fwd.dlogits, dres, BT, bc.vocab_size, H);

    // Backward through final RMSNorm.
    Tensor dfinal_norm_w = Tensor::zeros_f32({(std::size_t)H}, dev);
    Tensor dres_norm     = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
    ops::rms_norm_bwd(
        dres.bf16(), fwd.logits.bf16(), base.final_norm_w.bf16(),
        fwd.final_rms.f32(),
        dres_norm.bf16(), dfinal_norm_w.f32(),
        B, T, H, eps, dev.stream());

    Tensor grad = std::move(dres_norm);

    for (int l = (int)bc.num_layers - 1; l >= 0; l--) {
        const LayerWeights&    lw = base.layers[l];
        const Qwen2LayerCache& lc = fwd.layer_cache[l];
        LayerGrads&            lg = result[l];

        // ── FFN backward ──────────────────────────────────────

        // Copy gradient for FFN residual branch.
        Tensor dffn = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(dffn.data(), grad.data(), grad.nbytes(),
                        cudaMemcpyDeviceToDevice, dev.stream());

        // dact[BT,I] = dffn[BT,H] @ down_proj[H,I]
        //   W=down_proj[H,I], out_dim=H, in_dim=I
        Tensor dact = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        proj_bwd_x(dev, lw.down_proj_w, dffn, dact, BT, H, I);

        // Backward through SiLU gate.
        Tensor dgate = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        Tensor dup   = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)I}, dev);
        ops::silu_mul_bwd(dact.bf16(), lc.gate_out.bf16(), lc.up_out.bf16(),
                          dgate.bf16(), dup.bf16(), BT * I, dev.stream());

        // dx_normed2[BT,H] = dgate @ gate_proj + dup @ up_proj
        Tensor dx_normed2 = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)H}, core::DType::BF16, dev);
        proj_bwd_x(dev, lw.gate_proj_w, dgate, dx_normed2, BT, I, H, 0.f);
        proj_bwd_x(dev, lw.up_proj_w,   dup,   dx_normed2, BT, I, H, 1.f);

        lg.dx_ffn_in = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(lg.dx_ffn_in.data(), dx_normed2.data(),
                        dx_normed2.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

        // Backward through post-attention layernorm.
        Tensor dpost_norm_w = Tensor::zeros_f32({(std::size_t)H}, dev);
        Tensor dffn_in      = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_bwd(
            dx_normed2.bf16(), lc.x_normed2.bf16(), lw.post_norm_w.bf16(),
            lc.post_norm_rms.f32(),
            dffn_in.bf16(), dpost_norm_w.f32(),
            B, T, H, eps, dev.stream());

        ops::add_inplace(grad.bf16(), dffn_in.bf16(), BT * H, dev.stream());

        // ── Attention backward ────────────────────────────────

        Tensor do_in = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(do_in.data(), grad.data(), grad.nbytes(),
                        cudaMemcpyDeviceToDevice, dev.stream());

        // dx_o_in[BT, Hq*hd] = do_in[BT,H] @ o_proj[H, Hq*hd]
        //   W=o_proj[H, Hq*hd], out_dim=H, in_dim=Hq*hd
        lg.dx_o_in = Tensor::empty_bf16(
            {(std::size_t)BT, (std::size_t)(Hq * hd)}, dev);
        proj_bwd_x(dev, lw.o_proj_w, do_in, lg.dx_o_in, BT, H, Hq * hd);

        // Backward through SDPA.
        Tensor dQ = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)(Hq * hd)}, core::DType::BF16, dev);
        Tensor dK = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)kv_dim}, core::DType::BF16, dev);
        Tensor dV = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)kv_dim}, core::DType::BF16, dev);
        ops::sdpa_bwd(lg.dx_o_in.bf16(),
                      lc.Q.bf16(), lc.K.bf16(), lc.V.bf16(),
                      lc.attn_w.f32(),
                      dQ.bf16(), dK.bf16(), dV.bf16(),
                      B, T, Hq, Hkv, hd, dev.stream());

        // RoPE backward.
        ops::rope_bwd(dQ.bf16(), B, T, Hq,  hd, bc.rope_theta, dev.stream());
        ops::rope_bwd(dK.bf16(), B, T, Hkv, hd, bc.rope_theta, dev.stream());

        // dx_normed[BT,H] = dQ @ q_proj + dK @ k_proj + dV @ v_proj
        Tensor dx_normed = Tensor::zeros(
            {(std::size_t)BT, (std::size_t)H}, core::DType::BF16, dev);
        proj_bwd_x(dev, lw.q_proj_w, dQ, dx_normed, BT, Hq * hd, H, 0.f);
        proj_bwd_x(dev, lw.k_proj_w, dK, dx_normed, BT, kv_dim,   H, 1.f);
        proj_bwd_x(dev, lw.v_proj_w, dV, dx_normed, BT, kv_dim,   H, 1.f);

        lg.dx_attn_in = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        cudaMemcpyAsync(lg.dx_attn_in.data(), dx_normed.data(),
                        dx_normed.nbytes(), cudaMemcpyDeviceToDevice, dev.stream());

        // Backward through input layernorm.
        Tensor din_norm_w = Tensor::zeros_f32({(std::size_t)H}, dev);
        Tensor dattn_in   = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)H}, dev);
        ops::rms_norm_bwd(
            dx_normed.bf16(), lc.x_normed.bf16(), lw.input_norm_w.bf16(),
            lc.in_norm_rms.f32(),
            dattn_in.bf16(), din_norm_w.f32(),
            B, T, H, eps, dev.stream());

        ops::add_inplace(grad.bf16(), dattn_in.bf16(), BT * H, dev.stream());
    }

    return result;
}

} // namespace tensor::base::arch