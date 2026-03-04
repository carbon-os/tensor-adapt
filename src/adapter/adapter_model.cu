#include <tensor/adapter/adapter_model.hpp>
#include <tensor/backend/cuda/ops.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace tensor::adapter {

using namespace backend::cuda;
namespace ops = backend::cuda::ops;

// ── Kaiming uniform init for A, zero for B ───────────────────

__global__ void k_kaiming_init(
    __nv_bfloat16* A, int in_dim, int rank, unsigned seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = in_dim * rank;
    if (i >= n) return;
    // LCG random in [-bound, bound], bound = sqrt(1/in_dim)
    unsigned r  = seed ^ (unsigned)i * 2654435761u;
    r = r ^ (r >> 16); r *= 0x45d9f3b; r ^= (r >> 16);
    float u   = (float)(r & 0xFFFFFF) / (float)0xFFFFFF * 2.f - 1.f;
    float bound = sqrtf(1.f / (float)in_dim);
    A[i] = __float2bfloat16(u * bound);
}

// F32 kaiming init (for master weights).
__global__ void k_kaiming_init_f32(
    float* A, int in_dim, int rank, unsigned seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = in_dim * rank;
    if (i >= n) return;
    unsigned r  = seed ^ (unsigned)i * 2654435761u;
    r = r ^ (r >> 16); r *= 0x45d9f3b; r ^= (r >> 16);
    float u   = (float)(r & 0xFFFFFF) / (float)0xFFFFFF * 2.f - 1.f;
    float bound = sqrtf(1.f / (float)in_dim);
    A[i] = u * bound;
}

LoraPair AdapterModel::make_pair(
    int in_dim, int out_dim, int rank, const Device& dev)
{
    LoraPair lp;
    lp.in_dim  = in_dim;
    lp.out_dim = out_dim;
    lp.rank    = rank;

    // A: [rank, in_dim] — kaiming init
    lp.A_bf16 = Tensor::empty_bf16({(std::size_t)rank, (std::size_t)in_dim}, dev);
    lp.A_f32  = Tensor::empty_f32 ({(std::size_t)rank, (std::size_t)in_dim}, dev);
    int nA = rank * in_dim;
    int t = 256, b = (nA + t - 1) / t;
    k_kaiming_init    <<<b, t, 0, dev.stream()>>>(lp.A_bf16.bf16(), in_dim, rank, 0x1234u);
    k_kaiming_init_f32<<<b, t, 0, dev.stream()>>>(lp.A_f32.f32(),   in_dim, rank, 0x1234u);

    // B: [out_dim, rank] — zero init
    lp.B_bf16 = Tensor::zeros({(std::size_t)out_dim, (std::size_t)rank}, core::DType::BF16, dev);
    lp.B_f32  = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank}, dev);

    // Moments + gradients
    lp.mA = Tensor::zeros_f32({(std::size_t)rank,    (std::size_t)in_dim}, dev);
    lp.vA = Tensor::zeros_f32({(std::size_t)rank,    (std::size_t)in_dim}, dev);
    lp.mB = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank},   dev);
    lp.vB = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank},   dev);
    lp.gA = Tensor::zeros_f32({(std::size_t)rank,    (std::size_t)in_dim}, dev);
    lp.gB = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank},   dev);

    return lp;
}

AdapterModel AdapterModel::create(
    const base::FrozenBase& base,
    const AdapterConfig&    cfg,
    const Device&           dev)
{
    AdapterModel m;
    m.cfg_ = cfg;

    const base::BaseConfig& bc = base.config;
    int H   = bc.hidden_size;
    int kv  = bc.num_kv_heads * bc.head_dim;

    m.layers.resize(bc.num_layers);
    for (auto& la : m.layers) {
        la.lora_q = make_pair(H, H,  cfg.rank, dev);
        la.lora_k = make_pair(H, kv, cfg.rank, dev);
        la.lora_v = make_pair(H, kv, cfg.rank, dev);
        la.lora_o = make_pair(H, H,  cfg.rank, dev);  // o_proj: in=H (attn heads), out=H
    }

    return m;
}

void AdapterModel::apply_lora_fwd(
    std::size_t, LoraPair& lp,
    const Tensor& x, Tensor& out,
    int BT, const Device& dev)
{
    // h = A @ x^T → [rank, BT]
    Tensor h = Tensor::empty_bf16({(std::size_t)lp.rank, (std::size_t)BT}, dev);
    GemmParams p1;
    p1.transA = false; p1.transB = true;
    p1.M = lp.rank; p1.N = BT; p1.K = lp.in_dim;
    p1.alpha = 1.f; p1.beta = 0.f;
    gemm_bf16(dev, p1, lp.A_bf16.bf16(), lp.in_dim, x.bf16(), lp.in_dim, h.bf16(), BT);

    // delta = B @ h → [out_dim, BT]
    Tensor delta = Tensor::empty_bf16({(std::size_t)lp.out_dim, (std::size_t)BT}, dev);
    GemmParams p2;
    p2.transA = false; p2.transB = false;
    p2.M = lp.out_dim; p2.N = BT; p2.K = lp.rank;
    p2.alpha = 1.f; p2.beta = 0.f;
    gemm_bf16(dev, p2, lp.B_bf16.bf16(), lp.rank, h.bf16(), BT, delta.bf16(), BT);

    // out += scale * delta^T   (delta is [out_dim, BT], out is [BT, out_dim])
    // Add transposed — need to reinterpret. Since out is [BT, out_dim] and delta is
    // [out_dim, BT], use lora_add over the full buffer with a transpose.
    // Simplification: rerun as BT*out_dim element-wise add using the transposed layout.
    // For correctness, compute delta^T first.
    Tensor delta_t = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.out_dim}, dev);
    GemmParams pt;
    pt.transA = true; pt.transB = false;
    pt.M = BT; pt.N = lp.out_dim; pt.K = lp.out_dim; // actually just a transpose
    // Use a simple element-wise add instead: delta is [out_dim, BT] — we need to add
    // column-major to row-major out. Do it via a GEMM transpose trick:
    // delta_t[BT, out_dim] = delta^T via cublas with 1x1 alpha/beta.
    {
        GemmParams pT;
        pT.transA = true; pT.transB = false;
        pT.M = BT; pT.N = lp.out_dim; pT.K = 1;  // not right — use in-kernel transpose
        // Actually just use cudaMemcpy2D for transpose or re-order the GEMM.
        // Correct approach: B*h already gives [out_dim, BT] — transpose before adding.
        // Use cublasGeam if available, otherwise custom kernel.
    }
    // Simplest correct fix: reformulate as out += (h^T @ B^T)^T = B @ h transpose.
    // out[BT, out_dim] += scale * (x [BT, in_dim] @ A^T [in_dim, rank]) @ B^T [rank, out_dim]
    // i.e. h2[BT, rank] = x @ A^T,  delta2[BT, out_dim] = h2 @ B^T
    Tensor h2 = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    GemmParams pA;
    pA.transA = false; pA.transB = true;
    pA.M = BT; pA.N = lp.rank; pA.K = lp.in_dim;
    pA.alpha = 1.f; pA.beta = 0.f;
    gemm_bf16(dev, pA, x.bf16(), lp.in_dim, lp.A_bf16.bf16(), lp.in_dim, h2.bf16(), lp.rank);

    Tensor delta2 = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.out_dim}, dev);
    GemmParams pB;
    pB.transA = false; pB.transB = true;
    pB.M = BT; pB.N = lp.out_dim; pB.K = lp.rank;
    pB.alpha = 1.f; pB.beta = 0.f;
    gemm_bf16(dev, pB, h2.bf16(), lp.rank, lp.B_bf16.bf16(), lp.rank, delta2.bf16(), lp.out_dim);

    float scale = cfg_.alpha / (float)cfg_.rank;
    ops::lora_add(out.bf16(), delta2.bf16(), scale, BT * lp.out_dim, dev.stream());
}

void AdapterModel::apply_lora_bwd(
    LoraPair& lp,
    const Tensor& x,        // [BT, in_dim]
    const Tensor& grad_out, // [BT, out_dim]
    Tensor& grad_x,         // [BT, in_dim] — add into
    int BT,
    const Device& dev)
{
    float scale = cfg_.alpha / (float)cfg_.rank;

    // h = x @ A^T  [BT, rank]
    Tensor h = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    GemmParams pA;
    pA.transA = false; pA.transB = true;
    pA.M = BT; pA.N = lp.rank; pA.K = lp.in_dim;
    pA.alpha = 1.f; pA.beta = 0.f;
    gemm_bf16(dev, pA, x.bf16(), lp.in_dim, lp.A_bf16.bf16(), lp.in_dim, h.bf16(), lp.rank);

    // grad_B += scale * grad_out^T @ h  → [out_dim, rank]
    // gB is F32 — cast grad_out to F32 first for accumulation.
    Tensor g_out_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.out_dim}, dev);
    ops::cast_bf16_to_f32(grad_out.bf16(), g_out_f32.f32(), BT * lp.out_dim, dev.stream());

    Tensor h_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    ops::cast_bf16_to_f32(h.bf16(), h_f32.f32(), BT * lp.rank, dev.stream());

    // gB += scale * (grad_out^T @ h)   [out_dim, BT] @ [BT, rank] = [out_dim, rank]
    GemmParams pgB;
    pgB.transA = true; pgB.transB = false;
    pgB.M = lp.out_dim; pgB.N = lp.rank; pgB.K = BT;
    pgB.alpha = scale; pgB.beta = 1.f;  // accumulate
    gemm_f32(dev, pgB, g_out_f32.f32(), lp.out_dim, h_f32.f32(), lp.rank, lp.gB.f32(), lp.rank);

    // dh = grad_out @ B  [BT, out_dim] @ [out_dim, rank] = [BT, rank]  (scale already in gB)
    // For gA we need: gA += scale * dh^T @ x  [rank, BT] @ [BT, in_dim] = [rank, in_dim]
    Tensor dh_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    GemmParams pdh;
    pdh.transA = false; pdh.transB = false;
    pdh.M = BT; pdh.N = lp.rank; pdh.K = lp.out_dim;
    pdh.alpha = 1.f; pdh.beta = 0.f;
    // Use BF16 B for the projection (doesn't need to be F32).
    Tensor g_out_scaled = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    GemmParams pdh_bf;
    pdh_bf.transA = false; pdh_bf.transB = false;
    pdh_bf.M = BT; pdh_bf.N = lp.rank; pdh_bf.K = lp.out_dim;
    pdh_bf.alpha = 1.f; pdh_bf.beta = 0.f;
    gemm_bf16(dev, pdh_bf, grad_out.bf16(), lp.out_dim,
              lp.B_bf16.bf16(), lp.rank, g_out_scaled.bf16(), lp.rank);

    Tensor dh_tmp_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    ops::cast_bf16_to_f32(g_out_scaled.bf16(), dh_tmp_f32.f32(), BT * lp.rank, dev.stream());

    Tensor x_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.in_dim}, dev);
    ops::cast_bf16_to_f32(x.bf16(), x_f32.f32(), BT * lp.in_dim, dev.stream());

    // gA += scale * dh^T @ x  [rank, BT] @ [BT, in_dim] = [rank, in_dim]
    GemmParams pgA;
    pgA.transA = true; pgA.transB = false;
    pgA.M = lp.rank; pgA.N = lp.in_dim; pgA.K = BT;
    pgA.alpha = scale; pgA.beta = 1.f;
    gemm_f32(dev, pgA, dh_tmp_f32.f32(), lp.rank, x_f32.f32(), lp.in_dim,
             lp.gA.f32(), lp.in_dim);

    // grad_x += scale * A^T @ dh^T ... simplified: grad_x += (dh @ A) * scale
    // grad_x [BT, in_dim] += dh_scaled [BT, rank] @ A [rank, in_dim]
    GemmParams pgx;
    pgx.transA = false; pgx.transB = false;
    pgx.M = BT; pgx.N = lp.in_dim; pgx.K = lp.rank;
    pgx.alpha = scale; pgx.beta = 1.f;
    gemm_bf16(dev, pgx, g_out_scaled.bf16(), lp.rank,
              lp.A_bf16.bf16(), lp.in_dim, grad_x.bf16(), lp.in_dim);
}

void AdapterModel::zero_grad(const Device& dev) {
    for (auto& la : layers) {
        for (auto* lp : {&la.lora_q, &la.lora_k, &la.lora_v, &la.lora_o}) {
            lp->gA.zero_(dev);
            lp->gB.zero_(dev);
        }
    }
}

std::size_t AdapterModel::param_count() const {
    std::size_t n = 0;
    for (const auto& la : layers) {
        for (const auto* lp : {&la.lora_q, &la.lora_k, &la.lora_v, &la.lora_o}) {
            n += lp->rank * lp->in_dim + lp->out_dim * lp->rank;
        }
    }
    return n;
}

} // namespace tensor::adapter