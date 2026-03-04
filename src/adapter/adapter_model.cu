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
    unsigned r  = seed ^ (unsigned)i * 2654435761u;
    r = r ^ (r >> 16); r *= 0x45d9f3b; r ^= (r >> 16);
    float u     = (float)(r & 0xFFFFFF) / (float)0xFFFFFF * 2.f - 1.f;
    float bound = sqrtf(1.f / (float)in_dim);
    A[i] = __float2bfloat16(u * bound);
}

__global__ void k_kaiming_init_f32(
    float* A, int in_dim, int rank, unsigned seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = in_dim * rank;
    if (i >= n) return;
    unsigned r  = seed ^ (unsigned)i * 2654435761u;
    r = r ^ (r >> 16); r *= 0x45d9f3b; r ^= (r >> 16);
    float u     = (float)(r & 0xFFFFFF) / (float)0xFFFFFF * 2.f - 1.f;
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

    lp.A_bf16 = Tensor::empty_bf16({(std::size_t)rank, (std::size_t)in_dim}, dev);
    lp.A_f32  = Tensor::empty_f32 ({(std::size_t)rank, (std::size_t)in_dim}, dev);
    int nA = rank * in_dim;
    int t = 256, b = (nA + t - 1) / t;
    k_kaiming_init    <<<b, t, 0, dev.stream()>>>(lp.A_bf16.bf16(), in_dim, rank, 0x1234u);
    k_kaiming_init_f32<<<b, t, 0, dev.stream()>>>(lp.A_f32.f32(),   in_dim, rank, 0x1234u);

    lp.B_bf16 = Tensor::zeros({(std::size_t)out_dim, (std::size_t)rank}, core::DType::BF16, dev);
    lp.B_f32  = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank}, dev);

    lp.mA = Tensor::zeros_f32({(std::size_t)rank,    (std::size_t)in_dim},  dev);
    lp.vA = Tensor::zeros_f32({(std::size_t)rank,    (std::size_t)in_dim},  dev);
    lp.mB = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank},    dev);
    lp.vB = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank},    dev);
    lp.gA = Tensor::zeros_f32({(std::size_t)rank,    (std::size_t)in_dim},  dev);
    lp.gB = Tensor::zeros_f32({(std::size_t)out_dim, (std::size_t)rank},    dev);

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
    int H  = bc.hidden_size;
    int kv = bc.num_kv_heads * bc.head_dim;

    m.layers.resize(bc.num_layers);
    for (auto& la : m.layers) {
        la.lora_q = make_pair(H, H,  cfg.rank, dev);
        la.lora_k = make_pair(H, kv, cfg.rank, dev);
        la.lora_v = make_pair(H, kv, cfg.rank, dev);
        la.lora_o = make_pair(H, H,  cfg.rank, dev);
    }

    return m;
}

// ─────────────────────────────────────────────────────────────
//  cuBLAS GEMM convention for row-major matrices
//
//  cuBLAS is column-major. A row-major matrix X[r,c] passed
//  with lda=c appears to cuBLAS as the col-major matrix X^T[c,r].
//
//  To compute row-major C[m,n] = op_A(A) * op_B(B):
//    C^T[n,m] = op_B(B)^T * op_A(A)^T
//
//  Practical rule per operation used here:
//
//    C[m,n]  = A[m,k]   @ B[k,n]     → M=n, N=m, K=k, tA=N, tB=N, args=(B,n,A,k,C,n)
//    C[m,n]  = A[k,m]^T @ B[k,n]     → M=n, N=m, K=k, tA=N, tB=T, args=(B,n,A,m,C,n)
//    C[m,n]  = A[m,k]   @ B[n,k]^T   → M=n, N=m, K=k, tA=T, tB=N, args=(B,k,A,k,C,n)
//    C[m,n] += A[m,k]   @ B[k,n]     → same as above with beta=1
//
//  Leading dimensions are always the col-count of the row-major matrix
//  (= the memory stride), regardless of transpose flag.
// ─────────────────────────────────────────────────────────────

void AdapterModel::apply_lora_fwd(
    std::size_t, LoraPair& lp,
    const Tensor& x, Tensor& out,
    int BT, const Device& dev)
{
    // h[BT, rank] = x[BT, in_dim] @ A[rank, in_dim]^T
    //
    // C^T[rank,BT] = A[rank,in_dim] @ x^T[in_dim,BT]
    // cuBLAS: M=rank, N=BT, K=in_dim, tA=T(A), tB=F(x)
    //   A_arg=A_bf16 (rm[rank,in]→cm[in,rank], transT→[rank,in]) lda=in_dim
    //   B_arg=x      (rm[BT,in]→cm[in,BT],     transF→[in,BT])   ldb=in_dim
    //   C_arg=h      (rm[BT,rank]→cm[rank,BT])                    ldc=rank
    Tensor h = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    {
        GemmParams p;
        p.transA = true;  p.transB = false;
        p.M = lp.rank;    p.N = BT;          p.K = lp.in_dim;
        p.alpha = 1.f;    p.beta  = 0.f;
        gemm_bf16(dev, p,
                  lp.A_bf16.bf16(), lp.in_dim,
                  x.bf16(),         lp.in_dim,
                  h.bf16(),         lp.rank);
    }

    // delta[BT, out_dim] = h[BT, rank] @ B[out_dim, rank]^T
    //
    // C^T[out_dim,BT] = B[out_dim,rank] @ h^T[rank,BT]
    // cuBLAS: M=out_dim, N=BT, K=rank, tA=T(B), tB=F(h)
    //   A_arg=B_bf16 (rm[out,rank]→cm[rank,out], transT→[out,rank]) lda=rank
    //   B_arg=h      (rm[BT,rank]→cm[rank,BT],   transF→[rank,BT])  ldb=rank
    //   C_arg=delta  (rm[BT,out]→cm[out,BT])                         ldc=out_dim
    Tensor delta = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.out_dim}, dev);
    {
        GemmParams p;
        p.transA = true;     p.transB = false;
        p.M = lp.out_dim;    p.N = BT;        p.K = lp.rank;
        p.alpha = 1.f;       p.beta  = 0.f;
        gemm_bf16(dev, p,
                  lp.B_bf16.bf16(), lp.rank,
                  h.bf16(),         lp.rank,
                  delta.bf16(),     lp.out_dim);
    }

    float scale = cfg_.alpha / (float)cfg_.rank;
    ops::lora_add(out.bf16(), delta.bf16(), scale, BT * lp.out_dim, dev.stream());
}

void AdapterModel::apply_lora_bwd(
    LoraPair& lp,
    const Tensor& x,        // [BT, in_dim]
    const Tensor& grad_out, // [BT, out_dim]
    Tensor& grad_x,         // [BT, in_dim] — accumulated into (beta=1)
    int BT,
    const Device& dev)
{
    float scale = cfg_.alpha / (float)cfg_.rank;

    // ── Recompute forward h ───────────────────────────────────
    // h[BT, rank] = x[BT, in_dim] @ A[rank, in_dim]^T
    // (same GEMM as fwd — see apply_lora_fwd comment above)
    Tensor h = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    {
        GemmParams p;
        p.transA = true;  p.transB = false;
        p.M = lp.rank;    p.N = BT;          p.K = lp.in_dim;
        p.alpha = 1.f;    p.beta  = 0.f;
        gemm_bf16(dev, p,
                  lp.A_bf16.bf16(), lp.in_dim,
                  x.bf16(),         lp.in_dim,
                  h.bf16(),         lp.rank);
    }

    // ── Promote to F32 for gradient accumulation ─────────────
    Tensor h_f32      = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.rank},    dev);
    Tensor go_f32     = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.out_dim}, dev);
    Tensor x_f32      = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.in_dim},  dev);
    ops::cast_bf16_to_f32(h.bf16(),         h_f32.f32(),  BT * lp.rank,    dev.stream());
    ops::cast_bf16_to_f32(grad_out.bf16(),  go_f32.f32(), BT * lp.out_dim, dev.stream());
    ops::cast_bf16_to_f32(x.bf16(),         x_f32.f32(),  BT * lp.in_dim,  dev.stream());

    // ── gB[out_dim, rank] += scale * grad_out[BT,out_dim]^T @ h[BT,rank] ──
    //
    // gB^T[rank,out_dim] = scale * h^T[rank,BT]_cm @ grad_out[BT,out_dim]_cm(transB)
    // cuBLAS: M=rank, N=out_dim, K=BT, tA=F(h), tB=T(go)
    //   A_arg=h_f32  (rm[BT,rank]→cm[rank,BT],   transF→[rank,BT])   lda=rank
    //   B_arg=go_f32 (rm[BT,out]→cm[out,BT],     transT→[BT,out])    ldb=out_dim
    //   C_arg=gB     (rm[out,rank]→cm[rank,out])                       ldc=rank
    {
        GemmParams p;
        p.transA = false; p.transB = true;
        p.M = lp.rank;    p.N = lp.out_dim;  p.K = BT;
        p.alpha = scale;  p.beta  = 1.f;
        gemm_f32(dev, p,
                 h_f32.f32(),  lp.rank,
                 go_f32.f32(), lp.out_dim,
                 lp.gB.f32(),  lp.rank);
    }

    // ── dh[BT, rank] = grad_out[BT, out_dim] @ B[out_dim, rank] ─
    //
    // dh^T[rank,BT] = B^T[rank,out_dim]_cm @ grad_out^T[out_dim,BT]_cm
    // = B[out_dim,rank]_rm_as_cm[rank,out_dim](transF) @ go_cm[out,BT](transF)
    // cuBLAS: M=rank, N=BT, K=out_dim, tA=F(B), tB=F(go)
    //   A_arg=B_bf16 (rm[out,rank]→cm[rank,out], transF→[rank,out]) lda=rank
    //   B_arg=go     (rm[BT,out]→cm[out,BT],     transF→[out,BT])   ldb=out_dim
    //   C_arg=dh     (rm[BT,rank]→cm[rank,BT])                       ldc=rank
    Tensor dh = Tensor::empty_bf16({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    {
        GemmParams p;
        p.transA = false;  p.transB = false;
        p.M = lp.rank;     p.N = BT;           p.K = lp.out_dim;
        p.alpha = 1.f;     p.beta  = 0.f;
        gemm_bf16(dev, p,
                  lp.B_bf16.bf16(), lp.rank,
                  grad_out.bf16(),  lp.out_dim,
                  dh.bf16(),        lp.rank);
    }
    Tensor dh_f32 = Tensor::empty_f32({(std::size_t)BT, (std::size_t)lp.rank}, dev);
    ops::cast_bf16_to_f32(dh.bf16(), dh_f32.f32(), BT * lp.rank, dev.stream());

    // ── gA[rank, in_dim] += scale * dh[BT,rank]^T @ x[BT,in_dim] ──
    //
    // gA^T[in_dim,rank] = scale * x^T[in_dim,BT]_cm @ dh[BT,rank]_cm(transB)
    // cuBLAS: M=in_dim, N=rank, K=BT, tA=F(x), tB=T(dh)
    //   A_arg=x_f32  (rm[BT,in]→cm[in,BT],   transF→[in,BT])   lda=in_dim
    //   B_arg=dh_f32 (rm[BT,rank]→cm[rank,BT], transT→[BT,rank]) ldb=rank
    //   C_arg=gA     (rm[rank,in]→cm[in,rank])                    ldc=in_dim
    {
        GemmParams p;
        p.transA = false;   p.transB = true;
        p.M = lp.in_dim;    p.N = lp.rank;    p.K = BT;
        p.alpha = scale;    p.beta  = 1.f;
        gemm_f32(dev, p,
                 x_f32.f32(),   lp.in_dim,
                 dh_f32.f32(),  lp.rank,
                 lp.gA.f32(),   lp.in_dim);
    }

    // ── grad_x[BT, in_dim] += scale * dh[BT,rank] @ A[rank,in_dim] ──
    //
    // grad_x^T[in_dim,BT] = scale * A^T[in_dim,rank]_cm @ dh^T[rank,BT]_cm
    // = A[rank,in_dim]_rm_as_cm[in,rank](transF) @ dh_cm[rank,BT](transF)
    // cuBLAS: M=in_dim, N=BT, K=rank, tA=F(A), tB=F(dh)
    //   A_arg=A_bf16 (rm[rank,in]→cm[in,rank], transF→[in,rank]) lda=in_dim
    //   B_arg=dh     (rm[BT,rank]→cm[rank,BT], transF→[rank,BT]) ldb=rank
    //   C_arg=grad_x (rm[BT,in]→cm[in,BT])                        ldc=in_dim
    {
        GemmParams p;
        p.transA = false;  p.transB = false;
        p.M = lp.in_dim;   p.N = BT;          p.K = lp.rank;
        p.alpha = scale;   p.beta  = 1.f;
        gemm_bf16(dev, p,
                  lp.A_bf16.bf16(), lp.in_dim,
                  dh.bf16(),        lp.rank,
                  grad_x.bf16(),    lp.in_dim);
    }
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