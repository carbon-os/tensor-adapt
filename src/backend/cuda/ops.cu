// ops.cu
#include <tensor/backend/cuda/ops.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <float.h>

namespace tensor::backend::cuda::ops {

// ─────────────────────────────────────────────────────────────
//  AtomicAdd shim for BF16 on pre-Ampere hardware (sm < 80)
// ─────────────────────────────────────────────────────────────

__device__ __forceinline__
void atomic_add_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    unsigned short* addr = (unsigned short*)address;
    unsigned short old = *addr, assumed;
    do {
        assumed = old;
        float fsum = __bfloat162float(*(__nv_bfloat16*)&assumed)
                   + __bfloat162float(val);
        __nv_bfloat16 bsum = __float2bfloat16(fsum);
        unsigned short nv  = *(unsigned short*)&bsum;
        old = atomicCAS(addr, assumed, nv);
    } while (assumed != old);
#else
    atomicAdd(address, val);
#endif
}

// ─────────────────────────────────────────────────────────────
//  Warp-level reductions
// ─────────────────────────────────────────────────────────────

static constexpr int WARP = 32;

__device__ __forceinline__
float warp_reduce_sum(float v) {
    for (int mask = WARP/2; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

__device__ __forceinline__
float warp_reduce_max(float v) {
    for (int mask = WARP/2; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}

// ─────────────────────────────────────────────────────────────
//  RMSNorm forward
// ─────────────────────────────────────────────────────────────

__global__ void k_rms_norm_fwd(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__       out,
    float* __restrict__               rms_out,
    int D, float eps)
{
    int row = blockIdx.x;
    const __nv_bfloat16* xr  = x   + row * D;
    __nv_bfloat16*       orr = out  + row * D;

    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xi = __bfloat162float(xr[i]);
        ss += xi * xi;
    }
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    ss = warp_reduce_sum(ss);
    if (lane == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : 0.f;
        ss = warp_reduce_sum(ss);
        if (threadIdx.x == 0) {
            float rms = rsqrtf(ss / (float)D + eps);
            smem[0]      = rms;
            rms_out[row] = rms;
        }
    }
    __syncthreads();
    float rms = smem[0];

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xi = __bfloat162float(xr[i]);
        float wi = __bfloat162float(w[i]);
        orr[i] = __float2bfloat16(xi * rms * wi);
    }
}

void rms_norm_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* w,
    __nv_bfloat16* out, float* rms_out,
    int B, int T, int D, float eps, cudaStream_t stream)
{
    int rows    = B * T;
    int threads = min(((D + WARP - 1) / WARP) * WARP, 1024);
    k_rms_norm_fwd<<<rows, threads, 0, stream>>>(x, w, out, rms_out, D, eps);
}

// ─────────────────────────────────────────────────────────────
//  RMSNorm backward
// ─────────────────────────────────────────────────────────────

__global__ void k_rms_norm_bwd(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const float* __restrict__         rms_saved,
    __nv_bfloat16* __restrict__       dx,
    float* __restrict__               dw,
    int D, float eps)
{
    int row = blockIdx.x;
    const __nv_bfloat16* dyr = dy  + row * D;
    const __nv_bfloat16* xr  = x   + row * D;
    __nv_bfloat16*       dxr = dx  + row * D;
    float rms = rms_saved[row];

    float dot = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float dyi   = __bfloat162float(dyr[i]);
        float xi    = __bfloat162float(xr[i]);
        float wi    = __bfloat162float(w[i]);
        float x_hat = xi * rms;
        dot += dyi * wi * x_hat;
    }
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    dot = warp_reduce_sum(dot);
    if (lane == 0) smem[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        dot = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : 0.f;
        dot = warp_reduce_sum(dot);
        if (threadIdx.x == 0) smem[0] = dot;
    }
    __syncthreads();
    float correction = smem[0] / (float)D;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float dyi   = __bfloat162float(dyr[i]);
        float xi    = __bfloat162float(xr[i]);
        float wi    = __bfloat162float(w[i]);
        float x_hat = xi * rms;
        float dxi   = rms * (wi * dyi - x_hat * correction);
        dxr[i] = __float2bfloat16(dxi);
        atomicAdd(&dw[i], dyi * x_hat);
    }
}

void rms_norm_bwd(
    const __nv_bfloat16* dy, const __nv_bfloat16* x,
    const __nv_bfloat16* w, const float* rms_saved,
    __nv_bfloat16* dx, float* dw,
    int B, int T, int D, float eps, cudaStream_t stream)
{
    int rows    = B * T;
    int threads = min(((D + WARP - 1) / WARP) * WARP, 1024);
    k_rms_norm_bwd<<<rows, threads, 0, stream>>>(
        dy, x, w, rms_saved, dx, dw, D, eps);
}

// ─────────────────────────────────────────────────────────────
//  RoPE
// ─────────────────────────────────────────────────────────────

__global__ void k_rope(
    __nv_bfloat16* __restrict__ x,
    int T, int H, int hd, float theta, bool inverse)
{
    int b   = blockIdx.z, t = blockIdx.y, h = blockIdx.x;
    int idx = (b * T * H + t * H + h) * hd;
    for (int i = threadIdx.x; i < hd / 2; i += blockDim.x) {
        float freq  = powf(theta, -2.f * (float)i / (float)hd);
        float angle = (float)t * freq;
        if (inverse) angle = -angle;
        float cs = cosf(angle), sn = sinf(angle);
        float x0 = __bfloat162float(x[idx + 2*i    ]);
        float x1 = __bfloat162float(x[idx + 2*i + 1]);
        x[idx + 2*i    ] = __float2bfloat16(x0 * cs - x1 * sn);
        x[idx + 2*i + 1] = __float2bfloat16(x0 * sn + x1 * cs);
    }
}

void rope_fwd(__nv_bfloat16* x,
              int B, int T, int H, int hd, float theta, cudaStream_t s) {
    dim3 grid(H, T, B);
    k_rope<<<grid, min(hd/2, 256), 0, s>>>(x, T, H, hd, theta, false);
}

void rope_bwd(__nv_bfloat16* dx,
              int B, int T, int H, int hd, float theta, cudaStream_t s) {
    dim3 grid(H, T, B);
    k_rope<<<grid, min(hd/2, 256), 0, s>>>(dx, T, H, hd, theta, true);
}

// ─────────────────────────────────────────────────────────────
//  Scalar SDPA kernels — Volta / Turing fallback (sm < 80)
// ─────────────────────────────────────────────────────────────

__global__ void k_sdpa_fwd_scalar(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__       out,
    float* __restrict__               attn,
    int T, int Hq, int Hkv, int hd)
{
    int b = blockIdx.z, hq = blockIdx.y, tq = blockIdx.x;
    int hkv = hq / (Hq / Hkv);
    float scale = rsqrtf((float)hd);
    const __nv_bfloat16* Qrow = Q + (b * T * Hq + tq * Hq + hq) * hd;
    extern __shared__ float smem[];

    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        const __nv_bfloat16* Krow = K + (b * T * Hkv + tk * Hkv + hkv) * hd;
        float dot = 0.f;
        for (int i = 0; i < hd; i++)
            dot += __bfloat162float(Qrow[i]) * __bfloat162float(Krow[i]);
        smem[tk] = dot * scale;
    }
    for (int tk = tq + 1 + threadIdx.x; tk < T; tk += blockDim.x)
        smem[tk] = -1e38f;
    __syncthreads();

    __shared__ float smax;
    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int i = 0; i < T; i++) mx = fmaxf(mx, smem[i]);
        smax = mx;
    }
    __syncthreads();
    float mx = smax;

    float sum = 0.f;
    for (int tk = threadIdx.x; tk < T; tk += blockDim.x) {
        smem[tk] = expf(smem[tk] - mx);
        sum += smem[tk];
    }
    __syncthreads();
    __shared__ float ssum;
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < T; i++) s += smem[i];
        ssum = s;
    }
    __syncthreads();
    float inv_sum = 1.f / ssum;

    int attn_row = (b * Hq + hq) * T * T + tq * T;
    for (int tk = threadIdx.x; tk < T; tk += blockDim.x) {
        float aw = smem[tk] * inv_sum;
        smem[tk]          = aw;
        attn[attn_row+tk] = aw;
    }
    __syncthreads();

    __nv_bfloat16* outrow = out + (b * T * Hq + tq * Hq + hq) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int tk = 0; tk <= tq; tk++) {
            const __nv_bfloat16* Vrow = V + (b * T * Hkv + tk * Hkv + hkv) * hd;
            acc += smem[tk] * __bfloat162float(Vrow[i]);
        }
        outrow[i] = __float2bfloat16(acc);
    }
}

__global__ void k_sdpa_bwd_scalar(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__         attn_w,
    __nv_bfloat16* __restrict__       dQ,
    __nv_bfloat16* __restrict__       dK,
    __nv_bfloat16* __restrict__       dV,
    int T, int Hq, int Hkv, int hd)
{
    int b = blockIdx.z, hq = blockIdx.y, tq = blockIdx.x;
    int hkv = hq / (Hq / Hkv);
    float scale = rsqrtf((float)hd);

    const float* A = attn_w + (b * Hq + hq) * T * T + tq * T;
    const __nv_bfloat16* dO  = dout + (b * T * Hq + tq * Hq + hq) * hd;
    const __nv_bfloat16* Qr  = Q    + (b * T * Hq + tq * Hq + hq) * hd;
    extern __shared__ float smem[];

    float dAq = 0.f;
    for (int i = 0; i < hd; i++) {
        float doi = __bfloat162float(dO[i]);
        float yi  = 0.f;
        for (int tk2 = 0; tk2 <= tq; tk2++) {
            const __nv_bfloat16* Vr2 = V + (b * T * Hkv + tk2 * Hkv + hkv) * hd;
            yi += A[tk2] * __bfloat162float(Vr2[i]);
        }
        dAq += doi * yi;
    }

    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        const __nv_bfloat16* Vr = V + (b * T * Hkv + tk * Hkv + hkv) * hd;
        float dA_raw = 0.f;
        for (int i = 0; i < hd; i++)
            dA_raw += __bfloat162float(dO[i]) * __bfloat162float(Vr[i]);
        smem[tk] = A[tk] * (dA_raw - dAq);
    }
    __syncthreads();

    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        __nv_bfloat16* dVr = dV + (b * T * Hkv + tk * Hkv + hkv) * hd;
        for (int i = 0; i < hd; i++)
            atomic_add_bf16(dVr + i,
                __float2bfloat16(A[tk] * __bfloat162float(dO[i])));
    }

    __nv_bfloat16* dQr = dQ + (b * T * Hq + tq * Hq + hq) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int tk = 0; tk <= tq; tk++) {
            const __nv_bfloat16* Kr = K + (b * T * Hkv + tk * Hkv + hkv) * hd;
            acc += smem[tk] * __bfloat162float(Kr[i]);
        }
        dQr[i] = __float2bfloat16(__bfloat162float(dQr[i]) + acc * scale);
    }

    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        __nv_bfloat16* dKr = dK + (b * T * Hkv + tk * Hkv + hkv) * hd;
        for (int i = 0; i < hd; i++) {
            float dki = smem[tk] * __bfloat162float(Qr[i]) * scale;
            atomic_add_bf16(dKr + i, __float2bfloat16(dki));
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Causal softmax — in-place on F32 score matrix
//
//  Input/output: S[B*Hq, T, T] row-major F32.
//  Each block handles one (batch*head, tq) row of length T.
//  Positions tk > tq are masked to -inf before softmax.
// ─────────────────────────────────────────────────────────────

__global__ void k_causal_softmax_inplace(float* S, int T) {
    int row_idx = blockIdx.x;      // linear over (B*Hq*T) rows
    int tq      = row_idx % T;
    float* row  = S + row_idx * T;

    for (int tk = threadIdx.x; tk < T; tk += blockDim.x)
        if (tk > tq) row[tk] = -FLT_MAX;
    __syncthreads();

    float mx = -FLT_MAX;
    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x)
        mx = fmaxf(mx, row[tk]);
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    mx = warp_reduce_max(mx);
    if (lane == 0) smem[wid] = mx;
    __syncthreads();
    if (wid == 0) {
        mx = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : -FLT_MAX;
        mx = warp_reduce_max(mx);
        if (threadIdx.x == 0) smem[0] = mx;
    }
    __syncthreads();
    mx = smem[0];

    float s = 0.f;
    for (int tk = threadIdx.x; tk < T; tk += blockDim.x) {
        float v = (tk <= tq) ? expf(row[tk] - mx) : 0.f;
        row[tk] = v;
        s += v;
    }
    s = warp_reduce_sum(s);
    if (lane == 0) smem[wid] = s;
    __syncthreads();
    if (wid == 0) {
        s = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : 0.f;
        s = warp_reduce_sum(s);
        if (threadIdx.x == 0) smem[0] = s;
    }
    __syncthreads();
    float inv_s = 1.f / smem[0];

    for (int tk = threadIdx.x; tk < T; tk += blockDim.x)
        row[tk] *= inv_s;
}

// ─────────────────────────────────────────────────────────────
//  Softmax backward — in-place on dP buffer
//
//  dS[tq,tk] = P[tq,tk] * (dP[tq,tk] - dot)
//  where dot = sum_j P[tq,j] * dP[tq,j]
// ─────────────────────────────────────────────────────────────

__global__ void k_softmax_bwd_inplace(
    const float* __restrict__ P,
    float* __restrict__       dP,
    int T)
{
    int row_idx = blockIdx.x;
    int tq      = row_idx % T;
    const float* Prow  = P  + row_idx * T;
    float*       dProw = dP + row_idx * T;

    float dot = 0.f;
    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x)
        dot += Prow[tk] * dProw[tk];
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    dot = warp_reduce_sum(dot);
    if (lane == 0) smem[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        dot = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : 0.f;
        dot = warp_reduce_sum(dot);
        if (threadIdx.x == 0) smem[0] = dot;
    }
    __syncthreads();
    dot = smem[0];

    for (int tk = threadIdx.x; tk < T; tk += blockDim.x) {
        float ds  = (tk <= tq) ? Prow[tk] * (dProw[tk] - dot) : 0.f;
        dProw[tk] = ds;
    }
}

// ─────────────────────────────────────────────────────────────
//  Layout kernels for the batched SDPA path
//
//  All tensors in the rest of the codebase use these layouts:
//    Q, K, V, dQ, dK, dV, out:  [BT, H*hd]  where BT = B*T
//    attn_w:                     [B*Hq, T, T] (= [B, Hq, T, T] contiguous)
//
//  The batched GEMM path needs:
//    Qp, Kp, Vp:   [B*Hq, T, hd]  (each query head as a separate batch)
//
//  pack_q    : [BT, H*hd]    → [B*H, T, hd]         (reorder dims)
//  unpack_out: [B*H, T, hd]  → [BT, H*hd]           (reverse)
//  expand_kv : [BT, Hkv*hd]  → [B*Hq, T, hd]        (GQA broadcast)
//  contract_kv_f32: [B*Hq, T, hd] F32 → [BT, Hkv*hd] F32  (GQA sum)
//
//  Index identity used throughout:
//    [BT, H*hd] element (bt, h, d)  →  bt*(H*hd) + h*hd + d
//    where bt = b*T + t, so this equals b*(T*H*hd) + t*(H*hd) + h*hd + d.
//    [B*H, T, hd] element (b*H+h, t, d) → (b*H+h)*(T*hd) + t*hd + d
//                                         = b*(H*T*hd) + h*(T*hd) + t*hd + d.
//    Both address the same (b,t,h,d) element — the kernels below just
//    reorder which dimension varies fastest, which is the transpose.
// ─────────────────────────────────────────────────────────────

// [BT, H*hd]  →  [B*H, T, hd]
__global__ void k_pack_q(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__       dst,
    int B, int T, int H, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * T * hd) return;

    int d = idx % hd;
    int t = (idx / hd) % T;
    int h = (idx / (hd * T)) % H;
    int b = idx / (hd * T * H);

    // src layout: [BT, H*hd]  element (b*T+t, h, d)
    int src_idx = (b * T + t) * (H * hd) + h * hd + d;
    dst[idx] = src[src_idx];
}

// [B*H, T, hd]  →  [BT, H*hd]
__global__ void k_unpack_out(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__       dst,
    int B, int T, int H, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * T * hd) return;

    int d = idx % hd;
    int t = (idx / hd) % T;
    int h = (idx / (hd * T)) % H;
    int b = idx / (hd * T * H);

    // dst layout: [BT, H*hd]  element (b*T+t, h, d)
    int dst_idx = (b * T + t) * (H * hd) + h * hd + d;
    dst[dst_idx] = src[idx];
}

// F32 version of unpack for gradient output
__global__ void k_unpack_out_f32_to_bf16(
    const float* __restrict__   src,
    __nv_bfloat16* __restrict__ dst,
    int B, int T, int H, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * T * hd) return;

    int d = idx % hd;
    int t = (idx / hd) % T;
    int h = (idx / (hd * T)) % H;
    int b = idx / (hd * T * H);

    int dst_idx = (b * T + t) * (H * hd) + h * hd + d;
    dst[dst_idx] = __float2bfloat16(src[idx]);
}

// [BT, Hkv*hd]  →  [B*Hq, T, hd]   (GQA broadcast: each kv head fans out to ratio query heads)
__global__ void k_expand_kv_pack(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__       dst,
    int B, int T, int Hkv, int Hq, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * Hq * T * hd) return;

    int d   = idx % hd;
    int t   = (idx / hd) % T;
    int hq  = (idx / (hd * T)) % Hq;
    int b   = idx / (hd * T * Hq);
    int hkv = hq / (Hq / Hkv);

    // src layout: [BT, Hkv*hd]  element (b*T+t, hkv, d)
    int src_idx = (b * T + t) * (Hkv * hd) + hkv * hd + d;
    dst[idx] = src[src_idx];
}

// Accumulate [B*Hq, T, hd] F32  →  [BT, Hkv*hd] F32  (GQA reduce)
// Caller must zero the destination before calling.
__global__ void k_contract_kv_f32(
    const float* __restrict__ src,   // [B*Hq, T, hd] F32
    float* __restrict__       dst,   // [BT, Hkv*hd]  F32
    int B, int T, int Hkv, int Hq, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * Hq * T * hd) return;

    int d   = idx % hd;
    int t   = (idx / hd) % T;
    int hq  = (idx / (hd * T)) % Hq;
    int b   = idx / (hd * T * Hq);
    int hkv = hq / (Hq / Hkv);

    int dst_idx = (b * T + t) * (Hkv * hd) + hkv * hd + d;
    atomicAdd(&dst[dst_idx], src[idx]);
}

// ─────────────────────────────────────────────────────────────
//  Batched SDPA forward — Ampere / Ada (sm >= 80)
//
//  All GEMMs run in F32 to avoid mixed-precision type mismatches.
//  BF16 inputs are cast to F32 immediately after packing; the final
//  attention output is cast back to BF16 before unpacking.
//
//  GEMM argument order — row-major convention used throughout:
//
//    S[T,T]   = scale * Q[T,K] @ K[T,K]^T
//      Rule: C[m,n]=A[m,k]@B[n,k]^T → M=n, N=m, K=k, opA=T, opB=N,
//            cuBLAS A_arg = B_rm (the transposed matrix = Kp),
//            cuBLAS B_arg = A_rm (Qp)
//      → opA=T, opB=N, M=T, N=T, K=hd, A_arg=Kp, lda=hd, B_arg=Qp, ldb=hd
//
//    O[T,hd]  = P[T,T] @ V[T,hd]
//      Rule: C[m,n]=A[m,k]@B[k,n] → M=n, N=m, K=k, opA=N, opB=N,
//            cuBLAS A_arg = B_rm (Vp), cuBLAS B_arg = A_rm (P)
//      → opA=N, opB=N, M=hd, N=T, K=T, A_arg=Vp, lda=hd, B_arg=P, ldb=T
// ─────────────────────────────────────────────────────────────

static void sdpa_fwd_batched(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16*       out,
    float*               attn_w,
    int B, int T, int Hq, int Hkv, int hd,
    const Device& dev)
{
    cudaStream_t s = dev.stream();
    int BHq   = B * Hq;
    int nQ    = B * Hq  * T * hd;   // elements in packed Q / output
    int nKVex = B * Hq  * T * hd;   // elements in expanded K or V
    float scale = 1.f / sqrtf((float)hd);

    // ── 1. Pack Q: [BT, Hq*hd] → [B*Hq, T, hd] BF16 ─────────
    __nv_bfloat16 *Qp_bf16, *Kp_bf16, *Vp_bf16;
    CUDA_CHECK(cudaMalloc(&Qp_bf16, nQ    * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Kp_bf16, nKVex * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Vp_bf16, nKVex * sizeof(__nv_bfloat16)));

    k_pack_q       <<<(nQ   +255)/256,256,0,s>>>(Q, Qp_bf16, B,T,Hq,  hd);
    k_expand_kv_pack<<<(nKVex+255)/256,256,0,s>>>(K, Kp_bf16, B,T,Hkv,Hq,hd);
    k_expand_kv_pack<<<(nKVex+255)/256,256,0,s>>>(V, Vp_bf16, B,T,Hkv,Hq,hd);

    // ── 2. Cast packed tensors to F32 ─────────────────────────
    // All GEMMs run in F32 — avoids mixed BF16/F32 type combinations
    // that are not guaranteed to be supported by cublasGemmStridedBatchedEx.
    float *Qp, *Kp, *Vp;
    CUDA_CHECK(cudaMalloc(&Qp, nQ    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Kp, nKVex * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Vp, nKVex * sizeof(float)));

    cast_bf16_to_f32(Qp_bf16, Qp, nQ,    s);
    cast_bf16_to_f32(Kp_bf16, Kp, nKVex, s);
    cast_bf16_to_f32(Vp_bf16, Vp, nKVex, s);

    cudaFree(Qp_bf16); cudaFree(Kp_bf16); cudaFree(Vp_bf16);

    // ── 3. S[B*Hq, T, T] = scale * Q[T,hd] @ K[T,hd]^T ──────
    //
    // Convention: C_rm[m,n] = A_rm[m,k] @ B_rm[n,k]^T
    //   → cuBLAS: M=n, N=m, K=k, opA=T, opB=N,
    //             A_arg = B_rm = Kp  (the matrix being transposed)
    //             B_arg = A_rm = Qp
    //   → M=T, N=T, K=hd, A_arg=Kp, lda=hd, B_arg=Qp, ldb=hd, C=S, ldc=T
    float* S;
    CUDA_CHECK(cudaMalloc(&S, (size_t)BHq * T * T * sizeof(float)));
    {
        float beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            dev.cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, hd,
            &scale,
            Kp, CUDA_R_32F, hd, (long long)(T * hd),   // A_arg = Kp
            Qp, CUDA_R_32F, hd, (long long)(T * hd),   // B_arg = Qp
            &beta,
            S,  CUDA_R_32F, T,  (long long)(T * T),
            BHq,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // ── 4. Causal softmax in-place (S → P) ────────────────────
    {
        int rows    = BHq * T;
        int threads = min(((T + WARP - 1) / WARP) * WARP, 1024);
        k_causal_softmax_inplace<<<rows, threads, 0, s>>>(S, T);
    }
    CUDA_CHECK(cudaMemcpyAsync(attn_w, S,
        (size_t)BHq * T * T * sizeof(float),
        cudaMemcpyDeviceToDevice, s));

    // ── 5. O[B*Hq, T, hd] = P[T,T] @ V[T,hd] ─────────────────
    //
    // Convention: C_rm[m,n] = A_rm[m,k] @ B_rm[k,n]
    //   → cuBLAS: M=n, N=m, K=k, opA=N, opB=N,
    //             A_arg = B_rm = Vp,  B_arg = A_rm = P
    //   → M=hd, N=T, K=T, A_arg=Vp, lda=hd, B_arg=P, ldb=T, C=Op, ldc=hd
    float* Op_f32;
    CUDA_CHECK(cudaMalloc(&Op_f32, (size_t)nQ * sizeof(float)));
    {
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            dev.cublas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            hd, T, T,
            &alpha,
            Vp, CUDA_R_32F, hd, (long long)(T * hd),   // A_arg = Vp
            S,  CUDA_R_32F, T,  (long long)(T * T),    // B_arg = P (S after softmax)
            &beta,
            Op_f32, CUDA_R_32F, hd, (long long)(T * hd),
            BHq,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // ── 6. Cast + unpack O: [B*Hq, T, hd] → [BT, Hq*hd] ─────
    k_unpack_out_f32_to_bf16<<<(nQ+255)/256, 256, 0, s>>>(Op_f32, out, B, T, Hq, hd);

    cudaFree(Qp); cudaFree(Kp); cudaFree(Vp);
    cudaFree(S);  cudaFree(Op_f32);
}

// ─────────────────────────────────────────────────────────────
//  Batched SDPA backward — Ampere / Ada (sm >= 80)
//
//  All GEMMs in F32 (same reasoning as forward).
//
//  GEMM argument order:
//
//    dV[T,hd]  = P^T[T,T] @ dO[T,hd]
//      C[m,n]=A[k,m]^T@B[k,n] → M=n=hd, N=m=T, K=k=T, opA=N, opB=T,
//        A_arg=B_rm=dO (lda=hd), B_arg=A_rm=P (ldb=T)
//
//    dP[T,T]   = dO[T,hd] @ V[T,hd]^T
//      C[m,n]=A[m,k]@B[n,k]^T → M=n=T, N=m=T, K=k=hd, opA=T, opB=N,
//        A_arg=B_rm=Vp (lda=hd), B_arg=A_rm=dO (ldb=hd)
//
//    dQ[T,hd]  = scale * dS[T,T] @ K[T,hd]
//      C[m,n]=A[m,k]@B[k,n] → M=n=hd, N=m=T, K=k=T, opA=N, opB=N,
//        A_arg=B_rm=Kp (lda=hd), B_arg=A_rm=dS (ldb=T)
//
//    dK[T,hd]  = scale * dS^T[T,T] @ Q[T,hd]
//      C[m,n]=A[k,m]^T@B[k,n] → M=n=hd, N=m=T, K=k=T, opA=N, opB=T,
//        A_arg=B_rm=Qp (lda=hd), B_arg=A_rm=dS (ldb=T)
// ─────────────────────────────────────────────────────────────

static void sdpa_bwd_batched(
    const __nv_bfloat16* dout,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float*         attn_w,
    __nv_bfloat16*       dQ,
    __nv_bfloat16*       dK,
    __nv_bfloat16*       dV,
    int B, int T, int Hq, int Hkv, int hd,
    const Device& dev)
{
    cudaStream_t s = dev.stream();
    int BHq   = B * Hq;
    int nQ    = B * Hq  * T * hd;
    int nKVex = B * Hq  * T * hd;
    int nKV   = B * Hkv * T * hd;

    // ── 1. Pack inputs to [B*Hq, T, hd] BF16, then cast to F32 ──
    __nv_bfloat16 *dOp_bf16, *Qp_bf16, *Kp_bf16, *Vp_bf16;
    CUDA_CHECK(cudaMalloc(&dOp_bf16, nQ    * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Qp_bf16,  nQ    * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Kp_bf16,  nKVex * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Vp_bf16,  nKVex * sizeof(__nv_bfloat16)));

    k_pack_q        <<<(nQ   +255)/256,256,0,s>>>(dout, dOp_bf16, B,T,Hq,  hd);
    k_pack_q        <<<(nQ   +255)/256,256,0,s>>>(Q,    Qp_bf16,  B,T,Hq,  hd);
    k_expand_kv_pack<<<(nKVex+255)/256,256,0,s>>>(K,    Kp_bf16,  B,T,Hkv,Hq,hd);
    k_expand_kv_pack<<<(nKVex+255)/256,256,0,s>>>(V,    Vp_bf16,  B,T,Hkv,Hq,hd);

    float *dOp, *Qp, *Kp, *Vp;
    CUDA_CHECK(cudaMalloc(&dOp, nQ    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Qp,  nQ    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Kp,  nKVex * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Vp,  nKVex * sizeof(float)));

    cast_bf16_to_f32(dOp_bf16, dOp, nQ,    s);
    cast_bf16_to_f32(Qp_bf16,  Qp,  nQ,    s);
    cast_bf16_to_f32(Kp_bf16,  Kp,  nKVex, s);
    cast_bf16_to_f32(Vp_bf16,  Vp,  nKVex, s);

    cudaFree(dOp_bf16); cudaFree(Qp_bf16);
    cudaFree(Kp_bf16);  cudaFree(Vp_bf16);

    const float* P = attn_w;   // [B*Hq, T, T] F32 — saved from forward

    // ── 2. dV[T,hd] = P^T[T,T] @ dO[T,hd] ───────────────────
    // C[m,n]=A[k,m]^T@B[k,n]: M=hd, N=T, K=T, opA=N, opB=T,
    //   A_arg=dO (lda=hd), B_arg=P (ldb=T)
    float* dVp;
    CUDA_CHECK(cudaMalloc(&dVp, (size_t)nKVex * sizeof(float)));
    {
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            dev.cublas(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            hd, T, T,
            &alpha,
            dOp, CUDA_R_32F, hd, (long long)(T * hd),  // A_arg = dO
            P,   CUDA_R_32F, T,  (long long)(T * T),   // B_arg = P
            &beta,
            dVp, CUDA_R_32F, hd, (long long)(T * hd),
            BHq,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Contract dVp [B*Hq, T, hd] F32  →  dV [BT, Hkv*hd] BF16
    {
        float* dV_f32;
        CUDA_CHECK(cudaMalloc(&dV_f32, (size_t)nKV * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(dV_f32, 0, (size_t)nKV * sizeof(float), s));
        k_contract_kv_f32<<<(nKVex+255)/256,256,0,s>>>(dVp, dV_f32, B,T,Hkv,Hq,hd);
        cast_f32_to_bf16(dV_f32, dV, nKV, s);
        cudaFree(dV_f32);
    }
    cudaFree(dVp);

    // ── 3. dP[T,T] = dO[T,hd] @ V[T,hd]^T ───────────────────
    // C[m,n]=A[m,k]@B[n,k]^T: M=T, N=T, K=hd, opA=T, opB=N,
    //   A_arg=Vp (lda=hd), B_arg=dO (ldb=hd)
    float* dPbuf;
    CUDA_CHECK(cudaMalloc(&dPbuf, (size_t)BHq * T * T * sizeof(float)));
    {
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            dev.cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, hd,
            &alpha,
            Vp,  CUDA_R_32F, hd, (long long)(T * hd),  // A_arg = Vp  ← fixed (was dO)
            dOp, CUDA_R_32F, hd, (long long)(T * hd),  // B_arg = dO  ← fixed (was Vp)
            &beta,
            dPbuf, CUDA_R_32F, T, (long long)(T * T),
            BHq,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // ── 4. dS = softmax_bwd(P, dP) in-place ──────────────────
    {
        int rows    = BHq * T;
        int threads = min(((T + WARP - 1) / WARP) * WARP, 1024);
        k_softmax_bwd_inplace<<<rows, threads, 0, s>>>(P, dPbuf, T);
    }
    // dPbuf now holds dS [B*Hq, T, T]

    // ── 5. dQ[T,hd] = scale * dS[T,T] @ K[T,hd] ─────────────
    // C[m,n]=A[m,k]@B[k,n]: M=hd, N=T, K=T, opA=N, opB=N,
    //   A_arg=Kp (lda=hd), B_arg=dS (ldb=T)
    float* dQp;
    CUDA_CHECK(cudaMalloc(&dQp, (size_t)nQ * sizeof(float)));
    {
        float alpha = 1.f / sqrtf((float)hd), beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            dev.cublas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            hd, T, T,
            &alpha,
            Kp,    CUDA_R_32F, hd, (long long)(T * hd),  // A_arg = Kp
            dPbuf, CUDA_R_32F, T,  (long long)(T * T),   // B_arg = dS
            &beta,
            dQp, CUDA_R_32F, hd, (long long)(T * hd),
            BHq,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Unpack dQp [B*Hq, T, hd] F32  →  dQ [BT, Hq*hd] BF16
    k_unpack_out_f32_to_bf16<<<(nQ+255)/256, 256, 0, s>>>(dQp, dQ, B, T, Hq, hd);
    cudaFree(dQp);

    // ── 6. dK[T,hd] = scale * dS^T[T,T] @ Q[T,hd] ───────────
    // C[m,n]=A[k,m]^T@B[k,n]: M=hd, N=T, K=T, opA=N, opB=T,
    //   A_arg=Qp (lda=hd), B_arg=dS (ldb=T)
    float* dKp;
    CUDA_CHECK(cudaMalloc(&dKp, (size_t)nKVex * sizeof(float)));
    {
        float alpha = 1.f / sqrtf((float)hd), beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            dev.cublas(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            hd, T, T,
            &alpha,
            Qp,    CUDA_R_32F, hd, (long long)(T * hd),  // A_arg = Qp
            dPbuf, CUDA_R_32F, T,  (long long)(T * T),   // B_arg = dS
            &beta,
            dKp, CUDA_R_32F, hd, (long long)(T * hd),
            BHq,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Contract dKp [B*Hq, T, hd] F32  →  dK [BT, Hkv*hd] BF16
    {
        float* dK_f32;
        CUDA_CHECK(cudaMalloc(&dK_f32, (size_t)nKV * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(dK_f32, 0, (size_t)nKV * sizeof(float), s));
        k_contract_kv_f32<<<(nKVex+255)/256,256,0,s>>>(dKp, dK_f32, B,T,Hkv,Hq,hd);
        cast_f32_to_bf16(dK_f32, dK, nKV, s);
        cudaFree(dK_f32);
    }
    cudaFree(dKp);

    cudaFree(dOp); cudaFree(Qp); cudaFree(Kp); cudaFree(Vp);
    cudaFree(dPbuf);
}

// ─────────────────────────────────────────────────────────────
//  Public SDPA entry points — dispatch on sm_major
// ─────────────────────────────────────────────────────────────

void sdpa_fwd(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* out, float* attn_w,
    int B, int T, int Hq, int Hkv, int hd,
    const Device& dev)
{
    if (dev.sm_major() >= 8) {
        sdpa_fwd_batched(Q, K, V, out, attn_w, B, T, Hq, Hkv, hd, dev);
    } else {
        dim3 grid(T, Hq, B);
        k_sdpa_fwd_scalar<<<grid, min(T, 256), T * sizeof(float), dev.stream()>>>(
            Q, K, V, out, attn_w, T, Hq, Hkv, hd);
    }
}

void sdpa_bwd(
    const __nv_bfloat16* dout,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const float* attn_w,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int T, int Hq, int Hkv, int hd,
    const Device& dev)
{
    if (dev.sm_major() >= 8) {
        sdpa_bwd_batched(dout, Q, K, V, attn_w, dQ, dK, dV,
                         B, T, Hq, Hkv, hd, dev);
    } else {
        dim3 grid(T, Hq, B);
        k_sdpa_bwd_scalar<<<grid, min(T, 64), T * sizeof(float), dev.stream()>>>(
            dout, Q, K, V, attn_w, dQ, dK, dV, T, Hq, Hkv, hd);
    }
}

// ─────────────────────────────────────────────────────────────
//  SiLU gated activation
// ─────────────────────────────────────────────────────────────

__global__ void k_silu_mul_fwd(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float s = g / (1.f + expf(-g));
    out[i]  = __float2bfloat16(s * u);
}

__global__ void k_silu_mul_bwd(
    const __nv_bfloat16* dout,
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* dgate, __nv_bfloat16* dup, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g   = __bfloat162float(gate[i]);
    float u   = __bfloat162float(up[i]);
    float do_ = __bfloat162float(dout[i]);
    float sig = 1.f / (1.f + expf(-g));
    float s   = g * sig;
    float ds  = sig * (1.f + g * (1.f - sig));
    dgate[i] = __float2bfloat16(do_ * u * ds);
    dup[i]   = __float2bfloat16(do_ * s);
}

void silu_mul_fwd(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                  __nv_bfloat16* out, int N, cudaStream_t s) {
    k_silu_mul_fwd<<<(N+255)/256, 256, 0, s>>>(gate, up, out, N);
}

void silu_mul_bwd(const __nv_bfloat16* dout, const __nv_bfloat16* gate,
                  const __nv_bfloat16* up,
                  __nv_bfloat16* dgate, __nv_bfloat16* dup,
                  int N, cudaStream_t s) {
    k_silu_mul_bwd<<<(N+255)/256, 256, 0, s>>>(dout, gate, up, dgate, dup, N);
}

// ─────────────────────────────────────────────────────────────
//  Cross-entropy loss
// ─────────────────────────────────────────────────────────────

__global__ void k_cross_entropy(
    const __nv_bfloat16* __restrict__ logits,
    const int* __restrict__           targets,
    float* __restrict__               loss_per_token,
    __nv_bfloat16* __restrict__       dlogits,
    int V)
{
    int n = blockIdx.x;
    const __nv_bfloat16* L  = logits  + n * V;
    __nv_bfloat16*       dL = dlogits + n * V;
    int tgt = targets[n];

    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        mx = fmaxf(mx, __bfloat162float(L[i]));
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    mx = warp_reduce_max(mx);
    if (lane == 0) smem[wid] = mx;
    __syncthreads();
    if (wid == 0) {
        mx = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : -FLT_MAX;
        mx = warp_reduce_max(mx);
        if (threadIdx.x == 0) smem[0] = mx;
    }
    __syncthreads();
    mx = smem[0];

    float sum = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        sum += expf(__bfloat162float(L[i]) - mx);
    sum = warp_reduce_sum(sum);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : 0.f;
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) {
            float log_sum   = logf(sum) + mx;
            float tgt_logit = (tgt >= 0) ? __bfloat162float(L[tgt]) : 0.f;
            loss_per_token[n] = -(tgt_logit - log_sum);
            smem[0] = log_sum;
        }
    }
    __syncthreads();
    float log_sum = smem[0];

    float inv_N = 1.f / (float)gridDim.x;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float p = expf(__bfloat162float(L[i]) - log_sum);
        float d = p - (float)(i == tgt);
        dL[i] = __float2bfloat16(d * inv_N);
    }
}

void cross_entropy_fwd(
    const __nv_bfloat16* logits, const int* targets,
    float* loss_out, __nv_bfloat16* dlogits,
    int N, int V, cudaStream_t s)
{
    int threads = min(((V + WARP - 1) / WARP) * WARP, 1024);
    k_cross_entropy<<<N, threads, 0, s>>>(logits, targets, loss_out, dlogits, V);
}

// ─────────────────────────────────────────────────────────────
//  Embedding
// ─────────────────────────────────────────────────────────────

__global__ void k_embed_fwd(
    const __nv_bfloat16* table, const int* tokens,
    __nv_bfloat16* out, int D)
{
    int bt = blockIdx.x, tok = tokens[bt];
    const __nv_bfloat16* row = table + tok * D;
    __nv_bfloat16*       dst = out   + bt  * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = row[i];
}

__global__ void k_embed_bwd(
    const __nv_bfloat16* dout, const int* tokens,
    float* dtable, int D)
{
    int bt = blockIdx.x, tok = tokens[bt];
    const __nv_bfloat16* src = dout   + bt  * D;
    float*               dst = dtable + tok * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        atomicAdd(&dst[i], __bfloat162float(src[i]));
}

void embed_fwd(const __nv_bfloat16* table, const int* tokens,
               __nv_bfloat16* out, int BT, int D, cudaStream_t s) {
    k_embed_fwd<<<BT, min(D, 512), 0, s>>>(table, tokens, out, D);
}

void embed_bwd(const __nv_bfloat16* dout, const int* tokens,
               float* dtable, int BT, int D, cudaStream_t s) {
    k_embed_bwd<<<BT, min(D, 512), 0, s>>>(dout, tokens, dtable, D);
}

// ─────────────────────────────────────────────────────────────
//  Bias add
// ─────────────────────────────────────────────────────────────

__global__ void k_add_bias(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ bias,
    int D)
{
    int bt = blockIdx.x;
    __nv_bfloat16* row = out + bt * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = __bfloat162float(row[i]) + __bfloat162float(bias[i]);
        row[i] = __float2bfloat16(val);
    }
}

void add_bias(
    __nv_bfloat16* out, const __nv_bfloat16* bias,
    int BT, int D, cudaStream_t s)
{
    int threads = min(D, 512);
    k_add_bias<<<BT, threads, 0, s>>>(out, bias, D);
}

// ─────────────────────────────────────────────────────────────
//  Residual + LoRA adds
// ─────────────────────────────────────────────────────────────

__global__ void k_add_inplace(
    __nv_bfloat16* out, const __nv_bfloat16* delta, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2bfloat16(
        __bfloat162float(out[i]) + __bfloat162float(delta[i]));
}

void add_inplace(__nv_bfloat16* out, const __nv_bfloat16* delta,
                 int N, cudaStream_t s) {
    k_add_inplace<<<(N+255)/256, 256, 0, s>>>(out, delta, N);
}

__global__ void k_lora_add(
    __nv_bfloat16* out, const __nv_bfloat16* delta, float scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2bfloat16(
        __bfloat162float(out[i]) + scale * __bfloat162float(delta[i]));
}

void lora_add(__nv_bfloat16* out, const __nv_bfloat16* delta,
              float scale, int N, cudaStream_t s) {
    k_lora_add<<<(N+255)/256, 256, 0, s>>>(out, delta, scale, N);
}

// ─────────────────────────────────────────────────────────────
//  Cast
// ─────────────────────────────────────────────────────────────

__global__ void k_bf16_to_f32(const __nv_bfloat16* s, float* d, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d[i] = __bfloat162float(s[i]);
}
__global__ void k_f32_to_bf16(const float* s, __nv_bfloat16* d, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d[i] = __float2bfloat16(s[i]);
}

void cast_bf16_to_f32(const __nv_bfloat16* src, float* dst, int N, cudaStream_t s) {
    k_bf16_to_f32<<<(N+255)/256, 256, 0, s>>>(src, dst, N);
}
void cast_f32_to_bf16(const float* src, __nv_bfloat16* dst, int N, cudaStream_t s) {
    k_f32_to_bf16<<<(N+255)/256, 256, 0, s>>>(src, dst, N);
}

// ─────────────────────────────────────────────────────────────
//  AdamW step
// ─────────────────────────────────────────────────────────────

__global__ void k_adamw(
    float* master, __nv_bfloat16* working,
    float* m, float* v, const float* grad,
    int N,
    float lr, float b1, float b2, float eps,
    float wd, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g  = grad[i];
    float mi = b1 * m[i] + (1.f - b1) * g;
    float vi = b2 * v[i] + (1.f - b2) * g * g;
    m[i] = mi; v[i] = vi;
    float bc1  = 1.f - powf(b1, (float)step);
    float bc2  = 1.f - powf(b2, (float)step);
    float mhat = mi / bc1, vhat = vi / bc2;
    float p = master[i] - lr * (mhat / (sqrtf(vhat) + eps) + wd * master[i]);
    master[i]  = p;
    working[i] = __float2bfloat16(p);
}

void adamw_step(
    float* master, __nv_bfloat16* working,
    float* m, float* v, const float* grad, int N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int step, cudaStream_t s)
{
    k_adamw<<<(N+255)/256, 256, 0, s>>>(
        master, working, m, v, grad, N,
        lr, beta1, beta2, eps, weight_decay, step);
}

// ─────────────────────────────────────────────────────────────
//  Gradient squared-sum accumulator
// ─────────────────────────────────────────────────────────────

__global__ void k_sum_sq_into(const float* __restrict__ g, float* accum, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float s = 0.f;
    for (int i = tid; i < N; i += blockDim.x)
        s += g[i] * g[i];
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(accum, smem[0]);
}

void sum_sq_into(const float* g, float* accum, int N, cudaStream_t s) {
    int t = min(((N + 31) / 32) * 32, 1024);
    k_sum_sq_into<<<1, t, t * sizeof(float), s>>>(g, accum, N);
}

} // namespace tensor::backend::cuda::ops