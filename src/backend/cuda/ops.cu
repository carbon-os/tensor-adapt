#include <tensor/backend/cuda/ops.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

namespace tensor::backend::cuda::ops {

// ─────────────────────────────────────────────────────────────
//  Utility
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
// One block per (b,t) row.  Threads reduce over D.

__global__ void k_rms_norm_fwd(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__       out,
    float*         __restrict__       rms_out,
    int D, float eps)
{
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x   + row * D;
    __nv_bfloat16*       or_ = out + row * D;

    // Compute sum of squares using parallel reduction over threads.
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xi = __bfloat162float(xr[i]);
        ss += xi * xi;
    }
    // Block reduce.
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP;
    int wid  = threadIdx.x / WARP;
    ss = warp_reduce_sum(ss);
    if (lane == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = (threadIdx.x < (blockDim.x / WARP)) ? smem[threadIdx.x] : 0.f;
        ss = warp_reduce_sum(ss);
        if (threadIdx.x == 0) {
            float rms = rsqrtf(ss / (float)D + eps);
            smem[0] = rms;
            rms_out[row] = rms;
        }
    }
    __syncthreads();
    float rms = smem[0];

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xi = __bfloat162float(xr[i]);
        float wi = __bfloat162float(w[i]);
        or_[i] = __float2bfloat16(xi * rms * wi);
    }
}

void rms_norm_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* w,
    __nv_bfloat16* out, float* rms_out,
    int B, int T, int D, float eps, cudaStream_t stream)
{
    int rows = B * T;
    int threads = min(((D + WARP - 1) / WARP) * WARP, 1024);
    k_rms_norm_fwd<<<rows, threads, 0, stream>>>(x, w, out, rms_out, D, eps);
}

// ─────────────────────────────────────────────────────────────
//  RMSNorm backward
// ─────────────────────────────────────────────────────────────
// dx[i] = w[i]*rms * (dy[i] - x_hat[i] * sum_j(dy[j]*w[j]*x_hat[j]) / D)
// dw[i] += sum_{b,t} dy[b,t,i] * x_hat[b,t,i]

__global__ void k_rms_norm_bwd(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const float*         __restrict__ rms_saved,
    __nv_bfloat16*       __restrict__ dx,
    float*               __restrict__ dw,
    int D, float eps)
{
    int row = blockIdx.x;
    const __nv_bfloat16* dyr = dy  + row * D;
    const __nv_bfloat16* xr  = x   + row * D;
    __nv_bfloat16*       dxr = dx  + row * D;
    float rms = rms_saved[row]; // this is rsqrt(mean(x^2)+eps)

    // dot = sum_i dy[i] * w[i] * x[i] * rms
    float dot = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float dyi = __bfloat162float(dyr[i]);
        float xi  = __bfloat162float(xr[i]);
        float wi  = __bfloat162float(w[i]);
        dot += dyi * wi * (xi * rms);
    }
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP;
    int wid  = threadIdx.x / WARP;
    dot = warp_reduce_sum(dot);
    if (lane == 0) smem[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        dot = (threadIdx.x < (blockDim.x / WARP)) ? smem[threadIdx.x] : 0.f;
        dot = warp_reduce_sum(dot);
        if (threadIdx.x == 0) smem[0] = dot;
    }
    __syncthreads();
    dot = smem[0] / (float)D;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float dyi     = __bfloat162float(dyr[i]);
        float xi      = __bfloat162float(xr[i]);
        float wi      = __bfloat162float(w[i]);
        float x_hat   = xi * rms;
        float dxi     = rms * wi * (dyi - x_hat * dot);
        dxr[i]        = __float2bfloat16(dxi);
        // dw: accumulate in global F32 (atomicAdd across rows)
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
//  RoPE forward
// ─────────────────────────────────────────────────────────────
// Rotate pairs (x[2i], x[2i+1]) by angle pos * theta^(-2i/head_dim).

__global__ void k_rope(
    __nv_bfloat16* __restrict__ x,
    int T, int H, int hd, float theta, bool inverse)
{
    // Grid: [B, T, H], threads over hd/2
    int b   = blockIdx.z;
    int t   = blockIdx.y;
    int h   = blockIdx.x;
    int idx = (b * T * H + t * H + h) * hd;

    for (int i = threadIdx.x; i < hd / 2; i += blockDim.x) {
        float freq = powf(theta, -2.f * (float)i / (float)hd);
        float angle = (float)t * freq;
        if (inverse) angle = -angle;
        float cs = cosf(angle);
        float sn = sinf(angle);

        float x0 = __bfloat162float(x[idx + 2*i    ]);
        float x1 = __bfloat162float(x[idx + 2*i + 1]);
        x[idx + 2*i    ] = __float2bfloat16(x0 * cs - x1 * sn);
        x[idx + 2*i + 1] = __float2bfloat16(x0 * sn + x1 * cs);
    }
}

void rope_fwd(
    __nv_bfloat16* x,
    int B, int T, int H, int hd, float theta, cudaStream_t s)
{
    dim3 grid(H, T, B);
    k_rope<<<grid, min(hd/2, 256), 0, s>>>(x, T, H, hd, theta, false);
}

void rope_bwd(
    __nv_bfloat16* dx,
    int B, int T, int H, int hd, float theta, cudaStream_t s)
{
    dim3 grid(H, T, B);
    k_rope<<<grid, min(hd/2, 256), 0, s>>>(dx, T, H, hd, theta, true);
}

// ─────────────────────────────────────────────────────────────
//  Scaled dot-product attention — naive, O(T^2)
// ─────────────────────────────────────────────────────────────
// GQA: each KV head serves (Hq / Hkv) Q heads.

__global__ void k_sdpa_fwd(
    const __nv_bfloat16* __restrict__ Q,    // [B, T, Hq, hd]
    const __nv_bfloat16* __restrict__ K,    // [B, T, Hkv, hd]
    const __nv_bfloat16* __restrict__ V,    // [B, T, Hkv, hd]
    __nv_bfloat16* __restrict__       out,  // [B, T, Hq, hd]
    float*         __restrict__       attn, // [B, Hq, T, T]
    int T, int Hq, int Hkv, int hd)
{
    // One block per (b, hq, tq) triple, threads over tk.
    int b  = blockIdx.z;
    int hq = blockIdx.y;
    int tq = blockIdx.x;
    int hkv = hq / (Hq / Hkv);

    float scale = rsqrtf((float)hd);

    // Q row for this query token.
    const __nv_bfloat16* Qrow = Q + (b * T * Hq + tq * Hq + hq) * hd;

    extern __shared__ float smem[]; // [T] score buffer

    // Compute attention scores for all keys t <= tq (causal).
    float row_max = -FLT_MAX;
    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        const __nv_bfloat16* Krow = K + (b * T * Hkv + tk * Hkv + hkv) * hd;
        float dot = 0.f;
        for (int i = 0; i < hd; i++) {
            dot += __bfloat162float(Qrow[i]) * __bfloat162float(Krow[i]);
        }
        smem[tk] = dot * scale;
        row_max = fmaxf(row_max, smem[tk]);
    }
    // Mask future positions.
    for (int tk = tq + 1 + threadIdx.x; tk < T; tk += blockDim.x) {
        smem[tk] = -1e38f;
    }
    __syncthreads();

    // Reduce max across block.
    __shared__ float smax;
    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int i = 0; i < T; i++) mx = fmaxf(mx, smem[i]);
        smax = mx;
    }
    __syncthreads();
    float mx = smax;

    // Softmax numerator + denominator.
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

    // Normalise and save attention weights.
    int attn_row = (b * Hq + hq) * T * T + tq * T;
    for (int tk = threadIdx.x; tk < T; tk += blockDim.x) {
        float aw = smem[tk] * inv_sum;
        smem[tk] = aw;
        attn[attn_row + tk] = aw;
    }
    __syncthreads();

    // Weighted sum of V.
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

void sdpa_fwd(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* out, float* attn_w,
    int B, int T, int Hq, int Hkv, int hd, cudaStream_t s)
{
    dim3 grid(T, Hq, B);
    int  threads  = min(T, 256);
    int  smem_bytes = T * sizeof(float);
    k_sdpa_fwd<<<grid, threads, smem_bytes, s>>>(
        Q, K, V, out, attn_w, T, Hq, Hkv, hd);
}

__global__ void k_sdpa_bwd(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float*         __restrict__ attn_w,
    __nv_bfloat16* __restrict__       dQ,
    __nv_bfloat16* __restrict__       dK,
    __nv_bfloat16* __restrict__       dV,
    int T, int Hq, int Hkv, int hd)
{
    int b  = blockIdx.z;
    int hq = blockIdx.y;
    int tq = blockIdx.x;
    int hkv = hq / (Hq / Hkv);
    float scale = rsqrtf((float)hd);

    const float* A = attn_w + (b * Hq + hq) * T * T + tq * T;
    const __nv_bfloat16* dO = dout + (b * T * Hq + tq * Hq + hq) * hd;
    const __nv_bfloat16* Qr = Q    + (b * T * Hq + tq * Hq + hq) * hd;

    extern __shared__ float smem[]; // [T] for dS

    // dA[tk] = dot(dO, V[tk])
    float dAq = 0.f; // = sum_i dO[i] * y[i], y = attn*V
    for (int i = 0; i < hd; i++) {
        float doi = __bfloat162float(dO[i]);
        float yi  = 0.f;
        // y[i] = sum_k A[k] * V[k,i]
        for (int tk2 = 0; tk2 <= tq; tk2++) {
            const __nv_bfloat16* Vr2 = V + (b * T * Hkv + tk2 * Hkv + hkv) * hd;
            yi += A[tk2] * __bfloat162float(Vr2[i]);
        }
        dAq += doi * yi;
    }

    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        const __nv_bfloat16* Vr = V + (b * T * Hkv + tk * Hkv + hkv) * hd;
        // dA_raw[tk] = dot(dO, V[tk])
        float dA_raw = 0.f;
        for (int i = 0; i < hd; i++) {
            dA_raw += __bfloat162float(dO[i]) * __bfloat162float(Vr[i]);
        }
        // Softmax backward: dS[tk] = A[tk] * (dA_raw - dAq)
        smem[tk] = A[tk] * (dA_raw - dAq);
    }
    __syncthreads();

    // dV[tk,i] += A[tq,tk] * dO[i]
    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        __nv_bfloat16* dVr = dV + (b * T * Hkv + tk * Hkv + hkv) * hd;
        for (int i = 0; i < hd; i++) {
            atomicAdd((__nv_bfloat16*)dVr + i,
                __float2bfloat16(A[tk] * __bfloat162float(dO[i])));
        }
    }

    // dQ[tq,i] += sum_k dS[k] * K[k,i] * scale
    __nv_bfloat16* dQr = dQ + (b * T * Hq + tq * Hq + hq) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int tk = 0; tk <= tq; tk++) {
            const __nv_bfloat16* Kr = K + (b * T * Hkv + tk * Hkv + hkv) * hd;
            acc += smem[tk] * __bfloat162float(Kr[i]);
        }
        dQr[i] = __float2bfloat16(__bfloat162float(dQr[i]) + acc * scale);
    }

    // dK[tk,i] += dS[tk] * Q[tq,i] * scale
    for (int tk = threadIdx.x; tk <= tq; tk += blockDim.x) {
        __nv_bfloat16* dKr = dK + (b * T * Hkv + tk * Hkv + hkv) * hd;
        for (int i = 0; i < hd; i++) {
            float dki = smem[tk] * __bfloat162float(Qr[i]) * scale;
            atomicAdd((__nv_bfloat16*)dKr + i, __float2bfloat16(dki));
        }
    }
}

void sdpa_bwd(
    const __nv_bfloat16* dout,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const float* attn_w,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int T, int Hq, int Hkv, int hd, cudaStream_t s)
{
    dim3 grid(T, Hq, B);
    int threads   = min(T, 64);
    int smem_bytes = T * sizeof(float);
    k_sdpa_bwd<<<grid, threads, smem_bytes, s>>>(
        dout, Q, K, V, attn_w, dQ, dK, dV, T, Hq, Hkv, hd);
}

// ─────────────────────────────────────────────────────────────
//  SiLU gated activation
// ─────────────────────────────────────────────────────────────

__global__ void k_silu_mul_fwd(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float s = g / (1.f + expf(-g)); // silu(g)
    out[i]  = __float2bfloat16(s * u);
}

__global__ void k_silu_mul_bwd(
    const __nv_bfloat16* dout,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* dgate,
    __nv_bfloat16* dup,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g  = __bfloat162float(gate[i]);
    float u  = __bfloat162float(up[i]);
    float do_ = __bfloat162float(dout[i]);

    float sig = 1.f / (1.f + expf(-g));
    float s   = g * sig;
    float ds  = sig * (1.f + g * (1.f - sig)); // d silu(g) / dg

    dgate[i] = __float2bfloat16(do_ * u * ds);
    dup[i]   = __float2bfloat16(do_ * s);
}

void silu_mul_fwd(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, int N, cudaStream_t s)
{
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    k_silu_mul_fwd<<<blocks, threads, 0, s>>>(gate, up, out, N);
}

void silu_mul_bwd(
    const __nv_bfloat16* dout, const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* dgate, __nv_bfloat16* dup,
    int N, cudaStream_t s)
{
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    k_silu_mul_bwd<<<blocks, threads, 0, s>>>(dout, gate, up, dgate, dup, N);
}

// ─────────────────────────────────────────────────────────────
//  Cross-entropy loss (fused forward + dlogits)
// ─────────────────────────────────────────────────────────────
// One block per token. Threads reduce over vocab V.

__global__ void k_cross_entropy(
    const __nv_bfloat16* __restrict__ logits,  // [N, V]
    const int*           __restrict__ targets, // [N]
    float*               __restrict__ loss_per_token, // [N]
    __nv_bfloat16*       __restrict__ dlogits, // [N, V]
    int V)
{
    int n = blockIdx.x;
    const __nv_bfloat16* L = logits  + n * V;
    __nv_bfloat16*       dL = dlogits + n * V;
    int tgt = targets[n];

    // Find max for numerical stability.
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        mx = fmaxf(mx, __bfloat162float(L[i]));
    }
    __shared__ float smem[32];
    int lane = threadIdx.x % WARP;
    int wid  = threadIdx.x / WARP;
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

    // Sum of exp(logit - max).
    float sum = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        sum += expf(__bfloat162float(L[i]) - mx);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / WARP) ? smem[threadIdx.x] : 0.f;
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) {
            float log_sum = logf(sum) + mx;
            float tgt_logit = (tgt >= 0) ? __bfloat162float(L[tgt]) : 0.f;
            loss_per_token[n] = -(tgt_logit - log_sum);
            smem[0] = log_sum;
        }
    }
    __syncthreads();
    float log_sum = smem[0];

    // dlogits[i] = softmax[i] - (i == tgt ? 1 : 0), scaled by 1/N.
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
    // loss_out is [N] per-token losses; caller averages.
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
    int bt = blockIdx.x;
    int tok = tokens[bt];
    const __nv_bfloat16* row = table + tok * D;
    __nv_bfloat16*       dst = out   + bt  * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = row[i];
}

__global__ void k_embed_bwd(
    const __nv_bfloat16* dout, const int* tokens,
    float* dtable, int D)
{
    int bt  = blockIdx.x;
    int tok = tokens[bt];
    const __nv_bfloat16* src = dout   + bt  * D;
    float*               dst = dtable + tok * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        atomicAdd(&dst[i], __bfloat162float(src[i]));
}

void embed_fwd(
    const __nv_bfloat16* table, const int* tokens,
    __nv_bfloat16* out, int BT, int D, cudaStream_t s)
{
    k_embed_fwd<<<BT, min(D, 512), 0, s>>>(table, tokens, out, D);
}

void embed_bwd(
    const __nv_bfloat16* dout, const int* tokens,
    float* dtable, int BT, int D, cudaStream_t s)
{
    k_embed_bwd<<<BT, min(D, 512), 0, s>>>(dout, tokens, dtable, D);
}

// ─────────────────────────────────────────────────────────────
//  Residual + LoRA add
// ─────────────────────────────────────────────────────────────

__global__ void k_add_inplace(
    __nv_bfloat16* out, const __nv_bfloat16* delta, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float a = __bfloat162float(out[i]) + __bfloat162float(delta[i]);
    out[i] = __float2bfloat16(a);
}

void add_inplace(
    __nv_bfloat16* out, const __nv_bfloat16* delta, int N, cudaStream_t s)
{
    int t = 256, b = (N + t - 1) / t;
    k_add_inplace<<<b, t, 0, s>>>(out, delta, N);
}

__global__ void k_lora_add(
    __nv_bfloat16* out, const __nv_bfloat16* delta, float scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float a = __bfloat162float(out[i]) + scale * __bfloat162float(delta[i]);
    out[i] = __float2bfloat16(a);
}

void lora_add(
    __nv_bfloat16* out, const __nv_bfloat16* delta, float scale, int N, cudaStream_t s)
{
    int t = 256, b = (N + t - 1) / t;
    k_lora_add<<<b, t, 0, s>>>(out, delta, scale, N);
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
    float* m, float* v,
    const float* grad,
    int N,
    float lr, float b1, float b2, float eps,
    float wd, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float g  = grad[i];
    float mi = b1 * m[i] + (1.f - b1) * g;
    float vi = b2 * v[i] + (1.f - b2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float bc1 = 1.f - powf(b1, (float)step);
    float bc2 = 1.f - powf(b2, (float)step);
    float mhat = mi / bc1;
    float vhat = vi / bc2;

    float p = master[i];
    p = p - lr * (mhat / (sqrtf(vhat) + eps) + wd * p);
    master[i]  = p;
    working[i] = __float2bfloat16(p);
}

void adamw_step(
    float* master, __nv_bfloat16* working,
    float* m, float* v, const float* grad,
    int N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int step, cudaStream_t s)
{
    int threads = 256, blocks = (N + threads - 1) / threads;
    k_adamw<<<blocks, threads, 0, s>>>(
        master, working, m, v, grad, N,
        lr, beta1, beta2, eps, weight_decay, step);
}

} // namespace tensor::backend::cuda::ops