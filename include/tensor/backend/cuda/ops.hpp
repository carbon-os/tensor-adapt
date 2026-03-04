// ops.hpp
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tensor::backend::cuda::ops {

// ── RMSNorm ──────────────────────────────────────────────────
void rms_norm_fwd(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16*       out,
    float*               rms_out,     // [B*T] — saved for backward
    int B, int T, int D, float eps,
    cudaStream_t stream);

void rms_norm_bwd(
    const __nv_bfloat16* dy,
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    const float*         rms_saved,   // [B*T]
    __nv_bfloat16*       dx,
    float*               dw,          // [D] accumulated in F32
    int B, int T, int D, float eps,
    cudaStream_t stream);

// ── RoPE ─────────────────────────────────────────────────────
void rope_fwd(
    __nv_bfloat16* x,
    int B, int T, int H, int head_dim,
    float theta,
    cudaStream_t stream);

void rope_bwd(
    __nv_bfloat16* dx,
    int B, int T, int H, int head_dim,
    float theta,
    cudaStream_t stream);

// ── Attention ─────────────────────────────────────────────────
void sdpa_fwd(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16*       out,
    float*               attn_w,
    int B, int T, int Hq, int Hkv, int head_dim,
    cudaStream_t stream);

void sdpa_bwd(
    const __nv_bfloat16* dout,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float*         attn_w,
    __nv_bfloat16*       dQ,
    __nv_bfloat16*       dK,
    __nv_bfloat16*       dV,
    int B, int T, int Hq, int Hkv, int head_dim,
    cudaStream_t stream);

// ── Gated SiLU ────────────────────────────────────────────────
void silu_mul_fwd(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16*       out,
    int N,
    cudaStream_t stream);

void silu_mul_bwd(
    const __nv_bfloat16* dout,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16*       dgate,
    __nv_bfloat16*       dup,
    int N,
    cudaStream_t stream);

// ── Cross-entropy loss ────────────────────────────────────────
void cross_entropy_fwd(
    const __nv_bfloat16* logits,
    const int*           targets,
    float*               loss_out,
    __nv_bfloat16*       dlogits,
    int N, int V,
    cudaStream_t stream);

// ── Embedding ─────────────────────────────────────────────────
void embed_fwd(
    const __nv_bfloat16* table,
    const int*           tokens,
    __nv_bfloat16*       out,
    int BT, int D,
    cudaStream_t stream);

void embed_bwd(
    const __nv_bfloat16* dout,
    const int*           tokens,
    float*               dtable,
    int BT, int D,
    cudaStream_t stream);

// ── Bias add ─────────────────────────────────────────────────
// out[BT, D] += bias[D]   (broadcast row vector over batch dimension)
void add_bias(
    __nv_bfloat16*       out,
    const __nv_bfloat16* bias,
    int BT, int D,
    cudaStream_t stream);

// ── Residual / LoRA adds ──────────────────────────────────────
void add_inplace(
    __nv_bfloat16*       out,
    const __nv_bfloat16* delta,
    int N,
    cudaStream_t stream);

void lora_add(
    __nv_bfloat16*       out,
    const __nv_bfloat16* lora_delta,
    float                scale,
    int N,
    cudaStream_t stream);

// ── BF16 ↔ F32 cast ──────────────────────────────────────────
void cast_bf16_to_f32(const __nv_bfloat16* src, float* dst, int N, cudaStream_t s);
void cast_f32_to_bf16(const float* src, __nv_bfloat16* dst, int N, cudaStream_t s);

// ── AdamW step ────────────────────────────────────────────────
void adamw_step(
    float*               master,
    __nv_bfloat16*       working,
    float*               m,
    float*               v,
    const float*         grad,
    int                  N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream);

} // namespace tensor::backend::cuda::ops