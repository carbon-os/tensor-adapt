#pragma once
// Internal kernel declarations — not part of public API.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tensor::backend::cuda::ops {

// ── RMSNorm ──────────────────────────────────────────────────
// Forward: out[b,t,d] = x[b,t,d] / rms(x[b,t,:]) * w[d]
void rms_norm_fwd(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16*       out,
    float*               rms_out,     // [B*T] — saved for backward
    int B, int T, int D, float eps,
    cudaStream_t stream);

// Backward: given dy → dx, dw
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
// Apply rotary position embeddings in-place on q or k.
// x: [B, T, H, head_dim], positions: [T]
void rope_fwd(
    __nv_bfloat16* x,
    int B, int T, int H, int head_dim,
    float theta,
    cudaStream_t stream);

// RoPE backward = inverse RoPE (same rotation with negative angle).
void rope_bwd(
    __nv_bfloat16* dx,
    int B, int T, int H, int head_dim,
    float theta,
    cudaStream_t stream);

// ── Attention ────────────────────────────────────────────────
// Scaled dot-product attention with causal mask + GQA expansion.
// Q: [B, T, Hq, D], K/V: [B, T, Hkv, D]  (Hkv divides Hq)
// out: [B, T, Hq, D],  attn_weights: [B, Hq, T, T] (saved for bwd)
void sdpa_fwd(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16*       out,
    float*               attn_w,      // saved weights [B, Hq, T, T]
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

// ── Gated SiLU ───────────────────────────────────────────────
// out[i] = silu(gate[i]) * up[i]
// gate, up: [B*T, D_ff]
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

// ── Cross-entropy loss ───────────────────────────────────────
// logits: [B*T, V],  targets: [B*T]  (int32)
// loss_out: scalar F32,  dlogits: [B*T, V]
void cross_entropy_fwd(
    const __nv_bfloat16* logits,
    const int*           targets,
    float*               loss_out,
    __nv_bfloat16*       dlogits,
    int N, int V,
    cudaStream_t stream);

// ── Embedding ────────────────────────────────────────────────
// Forward lookup: out[b,t,:] = table[tokens[b,t],:]
void embed_fwd(
    const __nv_bfloat16* table,
    const int*           tokens,
    __nv_bfloat16*       out,
    int BT, int D,
    cudaStream_t stream);

// Backward: scatter dout gradients into dtable (atomicAdd).
void embed_bwd(
    const __nv_bfloat16* dout,
    const int*           tokens,
    float*               dtable,    // F32 accumulation
    int BT, int D,
    cudaStream_t stream);

// ── Residual add ─────────────────────────────────────────────
// out[i] += delta[i]   (in-place, BF16)
void add_inplace(
    __nv_bfloat16*       out,
    const __nv_bfloat16* delta,
    int N,
    cudaStream_t stream);

// ── LoRA scale-add ───────────────────────────────────────────
// out[i] += scale * lora_delta[i]
void lora_add(
    __nv_bfloat16*       out,
    const __nv_bfloat16* lora_delta,
    float                scale,
    int N,
    cudaStream_t stream);

// ── BF16 ↔ F32 cast ─────────────────────────────────────────
void cast_bf16_to_f32(const __nv_bfloat16* src, float* dst, int N, cudaStream_t s);
void cast_f32_to_bf16(const float* src, __nv_bfloat16* dst, int N, cudaStream_t s);

// ── AdamW step ───────────────────────────────────────────────
// Update F32 master weight, then cast back to BF16 working copy.
void adamw_step(
    float*               master,     // F32 master weights — updated in-place
    __nv_bfloat16*       working,    // BF16 working copy — updated in-place
    float*               m,          // first moment
    float*               v,          // second moment
    const float*         grad,       // F32 gradients
    int                  N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream);

} // namespace tensor::backend::cuda::ops