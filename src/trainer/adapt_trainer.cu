#include <tensor/trainer/adapt_trainer.hpp>
#include <tensor/trainer/adamw.hpp>
#include <tensor/trainer/cosine_schedule.hpp>
#include <tensor/base/arch/qwen2.hpp>
#include <tensor/centroid/centroid_accumulator.hpp>
#include <tensor/checkpoint/adapter_writer.hpp>
#include <tensor/backend/cuda/tensor.hpp>
#include <tensor/backend/cuda/ops.hpp>

#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace tensor::trainer {

using namespace backend::cuda;
using namespace adapter;
using namespace base;
using namespace base::arch;
namespace ops = backend::cuda::ops;
using Clock = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────
//  Gradient norm — fully GPU-side
// ─────────────────────────────────────────────────────────────

static float compute_grad_norm(const AdapterModel& model, const Device& dev) {
    Tensor accum = Tensor::zeros_f32({1}, dev);
    for (const auto& la : model.layers) {
        for (const auto* lp : {&la.lora_q, &la.lora_k, &la.lora_v, &la.lora_o}) {
            ops::sum_sq_into(lp->gA.f32(), accum.f32(),
                             lp->rank * lp->in_dim,  dev.stream());
            ops::sum_sq_into(lp->gB.f32(), accum.f32(),
                             lp->out_dim * lp->rank, dev.stream());
        }
    }
    dev.sync();
    return std::sqrt(accum.to_host_f32()[0]);
}

// ─────────────────────────────────────────────────────────────
//  Gradient clip
// ─────────────────────────────────────────────────────────────

__global__ static void k_grad_scale(float* g, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) g[i] *= scale;
}

static void clip_gradients(AdapterModel& model, float max_norm,
                            float actual_norm, const Device& dev)
{
    if (actual_norm <= max_norm) return;
    float scale = max_norm / (actual_norm + 1e-6f);
    for (auto& la : model.layers) {
        for (auto* lp : {&la.lora_q, &la.lora_k, &la.lora_v, &la.lora_o}) {
            int nA = lp->rank * lp->in_dim;
            int nB = lp->out_dim * lp->rank;
            k_grad_scale<<<(nA+255)/256, 256, 0, dev.stream()>>>(
                lp->gA.f32(), nA, scale);
            k_grad_scale<<<(nB+255)/256, 256, 0, dev.stream()>>>(
                lp->gB.f32(), nB, scale);
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Diagnostics
// ─────────────────────────────────────────────────────────────

static void run_diagnostics(const Qwen2ForwardResult& fwd,
                             const std::vector<int32_t>& targets,
                             const base::BaseConfig& bc,
                             int BT, const Device& dev) {
    // Basic NaN check on logits
    auto logits_h = fwd.logits.to_host_bf16(); 

    int nans = 0;
    for (auto v : logits_h) {
        float f = __bfloat162float(v);
        if (std::isnan(f) || std::isinf(f)) nans++;
    }
    if (nans > 0) {
        std::cerr << "[trainer] WARNING: " << nans << " NaNs in logits at step 0\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  Implementation
// ─────────────────────────────────────────────────────────────

struct AdaptTrainer::Impl {
    const FrozenBase* base;
    AdapterConfig       cfg;
    data::Dataset* dataset;
    AdaptOptions        opts;
    const Device* dev;

    AdapterModel        model;
    AdamW               optimizer;
    CosineSchedule      schedule;
    centroid::CentroidAccumulator centroid_acc;
    checkpoint::AdapterWriter     writer;

    std::size_t step_count  = 0;
    std::size_t tokens_seen = 0;
    std::size_t total_steps = 0;

    Clock::time_point   step_start;
    double              ema_ms_per_step = 0.0;
    static constexpr double EMA_ALPHA   = 0.1;

    Impl(const FrozenBase* b, const AdapterConfig& c,
         data::Dataset* ds, const AdaptOptions& o, const Device* d)
        : base(b), cfg(c), dataset(ds), opts(o), dev(d)
        , model(AdapterModel::create(*b, c, *d))
        , optimizer(c)
        , centroid_acc(b->config.hidden_size, o.output_dir + "/centroids")
        , writer(o.output_dir)
    {
        total_steps = opts.tokens / (cfg.batch_size * cfg.seq_len);
        schedule    = CosineSchedule{cfg.lr, cfg.warmup_steps, total_steps, 0.1f};
    }

    StepMetrics do_step() {
        step_start = Clock::now();

        const BaseConfig& bc = base->config;
        int B  = cfg.batch_size;
        int T  = cfg.seq_len;
        int BT = B * T;

        // ── 1. Fetch batch ─────────────────────────────────────
        auto [input_ids, target_ids] = dataset->next_batch(B, T);

        // ── 2. STATE-BASED MASKING ─────────────────────────────
        // Only train on Assistant response content.
        // Qwen2.5 Tokens: 151644=<|im_start|>, 151645=<|im_end|>, 77091="assistant"
        
        for (int b = 0; b < B; ++b) {
            bool in_assistant_response = false;
            int offset = b * T;

            for (int t = 0; t < T; ++t) {
                int idx = offset + t;
                int token = input_ids[idx];

                // Detect start of a turn
                if (token == 151644) {
                    // Peek at next token to check role
                    if (t + 1 < T && input_ids[offset + t + 1] == 77091) {
                        in_assistant_response = true;
                    } else {
                        in_assistant_response = false;
                    }
                }

                bool mask_this = !in_assistant_response;

                // Mask header tokens (<|im_start|>, assistant)
                if (token == 151644 || token == 77091) {
                    mask_this = true;
                }

                // If end token, mask it and close the turn
                if (token == 151645) { // <|im_end|>
                    in_assistant_response = false;
                    // FIX 1: We MUST train on the stop token so the model learns when to stop.
                    // Previous buggy code: mask_this = true;
                    mask_this = false; 
                }

                if (mask_this) {
                    target_ids[idx] = -1; // Ignore in Cross Entropy
                }
            }
        }

        Tensor tok_dev = Tensor::from_host(
            core::TensorView{input_ids.data(),  core::DType::I32, {(std::size_t)BT}}, *dev);
        Tensor tgt_dev = Tensor::from_host(
            core::TensorView{target_ids.data(), core::DType::I32, {(std::size_t)BT}}, *dev);

        // ── 3. Zero gradients ──────────────────────────────────
        model.zero_grad(*dev);

        // ── 4. Forward ────────────────────────────────────────
        auto fwd = Qwen2Base::forward(*base, tok_dev, B, T, *dev, &model);

        // ── 5. Diagnostics ────────────────────────────────────
        if (step_count == 0) {
            run_diagnostics(fwd, target_ids, bc, BT, *dev);
        }

        // ── 6. Loss ───────────────────────────────────────────
        Tensor loss_per_tok = Tensor::empty_f32({(std::size_t)BT}, *dev);
        ops::cross_entropy_fwd(
            fwd.logits.bf16(), tgt_dev.i32(),
            loss_per_tok.f32(), fwd.dlogits.bf16(),
            BT, bc.vocab_size, dev->stream());

        // ── 7. Backward ───────────────────────────────────────
        auto layer_grads = Qwen2Base::backward(*base, fwd, B, T, *dev, &model);

        for (std::size_t l = 0; l < bc.num_layers; l++) {
             // FIX 2: Cast to F32 before Centroid Accumulation
             // CentroidAccumulator expects F32 inputs (it calls to_host_f32), 
             // but Qwen2 uses BF16. We must cast manually here to avoid "dtype mismatch".
             
             Tensor x_f32 = Tensor::empty_f32(fwd.layer_cache[l].x_normed.shape(), *dev);
             ops::cast_bf16_to_f32(
                 fwd.layer_cache[l].x_normed.bf16(), 
                 x_f32.f32(), 
                 x_f32.numel(), 
                 dev->stream()
             );

             Tensor g_f32 = Tensor::empty_f32(layer_grads[l].dx_attn_in.shape(), *dev);
             ops::cast_bf16_to_f32(
                 layer_grads[l].dx_attn_in.bf16(), 
                 g_f32.f32(), 
                 g_f32.numel(), 
                 dev->stream()
             );

             centroid_acc.accumulate(x_f32, g_f32, *dev);
        }

        // ── 8. Stats & Updates ────────────────────────────────
        dev->sync();

        auto loss_h = loss_per_tok.to_host_f32();
        float mean_loss = 0.f;
        int valid_toks = 0;
        for (float l : loss_h) {
            if (l > 1e-9f) { // Filter out masked tokens
                mean_loss += l;
                valid_toks++;
            }
        }
        if (valid_toks > 0) mean_loss /= (float)valid_toks;

        float gnorm = compute_grad_norm(model, *dev);
        clip_gradients(model, cfg.grad_clip, gnorm, *dev);
        gnorm = std::min(gnorm, cfg.grad_clip);

        float lr = schedule.lr_at(step_count);
        optimizer.step_all(model, lr, (int)step_count + 1, *dev);

        step_count++;
        tokens_seen += BT;

        auto now = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - step_start).count();
        if (step_count == 1) ema_ms_per_step = ms;
        else ema_ms_per_step = EMA_ALPHA * ms + (1.0 - EMA_ALPHA) * ema_ms_per_step;

        double tokens_per_sec = (ema_ms_per_step > 0.0) ? (BT * 1000.0 / ema_ms_per_step) : 0.0;

        if (step_count % opts.checkpoint_every == 0) {
            dev->sync();
            std::string ckpt = opts.output_dir + "/step-" + std::to_string(step_count);
            centroid_acc.write_snapshot(step_count);
            writer.save_checkpoint(model, *base, opts.domain, step_count, tokens_seen, ckpt);
        }

        return StepMetrics{
            step_count, tokens_seen,
            mean_loss, lr, gnorm,
            ms, ema_ms_per_step, tokens_per_sec
        };
    }
};

AdaptTrainer AdaptTrainer::create(
    const FrozenBase& base, const AdapterConfig& cfg,
    data::Dataset& dataset, const AdaptOptions& opts, const Device& dev)
{
    fs::create_directories(opts.output_dir);
    return AdaptTrainer(std::make_unique<Impl>(&base, cfg, &dataset, opts, &dev));
}

AdaptTrainer::AdaptTrainer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AdaptTrainer::~AdaptTrainer() = default;
AdaptTrainer::AdaptTrainer(AdaptTrainer&&) noexcept = default;
AdaptTrainer& AdaptTrainer::operator=(AdaptTrainer&&) noexcept = default;

bool        AdaptTrainer::done() const { return impl_->tokens_seen >= impl_->opts.tokens; }
StepMetrics AdaptTrainer::step()       { return impl_->do_step(); }

void AdaptTrainer::run() {
    while (!done()) {
        auto m = step();
        if (impl_->opts.log_to_stdout && impl_->step_count % 10 == 0) {
            std::cerr << "[step " << m.step << "/" << impl_->total_steps << "]"
                      << " loss="  << m.loss
                      << " lr="    << m.learning_rate
                      << " grad="  << m.grad_norm
                      << " tok="   << m.tokens_consumed
                      << " ms="    << static_cast<int>(m.step_ms)
                      << " ema="   << static_cast<int>(m.ema_ms)
                      << " tok/s=" << static_cast<int>(m.tokens_per_sec)
                      << "\n";
        }
    }
    save_adapter(impl_->opts.output_dir);
}

void AdaptTrainer::save_checkpoint(const std::string& dir) {
    impl_->dev->sync();
    impl_->centroid_acc.write_snapshot(impl_->step_count);
    impl_->writer.save_checkpoint(
        impl_->model, *impl_->base, impl_->opts.domain,
        impl_->step_count, impl_->tokens_seen, dir);
}

void AdaptTrainer::save_adapter(const std::string& dir) {
    impl_->dev->sync();
    impl_->centroid_acc.write_snapshot(impl_->step_count);
    impl_->centroid_acc.merge_and_write(16, dir + "/adapter.centroid");
    impl_->writer.save_final(
        impl_->model, *impl_->base, impl_->opts.domain,
        impl_->step_count, impl_->tokens_seen, dir);
    std::cerr << "[trainer] adapter saved → " << dir << "\n";
}

} // namespace tensor::trainer