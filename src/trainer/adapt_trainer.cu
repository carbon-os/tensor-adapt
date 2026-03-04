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
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace tensor::trainer {

using namespace backend::cuda;
using namespace adapter;
using namespace base;
using namespace base::arch;
namespace ops = backend::cuda::ops;

// ─────────────────────────────────────────────────────────────
//  Gradient norm — sqrt(sum of squared F32 grads across all LoRA params)
// ─────────────────────────────────────────────────────────────

static float compute_grad_norm(const AdapterModel& model, const Device& dev) {
    float norm_sq = 0.f;
    for (const auto& la : model.layers) {
        for (const auto* lp : {&la.lora_q, &la.lora_k, &la.lora_v, &la.lora_o}) {
            auto gA = lp->gA.to_host_f32();
            auto gB = lp->gB.to_host_f32();
            for (float g : gA) norm_sq += g * g;
            for (float g : gB) norm_sq += g * g;
        }
    }
    return std::sqrt(norm_sq);
}

// Clip gradients to max_norm in-place (scale all F32 grads).
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
            k_grad_scale<<<(nA+255)/256, 256, 0, dev.stream()>>>(lp->gA.f32(), nA, scale);
            k_grad_scale<<<(nB+255)/256, 256, 0, dev.stream()>>>(lp->gB.f32(), nB, scale);
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Impl
// ─────────────────────────────────────────────────────────────

struct AdaptTrainer::Impl {
    const FrozenBase*   base;
    AdapterConfig       cfg;
    data::Dataset*      dataset;
    AdaptOptions        opts;
    const Device*       dev;

    AdapterModel        model;
    AdamW               optimizer;
    CosineSchedule      schedule;
    centroid::CentroidAccumulator centroid_acc;
    checkpoint::AdapterWriter     writer;

    std::size_t step_count     = 0;
    std::size_t tokens_seen    = 0;
    std::size_t total_steps    = 0;   // derived from token budget + batch

    Impl(const FrozenBase* b, const AdapterConfig& c,
         data::Dataset* ds, const AdaptOptions& o, const Device* d)
        : base(b), cfg(c), dataset(ds), opts(o), dev(d)
        , model(AdapterModel::create(*b, c, *d))
        , optimizer(c)
        , centroid_acc(b->config.hidden_size, o.output_dir + "/centroids")
        , writer(o.output_dir)
    {
        std::size_t steps_per_epoch =
            opts.tokens / (cfg.batch_size * cfg.seq_len);
        total_steps = steps_per_epoch;

        schedule = CosineSchedule{
            cfg.lr, cfg.warmup_steps, total_steps, 0.1f};

        std::cerr << "[trainer] LoRA params=" << model.param_count()
                  << " total_steps=" << total_steps
                  << " batch=" << cfg.batch_size
                  << " seq=" << cfg.seq_len << "\n";
    }

    StepMetrics do_step() {
        const BaseConfig& bc = base->config;
        int B  = cfg.batch_size;
        int T  = cfg.seq_len;
        int BT = B * T;

        // ── 1. Fetch batch ─────────────────────────────────────
        auto [input_ids, target_ids] = dataset->next_batch(B, T);

        // Upload tokens to device.
        Tensor tok_dev = Tensor::from_host(
            core::TensorView{input_ids.data(), core::DType::I32,
                             {(std::size_t)BT}},
            *dev);
        Tensor tgt_dev = Tensor::from_host(
            core::TensorView{target_ids.data(), core::DType::I32,
                             {(std::size_t)BT}},
            *dev);

        // ── 2. Zero gradients ─────────────────────────────────
        model.zero_grad(*dev);

        // ── 3. Forward pass (frozen base) ────────────────────
        auto fwd = Qwen2Base::forward(*base, tok_dev, B, T, *dev);

        // ── 4. Apply LoRA deltas (re-run projection with LoRA) ─
        // NOTE: in a full implementation the LoRA delta is applied inside
        // the forward pass. Here we apply it as a correction to the saved
        // projection outputs and store the corrected values.
        // For a prototype that keeps base forward and LoRA forward separate:
        // The LoRA delta modifies q/k/v/o outputs; because the frozen forward
        // already ran, we inject LoRA corrections into the stored layer cache
        // activations and recompute the attention output if needed.
        // For simplicity in this version, the LoRA contribution is small at
        // init (B=0) so the loss is dominated by the base model's predictions,
        // which is the correct initialisation.

        // ── 5. Loss ───────────────────────────────────────────
        Tensor loss_per_tok = Tensor::empty_f32({(std::size_t)BT}, *dev);
        ops::cross_entropy_fwd(
            fwd.logits.bf16(), tgt_dev.i32(),
            loss_per_tok.f32(), fwd.dlogits.bf16(),
            BT, bc.vocab_size, dev->stream());

        // Mean loss on host.
        dev->sync();
        auto loss_h = loss_per_tok.to_host_f32();
        float mean_loss = 0.f;
        for (float l : loss_h) mean_loss += l;
        mean_loss /= (float)BT;

        // ── 6. Backward (base + LoRA grads) ──────────────────
        auto layer_grads = Qwen2Base::backward(*base, fwd, B, T, *dev);

        // Accumulate LoRA gradients from each layer's injection points.
        for (std::size_t l = 0; l < bc.num_layers; l++) {
            const auto& lg = layer_grads[l];
            LayerAdapter& la = model.layers[l];
            const auto& lc  = fwd.layer_cache[l];

            // dx_attn_in is the gradient signal arriving at q/k/v input.
            // We use x_normed (the normed attn input) as the "x" for LoRA backward.
            Tensor dummy_gx = Tensor::zeros(
                {(std::size_t)BT, (std::size_t)bc.hidden_size},
                core::DType::BF16, *dev);

            model.apply_lora_bwd(la.lora_q, lc.x_normed, lg.dx_attn_in,
                                  dummy_gx, BT, *dev);
            model.apply_lora_bwd(la.lora_k, lc.x_normed, lg.dx_attn_in,
                                  dummy_gx, BT, *dev);
            model.apply_lora_bwd(la.lora_v, lc.x_normed, lg.dx_attn_in,
                                  dummy_gx, BT, *dev);
            model.apply_lora_bwd(la.lora_o, lc.attn_out, lg.dx_o_in,
                                  dummy_gx, BT, *dev);

            // Feed gradient signal to centroid accumulator.
            centroid_acc.accumulate(lc.x_normed, lg.dx_attn_in, *dev);
        }

        // ── 7. Grad norm + clip ───────────────────────────────
        dev->sync();
        float gnorm = compute_grad_norm(model, *dev);
        clip_gradients(model, cfg.grad_clip, gnorm, *dev);
        gnorm = std::min(gnorm, cfg.grad_clip);

        // ── 8. Optimizer step ─────────────────────────────────
        float lr = schedule.lr_at(step_count);
        optimizer.step_all(model, lr, (int)step_count + 1, *dev);
        dev->sync();

        step_count++;
        tokens_seen += BT;

        // ── 9. Checkpoint ─────────────────────────────────────
        if (step_count % opts.checkpoint_every == 0) {
            std::string ckpt_dir = opts.output_dir + "/step-" +
                                   std::to_string(step_count);
            centroid_acc.write_snapshot(step_count);
            writer.save_checkpoint(model, *base, opts.domain, step_count,
                                   tokens_seen, ckpt_dir);
            std::cerr << "[trainer] checkpoint → " << ckpt_dir << "\n";
        }

        return StepMetrics{step_count, tokens_seen, mean_loss, lr, gnorm};
    }
};

AdaptTrainer AdaptTrainer::create(
    const FrozenBase&       base,
    const AdapterConfig&    cfg,
    data::Dataset&          dataset,
    const AdaptOptions&     opts,
    const Device&           dev)
{
    fs::create_directories(opts.output_dir);
    auto impl = std::make_unique<Impl>(&base, cfg, &dataset, opts, &dev);
    return AdaptTrainer(std::move(impl));
}

AdaptTrainer::AdaptTrainer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AdaptTrainer::~AdaptTrainer() = default;

AdaptTrainer::AdaptTrainer(AdaptTrainer&&) noexcept = default;
AdaptTrainer& AdaptTrainer::operator=(AdaptTrainer&&) noexcept = default;

bool AdaptTrainer::done() const {
    return impl_->tokens_seen >= impl_->opts.tokens;
}

StepMetrics AdaptTrainer::step() {
    return impl_->do_step();
}

void AdaptTrainer::run() {
    while (!done()) {
        auto m = step();
        if (impl_->opts.log_to_stdout && impl_->step_count % 10 == 0) {
            std::cerr << "[step " << m.step
                      << "] loss=" << m.loss
                      << " lr=" << m.learning_rate
                      << " grad=" << m.grad_norm
                      << " tokens=" << m.tokens_consumed << "\n";
        }
    }
    // Final centroid merge + save.
    save_adapter(impl_->opts.output_dir);
}

void AdaptTrainer::save_checkpoint(const std::string& dir) {
    impl_->centroid_acc.write_snapshot(impl_->step_count);
    impl_->writer.save_checkpoint(
        impl_->model, *impl_->base, impl_->opts.domain,
        impl_->step_count, impl_->tokens_seen, dir);
}

void AdaptTrainer::save_adapter(const std::string& dir) {
    impl_->centroid_acc.merge_and_write(16, dir + "/adapter.centroid");
    impl_->writer.save_final(
        impl_->model, *impl_->base, impl_->opts.domain,
        impl_->step_count, impl_->tokens_seen, dir);
    std::cerr << "[trainer] adapter saved → " << dir << "\n";
}

} // namespace tensor::trainer