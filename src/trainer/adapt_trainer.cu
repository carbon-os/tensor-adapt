// adapt_trainer.cu
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
//  Debug helpers
// ─────────────────────────────────────────────────────────────

struct TensorStats {
    float mn, mx, mean, std_dev;
    int   nan_count, inf_count;
    bool  all_zero;
};

static TensorStats compute_stats_f32(const std::vector<float>& v) {
    TensorStats s{};
    if (v.empty()) return s;
    s.mn = s.mx = v[0];
    double sum = 0.0, sum_sq = 0.0;
    for (float x : v) {
        if (std::isnan(x)) { s.nan_count++; continue; }
        if (std::isinf(x)) { s.inf_count++; continue; }
        s.mn = std::min(s.mn, x);
        s.mx = std::max(s.mx, x);
        sum    += x;
        sum_sq += (double)x * x;
    }
    std::size_t valid = v.size() - s.nan_count - s.inf_count;
    if (valid > 0) {
        s.mean    = (float)(sum / valid);
        s.std_dev = (float)std::sqrt(sum_sq / valid - (sum / valid) * (sum / valid));
    }
    s.all_zero = (s.mn == 0.f && s.mx == 0.f);
    return s;
}

static TensorStats bf16_tensor_stats(const Tensor& t, const Device& dev,
                                     int max_elements = -1)
{
    int n = (max_elements > 0)
        ? std::min((int)t.numel(), max_elements)
        : (int)t.numel();
    Tensor f32 = Tensor::empty_f32({(std::size_t)n}, dev);
    ops::cast_bf16_to_f32(t.bf16(), f32.f32(), n, dev.stream());
    dev.sync();
    return compute_stats_f32(f32.to_host_f32());
}

static void print_stats(const char* name, const TensorStats& s) {
    std::cerr << "[debug] " << name
              << ": min=" << s.mn << " max=" << s.mx
              << " mean=" << s.mean << " std=" << s.std_dev;
    if (s.nan_count) std::cerr << " NAN=" << s.nan_count;
    if (s.inf_count) std::cerr << " INF=" << s.inf_count;
    if (s.all_zero)  std::cerr << " *** ALL ZERO ***";
    std::cerr << "\n";
}

static void run_diagnostics(const Qwen2ForwardResult& fwd,
                             const std::vector<int32_t>& targets,
                             const base::BaseConfig& bc,
                             int BT, const Device& dev)
{
    std::cerr << "\n[diag] ════════ Forward pass diagnostic ════════\n";

    {
        auto s = bf16_tensor_stats(fwd.logits, dev, BT * 256);
        print_stats("logits (sample)", s);

        Tensor row_f32 = Tensor::empty_f32({(std::size_t)bc.vocab_size}, dev);
        ops::cast_bf16_to_f32(fwd.logits.bf16(), row_f32.f32(),
                              bc.vocab_size, dev.stream());
        dev.sync();
        auto row = row_f32.to_host_f32();
        auto rs  = compute_stats_f32(row);
        print_stats("logits[token0] full vocab", rs);

        int argmax = 0; float argmax_v = row[0];
        for (int i = 1; i < (int)row.size(); i++)
            if (row[i] > argmax_v) { argmax_v = row[i]; argmax = i; }
        std::cerr << "[diag] logits[token0] argmax=" << argmax
                  << " logit=" << argmax_v << "\n";

        if (!targets.empty()) {
            int tgt = targets[0];
            float tgt_logit = (tgt < (int)row.size()) ? row[tgt] : -999.f;
            std::cerr << "[diag] target[0]=" << tgt
                      << " target_logit=" << tgt_logit
                      << " max_logit=" << argmax_v
                      << " gap=" << (argmax_v - tgt_logit) << "\n";
        }

        float sum_exp = 0.f;
        for (float v : row) sum_exp += std::exp(v - argmax_v);
        float log_sum = std::log(sum_exp) + argmax_v;
        float entropy = 0.f;
        for (float v : row) {
            float p = std::exp(v - log_sum);
            if (p > 1e-10f) entropy -= p * std::log(p);
        }
        float max_entropy = std::log((float)bc.vocab_size);
        std::cerr << "[diag] token0 softmax entropy=" << entropy
                  << " / max=" << max_entropy
                  << " (" << (100.f * entropy / max_entropy) << "% of uniform)\n";
    }

    print_stats("embed_in", bf16_tensor_stats(fwd.embed_in, dev));

    if (!fwd.layer_cache.empty()) {
        const auto& lc0 = fwd.layer_cache[0];
        print_stats("layer0.x_normed",  bf16_tensor_stats(lc0.x_normed,  dev));
        print_stats("layer0.Q",         bf16_tensor_stats(lc0.Q,         dev));
        print_stats("layer0.K",         bf16_tensor_stats(lc0.K,         dev));
        print_stats("layer0.V",         bf16_tensor_stats(lc0.V,         dev));
        print_stats("layer0.attn_out",  bf16_tensor_stats(lc0.attn_out,  dev));
        print_stats("layer0.h_mid",     bf16_tensor_stats(lc0.h_mid,     dev));
        print_stats("layer0.x_normed2", bf16_tensor_stats(lc0.x_normed2, dev));
        print_stats("layer0.gate_out",  bf16_tensor_stats(lc0.gate_out,  dev));
        print_stats("layer0.up_out",    bf16_tensor_stats(lc0.up_out,    dev));
        print_stats("layer0.act_out",   bf16_tensor_stats(lc0.act_out,   dev));
        print_stats("layer0.ffn_out",   bf16_tensor_stats(lc0.ffn_out,   dev));
    }
    if (fwd.layer_cache.size() > 1) {
        const auto& lcN = fwd.layer_cache.back();
        print_stats("layerN.x_normed", bf16_tensor_stats(lcN.x_normed, dev));
        print_stats("layerN.attn_out", bf16_tensor_stats(lcN.attn_out, dev));
        print_stats("layerN.ffn_out",  bf16_tensor_stats(lcN.ffn_out,  dev));
    }
    {
        dev.sync();
        auto s = compute_stats_f32(fwd.final_rms.to_host_f32());
        print_stats("final_rms_values", s);
    }

    std::cerr << "[diag] ═══════════════════════════════════════════\n\n";
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

        std::cerr << "[trainer] LoRA params=" << model.param_count()
                  << " total_steps=" << total_steps
                  << " batch=" << cfg.batch_size
                  << " seq=" << cfg.seq_len << "\n";
    }

    StepMetrics do_step() {
        step_start = Clock::now();

        const BaseConfig& bc = base->config;
        int B  = cfg.batch_size;
        int T  = cfg.seq_len;
        int BT = B * T;

        // ── 1. Fetch batch ─────────────────────────────────────
        auto [input_ids, target_ids] = dataset->next_batch(B, T);

        Tensor tok_dev = Tensor::from_host(
            core::TensorView{input_ids.data(),  core::DType::I32, {(std::size_t)BT}}, *dev);
        Tensor tgt_dev = Tensor::from_host(
            core::TensorView{target_ids.data(), core::DType::I32, {(std::size_t)BT}}, *dev);

        // ── 2. Zero gradients ──────────────────────────────────
        model.zero_grad(*dev);

        // ── 3. Forward — pass adapter so LoRA deltas are injected ──
        auto fwd = Qwen2Base::forward(*base, tok_dev, B, T, *dev, &model);

        // ── 4. Step-1 diagnostics ─────────────────────────────
        if (step_count == 0) {
            dev->sync();
            std::cerr << "[diag] input token IDs (first 16): ";
            for (int i = 0; i < std::min(16, BT); i++)
                std::cerr << input_ids[i] << " ";
            std::cerr << "\n";
            std::cerr << "[diag] target token IDs (first 16): ";
            for (int i = 0; i < std::min(16, BT); i++)
                std::cerr << target_ids[i] << " ";
            std::cerr << "\n";
            std::cerr << "[diag] vocab_size=" << bc.vocab_size
                      << " hidden=" << bc.hidden_size << " BT=" << BT << "\n";
            run_diagnostics(fwd, target_ids, bc, BT, *dev);
        }

        // ── 5. Loss ───────────────────────────────────────────
        Tensor loss_per_tok = Tensor::empty_f32({(std::size_t)BT}, *dev);
        ops::cross_entropy_fwd(
            fwd.logits.bf16(), tgt_dev.i32(),
            loss_per_tok.f32(), fwd.dlogits.bf16(),
            BT, bc.vocab_size, dev->stream());

        // ── 6. Backward ───────────────────────────────────────
        auto layer_grads = Qwen2Base::backward(*base, fwd, B, T, *dev);

        for (std::size_t l = 0; l < bc.num_layers; l++) {
            const auto& lg = layer_grads[l];
            LayerAdapter& la = model.layers[l];
            const auto& lc  = fwd.layer_cache[l];

            Tensor dummy_gx = Tensor::zeros(
                {(std::size_t)BT, (std::size_t)bc.hidden_size},
                core::DType::BF16, *dev);

            model.apply_lora_bwd(la.lora_q, lc.x_normed,  lg.dQ,      dummy_gx, BT, *dev);
            model.apply_lora_bwd(la.lora_k, lc.x_normed,  lg.dK,      dummy_gx, BT, *dev);
            model.apply_lora_bwd(la.lora_v, lc.x_normed,  lg.dV,      dummy_gx, BT, *dev);
            model.apply_lora_bwd(la.lora_o, lc.attn_out,  lg.do_proj, dummy_gx, BT, *dev);

            centroid_acc.accumulate(lc.x_normed, lg.dx_attn_in, *dev);
        }

        // ── 7. Single sync — read loss + compute grad norm ────
        dev->sync();

        auto loss_h = loss_per_tok.to_host_f32();
        float mean_loss = 0.f;
        for (float l : loss_h) mean_loss += l;
        mean_loss /= (float)BT;

        if (step_count > 0 && (std::isnan(mean_loss) || mean_loss > 20.f)) {
            std::cerr << "[diag] anomalous loss=" << mean_loss
                      << " at step=" << step_count << "\n";
            run_diagnostics(fwd, target_ids, bc, BT, *dev);
        }

        float gnorm = compute_grad_norm(model, *dev);

        if (step_count == 0) {
            std::cerr << "[diag] grad_norm before clip=" << gnorm << "\n";
            auto gB0 = model.layers[0].lora_q.gB.to_host_f32();
            print_stats("layer0.lora_q.gB (post-bwd)", compute_stats_f32(gB0));
        }

        // ── 8. Clip + optimizer step ───────────────────────────
        clip_gradients(model, cfg.grad_clip, gnorm, *dev);
        gnorm = std::min(gnorm, cfg.grad_clip);

        float lr = schedule.lr_at(step_count);
        optimizer.step_all(model, lr, (int)step_count + 1, *dev);

        step_count++;
        tokens_seen += BT;

        // ── 9. Wall-clock timing ───────────────────────────────
        auto now = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(
                        now - step_start).count();

        if (step_count == 1) {
            ema_ms_per_step = ms;
        } else {
            ema_ms_per_step = EMA_ALPHA * ms + (1.0 - EMA_ALPHA) * ema_ms_per_step;
        }

        double tokens_per_sec = (ema_ms_per_step > 0.0)
            ? (BT * 1000.0 / ema_ms_per_step)
            : 0.0;

        // ── 10. Checkpoint ─────────────────────────────────────
        if (step_count % opts.checkpoint_every == 0) {
            dev->sync();
            std::string ckpt = opts.output_dir + "/step-" + std::to_string(step_count);
            centroid_acc.write_snapshot(step_count);
            writer.save_checkpoint(model, *base, opts.domain,
                                   step_count, tokens_seen, ckpt);
            std::cerr << "[trainer] checkpoint → " << ckpt << "\n";
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