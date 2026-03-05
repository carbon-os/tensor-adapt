#include <tensor/base/base_loader.hpp>
#include <tensor/base/frozen_base.hpp>
#include <tensor/adapter/adapter_config.hpp>
#include <tensor/data/dataset.hpp>
#include <tensor/trainer/adapt_trainer.hpp>
#include <tensor/backend/cuda/device.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace tensor;
using namespace tensor::backend::cuda;
using namespace tensor::base;
using namespace tensor::adapter;
using namespace tensor::data;
using namespace tensor::trainer;

// ─────────────────────────────────────────────────────────────
//  Argument parser
// ─────────────────────────────────────────────────────────────

// Sentinel: -1 means "not set by user, use auto heuristic".
static constexpr int    UNSET_INT   = -1;
static constexpr float  UNSET_FLOAT = -1.f;

struct Args {
    std::string arch;
    std::string base;
    std::string data;
    std::string domain;
    std::string output;
    std::string device = "cuda:0";
    std::string resume;

    std::size_t tokens           = 50'000'000;
    std::size_t checkpoint_every = 1000;
    uint64_t    seed             = 42;

    // LoRA capacity overrides (UNSET = use auto heuristic)
    int   rank   = UNSET_INT;
    float alpha  = UNSET_FLOAT;   // defaults to rank if rank is set and alpha is not

    // Optimiser / schedule overrides
    float lr          = UNSET_FLOAT;
    float grad_clip   = UNSET_FLOAT;
    int   warmup      = UNSET_INT;

    // Batch / sequence overrides
    int   batch = UNSET_INT;
    int   seq   = UNSET_INT;

    // Injection target overrides (empty = keep auto)
    // Accepted: "qkvo", "qkv", "q", "o", ...
    std::string inject;
};

static std::size_t parse_token_budget(const std::string& s) {
    if (s.empty()) throw std::invalid_argument("empty token budget");
    char suffix = s.back();
    if (suffix == 'M' || suffix == 'm')
        return (std::size_t)std::stoull(s.substr(0, s.size()-1)) * 1'000'000ULL;
    if (suffix == 'G' || suffix == 'g')
        return (std::size_t)std::stoull(s.substr(0, s.size()-1)) * 1'000'000'000ULL;
    return std::stoull(s);
}

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Required:\n"
        << "  --base    <dir>         model directory (safetensors + config.json)\n"
        << "  --data    <dir|file>    pre-tokenized .bin dataset path\n"
        << "  --domain  <name>        adapter domain label, e.g. golang/gin\n"
        << "  --output  <dir>         adapter output directory\n"
        << "\n"
        << "Optional — infrastructure:\n"
        << "  --arch       <name>     architecture hint (default: auto from config.json)\n"
        << "  --tokens     <N>[M|G]   token training budget (default: 50M)\n"
        << "  --device     <cuda:N>   CUDA device (default: cuda:0)\n"
        << "  --resume     <ckpt-dir> resume from checkpoint\n"
        << "  --ckpt-every <N>        steps between checkpoints (default: 1000)\n"
        << "  --seed       <N>        RNG seed (default: 42)\n"
        << "\n"
        << "Optional — adapter capacity (default: auto from model size):\n"
        << "  --rank    <N>           LoRA rank, e.g. 4 8 16 32 64\n"
        << "  --alpha   <F>           LoRA alpha (default: same as rank)\n"
        << "  --inject  <targets>     which projections to adapt, e.g. qkvo qkv q o\n"
        << "                          (default: auto — qkvo for small models)\n"
        << "\n"
        << "Optional — optimiser / schedule (default: auto from model size + VRAM):\n"
        << "  --lr         <F>        peak learning rate, e.g. 2e-4\n"
        << "  --warmup     <N>        warmup steps (default: 100-500 by model size)\n"
        << "  --grad-clip  <F>        gradient clip norm (default: 1.0)\n"
        << "  --batch      <N>        batch size override\n"
        << "  --seq        <N>        sequence length override\n"
        << "\n"
        << "Examples:\n"
        << "  # Baseline — let everything auto-configure:\n"
        << "  " << prog << " \\\n"
        << "    --base   ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \\\n"
        << "    --data   ~/.cache/tensor/datasets/bigcode/the-stack-v2/go \\\n"
        << "    --domain golang/gin --tokens 50M --output ./adapters/qwen-gin\n"
        << "\n"
        << "  # Bump rank to fix hallucination on a low-capacity adapter:\n"
        << "  " << prog << " \\\n"
        << "    --base   ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \\\n"
        << "    --data   ~/.cache/tensor/datasets/bigcode/the-stack-v2/go \\\n"
        << "    --domain golang/gin --tokens 50M --output ./adapters/qwen-gin-r16 \\\n"
        << "    --rank 16 --alpha 16\n"
        << "\n"
        << "  # Aggressive experiment — high rank, slow lr, long warmup:\n"
        << "  " << prog << " \\\n"
        << "    --base   ~/.cache/tensor/models/Qwen/Qwen2.5-7B \\\n"
        << "    --data   ~/.cache/tensor/datasets/bigcode/the-stack-v2/go \\\n"
        << "    --domain golang/gin --tokens 200M --output ./adapters/qwen7b-gin-r32 \\\n"
        << "    --rank 32 --alpha 32 --lr 5e-5 --warmup 500 --inject qkvo\n";
}

static Args parse(int argc, char** argv) {
    Args a;

    std::unordered_map<std::string, std::string*> str_flags = {
        {"--arch",   &a.arch},
        {"--base",   &a.base},
        {"--data",   &a.data},
        {"--domain", &a.domain},
        {"--output", &a.output},
        {"--device", &a.device},
        {"--resume", &a.resume},
        {"--inject", &a.inject},
    };

    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];

        if (key == "-h" || key == "--help") { usage(argv[0]); std::exit(0); }

        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << key << " requires an argument\n";
                usage(argv[0]);
                std::exit(1);
            }
            return argv[++i];
        };

        if      (key == "--tokens")    { a.tokens           = parse_token_budget(next()); }
        else if (key == "--ckpt-every"){ a.checkpoint_every = std::stoull(next()); }
        else if (key == "--seed")      { a.seed             = std::stoull(next()); }
        else if (key == "--rank")      { a.rank             = std::stoi(next()); }
        else if (key == "--alpha")     { a.alpha            = std::stof(next()); }
        else if (key == "--lr")        { a.lr               = std::stof(next()); }
        else if (key == "--grad-clip") { a.grad_clip        = std::stof(next()); }
        else if (key == "--warmup")    { a.warmup           = std::stoi(next()); }
        else if (key == "--batch")     { a.batch            = std::stoi(next()); }
        else if (key == "--seq")       { a.seq              = std::stoi(next()); }
        else if (str_flags.count(key)) { *str_flags[key]   = next(); }
        else {
            std::cerr << "unknown option: " << key << "\n";
            usage(argv[0]);
            std::exit(1);
        }
    }

    bool ok = true;
    if (a.base.empty())   { std::cerr << "missing --base\n";   ok = false; }
    if (a.data.empty())   { std::cerr << "missing --data\n";   ok = false; }
    if (a.domain.empty()) { std::cerr << "missing --domain\n"; ok = false; }
    if (a.output.empty()) { std::cerr << "missing --output\n"; ok = false; }

    if (a.rank != UNSET_INT && a.rank <= 0) {
        std::cerr << "--rank must be > 0\n"; ok = false;
    }
    if (a.batch != UNSET_INT && a.batch <= 0) {
        std::cerr << "--batch must be > 0\n"; ok = false;
    }
    if (a.seq != UNSET_INT && a.seq <= 0) {
        std::cerr << "--seq must be > 0\n"; ok = false;
    }

    if (!ok) { usage(argv[0]); std::exit(1); }

    return a;
}

// ─────────────────────────────────────────────────────────────
//  Apply CLI overrides on top of auto-derived AdapterConfig
// ─────────────────────────────────────────────────────────────

static void apply_overrides(AdapterConfig& cfg, const Args& args) {
    bool changed = false;

    if (args.rank != UNSET_INT) {
        cfg.rank  = (std::size_t)args.rank;
        // Default alpha = rank unless the user also set alpha.
        cfg.alpha = (args.alpha != UNSET_FLOAT) ? args.alpha : (float)args.rank;
        changed = true;
    } else if (args.alpha != UNSET_FLOAT) {
        cfg.alpha = args.alpha;
        changed = true;
    }

    if (args.lr != UNSET_FLOAT) {
        cfg.lr = args.lr;
        changed = true;
    }
    if (args.grad_clip != UNSET_FLOAT) {
        cfg.grad_clip = args.grad_clip;
        changed = true;
    }
    if (args.warmup != UNSET_INT) {
        cfg.warmup_steps = (std::size_t)args.warmup;
        changed = true;
    }
    if (args.batch != UNSET_INT) {
        cfg.batch_size = (std::size_t)args.batch;
        changed = true;
    }
    if (args.seq != UNSET_INT) {
        cfg.seq_len = (std::size_t)args.seq;
        changed = true;
    }

    // --inject qkvo / qkv / q / o etc.
    if (!args.inject.empty()) {
        const std::string& s = args.inject;
        cfg.inject_q = s.find('q') != std::string::npos;
        cfg.inject_k = s.find('k') != std::string::npos;
        cfg.inject_v = s.find('v') != std::string::npos;
        cfg.inject_o = s.find('o') != std::string::npos;
        changed = true;
    }

    if (changed) {
        std::cerr << "[tensor-adapt] overrides applied:"
                  << " rank="     << cfg.rank
                  << " alpha="    << cfg.alpha
                  << " lr="       << cfg.lr
                  << " grad_clip="<< cfg.grad_clip
                  << " warmup="   << cfg.warmup_steps
                  << " batch="    << cfg.batch_size
                  << " seq="      << cfg.seq_len
                  << " inject Q=" << cfg.inject_q
                  << " K="        << cfg.inject_k
                  << " V="        << cfg.inject_v
                  << " O="        << cfg.inject_o
                  << "\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args = parse(argc, argv);

    try {
        // 1. Device.
        auto dev = Device::open(args.device);
        std::cerr << "[tensor-adapt] device: " << args.device << "\n";

        // 2. Load base model.
        std::cerr << "[tensor-adapt] loading base: " << args.base << "\n";
        auto base = BaseLoader::load(args.base, dev, args.arch);
        std::cerr << "[tensor-adapt] base loaded — "
                  << base.config.arch_name
                  << " layers=" << base.config.num_layers
                  << " hidden=" << base.config.hidden_size << "\n";

        // 3. Derive adapter config from base + VRAM, then apply CLI overrides.
        auto cfg = AdapterConfig::for_base(base, dev);
        apply_overrides(cfg, args);

        std::cerr << "[tensor-adapt] final config:"
                  << " rank="  << cfg.rank
                  << " alpha=" << cfg.alpha
                  << " lr="    << cfg.lr
                  << " inject Q=" << cfg.inject_q
                  << " K=" << cfg.inject_k
                  << " V=" << cfg.inject_v
                  << " O=" << cfg.inject_o << "\n";

        // 4. Dataset.
        std::cerr << "[tensor-adapt] loading dataset: " << args.data << "\n";
        auto dataset = Dataset::load(args.data);
        std::cerr << "[tensor-adapt] dataset: "
                  << dataset.total_tokens() << " tokens\n";

        if (dataset.total_tokens() < (std::size_t)(cfg.seq_len + 1)) {
            throw std::runtime_error(
                "dataset too small: need at least " +
                std::to_string(cfg.seq_len + 1) + " tokens");
        }

        // 5. Options.
        AdaptOptions opts;
        opts.domain           = args.domain;
        opts.output_dir       = args.output;
        opts.checkpoint_every = args.checkpoint_every;
        opts.resume_from      = args.resume;
        opts.device           = args.device;
        opts.seed             = args.seed;
        opts.log_to_stdout    = true;
        opts.tokens           = args.tokens;

        std::cerr << "[tensor-adapt] token budget: " << opts.tokens
                  << " domain: " << opts.domain << "\n";

        // 6. Train.
        auto trainer = AdaptTrainer::create(base, cfg, dataset, opts, dev);
        trainer.run();

        std::cerr << "[tensor-adapt] done.\n";

    } catch (const std::exception& e) {
        std::cerr << "[tensor-adapt] error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}