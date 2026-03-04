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

struct Args {
    std::string arch;
    std::string base;
    std::string data;
    std::string domain;
    std::string output;
    std::string device = "cuda:0";
    std::string resume;
    std::size_t tokens            = 50'000'000;
    std::size_t checkpoint_every  = 1000;
    uint64_t    seed              = 42;
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
        << "Optional:\n"
        << "  --arch       <name>     architecture hint (default: auto from config.json)\n"
        << "  --tokens     <N>[M|G]   token training budget (default: 50M)\n"
        << "  --device     <cuda:N>   CUDA device (default: cuda:0)\n"
        << "  --resume     <ckpt-dir> resume from checkpoint\n"
        << "  --ckpt-every <N>        steps between checkpoints (default: 1000)\n"
        << "  --seed       <N>        RNG seed (default: 42)\n"
        << "\n"
        << "Example:\n"
        << "  " << prog << " \\\n"
        << "    --arch    qwen2 \\\n"
        << "    --base    ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \\\n"
        << "    --data    ~/.cache/tensor/datasets/bigcode/the-stack-v2/go \\\n"
        << "    --domain  golang/gin \\\n"
        << "    --tokens  50M \\\n"
        << "    --output  ./adapters/qwen2.5-0.5b-golang-gin \\\n"
        << "    --device  cuda:0\n";
}

static Args parse(int argc, char** argv) {
    Args a;

    std::unordered_map<std::string, std::string*> flags = {
        {"--arch",   &a.arch},
        {"--base",   &a.base},
        {"--data",   &a.data},
        {"--domain", &a.domain},
        {"--output", &a.output},
        {"--device", &a.device},
        {"--resume", &a.resume},
    };

    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];

        if (key == "-h" || key == "--help") { usage(argv[0]); std::exit(0); }

        if (key == "--tokens" && i + 1 < argc) {
            a.tokens = parse_token_budget(argv[++i]);
        } else if (key == "--ckpt-every" && i + 1 < argc) {
            a.checkpoint_every = std::stoull(argv[++i]);
        } else if (key == "--seed" && i + 1 < argc) {
            a.seed = std::stoull(argv[++i]);
        } else if (flags.count(key) && i + 1 < argc) {
            *flags[key] = argv[++i];
        } else {
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
    if (!ok) { usage(argv[0]); std::exit(1); }

    return a;
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

        // 3. Derive adapter config from base.
        auto cfg = AdapterConfig::for_base(base);
        std::cerr << "[tensor-adapt] rank=" << cfg.rank
                  << " alpha=" << cfg.alpha
                  << " lr=" << cfg.lr
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