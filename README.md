<div align="center">
  <img src="./assets/logo_.png" alt="Tensor Adapt" width="400px">

  <div>

# Tensor Adapt

  </div>

![Framework](https://img.shields.io/badge/Tensor-Framework-blue)
![Adapters](https://img.shields.io/badge/Adapters-Low--Rank%20Delta-blue)
![Precision](https://img.shields.io/badge/Precision-BF16%20%2B%20FP32-blue)
![Backend](https://img.shields.io/badge/Backend-CUDA-blue)

</div>

<br>

Native C++ adapter training library built for performance. Freeze your base model, 
train low-rank delta weights, and hot-swap adapters at deployment — no full fine-tune 
required.

---

## Router-Native Adapters

Tensor adapters are structurally compatible with standard LoRA — the A/B weight 
matrices are the same, the training mechanics are the same. But they are not 
general-purpose LoRA adapters. They are built specifically to run inside 
`tensor-inference`, which operates a modified Mixture-of-Experts router capable 
of managing millions of adapters simultaneously.

For that router to work, it needs to know what each adapter actually learned — 
not a label you wrote by hand, not a filename, not a domain tag in a config. 
It needs a mathematical description of the adapter's learned expertise, derived 
automatically from the training process itself.

That description is the **Semantic Centroid**.

---

## Semantic Centroids

Every tensor adapter produces two output files:

```
adapter.safetensors   — the A/B weight matrices (what the adapter knows)
adapter.centroid      — the semantic centroid   (where it lives in latent space)
```

The `.centroid` file is a 512-dimensional vector. It is not hand-crafted. 
It is not an embedding of the domain label. It is computed automatically 
during training by the **Centroid Accumulator** — a component that runs 
as a side-channel inside the training loop.

### How it is produced

During the backward pass, the Centroid Accumulator monitors gradient norms 
token by token. Hidden states that produce the highest gradient signal — the 
tokens the model learned from most — are weighted heavily and accumulated into 
a running centroid:

```
C = Σ ( hₜ · ‖∇Lₜ‖ )
```

At the end of training, this centroid is normalized to the unit sphere and 
written to `adapter.centroid`. It represents the **semantic center of mass** 
of everything the adapter absorbed — not what you told it to learn, but what 
the gradient signal says it actually learned.

### Why this matters

`tensor-inference` loads `.centroid` files at startup and builds a 
**Product Key Memory (PKM) index** — a two-codebook structure that maps 
every centroid to a discrete address in a 512×512 = 262,144 bucket space. 
At inference time, the router computes the nearest PKM address from the 
model's current residual stream and retrieves the matching adapter in O(1).

No embedding call. No keyword search. No manual routing rules.

The residual stream at inference time lives in the same vector space as the 
centroid. The query and the index are naturally aligned. The router finds the 
right adapter because the centroid was built from the same signal the base 
model uses to process tokens.

A standard LoRA adapter has no centroid. It can be loaded manually or by name, 
but it cannot be discovered automatically by a semantic router. Tensor adapters 
can — that is the distinction.

---

## The Tensor Framework
```
tensor-pretrain     train base models from scratch
tensor-adapt        extend base knowledge with tensor adapters          ← you are here
tensor-inference    run base models + hot-swap adapters at runtime
```

Each tool is independent. Together they form a complete pipeline — from training
a base model to deploying a system that loads and unloads domain knowledge on 
demand.

---

## Why

Base models are general. Production systems are not.

A code assistant needs deep knowledge of specific libraries.
A reasoning assistant needs familiarity with specific document structures.
A support agent needs to know a specific product inside out.

Retraining a base model for each domain is expensive, slow, and produces
a separate model per domain — which means separate memory, separate deployment,
separate maintenance.

Tensor adapters solve this differently. The base stays frozen. Each domain
becomes a small, independently loadable artifact — a few megabytes — that
`tensor-inference` injects into the base at runtime in microseconds.
Swap a domain in, answer a question, swap it out. The base never moves.
Thousands of adapters can share a single base in memory.

`tensor-adapt` is the tool that produces those artifacts — against any supported 
base model.

---

## What it produces
```
adapters/golang-gin/
├── adapter.safetensors     # A/B weight matrices for every injected layer
├── adapter.centroid        # 512-dim semantic centroid — router index key
├── adapter.json            # metadata — base ref, architecture, rank, alpha, 
│                           #            layer range, domain
└── tokenizer/              # symlink to base model tokenizer
```

`adapter.json` is the contract between `tensor-adapt` and `tensor-inference`.
`tensor-inference` reads it at load time and rejects the adapter if anything
doesn't match the loaded base.

`adapter.centroid` is the contract between `tensor-adapt` and the PKM router.
Without it, the adapter exists but cannot be discovered at inference time.

---

## What is a tensor adapter?

A tensor adapter is a low-rank weight delta trained on top of a frozen base model.

During training, the base weights are fixed. Two small matrices — A and B —
are inserted alongside each target layer. The adapter learns the residual
signal the base doesn't already have for the target domain.

At inference time, `tensor-inference` injects the A/B matrices and adds
their product to the frozen base activations. The base memory footprint
is unchanged. Rank is kept low by design — the adapter is a focused delta,
not a general update.

This is not a general LoRA framework. Adapter rank, alpha, target layers,
and optimizer settings are derived from the detected base model architecture
and locked for that configuration. The output is a validated artifact
with a known format that `tensor-inference` can load and trust.

---

## Supported base models

`tensor-adapt` loads any model that `tensor-inference` can run.
Tensor adapters can be trained against any of these architectures:
```
Decoder-only     LLaMA 3 / 3.1 / 3.2 / 3.3
                 Qwen 2 / 2.5
                 Mistral / Mixtral
                 DeepSeek V2 / V3
                 Phi-3 / Phi-4
                 Gemma / Gemma 2
                 Falcon
                 Tensor Series (tensor-pretrain output)

Encoder-only     BERT / RoBERTa / DeBERTa / ModernBERT

Encoder-Decoder  T5 / BART
```

Base models are loaded from a local directory in `.safetensors` or `.gguf` format —
the same formats `tensor-inference` accepts. Any model `tensor-inference` can run,
`tensor-adapt` can train adapters against.

---

## Structure
```
tensor-adapt/
│
├── core/           — zero-dependency primitives
│                     DType, Shape, TensorView
│                     shared across the Tensor Framework
│
├── backend/        — compute accelerators
│   └── cuda/       — CUDA kernels and device memory management
│                     matmul, softmax, rms_norm, rope, silu, element-wise ops
│                     device allocation, host↔device transfers, stream management
│                     CUDA headers live here and only here
│
├── resolve/        — hf:// URI resolution, HTTP download, cache management
│                     same resolver used by tensor-pretrain
│                     parses hf://org/dataset URIs
│                     downloads to ~/.cache/tensor/{org}/{dataset}/
│
├── data/           — dataset loading, streaming, tokenization
│   └── loader/     — Dataset::load("org/dataset") — reads from cache,
│                     validates layout, exposes a token stream
│
├── parser/         — safetensors, GGUF, config.json, tokenizer.json
│                     memory-maps files, parses headers
│                     returns TensorView — never allocates weight memory
│                     no dependency on backend — no compute, no platform headers
│                     shared format support with tensor-inference
│
│                     parser/ knows about FILE FORMATS only.
│                     it does not know what a layer is, what attention is,
│                     or what any weight tensor means. that lives in base/arch/.
│
├── base/           — base model loading, architecture dispatch, frozen forward pass
│   │
│   ├── base_loader.hpp     — load(path, device) → FrozenBase
│   │                         reads config.json via parser/, detects architecture,
│   │                         dispatches to the correct arch/ implementation
│   │
│   ├── frozen_base.hpp     — frozen weights, architecture info,
│   │                         layer target enumeration, base_sha
│   │
│   └── arch/               — one implementation per model family
│       │                     each arch defines its own layer structure,
│       │                     attention variant, FFN layout, and norm positions.
│       │                     the file format (safetensors/GGUF) is parser/'s concern —
│       │                     what the weights mean and how they compose is arch/'s concern.
│       │                     adding a new architecture means adding a file in base/arch/.
│       │                     nothing in parser/ changes.
│       │
│       ├── llama.hpp        — LLaMA 3 / 3.1 / 3.2 / 3.3
│       ├── qwen2.hpp        — Qwen 2 / 2.5
│       ├── mistral.hpp      — Mistral / Mixtral (MoE dispatch included)
│       ├── deepseek.hpp     — DeepSeek V2 / V3 (MoE dispatch included)
│       ├── phi.hpp          — Phi-3 / Phi-4
│       ├── gemma.hpp        — Gemma / Gemma 2
│       ├── falcon.hpp       — Falcon
│       ├── bert.hpp         — BERT / RoBERTa / DeBERTa / ModernBERT
│       ├── t5.hpp           — T5 / BART
│       └── tensor.hpp       — Tensor Series (tensor-pretrain output)
│
├── centroid/       — semantic centroid accumulation and serialization
│                     CentroidAccumulator — gradient-weighted hidden state accumulator
│                     CentroidWriter      — normalizes and writes .centroid file
│                     hooks into trainer/ backward pass, zero overhead on forward
│
├── adapter/        — adapter architecture and forward pass
│                     AdapterConfig — rank, alpha, target layers derived from base
│                     AdapterModel  — A/B matrices, merged forward pass
│                     InitStrategy  — kaiming A, zero B (standard LoRA init)
│
├── trainer/        — adapter training loop, optimizer, LR schedule
│                     AdamW with adapter-specific settings
│                     cosine schedule with warmup
│                     gradient clipping, BF16 + FP32 master weights
│                     base weights excluded from all gradient ops
│
└── checkpoint/     — safetensors save, adapter.json + .centroid serialization
                      output is directly loadable by tensor-inference
```

---

## Namespace map
```
tensor::core::DType
tensor::core::Shape
tensor::core::TensorView

tensor::backend::cuda::Device
tensor::backend::cuda::Tensor
tensor::backend::cuda::Stream

tensor::resolve::Resolver
tensor::resolve::ResolveError
tensor::resolve::CacheLayout

tensor::data::Dataset
tensor::data::DatasetConfig
tensor::data::TokenStream
tensor::data::DataError

tensor::parser::WeightMap             — memory-mapped safetensors or GGUF
tensor::parser::ModelConfig           — parses config.json for any architecture
tensor::parser::ArchitectureType      — detected model family enum

tensor::base::BaseLoader              — load(path, device) → FrozenBase
tensor::base::FrozenBase              — frozen weights, architecture info,
                                        layer target enumeration, base_sha
tensor::base::BaseError               — unsupported architecture, corrupt weights

tensor::base::arch::LlamaBase         — LLaMA 3.x layer structure and forward pass
tensor::base::arch::Qwen2Base         — Qwen 2 / 2.5
tensor::base::arch::MistralBase       — Mistral / Mixtral
tensor::base::arch::DeepSeekBase      — DeepSeek V2 / V3
tensor::base::arch::PhiBase           — Phi-3 / Phi-4
tensor::base::arch::GemmaBase         — Gemma / Gemma 2
tensor::base::arch::FalconBase        — Falcon
tensor::base::arch::BertBase          — BERT / RoBERTa / DeBERTa / ModernBERT
tensor::base::arch::T5Base            — T5 / BART
tensor::base::arch::TensorBase        — Tensor Series

tensor::centroid::CentroidAccumulator — hooks into backward pass, accumulates
                                        gradient-weighted hidden states
tensor::centroid::CentroidWriter      — finalizes, normalizes, writes .centroid

tensor::adapter::AdapterConfig        — rank, alpha, target layers derived from base
tensor::adapter::AdapterModel         — A/B matrices, forward pass delta
tensor::adapter::AdaptOptions         — what the caller can actually change
tensor::adapter::InitStrategy         — kaiming A, zero B

tensor::trainer::AdaptTrainer         — the adapter training loop
tensor::trainer::AdamW                — AdamW, adapter settings
tensor::trainer::CosineSchedule       — warmup → cosine decay
tensor::trainer::GradClipper          — gradient norm clipping

tensor::checkpoint::AdapterWriter     — writes adapter.safetensors + adapter.json
                                        + adapter.centroid
tensor::checkpoint::AdapterLoader     — loads and validates an adapter checkpoint
```

---

## End-to-end example
```cpp
#include <tensor/base/base_loader.hpp>
#include <tensor/data/dataset.hpp>
#include <tensor/adapter/adapter_config.hpp>
#include <tensor/adapter/adapter_model.hpp>
#include <tensor/trainer/adapt_trainer.hpp>

using tensor::base::BaseLoader;
using tensor::data::Dataset;
using tensor::adapter::AdapterConfig;
using tensor::adapter::AdaptOptions;
using tensor::trainer::AdaptTrainer;

int main() {

    // 1. load the frozen base — architecture detected from config.json,
    //    dispatched to the correct base/arch/ implementation automatically.
    //    weights are read-only from this point forward.
    auto base = BaseLoader::load(
        "./models/Llama-3.1-8B/",
        "cuda:0"
    );

    // 2. adapter spec derived from detected architecture — read-only
    auto config = AdapterConfig::for_base(base);

    // 3. domain dataset — a focused slice, not a general corpus
    auto dataset = Dataset::load("bigcode/the-stack-v2:go")
        .max_tokens(50_M)
        .validate();

    // 4. what the caller can actually control
    auto options = AdaptOptions {
        .domain           = "golang/gin",
        .output_dir       = "./adapters/llama-3.1-8b-golang-gin/",
        .checkpoint_every = 1'000,
        .device           = "cuda:0",
        .seed             = 42,
    };

    // 5. train
    //    CentroidAccumulator runs automatically inside the training loop.
    //    adapter.centroid is written alongside adapter.safetensors on save.
    auto trainer = AdaptTrainer::create(base, config, dataset, options);
    trainer.run();
}
```

The centroid accumulator requires no configuration. It runs as part of every 
training loop and is not optional — an adapter without a `.centroid` file 
cannot be indexed by `tensor-inference`.

---

## AdapterConfig — derived from the base

`AdapterConfig::for_base()` inspects the loaded architecture — parameter count,
hidden size, number of layers, attention head structure — and selects the
appropriate locked configuration for that model class.
Nothing in it is settable at runtime.
```cpp
auto config = AdapterConfig::for_base(base);

// adapter architecture — read-only, derived from base
int         rank         = config.rank();
float       alpha        = config.alpha();
std::size_t target_begin = config.target_layer_begin();
std::size_t target_end   = config.target_layer_end();
bool        inject_q     = config.inject_q();
bool        inject_k     = config.inject_k();
bool        inject_v     = config.inject_v();
bool        inject_o     = config.inject_o();
bool        inject_up    = config.inject_up();
bool        inject_down  = config.inject_down();

// training config — preset per model class
float       lr           = config.learning_rate();
float       beta1        = config.adam_beta1();
float       beta2        = config.adam_beta2();
float       weight_decay = config.weight_decay();
float       grad_clip    = config.grad_clip_norm();
std::size_t batch_size   = config.batch_size();
std::size_t warmup_steps = config.warmup_steps();

// detected base info
std::string arch         = config.architecture();   // "llama", "qwen2", "tensor", ...
std::size_t params       = config.base_parameters();
```

### Config ladder

| Parameter Range | Rank | Alpha | Target Layers | Inject Attn | Inject FFN |
|---|---|---|---|---|---|
| < 200M | 2 | 2.0 | all | Q K V O | ❌ |
| 200M – 800M | 4 | 4.0 | all | Q K V O | ❌ |
| 800M – 2B | 8 | 8.0 | all | Q K V O | ✅ |
| 2B – 8B | 16 | 16.0 | all | Q K V O | ✅ |
| 8B – 20B | 32 | 32.0 | all | Q K V O | ✅ |
| 20B+ | 64 | 64.0 | all | Q K V O | ✅ |

Architecture family affects target layer selection where relevant — MoE models
(Mixtral, DeepSeek) inject into shared attention layers only, not individual
expert FFNs. This logic lives in the corresponding `base/arch/` implementation
and is surfaced through `FrozenBase::layer_targets()`.

---

## AdaptOptions — the only thing the caller controls
```cpp
struct AdaptOptions {
    std::string  domain;            // namespaced domain label, e.g. "golang/gin"
    std::string  output_dir;        // where adapter output is written
    std::size_t  checkpoint_every;  // steps between mid-run checkpoints
    std::string  resume_from;       // checkpoint dir to resume, or ""
    std::string  device;            // "cuda:0"
    uint64_t     seed;              // RNG seed for shuffle and init
    bool         log_to_stdout;     // default true
    std::string  wandb_project;     // "" disables W&B logging
};
```

Nothing about rank, alpha, target layers, optimizer settings, or schedule
belongs here. Those are in `AdapterConfig`. `AdaptOptions` is purely operational.

---

## Training loop
```cpp
auto trainer = AdaptTrainer::create(base, config, dataset, options);

// run to completion
trainer.run();

// or drive manually
while (!trainer.done()) {
    auto m = trainer.step();

    float       loss        = m.loss;
    float       lr          = m.learning_rate;
    float       grad_norm   = m.grad_norm;
    std::size_t step        = m.step;
    std::size_t tokens_seen = m.tokens_consumed;
}
```

### Cosine schedule
```
LR
 │
 │        ┌─────────────────────────────┐
 │       /                               \
 │      /                                 \
 │─────/                                   \──── floor
 │  warmup                               cosine decay
 │  200–500 steps  ◄── token budget ──►
 └──────────────────────────────────────────────── steps
```

Adapter training runs to a token budget, not a fixed step count.
The schedule scales to the dataset automatically.
Warmup step count scales with base model size — larger bases need a
gentler ramp. Minimum LR floor is `lr × 0.1`.

### AdamW — adapter settings
```
β1 = 0.9
β2 = 0.999   (tighter than base training — adapter signal is narrower)
ε  = 1e-8
λ  = 0.0     (no weight decay on adapter weights)
```

Gradient clipping to norm 1.0 before every optimizer step.
BF16 parameters maintained with FP32 master weights internally.
Base model weights receive no gradient signal at any point.

---

## Output format

All adapter output is safetensors. No other format.
Output is directly loadable by `tensor-inference`.
```cpp
trainer.save_checkpoint("./adapters/llama-3.1-8b-golang-gin/step-5000/");
trainer.save_adapter("./adapters/llama-3.1-8b-golang-gin/");
```

**Mid-run checkpoint layout:**
```
llama-3.1-8b-golang-gin-step-5000/
├── adapter.safetensors     # A/B matrices for all injected layers
├── adapter.centroid        # centroid snapshot at this step
├── optimizer.safetensors   # AdamW m, v, step tensors
├── train_state.json        # step, tokens_consumed, data position, RNG state
└── adapter.json            # AdapterConfig snapshot + base model ref
```

**Final adapter layout** (matches the tensor-inference load format):
```
llama-3.1-8b-golang-gin/
├── adapter.safetensors     # A/B matrices — inference-ready
├── adapter.centroid        # semantic centroid — PKM router index key
├── adapter.json            # metadata contract read by tensor-inference
└── tokenizer/              # symlink to base model tokenizer
```

### adapter.json
```json
{
  "domain":          "golang/gin",
  "architecture":    "llama",
  "base_model":      "meta-llama/Llama-3.1-8B",
  "base_sha":        "a3f9...",
  "rank":            16,
  "alpha":           16.0,
  "target_begin":    0,
  "target_end":      31,
  "inject_q":        true,
  "inject_k":        true,
  "inject_v":        true,
  "inject_o":        true,
  "inject_up":       true,
  "inject_down":     true,
  "tokens_trained":  48302080,
  "centroid_dim":    512,
  "tensor_adapt_version": "0.1.0"
}
```

`centroid_dim` is written by the CentroidAccumulator and validated by 
`tensor-inference` at index build time. A mismatch against the PKM codebook 
dimension is a hard reject.

---

## Resuming
```cpp
auto options = AdaptOptions {
    .domain      = "golang/gin",
    .output_dir  = "./adapters/llama-3.1-8b-golang-gin/",
    .resume_from = "./adapters/llama-3.1-8b-golang-gin/step-5000/",
    .device      = "cuda:0",
};

// seed is ignored on resume — RNG state is restored from checkpoint
// adapter.json is validated on load — mismatched base ref fails immediately
// centroid accumulator state is restored from the step-5000 snapshot
auto trainer = AdaptTrainer::create(base, config, dataset, options);
trainer.run();  // continues from step 5000, same data position
```

---

## Getting datasets

Same tooling as `tensor-pretrain`. `data-fetch` resolves `hf://` URIs and
stores everything under `~/.cache/tensor/`. Fetch once, train as many
adapters as needed.
```bash
./build/data-fetch fetch hf://bigcode/the-stack-v2 --subset go
./build/data-fetch fetch hf://bigcode/the-stack-v2 --subset python

export HF_TOKEN="hf_your_token_here"
./build/data-fetch fetch hf://some-org/some-dataset
```

---

## CLI tools

### data-fetch
```bash
./build/data-fetch fetch hf://bigcode/the-stack-v2 --subset go
./build/data-fetch fetch hf://HuggingFaceTB/smollm-corpus --subset python-edu

export HF_TOKEN="hf_your_token_here"
./build/data-fetch fetch hf://some-org/some-dataset
```

### tensor-adapt
```bash
# train — architecture detected automatically, centroid produced automatically
./build/tensor-adapt \
    --base   ./models/Llama-3.1-8B/ \
    --data   "bigcode/the-stack-v2:go" \
    --domain "golang/gin" \
    --tokens 50M \
    --output ./adapters/llama-3.1-8b-golang-gin/ \
    --device cuda:0

# same command, different base
./build/tensor-adapt \
    --base   ./models/Qwen2.5-7B/ \
    --data   "bigcode/the-stack-v2:go" \
    --domain "golang/gin" \
    --tokens 50M \
    --output ./adapters/qwen2.5-7b-golang-gin/ \
    --device cuda:0

# resume — centroid accumulator state restored from checkpoint
./build/tensor-adapt \
    --base    ./models/Llama-3.1-8B/ \
    --domain  "golang/gin" \
    --output  ./adapters/llama-3.1-8b-golang-gin/ \
    --resume  ./adapters/llama-3.1-8b-golang-gin/step-5000/ \
    --device  cuda:0
```

---

## Build

### Requirements

- CMake 3.21+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+)
- CUDA Toolkit 11.8+ — Volta or newer (sm_70+)

### Build
```bash
git clone https://github.com/netangular/tensor-adapt
cd tensor-adapt

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### CMake options

| Option | Default | Description |
|---|---|---|
| `TENSOR_BACKEND_CUDA` | `ON` if CUDA found | Enable CUDA backend |
| `TENSOR_BACKEND_CPU` | `ON` | CPU fallback — always built |
| `TENSOR_BUILD_TOOLS` | `ON` | Build CLI tools |
| `TENSOR_BUILD_TESTS` | `OFF` | Build tests |

---

## Status
```
core          ████████████████████  complete
resolve       ████████████████████  complete
parser        ████████████████████  complete
backend/cuda  ████████░░░░░░░░░░░░  in progress
data          ██████░░░░░░░░░░░░░░  in progress
base          ██████░░░░░░░░░░░░░░  in progress
base/arch     ████░░░░░░░░░░░░░░░░  in progress
centroid      ████░░░░░░░░░░░░░░░░  in progress
adapter       ████░░░░░░░░░░░░░░░░  in progress
trainer       ████░░░░░░░░░░░░░░░░  in progress
checkpoint    ████░░░░░░░░░░░░░░░░  in progress
CLI tools     ██░░░░░░░░░░░░░░░░░░  in progress
```

---

## Non-goals

- **Base model training** — use `tensor-pretrain`
- **Inference** — use `tensor-inference`
- **General LoRA framework** — tensor adapters for supported architectures only
- **Custom rank or alpha** — changes go through ablation and spec update
- **Full fine-tuning** — the base is always frozen
- **Configurable optimizer betas** — fixed per adapter spec
- **Alternative LR schedules** — cosine only
- **GGUF or pickle output** — safetensors only
- **Adapters without centroids** — every adapter must be router-discoverable

---

**Tensor Framework** is developed by [Netangular](https://github.com/netangular).  
*tensor-pretrain → tensor-adapt → tensor-inference*  
Apache 2.0 — free to use, modify, and build on, including for commercial purposes.