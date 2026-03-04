# Building tensor-adapt

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| CMake | 3.21 | |
| C++ compiler | GCC 11, Clang 14, or MSVC 19.30 | C++20 required |
| CUDA Toolkit | 11.8 | Volta or newer (sm_70+) — required for training |
| libcurl | any recent | with SSL support |
| nlohmann_json | 3.x | |
| OpenSSL | any recent | used for base model SHA fingerprinting |
| vcpkg | any | recommended for managing dependencies |

CUDA is optional at the CMake level — the resolver, `hf`, and `dataset-format`
tools build without it. The training stack (`tensor-adapt`, `centroid-merge`)
requires CUDA and will be skipped if a compiler is not found.

---

## Clone

```bash
git clone https://github.com/carbon-os/tensor-adapt
cd tensor-adapt
```

---

## Dependencies via vcpkg

If you don't have vcpkg yet:

```bash
apt-get install curl zip unzip tar pkg-config flex bison cmake
git clone --depth 1 https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
cd ..
```

Install the required packages:

```bash
./vcpkg/vcpkg install curl nlohmann-json openssl
```

---

## Configure

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
```

To disable CUDA explicitly (resolver, `hf`, and `dataset-format` only):

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DTENSOR_BACKEND_CUDA=OFF
```

CUDA architectures default to `70;80;86;89;90` (Volta through Hopper).
To target a specific GPU only, e.g. Ada (sm_89):

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_CUDA_ARCHITECTURES=89
```

---

## Build

```bash
cmake --build build --parallel
```

Binaries will be at:

```
build/hf
build/dataset-format
build/tensor-adapt
build/centroid-merge
```

`tensor-adapt` and `centroid-merge` are only produced when CUDA is found.
`hf` and `dataset-format` always build.

---

## Fetching Qwen2.5 0.5B

Qwen2.5 0.5B is a public model — no token required. It is the recommended
target for adapter development and testing.

### List the repo files first

```bash
./build/hf list hf://Qwen/Qwen2.5-0.5B
```

Expected output (approximate):

```
config.json
generation_config.json
merges.txt
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json

7 file(s)
```

Qwen2.5 0.5B fits in a single `model.safetensors` shard — no sharding needed.

### Download weights and config

```bash
./build/hf fetch hf://Qwen/Qwen2.5-0.5B --model \
    --include '*.safetensors' \
    --include '*.json'
```

### Verify the download

```bash
ls -lh ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B/
```

Expected:

```
config.json
generation_config.json
merges.txt
model.safetensors      ~  990 MB
tokenizer.json
tokenizer_config.json
vocab.json
```

---

## Preparing a dataset

`tensor-adapt` reads pre-tokenized datasets — flat files of `uint32_t` token
IDs in `.bin` format. The `dataset-format` tool handles fetching and
tokenizing raw datasets from HuggingFace into this format.

### Step 1 — fetch a raw dataset

For quick testing, `stas/openwebtext-10k` is small, open, and requires no
token:

```bash
./build/hf fetch hf://stas/openwebtext-10k --dataset
```

For a larger general-purpose text dataset:

```bash
./build/hf fetch hf://roneneldan/TinyStories --dataset \
    --include 'TinyStories-valid.txt'
```

For Go code (no token required):

```bash
./build/hf fetch hf://bigcode/the-stack-smol:go --dataset
```

### Step 2 — tokenize with dataset-format

`dataset-format` reads the downloaded files and writes a `train.bin` file
of raw `uint32_t` token IDs using the same tokenizer as your base model.

Supported input formats are detected automatically from file extension:

| Extension | Format |
|---|---|
| `.txt` | Plain text — splits on `<\|endoftext\|>` if present |
| `.jsonl` | One JSON object per line |
| `.json` | JSON array of objects |
| directory | Recursively processes all `.txt`, `.jsonl`, `.json` files |

```bash
# openwebtext-10k  (jsonl, "text" field)
./build/dataset-format \
    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --input     ~/.cache/tensor/datasets/stas/openwebtext-10k \
    --output    ./data/openwebtext-10k

# TinyStories  (txt with <|endoftext|> separators)
./build/dataset-format \
    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --input     ~/.cache/tensor/datasets/roneneldan/TinyStories \
    --output    ./data/tinystories

# the-stack-smol go subset  (jsonl, "content" field)
./build/dataset-format \
    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --input     ~/.cache/tensor/datasets/bigcode/the-stack-smol/go \
    --field     content \
    --output    ./data/stack-smol-go
```

Expected output:

```
[tokenizer] vocab=151936 merges=151387 added=208 eos=151643
[dataset-format] 1 file(s) to process
[dataset-format] appending eos=151643 between documents
[dataset-format] openwebtext-10k.jsonl  docs=10000  tokens=5021873

[dataset-format] done
  documents : 10000
  tokens    : 5021873
  output    : ./data/openwebtext-10k/train.bin
  size      : 19 MB
```

### dataset-format options

```
Required:
  --tokenizer <dir>   model dir with tokenizer.json (or vocab.json + merges.txt)
  --input     <path>  file or directory to tokenize
  --output    <dir>   output directory — writes train.bin

Optional:
  --field  <name>  JSON key containing text  (default: text)
  --format <fmt>   txt | jsonl | json | auto (default: auto)
  --no-eos         skip EOS token between documents
```

---

## Training an adapter

With the base model and tokenized dataset in place:

```bash
./build/tensor-adapt \
    --arch    qwen2 \
    --base    ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --data    ./data/tinystories \
    --domain  test/tinystories \
    --tokens  1M \
    --output  ./adapters/qwen2.5-0.5b-tinystories \
    --device  cuda:0
```

`--arch` is optional — architecture is detected from `config.json`
automatically. Pass it to override detection or when running non-standard
configs.

Adapter rank, alpha, and optimizer settings are derived from the base model
automatically. The only things you control are the ones in `--help`.

### Resume a training run

```bash
./build/tensor-adapt \
    --base    ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --domain  sample/openwebtext \
    --output  ./adapters/qwen2.5-0.5b-openwebtext \
    --resume  ./adapters/qwen2.5-0.5b-openwebtext/step-5000 \
    --device  cuda:0
```

Centroid snapshots from before the interruption are preserved and the
accumulator picks up from where it left off.

---

## Centroid merge

`tensor-adapt` calls `centroid-merge` automatically at the end of a
completed training run. To run it manually — for example to re-cluster
with a different `k` without retraining:

```bash
./build/centroid-merge \
    ./adapters/qwen2.5-0.5b-openwebtext/step-5000/centroids \
    --out      ./adapters/qwen2.5-0.5b-openwebtext/adapter.centroid \
    --clusters 16
```

---

## Cache layout

All downloads go to `~/.cache/tensor/` by default:

```
~/.cache/tensor/
├── models/
│   └── Qwen/
│       └── Qwen2.5-0.5B/
│           ├── config.json
│           ├── model.safetensors
│           └── ...
└── datasets/
    └── stas/
        └── openwebtext-10k/
            └── ...
```

Tokenized `.bin` files are written to whatever `--output` directory you
pass to `dataset-format` — they are not stored under the cache root.

Running `hf fetch` on a repo that already has a warm cache directory exits
immediately without making any network requests.

To use a different cache location:

```bash
./build/hf fetch hf://Qwen/Qwen2.5-0.5B --model \
    --cache-dir /mnt/models
```

To print the active cache root:

```bash
./build/hf cache
```

---

## Gated models

Some models (Llama 3, Mistral, etc.) require accepting a license on
HuggingFace before downloading.

1. Accept the license at `https://huggingface.co/<org>/<model>`
2. Generate a token at `https://huggingface.co/settings/tokens`
3. Export and fetch:

```bash
export HF_TOKEN="hf_your_token_here"

./build/hf fetch hf://meta-llama/Llama-3.1-8B --model \
    --include '*.safetensors' \
    --include '*.json'
```

---

## Quick reference

```bash
# list files in a repo without downloading
./build/hf list hf://Qwen/Qwen2.5-0.5B

# download a model — weights and config only
./build/hf fetch hf://Qwen/Qwen2.5-0.5B --model \
    --include '*.safetensors' \
    --include '*.json'

# download a dataset (open, no token)
./build/hf fetch hf://stas/openwebtext-10k --dataset

# download a dataset subset
./build/hf fetch hf://bigcode/the-stack-smol:go --dataset

# gated repo
HF_TOKEN=hf_xxx ./build/hf fetch hf://meta-llama/Llama-3.1-8B --model

# custom cache location
./build/hf fetch hf://Qwen/Qwen2.5-0.5B --model --cache-dir /mnt/models

# print cache root
./build/hf cache

# tokenize a dataset (jsonl, default "text" field)
./build/dataset-format \
    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --input     ~/.cache/tensor/datasets/stas/openwebtext-10k \
    --output    ./data/openwebtext-10k

# tokenize a dataset (jsonl, custom field)
./build/dataset-format \
    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --input     ~/.cache/tensor/datasets/bigcode/the-stack-smol/go \
    --field     content \
    --output    ./data/stack-smol-go

# tokenize a txt dataset with <|endoftext|> separators
./build/dataset-format \
    --tokenizer ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --input     ~/.cache/tensor/datasets/roneneldan/TinyStories \
    --output    ./data/tinystories

# train an adapter
./build/tensor-adapt \
    --arch   qwen2 \
    --base   ~/.cache/tensor/models/Qwen/Qwen2.5-0.5B \
    --data   ./data/openwebtext-10k \
    --domain sample/openwebtext \
    --tokens 50M \
    --output ./adapters/qwen2.5-0.5b-openwebtext \
    --device cuda:0

# merge centroid snapshots manually
./build/centroid-merge \
    ./adapters/qwen2.5-0.5b-openwebtext/step-5000/centroids \
    --out ./adapters/qwen2.5-0.5b-openwebtext/adapter.centroid \
    --clusters 16
```