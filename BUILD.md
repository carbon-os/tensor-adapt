# Building tensor-adapt

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| CMake | 3.21 | |
| C++ compiler | GCC 11, Clang 14, or MSVC 19.30 | C++20 required |
| libcurl | any recent | with SSL support |
| nlohmann_json | 3.x | |
| vcpkg | any | recommended for managing dependencies |
| CUDA Toolkit | 11.8 | optional — only needed for training |

CUDA is optional at this stage. The resolver and `hf-fetch` are pure C++
and build without it. The CMake configuration detects CUDA automatically
and enables it if found — if it isn't found the build continues without it.

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
./vcpkg/vcpkg install curl nlohmann-json
```

---

## Configure

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
```

To disable CUDA explicitly (pure CPU / resolve-only build):

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DTENSOR_BACKEND_CUDA=OFF
```

---

## Build

```bash
cmake --build build --parallel
```

The `hf-fetch` binary will be at:

```
build/hf-fetch
```

---

## Fetching Qwen2 0.5B

Qwen2 0.5B is a public model — no token required.

### List the repo files first

It's worth checking what's in the repo before downloading. This hits the
HF Hub API and prints the file list without downloading anything:

```bash
./build/hf-fetch list hf://Qwen/Qwen2-0.5B
```

Expected output (approximate — HF repos can add files over time):

```
.gitattributes
README.md
config.json
generation_config.json
merges.txt
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json

9 file(s)
```

Qwen2 0.5B is small enough to fit in a single `model.safetensors` shard —
no splitting needed.

### Download the full model

```bash
./build/hf-fetch fetch hf://Qwen/Qwen2-0.5B --model
```

This downloads everything in the repo into:

```
~/.cache/tensor/models/Qwen/Qwen2-0.5B/
```

### Download weights and config only

If you want to skip the README and git metadata and only pull what
tensor-adapt actually needs:

```bash
./build/hf-fetch fetch hf://Qwen/Qwen2-0.5B --model \
    --include '*.safetensors' \
    --include '*.json'
```

This pulls:

```
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

### Verify the download

```bash
ls -lh ~/.cache/tensor/models/Qwen/Qwen2-0.5B/
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

## Cache

All downloads go to `~/.cache/tensor/` by default, separated by kind:

```
~/.cache/tensor/
├── models/
│   └── Qwen/
│       └── Qwen2-0.5B/
│           ├── config.json
│           ├── model.safetensors
│           └── ...
└── datasets/
    └── bigcode/
        └── the-stack-v2/
            └── go/
                └── ...
```

Running `hf-fetch fetch` on a repo that already has a warm cache directory
exits immediately without making any network requests.

To use a different cache location:

```bash
./build/hf-fetch fetch hf://Qwen/Qwen2-0.5B --model \
    --cache-dir /mnt/models
```

To check the active cache root:

```bash
./build/hf-fetch cache
```

---

## Gated models

Some models (Llama 3, Mistral, etc.) require accepting a license on
HuggingFace before downloading.

1. Accept the license at `https://huggingface.co/meta-llama/Llama-3.1-8B`
2. Generate a token at `https://huggingface.co/settings/tokens`
3. Export the token and fetch:

```bash
export HF_TOKEN="hf_your_token_here"

./build/hf-fetch fetch hf://meta-llama/Llama-3.1-8B --model \
    --include '*.safetensors' \
    --include '*.json'
```

---

## Quick reference

```bash
# list files in a repo without downloading
./build/hf-fetch list hf://Qwen/Qwen2-0.5B

# download a model (auto-detected)
./build/hf-fetch fetch hf://Qwen/Qwen2-0.5B

# download a model (explicit), weights + config only
./build/hf-fetch fetch hf://Qwen/Qwen2-0.5B --model \
    --include '*.safetensors' \
    --include '*.json'

# download a dataset subset
./build/hf-fetch fetch hf://bigcode/the-stack-v2 --dataset --subset go

# same, using inline subset syntax
./build/hf-fetch fetch hf://bigcode/the-stack-v2:go --dataset

# gated repo
HF_TOKEN=hf_xxx ./build/hf-fetch fetch hf://meta-llama/Llama-3.1-8B --model

# custom cache location
./build/hf-fetch fetch hf://Qwen/Qwen2-0.5B --cache-dir /mnt/models

# print cache root
./build/hf-fetch cache
```