// base_loader.cpp
#include <tensor/base/base_loader.hpp>
#include <tensor/base/frozen_base.hpp>
#include <tensor/parser/config.hpp>
#include <tensor/parser/weight_map.hpp>

#include <openssl/sha.h>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

namespace tensor::base {

using namespace backend::cuda;
using namespace parser;

static ArchType detect_arch(const std::string& model_type, const std::string& hint) {
    std::string s = hint.empty() ? model_type : hint;
    if (s == "qwen2") return ArchType::Qwen2;
    if (s == "llama") return ArchType::LLaMA;
    if (s.find("qwen") != std::string::npos) return ArchType::Qwen2;
    if (s.find("llama") != std::string::npos) return ArchType::LLaMA;
    return ArchType::Unknown;
}

static std::string sha256_dir(const std::string& dir) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".safetensors")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    std::ostringstream oss;
    for (const auto& f : files)
        oss << f.filename().string() << ":" << fs::file_size(f) << ";";
    std::string content = oss.str();

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(content.c_str()),
           content.size(), hash);

    std::ostringstream hex;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        hex << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(hash[i]);
    return hex.str();
}

static Tensor upload_bf16(
    const TensorView& tv, const Device& dev, const std::string& name)
{
    if (tv.dtype == core::DType::BF16)
        return Tensor::from_host(tv, dev);

    if (tv.dtype == core::DType::F32) {
        std::size_t n = tv.numel();
        std::vector<__nv_bfloat16> buf(n);
        const float* src = static_cast<const float*>(tv.data);
        for (std::size_t i = 0; i < n; i++)
            buf[i] = __float2bfloat16(src[i]);
        core::TensorView bfv;
        bfv.data  = buf.data();
        bfv.dtype = core::DType::BF16;
        bfv.shape = tv.shape;
        return Tensor::from_host(bfv, dev);
    }
    throw BaseError("unsupported weight dtype for " + name);
}

FrozenBase BaseLoader::load(
    const std::string& dir,
    const Device&      dev,
    const std::string& arch_hint)
{
    std::cerr << "[base] loading from " << dir << "\n";

    auto cfg = parser::ModelConfig::from_dir(dir);
    auto wm  = parser::WeightMap::open(dir);

    FrozenBase base;
    BaseConfig& bc = base.config;

    bc.arch_name = cfg.model_type();
    bc.arch      = detect_arch(bc.arch_name, arch_hint);
    bc.base_sha  = sha256_dir(dir);

    if (bc.arch == ArchType::Unknown)
        throw BaseError("unknown architecture: " + bc.arch_name);

    bc.vocab_size        = cfg.vocab_size();
    bc.hidden_size       = cfg.hidden_size();
    bc.num_layers        = cfg.num_hidden_layers();
    bc.num_q_heads       = cfg.num_attention_heads();
    bc.num_kv_heads      = cfg.num_key_value_heads();
    bc.head_dim          = bc.hidden_size / bc.num_q_heads;
    bc.intermediate_size = cfg.intermediate_size();
    bc.rms_norm_eps      = cfg.rms_norm_eps();
    bc.rope_theta        = cfg.rope_theta();

    std::cerr << "[base] arch=" << bc.arch_name
              << " layers=" << bc.num_layers
              << " hidden=" << bc.hidden_size
              << " heads=" << bc.num_q_heads << "/" << bc.num_kv_heads
              << " intermediate=" << bc.intermediate_size << "\n";

    // ── Embeddings ────────────────────────────────────────────
    base.embed_w = upload_bf16(
        wm.tensor("model.embed_tokens.weight"), dev, "embed_tokens");

    // ── Transformer layers ────────────────────────────────────
    base.layers.reserve(bc.num_layers);
    for (std::size_t l = 0; l < bc.num_layers; l++) {
        auto p = [&](const std::string& k) {
            return "model.layers." + std::to_string(l) + "." + k;
        };

        LayerWeights lw;

        lw.input_norm_w = upload_bf16(
            wm.tensor(p("input_layernorm.weight")), dev, p("in_norm"));

        // ── Attention weights + QKV biases ────────────────────
        //
        // Qwen2 retains learned biases on Q, K, and V projections.
        // These shift the projected values before RoPE is applied,
        // giving the model more freedom to represent relative positions.
        // Loading them is mandatory: without the bias the model produces
        // completely wrong key/query/value vectors, and loss diverges
        // far above the uniform-random ceiling of ln(vocab_size) ≈ 11.9.
        lw.q_proj_w = upload_bf16(wm.tensor(p("self_attn.q_proj.weight")), dev, p("q_w"));
        lw.q_proj_b = upload_bf16(wm.tensor(p("self_attn.q_proj.bias")),   dev, p("q_b"));

        lw.k_proj_w = upload_bf16(wm.tensor(p("self_attn.k_proj.weight")), dev, p("k_w"));
        lw.k_proj_b = upload_bf16(wm.tensor(p("self_attn.k_proj.bias")),   dev, p("k_b"));

        lw.v_proj_w = upload_bf16(wm.tensor(p("self_attn.v_proj.weight")), dev, p("v_w"));
        lw.v_proj_b = upload_bf16(wm.tensor(p("self_attn.v_proj.bias")),   dev, p("v_b"));

        // O projection — no bias in Qwen2.
        lw.o_proj_w = upload_bf16(wm.tensor(p("self_attn.o_proj.weight")), dev, p("o_w"));

        lw.post_norm_w = upload_bf16(
            wm.tensor(p("post_attention_layernorm.weight")), dev, p("post_norm"));

        // MLP — no bias on any projection in Qwen2.
        lw.gate_proj_w = upload_bf16(wm.tensor(p("mlp.gate_proj.weight")), dev, p("gate"));
        lw.up_proj_w   = upload_bf16(wm.tensor(p("mlp.up_proj.weight")),   dev, p("up"));
        lw.down_proj_w = upload_bf16(wm.tensor(p("mlp.down_proj.weight")), dev, p("down"));

        base.layers.push_back(std::move(lw));

        if ((l + 1) % 4 == 0)
            std::cerr << "[base] loaded " << (l+1) << "/" << bc.num_layers << " layers\r";
    }
    std::cerr << "\n";

    base.final_norm_w = upload_bf16(wm.tensor("model.norm.weight"), dev, "final_norm");

    bc.tie_embeddings = !wm.contains("lm_head.weight");
    if (!bc.tie_embeddings)
        base.lm_head_w = upload_bf16(wm.tensor("lm_head.weight"), dev, "lm_head");

    dev.sync();
    std::cerr << "[base] loaded OK (sha=" << bc.base_sha.substr(0,12) << "...)\n";

    return base;
}

} // namespace tensor::base