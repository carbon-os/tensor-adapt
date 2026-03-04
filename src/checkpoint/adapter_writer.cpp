#include <tensor/checkpoint/adapter_writer.hpp>
#include <tensor/adapter/adapter_model.hpp>
#include <tensor/base/frozen_base.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace tensor::checkpoint {

using namespace adapter;
using namespace base;

AdapterWriter::AdapterWriter(const std::string& base_dir)
    : base_dir_(base_dir)
{
    fs::create_directories(base_dir);
}

// ── safetensors serialisation (minimal, header + raw data) ───

static void write_f32_to_bf16_file(
    const std::string& path,
    const std::vector<std::pair<std::string, std::vector<float>>>& tensors)
{
    // Build JSON header.
    // safetensors format: [8 bytes: header_len][header_len bytes: json][data]
    json header;
    std::size_t offset = 0;
    std::vector<std::vector<uint16_t>> bufs;

    for (const auto& [name, data] : tensors) {
        // Convert F32 → BF16 raw bits.
        std::vector<uint16_t> bf16(data.size());
        for (std::size_t i = 0; i < data.size(); i++) {
            __nv_bfloat16 v = __float2bfloat16(data[i]);
            std::memcpy(&bf16[i], &v, 2);
        }
        bufs.push_back(bf16);

        std::size_t nbytes = data.size() * 2;
        header[name] = {
            {"dtype", "BF16"},
            {"shape", json::array({data.size()})}, // flat 1-D for now
            {"data_offsets", json::array({offset, offset + nbytes})}
        };
        offset += nbytes;
    }

    std::string header_str = header.dump();
    // Pad to 8-byte alignment.
    while (header_str.size() % 8 != 0) header_str += ' ';
    uint64_t hlen = header_str.size();

    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(header_str.data(), hlen);
    for (const auto& buf : bufs) {
        f.write(reinterpret_cast<const char*>(buf.data()), buf.size() * 2);
    }
}

void AdapterWriter::write_safetensors(
    const AdapterModel& model, const std::string& path)
{
    std::vector<std::pair<std::string, std::vector<float>>> tensors;

    auto pull = [&](const std::string& prefix, const LoraPair& lp) {
        tensors.emplace_back(prefix + ".lora_A", lp.A_f32.to_host_f32());
        tensors.emplace_back(prefix + ".lora_B", lp.B_f32.to_host_f32());
    };

    for (std::size_t l = 0; l < model.layers.size(); l++) {
        std::string lp = "layers." + std::to_string(l);
        const auto& la = model.layers[l];
        pull(lp + ".self_attn.q_proj", la.lora_q);
        pull(lp + ".self_attn.k_proj", la.lora_k);
        pull(lp + ".self_attn.v_proj", la.lora_v);
        pull(lp + ".self_attn.o_proj", la.lora_o);
    }

    write_f32_to_bf16_file(path, tensors);
}

void AdapterWriter::write_adapter_json(
    const AdapterModel& model,
    const FrozenBase&   base,
    const std::string&  domain,
    std::size_t step, std::size_t tokens,
    const std::string& path)
{
    const auto& cfg = model.config();
    const auto& bc  = base.config;

    json j = {
        {"domain",          domain},
        {"architecture",    bc.arch_name},
        {"base_model",      bc.model_id.empty() ? "unknown" : bc.model_id},
        {"base_sha",        bc.base_sha},
        {"rank",            cfg.rank},
        {"alpha",           cfg.alpha},
        {"target_begin",    0},
        {"target_end",      (int)bc.num_layers - 1},
        {"inject_q",        cfg.inject_q},
        {"inject_k",        cfg.inject_k},
        {"inject_v",        cfg.inject_v},
        {"inject_o",        cfg.inject_o},
        {"inject_up",       cfg.inject_up},
        {"inject_down",     cfg.inject_down},
        {"tokens_trained",  tokens},
        {"centroid_dim",    bc.hidden_size},
        {"centroid_count",  16},
        {"step",            step},
        {"tensor_adapt_version", "0.1.0"}
    };

    std::ofstream f(path);
    f << j.dump(2);
}

void AdapterWriter::save_checkpoint(
    const AdapterModel& model,
    const FrozenBase&   base,
    const std::string&  domain,
    std::size_t step, std::size_t tokens,
    const std::string& ckpt_dir)
{
    fs::create_directories(ckpt_dir);
    write_safetensors(model, ckpt_dir + "/adapter.safetensors");
    write_adapter_json(model, base, domain, step, tokens, ckpt_dir + "/adapter.json");

    // Write train_state.json.
    json state = {{"step", step}, {"tokens_consumed", tokens}};
    std::ofstream(ckpt_dir + "/train_state.json") << state.dump(2);

    std::cerr << "[writer] checkpoint saved → " << ckpt_dir << "\n";
}

void AdapterWriter::save_final(
    const AdapterModel& model,
    const FrozenBase&   base,
    const std::string&  domain,
    std::size_t step, std::size_t tokens,
    const std::string& out_dir)
{
    fs::create_directories(out_dir);
    write_safetensors(model, out_dir + "/adapter.safetensors");
    write_adapter_json(model, base, domain, step, tokens, out_dir + "/adapter.json");
    std::cerr << "[writer] final adapter saved → " << out_dir << "\n";
}

} // namespace tensor::checkpoint