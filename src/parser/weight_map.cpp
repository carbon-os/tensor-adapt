#include <tensor/parser/weight_map.hpp>
#include <tensor/parser/safetensors.hpp>
#include <tensor/parser/errors.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace tensor::parser {

struct WeightMap::Impl {
    // Currently SafeTensors only. GGUF will be a second variant here.
    SafeTensors st;
    explicit Impl(SafeTensors s) : st(std::move(s)) {}
};

WeightMap::WeightMap(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
WeightMap::~WeightMap() = default;

// ── factory ──────────────────────────────────────────────────

WeightMap WeightMap::open(const std::string& dir) {
    fs::path d(dir);

    if (!fs::exists(d)) {
        throw ParseError("WeightMap::open: path does not exist: " + dir);
    }

    // Single model.safetensors
    if (fs::is_regular_file(d) && d.extension() == ".safetensors") {
        return from_safetensors(dir);
    }

    if (fs::is_directory(d)) {
        // model.safetensors (single shard)
        auto single = d / "model.safetensors";
        if (fs::exists(single)) {
            return from_safetensors(single.string());
        }

        // Sharded: any .safetensors files
        std::vector<fs::path> shards;
        for (const auto& e : fs::directory_iterator(d)) {
            if (e.is_regular_file() && e.path().extension() == ".safetensors") {
                shards.push_back(e.path());
            }
        }
        if (!shards.empty()) {
            std::sort(shards.begin(), shards.end(),
                [](const fs::path& a, const fs::path& b) {
                    return a.filename() < b.filename();
                });
            std::vector<std::string> paths;
            for (const auto& p : shards) paths.push_back(p.string());
            return from_safetensors(paths);
        }

        throw ParseError(
            "WeightMap::open: no .safetensors files found in: " + dir
        );
    }

    throw ParseError("WeightMap::open: not a file or directory: " + dir);
}

WeightMap WeightMap::from_safetensors(const std::string& path) {
    fs::path p(path);
    if (fs::is_directory(p)) {
        return WeightMap(std::make_unique<Impl>(SafeTensors::open_dir(path)));
    }
    return WeightMap(std::make_unique<Impl>(SafeTensors::open(path)));
}

WeightMap WeightMap::from_safetensors(const std::vector<std::string>& paths) {
    return WeightMap(std::make_unique<Impl>(SafeTensors::open(paths)));
}

// ── tensor access ────────────────────────────────────────────

bool WeightMap::contains(const std::string& name) const noexcept {
    return impl_->st.contains(name);
}

std::size_t WeightMap::size() const noexcept {
    return impl_->st.size();
}

std::size_t WeightMap::total_bytes() const noexcept {
    return impl_->st.total_bytes();
}

std::vector<std::string> WeightMap::names() const {
    return impl_->st.names();
}

TensorView WeightMap::tensor(const std::string& name) const {
    return impl_->st.tensor(name);
}

std::unordered_map<std::string, TensorView> WeightMap::tensors() const {
    return impl_->st.tensors();
}

} // namespace tensor::parser