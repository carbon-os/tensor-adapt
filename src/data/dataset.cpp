#include <tensor/data/dataset.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace tensor::data {

Dataset Dataset::load(const std::string& dir) {
    fs::path d(dir);
    if (!fs::exists(d)) throw DataError("path not found: " + dir);

    // Collect all .bin files.
    std::vector<fs::path> files;
    if (fs::is_regular_file(d) && d.extension() == ".bin") {
        files.push_back(d);
    } else if (fs::is_directory(d)) {
        for (const auto& e : fs::directory_iterator(d)) {
            if (e.is_regular_file() && e.path().extension() == ".bin") {
                files.push_back(e.path());
            }
        }
        std::sort(files.begin(), files.end());
    } else {
        throw DataError("expected a .bin file or directory of .bin files: " + dir);
    }

    if (files.empty()) throw DataError("no .bin files in: " + dir);

    Dataset ds;
    for (const auto& f : files) {
        std::ifstream ifs(f, std::ios::binary | std::ios::ate);
        if (!ifs) throw DataError("cannot open: " + f.string());
        std::size_t file_size = ifs.tellg();
        std::size_t n_tokens  = file_size / sizeof(uint32_t);
        ifs.seekg(0);
        std::size_t old_size = ds.tokens_.size();
        ds.tokens_.resize(old_size + n_tokens);
        ifs.read(reinterpret_cast<char*>(ds.tokens_.data() + old_size),
                 n_tokens * sizeof(uint32_t));
        std::cerr << "[data] loaded " << f.filename() << " (" << n_tokens << " tokens)\n";
    }

    ds.total_tokens_ = ds.tokens_.size();
    std::cerr << "[data] total tokens: " << ds.total_tokens_ << "\n";
    return ds;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>>
Dataset::next_batch(int B, int T) {
    std::size_t window = T + 1;
    std::vector<int32_t> inputs(B * T), targets(B * T);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            std::size_t idx = (pos_ + t) % total_tokens_;
            inputs [b * T + t] = static_cast<int32_t>(tokens_[idx]);
            targets[b * T + t] = static_cast<int32_t>(tokens_[(idx + 1) % total_tokens_]);
        }
        pos_ = (pos_ + T) % total_tokens_;
    }

    tokens_consumed_ += B * T;
    return {inputs, targets};
}

} // namespace tensor::data