#include <tensor/parser/safetensors.hpp>

#include "detail/mmap.hpp"

// nlohmann/json — single-header, JSON only, no other deps.
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace tensor::parser {

// ─────────────────────────────────────────────────────────────
//  Dtype string -> DType  (safetensors canonical strings)
// ─────────────────────────────────────────────────────────────

static DType parse_dtype(const std::string& s, const std::string& tensor_name) {
    if (s == "F64")     return DType::F64;
    if (s == "F32")     return DType::F32;
    if (s == "BF16")    return DType::BF16;
    if (s == "F16")     return DType::F16;
    if (s == "F8_E5M2") return DType::F8_E5M2;
    if (s == "F8_E4M3") return DType::F8_E4M3;
    if (s == "I64")     return DType::I64;
    if (s == "I32")     return DType::I32;
    if (s == "I16")     return DType::I16;
    if (s == "I8")      return DType::I8;
    if (s == "U8")      return DType::U8;
    if (s == "BOOL")    return DType::BOOL;

    throw FormatError(
        "unknown dtype \"" + s + "\" for tensor \"" + tensor_name + "\""
    );
}

// ─────────────────────────────────────────────────────────────
//  Per-shard parsed state
// ─────────────────────────────────────────────────────────────

struct ShardData {
    detail::MmapFile mmap;
    std::size_t      data_start = 0; // byte offset inside mmap where data buffer begins
};

// ─────────────────────────────────────────────────────────────
//  Internal entry: TensorInfo + pre-resolved data pointer
// ─────────────────────────────────────────────────────────────

struct TensorEntry {
    TensorInfo  info;
    const void* data_ptr = nullptr; // points into shard mmap, never owned
};

// ─────────────────────────────────────────────────────────────
//  SafeTensors::Impl
// ─────────────────────────────────────────────────────────────

struct SafeTensors::Impl {
    std::vector<ShardData>                       shards;
    std::unordered_map<std::string, std::string> meta;
    std::unordered_map<std::string, TensorEntry> entries;
};

// ─────────────────────────────────────────────────────────────
//  SafeTensors::parse_shard  (private static)
// ─────────────────────────────────────────────────────────────

void SafeTensors::parse_shard(const std::string& path, Impl& impl) {

    // ── 1. mmap the file ─────────────────────────────────────

    detail::MmapFile mmap(path); // throws ParseError on failure

    const auto*       base      = static_cast<const std::uint8_t*>(mmap.data());
    const std::size_t file_size = mmap.size();

    // ── 2. read 8-byte little-endian header length ───────────

    if (file_size < 8) {
        throw ParseError("file too small to be safetensors: " + path);
    }

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, base, sizeof(header_len));

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    header_len = __builtin_bswap64(header_len);
#endif

    if (header_len == 0) {
        throw ParseError("safetensors header length is zero: " + path);
    }

    const std::size_t data_start = 8 + static_cast<std::size_t>(header_len);

    if (data_start > file_size) {
        throw ParseError(
            "safetensors header length (" + std::to_string(header_len) +
            ") exceeds file size ("        + std::to_string(file_size) +
            "): "                          + path
        );
    }

    // ── 3. parse the JSON header ─────────────────────────────

    const char* json_start = reinterpret_cast<const char*>(base + 8);

    json root;
    try {
        root = json::parse(json_start, json_start + header_len);
    } catch (const json::exception& e) {
        throw ParseError(
            std::string("JSON parse error in safetensors header: ") +
            e.what() + " in file: " + path
        );
    }

    if (!root.is_object()) {
        throw ParseError("safetensors header is not a JSON object: " + path);
    }

    // ── 4. __metadata__ (optional, merged across shards) ─────

    if (root.contains("__metadata__")) {
        const auto& meta_node = root["__metadata__"];
        if (!meta_node.is_object()) {
            throw ParseError("__metadata__ is not a JSON object in: " + path);
        }
        for (auto& [k, v] : meta_node.items()) {
            if (!v.is_string()) {
                throw ParseError(
                    "__metadata__ value for key \"" + k +
                    "\" is not a string in: " + path
                );
            }
            impl.meta[k] = v.get<std::string>();
        }
    }

    // ── 5. tensor descriptors ────────────────────────────────

    for (auto& [name, desc] : root.items()) {
        if (name == "__metadata__") continue;

        if (!desc.is_object()) {
            throw ParseError(
                "tensor descriptor for \"" + name +
                "\" is not a JSON object in: " + path
            );
        }

        // dtype
        if (!desc.contains("dtype") || !desc["dtype"].is_string()) {
            throw ParseError(
                "missing or non-string dtype for tensor \"" + name +
                "\" in: " + path
            );
        }
        const DType dtype = parse_dtype(desc["dtype"].get<std::string>(), name);

        // shape
        if (!desc.contains("shape") || !desc["shape"].is_array()) {
            throw ParseError(
                "missing or non-array shape for tensor \"" + name +
                "\" in: " + path
            );
        }
        Shape shape;
        for (const auto& dim : desc["shape"]) {
            if (!dim.is_number_unsigned()) {
                throw ParseError(
                    "non-integer shape dimension for tensor \"" + name +
                    "\" in: " + path
                );
            }
            shape.push_back(dim.get<std::size_t>());
        }

        // data_offsets
        if (!desc.contains("data_offsets") ||
            !desc["data_offsets"].is_array() ||
            desc["data_offsets"].size() != 2)
        {
            throw ParseError(
                "missing or invalid data_offsets for tensor \"" + name +
                "\" in: " + path
            );
        }
        const std::uint64_t off_begin =
            desc["data_offsets"][0].get<std::uint64_t>();
        const std::uint64_t off_end =
            desc["data_offsets"][1].get<std::uint64_t>();

        if (off_end < off_begin) {
            throw ParseError(
                "data_offsets end < begin for tensor \"" + name +
                "\" in: " + path
            );
        }

        // Validate byte range fits inside the mapped file.
        const std::size_t abs_begin =
            data_start + static_cast<std::size_t>(off_begin);
        const std::size_t abs_end =
            data_start + static_cast<std::size_t>(off_end);

        if (abs_end > file_size) {
            throw ParseError(
                "tensor \"" + name + "\" data range [" +
                std::to_string(abs_begin) + ", " + std::to_string(abs_end) +
                ") exceeds file size " + std::to_string(file_size) +
                " in: " + path
            );
        }

        // Duplicate tensor names across shards are a malformed model.
        if (impl.entries.count(name)) {
            throw ParseError(
                "duplicate tensor name \"" + name + "\" found in: " + path
            );
        }

        TensorEntry entry;
        entry.info.dtype        = dtype;
        entry.info.shape        = std::move(shape);
        entry.info.offset_begin = off_begin;
        entry.info.offset_end   = off_end;
        // data_ptr resolves directly — base pointer is stable across the
        // MmapFile move below because the OS mapping address does not change,
        // only the MmapFile handle/size fields move.
        entry.data_ptr = base + abs_begin;

        impl.entries[name] = std::move(entry);
    }

    // ── 6. commit the shard ──────────────────────────────────

    ShardData sd;
    sd.data_start = data_start;
    sd.mmap       = std::move(mmap);
    impl.shards.push_back(std::move(sd));
}

// ─────────────────────────────────────────────────────────────
//  Private constructor
// ─────────────────────────────────────────────────────────────

SafeTensors::SafeTensors(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

// ─────────────────────────────────────────────────────────────
//  Destructor — defined here where Impl is complete
// ─────────────────────────────────────────────────────────────

SafeTensors::~SafeTensors() = default;
SafeTensors::SafeTensors(SafeTensors&&) noexcept = default;
SafeTensors& SafeTensors::operator=(SafeTensors&&) noexcept = default;

// ─────────────────────────────────────────────────────────────
//  Factory: single file
// ─────────────────────────────────────────────────────────────

SafeTensors SafeTensors::open(const std::string& path) {
    auto impl = std::make_unique<Impl>();
    parse_shard(path, *impl);
    return SafeTensors(std::move(impl));
}

// ─────────────────────────────────────────────────────────────
//  Factory: explicit shard list
// ─────────────────────────────────────────────────────────────

SafeTensors SafeTensors::open(const std::vector<std::string>& paths) {
    if (paths.empty()) {
        throw ParseError("SafeTensors::open(paths): path list is empty");
    }
    auto impl = std::make_unique<Impl>();
    for (const auto& p : paths) {
        parse_shard(p, *impl);
    }
    return SafeTensors(std::move(impl));
}

// ─────────────────────────────────────────────────────────────
//  Factory: directory — all .safetensors files, sorted
// ─────────────────────────────────────────────────────────────

SafeTensors SafeTensors::open_dir(const std::string& dir) {
    fs::path dir_path(dir);

    if (!fs::is_directory(dir_path)) {
        throw ParseError("not a directory: " + dir);
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".safetensors")
        {
            files.push_back(entry.path());
        }
    }

    if (files.empty()) {
        throw ParseError("no .safetensors files found in: " + dir);
    }

    std::sort(files.begin(), files.end(),
        [](const fs::path& a, const fs::path& b) {
            return a.filename() < b.filename();
        }
    );

    auto impl = std::make_unique<Impl>();
    for (const auto& p : files) {
        parse_shard(p.string(), *impl);
    }
    return SafeTensors(std::move(impl));
}

// ─────────────────────────────────────────────────────────────
//  Metadata
// ─────────────────────────────────────────────────────────────

const std::unordered_map<std::string, std::string>&
SafeTensors::metadata() const noexcept {
    return impl_->meta;
}

const std::string&
SafeTensors::metadata(const std::string& key) const {
    auto it = impl_->meta.find(key);
    if (it == impl_->meta.end()) throw MetaNotFound(key);
    return it->second;
}

std::string
SafeTensors::metadata(const std::string& key,
                      const std::string& default_val) const noexcept {
    auto it = impl_->meta.find(key);
    if (it == impl_->meta.end()) return default_val;
    return it->second;
}

bool SafeTensors::has_metadata(const std::string& key) const noexcept {
    return impl_->meta.count(key) != 0;
}

// ─────────────────────────────────────────────────────────────
//  Tensor inventory
// ─────────────────────────────────────────────────────────────

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    out.reserve(impl_->entries.size());
    for (const auto& [name, _] : impl_->entries) {
        out.push_back(name);
    }
    return out;
}

std::size_t SafeTensors::size() const noexcept {
    return impl_->entries.size();
}

bool SafeTensors::contains(const std::string& name) const noexcept {
    return impl_->entries.count(name) != 0;
}

std::size_t SafeTensors::total_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& [_, entry] : impl_->entries) {
        total += entry.info.nbytes();
    }
    return total;
}

// ─────────────────────────────────────────────────────────────
//  Tensor access
// ─────────────────────────────────────────────────────────────

const TensorInfo& SafeTensors::info(const std::string& name) const {
    auto it = impl_->entries.find(name);
    if (it == impl_->entries.end()) throw TensorNotFound(name);
    return it->second.info;
}

TensorView SafeTensors::tensor(const std::string& name) const {
    auto it = impl_->entries.find(name);
    if (it == impl_->entries.end()) throw TensorNotFound(name);

    const TensorEntry& entry = it->second;
    TensorView view;
    view.data  = entry.data_ptr;
    view.dtype = entry.info.dtype;
    view.shape = entry.info.shape;
    return view;
}

std::unordered_map<std::string, TensorView>
SafeTensors::tensors() const {
    std::unordered_map<std::string, TensorView> out;
    out.reserve(impl_->entries.size());
    for (const auto& [name, entry] : impl_->entries) {
        TensorView view;
        view.data  = entry.data_ptr;
        view.dtype = entry.info.dtype;
        view.shape = entry.info.shape;
        out[name]  = view;
    }
    return out;
}

} // namespace tensor::parser