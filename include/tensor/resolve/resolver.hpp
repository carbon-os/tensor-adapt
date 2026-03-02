#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace tensor::resolve {

// ─────────────────────────────────────────────────────────────────────────────
//  Error
// ─────────────────────────────────────────────────────────────────────────────

class ResolveError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ─────────────────────────────────────────────────────────────────────────────
//  RepoType
//
//  Model   — safetensors / GGUF weights + config.json / tokenizer.json
//  Dataset — parquet / arrow / jsonl training data
//  Auto    — try model endpoint first, fall back to dataset endpoint
// ─────────────────────────────────────────────────────────────────────────────

enum class RepoType {
    Model,
    Dataset,
    Auto,
};

// ─────────────────────────────────────────────────────────────────────────────
//  FetchOptions
// ─────────────────────────────────────────────────────────────────────────────

struct FetchOptions {
    // Repo type — drives which HF API endpoint is tried first.
    // Auto tries model, falls back to dataset.
    RepoType repo_type = RepoType::Auto;

    // Subset filter — applied to dataset repos.
    // Matches files whose path starts with "<subset>/" or contains "/<subset>/".
    // Ignored for model repos.
    std::optional<std::string> subset;

    // If non-empty, only files whose names match at least one pattern are
    // downloaded. Patterns are simple globs: * matches any sequence of
    // non-separator characters. Example: {"*.safetensors", "*.json"}
    // Empty means download everything (after subset filtering).
    std::vector<std::string> include_patterns;

    // Files whose names match any exclude pattern are skipped even if they
    // matched an include pattern. Example: {"*.msgpack", "flax_model*"}
    std::vector<std::string> exclude_patterns;

    // Show a per-file progress bar during download.
    bool show_progress = true;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Resolver
//
//  Resolves hf:// URIs to local paths, downloading if necessary.
//  Cache root: ~/.cache/tensor/  (shared with tensor-pretrain)
//
//  URI formats accepted:
//    hf://org/repo                   — auto-detect repo type
//    hf://org/repo:subset            — dataset with subset
//
//  The caller controls the cache root via set_cache_dir().
// ─────────────────────────────────────────────────────────────────────────────

class Resolver {
public:
    // Resolve a URI to a local directory.
    // Downloads any missing files into the cache and returns the cache dir.
    // Throws ResolveError on any unrecoverable failure.
    static std::filesystem::path fetch(
        const std::string&              uri,
        const std::optional<std::string>& token   = std::nullopt,
        const FetchOptions&               options = {});

    // Override the default cache root (~/.cache/tensor).
    static void set_cache_dir(std::filesystem::path path);

    // Return the active cache root.
    static std::filesystem::path get_cache_dir();

    // Fetch the raw file list for a repo from the HF Hub API.
    // Useful for callers that want to inspect what's available before fetching.
    static std::vector<std::string> list_files(
        const std::string&                repo_id,
        RepoType                          repo_type,
        const std::optional<std::string>& token = std::nullopt);

private:
    static std::filesystem::path default_cache_dir();

    static std::filesystem::path resolve_hf(
        const std::string&                repo_id,
        const std::optional<std::string>& subset,
        const std::optional<std::string>& token,
        const FetchOptions&               options);

    static std::vector<std::string> fetch_file_list(
        const std::string&                repo_id,
        RepoType                          repo_type,
        const std::optional<std::string>& token);

    // Internal: try a single HF API endpoint (/api/models or /api/datasets).
    // Returns the file list on success, empty vector on 404 (caller tries next).
    // Throws ResolveError on auth failure or unexpected HTTP errors.
    static std::vector<std::string> try_file_list(
        const std::string&                repo_id,
        const std::string&                api_kind,   // "models" or "datasets"
        const std::optional<std::string>& token);

    static std::optional<std::filesystem::path> custom_cache_dir_;
};

} // namespace tensor::resolve