/*
 * hf-fetch — general-purpose HuggingFace fetch tool for tensor-adapt
 *
 * Usage:
 *   hf-fetch fetch  <uri> [options]    download a model or dataset
 *   hf-fetch list   <uri> [options]    list files without downloading
 *   hf-fetch cache                     print the cache root path
 *
 * URI formats:
 *   hf://org/repo                  auto-detect type
 *   hf://org/repo:subset           dataset with subset (can also use --subset)
 *
 * Options:
 *   --model                        treat as model repo
 *   --dataset                      treat as dataset repo
 *   --subset  <name>               dataset subset filter
 *   --include <glob>               include only matching files  (repeatable)
 *   --exclude <glob>               exclude matching files       (repeatable)
 *   --no-progress                  disable progress bar
 *   --cache-dir <path>             override cache root
 */

#include <tensor/resolve/resolver.hpp>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace tensor::resolve;

// ─────────────────────────────────────────────────────────────────────────────

static void usage(const char* name) {
    std::cerr 
        "usage: " << name << " <command> [options]\n\n"
        "commands:\n"
        "  fetch <uri>    Download a model or dataset\n"
        "  list  <uri>    List repo files without downloading\n"
        "  cache          Print the cache root directory\n\n"
        "uri formats:\n"
        "  hf://org/repo              auto-detect type\n"
        "  hf://org/repo:subset       dataset with inline subset\n\n"
        "options:\n"
        "  --model                    treat as model repo\n"
        "  --dataset                  treat as dataset repo\n"
        "  --subset  <name>           dataset subset filter\n"
        "  --include <glob>           include matching files (repeatable)\n"
        "  --exclude <glob>           exclude matching files (repeatable)\n"
        "  --no-progress              disable progress bar\n"
        "  --cache-dir <path>         override ~/.cache/tensor\n\n"
        "env:\n"
        "  HF_TOKEN                   token for gated repos\n\n"
        "examples:\n"
        "  hf-fetch fetch hf://meta-llama/Llama-3.1-8B --model\n"
        "  hf-fetch fetch hf://bigcode/the-stack-v2 --dataset --subset go\n"
        "  hf-fetch fetch hf://mistralai/Mistral-7B-v0.3 --include '*.safetensors' --include '*.json'\n"
        "  hf-fetch list  hf://Qwen/Qwen2.5-7B\n";
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string command = argv[1];

    if (command == "cache") {
        std::cout << Resolver::get_cache_dir().string() << "\n";
        return 0;
    }

    if (command != "fetch" && command != "list") {
        std::cerr << "unknown command: " << command << "\n";
        usage(argv[0]);
        return 1;
    }

    if (argc < 3) {
        std::cerr << "error: " << command << " requires a URI argument\n";
        usage(argv[0]);
        return 1;
    }

    std::string  uri           = argv[2];
    FetchOptions options;
    options.show_progress = true;

    std::optional<std::string> token;
    if (const char* env = std::getenv("HF_TOKEN"))
        token = env;

    // ── Parse flags ───────────────────────────────────────────────────────
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model") {
            options.repo_type = RepoType::Model;
        } else if (arg == "--dataset") {
            options.repo_type = RepoType::Dataset;
        } else if (arg == "--no-progress") {
            options.show_progress = false;
        } else if (arg == "--subset" && i + 1 < argc) {
            options.subset = argv[++i];
        } else if (arg == "--include" && i + 1 < argc) {
            options.include_patterns.emplace_back(argv[++i]);
        } else if (arg == "--exclude" && i + 1 < argc) {
            options.exclude_patterns.emplace_back(argv[++i]);
        } else if (arg == "--cache-dir" && i + 1 < argc) {
            Resolver::set_cache_dir(argv[++i]);
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    // ── list command ─────────────────────────────────────────────────────
    if (command == "list") {
        static constexpr std::string_view hf_prefix = "hf://";
        if (!uri.starts_with(hf_prefix)) {
            std::cerr << "error: unsupported URI scheme: " << uri << "\n";
            return 1;
        }

        // Strip inline subset from the repo id for listing
        std::string repo_id = std::string(uri.substr(hf_prefix.size()));
        if (auto colon = repo_id.find(':'); colon != std::string::npos)
            repo_id = repo_id.substr(0, colon);

        try {
            auto files = Resolver::list_files(repo_id, options.repo_type, token);
            for (const auto& f : files)
                std::cout << f << "\n";
            std::cout << "\n" << files.size() << " file(s)\n";
        } catch (const ResolveError& e) {
            std::cerr << "\nfatal: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ── fetch command ────────────────────────────────────────────────────
    try {
        auto path = Resolver::fetch(uri, token, options);
        std::cout << "\n[ok] ready at: " << path.string() << "\n";
    } catch (const ResolveError& e) {
        std::cerr << "\nfatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}