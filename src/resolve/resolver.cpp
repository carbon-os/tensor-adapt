/*
 * resolver.cpp — general-purpose HuggingFace resolver for tensor-adapt
 *
 * Handles both model repos (safetensors / GGUF / config.json) and dataset
 * repos (parquet / arrow / jsonl) through a single fetch() entry point.
 *
 * HTTP / SSL
 * ----------
 * libcurl is used throughout — it handles OpenSSL + system CA bundle
 * automatically and is the only HTTP dependency in the project.
 *
 * HF REDIRECT BEHAVIOUR
 * ---------------------
 * Large files are served via XetHub CAS (cas-bridge.xethub.hf.co).
 * The resolve/main/{file} endpoint returns a 302 to a pre-signed CDN URL.
 * Pre-signed URLs reject Authorization headers, so we use a two-step approach:
 *
 *   Step 1 — fetch_metadata()  GET Range:bytes=0-0, redirects disabled.
 *             Extracts x-repo-commit, etag, content size, Location (CDN URL).
 *
 *   Step 2 — download_file()   GET to CDN URL directly, no Authorization.
 *
 * REPO TYPE DETECTION
 * -------------------
 * With RepoType::Auto the resolver tries /api/models/{id} first.
 * On 404 it falls back to /api/datasets/{id}.
 * Both endpoints return the same siblings[] file list structure.
 *
 * CACHING
 * -------
 * Cache root: ~/.cache/tensor/   (shared with tensor-pretrain)
 * Layout:
 *   models/   {org}/{repo}/
 *   datasets/ {org}/{repo}/
 *   datasets/ {org}/{repo}/{subset}/   (when subset is given)
 *
 * A directory is considered warm (skipped) if it already contains at least
 * one file with a recognised extension for its repo type.
 */

#include <tensor/resolve/resolver.hpp>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor::resolve {

std::optional<std::filesystem::path> Resolver::custom_cache_dir_ = std::nullopt;

// ─────────────────────────────────────────────────────────────────────────────
//  libcurl RAII wrappers
// ─────────────────────────────────────────────────────────────────────────────

struct CurlHandle {
    CURL* h;
    explicit CurlHandle() : h(curl_easy_init()) {
        if (!h) throw ResolveError("curl_easy_init() failed");
    }
    ~CurlHandle() { curl_easy_cleanup(h); }
    CurlHandle(const CurlHandle&)            = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
    operator CURL*() const { return h; }
};

struct CurlHeaders {
    curl_slist* list = nullptr;
    void append(const std::string& h) { list = curl_slist_append(list, h.c_str()); }
    ~CurlHeaders() { if (list) curl_slist_free_all(list); }
};

struct CurlGlobal {
    CurlGlobal()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobal() { curl_global_cleanup(); }
} g_curl_global;

// ─────────────────────────────────────────────────────────────────────────────
//  curl write callbacks
// ─────────────────────────────────────────────────────────────────────────────

static size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* ud) {
    static_cast<std::string*>(ud)->append(ptr, size * nmemb);
    return size * nmemb;
}

static size_t write_to_file(char* ptr, size_t size, size_t nmemb, void* ud) {
    static_cast<std::ofstream*>(ud)->write(ptr, static_cast<std::streamsize>(size * nmemb));
    return size * nmemb;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string fmt_size(double bytes) {
    char buf[64];
    if (bytes < 1024.0 * 1024.0)
        std::snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else if (bytes < 1024.0 * 1024.0 * 1024.0)
        std::snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
    else
        std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

// Minimal glob: only * is special, matches any run of non-separator characters.
// Used for include/exclude pattern matching against bare filenames.
static bool glob_match(const std::string& pattern, const std::string& name) {
    const char* p = pattern.c_str();
    const char* s = name.c_str();

    while (*p && *s) {
        if (*p == '*') {
            ++p;
            if (!*p) return true;           // trailing * matches rest
            while (*s) {
                if (glob_match(p, s)) return true;
                ++s;
            }
            return false;
        }
        if (*p != *s) return false;
        ++p; ++s;
    }
    while (*p == '*') ++p;
    return !*p && !*s;
}

static bool matches_any(const std::vector<std::string>& patterns,
                        const std::string& filename) {
    for (const auto& pat : patterns)
        if (glob_match(pat, filename)) return true;
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  FileMetadata — extracted from the HF redirect response
// ─────────────────────────────────────────────────────────────────────────────

struct FileMetadata {
    std::string commit_hash;
    std::string etag;
    uint64_t    size         = 0;
    std::string download_url;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Cache dir
// ─────────────────────────────────────────────────────────────────────────────

std::filesystem::path Resolver::default_cache_dir() {
    const char* home = std::getenv("HOME");
    if (!home) throw ResolveError("Resolver: HOME not set");
    return std::filesystem::path(home) / ".cache" / "tensor";
}

void Resolver::set_cache_dir(std::filesystem::path path) {
    custom_cache_dir_ = std::move(path);
}

std::filesystem::path Resolver::get_cache_dir() {
    return custom_cache_dir_.value_or(default_cache_dir());
}

// ─────────────────────────────────────────────────────────────────────────────
//  URI parsing + public entry point
// ─────────────────────────────────────────────────────────────────────────────

std::filesystem::path Resolver::fetch(const std::string& uri,
                                      const std::optional<std::string>& token,
                                      const FetchOptions& options) {
    static constexpr std::string_view hf_prefix = "hf://";

    if (!uri.starts_with(hf_prefix))
        throw ResolveError("Resolver: unsupported URI scheme: " + uri);

    std::string rest = uri.substr(hf_prefix.size());

    // Parse optional inline subset: hf://org/repo:subset
    std::optional<std::string> subset = options.subset;
    if (auto colon = rest.find(':'); colon != std::string::npos) {
        if (!subset)
            subset = rest.substr(colon + 1);
        rest = rest.substr(0, colon);
    }

    return resolve_hf(rest, subset, token, options);
}

// ─────────────────────────────────────────────────────────────────────────────
//  HF Hub API — file list, with repo-type detection
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::string> Resolver::try_file_list(
        const std::string& repo_id,
        const std::string& api_kind,
        const std::optional<std::string>& token) {

    CurlHandle  curl;
    CurlHeaders headers;
    headers.append("User-Agent: tensor-adapt/0.1.0");
    if (token && !token->empty())
        headers.append("Authorization: Bearer " + *token);

    std::string url  = "https://huggingface.co/api/" + api_kind + "/" + repo_id;
    std::string body;

    curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers.list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &body);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK)
        throw ResolveError(
            "Could not reach HuggingFace Hub: " +
            std::string(curl_easy_strerror(rc)) +
            "\nCheck your internet connection.");

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    // 404 with Auto detection — caller will try next endpoint
    if (status == 404) return {};

    if (status == 401 || status == 403)
        throw ResolveError(
            "'" + repo_id + "' requires authentication.\n"
            "Set HF_TOKEN to a valid token from https://huggingface.co/settings/tokens");

    if (status != 200)
        throw ResolveError(
            "HF Hub API returned HTTP " + std::to_string(status) +
            " for '" + repo_id + "'");

    try {
        auto json  = nlohmann::json::parse(body);
        std::vector<std::string> files;
        for (const auto& s : json.at("siblings"))
            files.push_back(s.at("rfilename").get<std::string>());
        return files;
    } catch (const nlohmann::json::exception& e) {
        throw ResolveError(
            "Failed to parse Hub API response for '" + repo_id + "': " +
            std::string(e.what()));
    }
}

std::vector<std::string> Resolver::fetch_file_list(
        const std::string& repo_id,
        RepoType repo_type,
        const std::optional<std::string>& token) {

    std::vector<std::string> files;

    if (repo_type == RepoType::Model) {
        files = try_file_list(repo_id, "models", token);
        if (files.empty())
            throw ResolveError(
                "Model '" + repo_id + "' not found.\n"
                "Check the repo at https://huggingface.co/" + repo_id);

    } else if (repo_type == RepoType::Dataset) {
        files = try_file_list(repo_id, "datasets", token);
        if (files.empty())
            throw ResolveError(
                "Dataset '" + repo_id + "' not found.\n"
                "Check the repo at https://huggingface.co/datasets/" + repo_id);

    } else {
        // Auto: try model first, then dataset
        files = try_file_list(repo_id, "models", token);
        if (files.empty()) {
            files = try_file_list(repo_id, "datasets", token);
            if (files.empty())
                throw ResolveError(
                    "'" + repo_id + "' not found as a model or dataset on HuggingFace.\n"
                    "Double-check the repo name.");
        }
    }

    if (files.empty())
        throw ResolveError("'" + repo_id + "' exists but contains no files.");

    return files;
}

std::vector<std::string> Resolver::list_files(
        const std::string& repo_id,
        RepoType repo_type,
        const std::optional<std::string>& token) {
    return fetch_file_list(repo_id, repo_type, token);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Metadata pre-fetch (step 1 of two-step download)
// ─────────────────────────────────────────────────────────────────────────────

static FileMetadata fetch_metadata(const std::string& hf_url,
                                   const std::string& auth_header) {
    CurlHandle  curl;
    CurlHeaders headers;
    headers.append("User-Agent: tensor-adapt/0.1.0");
    headers.append("Range: bytes=0-0");
    if (!auth_header.empty())
        headers.append(auth_header);

    std::string body, raw_headers;

    curl_easy_setopt(curl, CURLOPT_URL,            hf_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers.list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &body);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, write_to_string);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA,     &raw_headers);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);  // stop before CDN redirect

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK)
        throw std::runtime_error("metadata request failed: " +
                                 std::string(curl_easy_strerror(rc)));

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    if (status == 401 || status == 403)
        throw std::runtime_error("HTTP_AUTH");

    if (status != 200 && status != 206 &&
        status != 301 && status != 302 &&
        status != 307 && status != 308)
        throw std::runtime_error("metadata HTTP " + std::to_string(status));

    // Case-insensitive header extraction
    auto get_header = [&](const std::string& name) -> std::string {
        std::istringstream ss(raw_headers);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.size() <= name.size() + 1) continue;
            std::string prefix = line.substr(0, name.size());
            std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);
            std::string lower_name = name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
            if (prefix == lower_name && line[name.size()] == ':') {
                auto val   = line.substr(name.size() + 1);
                auto start = val.find_first_not_of(" \t");
                auto end   = val.find_last_not_of(" \t\r\n");
                if (start == std::string::npos) return "";
                return val.substr(start, end - start + 1);
            }
        }
        return "";
    };

    FileMetadata meta;

    meta.commit_hash = get_header("x-repo-commit");
    if (meta.commit_hash.empty())
        throw std::runtime_error("metadata missing x-repo-commit");

    meta.etag = get_header("x-linked-etag");
    if (meta.etag.empty()) meta.etag = get_header("etag");
    if (meta.etag.size() >= 2 && meta.etag.front() == '"')
        meta.etag = meta.etag.substr(1, meta.etag.size() - 2);

    std::string cr = get_header("content-range");
    if (cr.empty()) {
        auto cl = get_header("content-length");
        meta.size = cl.empty() ? 0 : std::stoull(cl);
    } else {
        auto slash = cr.rfind('/');
        meta.size = (slash != std::string::npos) ? std::stoull(cr.substr(slash + 1)) : 0;
    }

    meta.download_url = get_header("location");
    if (!meta.download_url.empty() && meta.download_url[0] == '/')
        meta.download_url = "https://huggingface.co" + meta.download_url;
    if (meta.download_url.empty())
        meta.download_url = hf_url;

    return meta;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Progress bar
// ─────────────────────────────────────────────────────────────────────────────

struct ProgressState { uint64_t expected_size; bool show; };

static int progress_callback(void* clientp,
                              curl_off_t dltotal, curl_off_t dlnow,
                              curl_off_t, curl_off_t) {
    auto* st = static_cast<ProgressState*>(clientp);
    if (!st->show) return 0;

    uint64_t total = dltotal > 0 ? static_cast<uint64_t>(dltotal) : st->expected_size;
    double   d     = static_cast<double>(dlnow);

    if (total == 0) {
        std::cout << "\r\033[K[=>] " << fmt_size(d) << " downloaded";
        std::cout.flush();
        return 0;
    }

    constexpr int W  = 40;
    float ratio = static_cast<float>(dlnow) / static_cast<float>(total);
    int   pos   = static_cast<int>(W * ratio);

    std::cout << "\r\033[K[";
    for (int i = 0; i < W; ++i) {
        if      (i <  pos) std::cout << '=';
        else if (i == pos) std::cout << '>';
        else               std::cout << ' ';
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf), "] %3d%%  %s / %s",
                  static_cast<int>(ratio * 100.f),
                  fmt_size(d).c_str(),
                  fmt_size(static_cast<double>(total)).c_str());
    std::cout << buf;
    std::cout.flush();
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  File download (step 2 — direct to CDN, no auth header)
// ─────────────────────────────────────────────────────────────────────────────

static void download_file(const std::string& url,
                          const std::filesystem::path& out_path,
                          uint64_t expected_size,
                          bool show_progress) {
    CurlHandle  curl;
    CurlHeaders headers;
    headers.append("User-Agent: tensor-adapt/0.1.0");
    // No Authorization — CDN pre-signed URLs reject it.

    std::ofstream out(out_path, std::ios::binary);
    if (!out.is_open())
        throw ResolveError("Cannot open for writing: " + out_path.string());

    ProgressState pst{ expected_size, show_progress };

    curl_easy_setopt(curl, CURLOPT_URL,              url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,       headers.list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,    write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,        &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,   1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,   15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,          0L);    // no cap — large files
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS,       0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA,     &pst);

    CURLcode rc = curl_easy_perform(curl);
    out.close();

    if (rc != CURLE_OK)
        throw ResolveError("Download failed: " + std::string(curl_easy_strerror(rc)));

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    if (status != 200 && status != 206)
        throw ResolveError("Download failed: HTTP " + std::to_string(status));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Warm-cache check
//
//  Returns true if the directory already contains at least one file we'd
//  consider a valid cached artifact — meaning we can skip re-downloading.
//  Deliberately permissive: any recognised extension is enough.
// ─────────────────────────────────────────────────────────────────────────────

static bool cache_is_warm(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir)) return false;

    static const std::vector<std::string> warm_exts = {
        // model weights
        ".safetensors", ".gguf",
        // model metadata
        ".json",
        // dataset formats
        ".parquet", ".arrow", ".jsonl",
    };

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        for (const auto& we : warm_exts)
            if (ext == we) return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  resolve_hf — main download orchestration
// ─────────────────────────────────────────────────────────────────────────────

std::filesystem::path Resolver::resolve_hf(
        const std::string& repo_id,
        const std::optional<std::string>& subset,
        const std::optional<std::string>& token,
        const FetchOptions& options) {

    // ── Cache directory layout ────────────────────────────────────────────
    //   models/   {org}/{repo}/
    //   datasets/ {org}/{repo}/            (no subset)
    //   datasets/ {org}/{repo}/{subset}/   (with subset)
    //
    // We don't know the repo type yet with Auto, so we resolve the dir
    // after detection.  For now build a tentative path; if the actual
    // detected type differs from what's cached that's fine — the warm check
    // prevents re-downloading.

    auto build_dir = [&](const std::string& kind) -> std::filesystem::path {
        auto d = get_cache_dir() / kind / repo_id;
        if (subset) d /= *subset;
        return d;
    };

    // ── Detect repo type and resolve file list ────────────────────────────
    RepoType detected = options.repo_type;

    // For Auto we probe both; record which kind we ended up using.
    std::string detected_kind;
    if (detected == RepoType::Model)         detected_kind = "models";
    else if (detected == RepoType::Dataset)  detected_kind = "datasets";

    std::vector<std::string> all_files;

    if (detected == RepoType::Auto) {
        auto mfiles = try_file_list(repo_id, "models", token);
        if (!mfiles.empty()) {
            all_files     = std::move(mfiles);
            detected_kind = "models";
            detected      = RepoType::Model;
        } else {
            auto dfiles = try_file_list(repo_id, "datasets", token);
            if (!dfiles.empty()) {
                all_files     = std::move(dfiles);
                detected_kind = "datasets";
                detected      = RepoType::Dataset;
            } else {
                throw ResolveError(
                    "'" + repo_id + "' not found as a model or dataset on HuggingFace.");
            }
        }
    } else {
        all_files = fetch_file_list(repo_id, detected, token);
    }

    std::filesystem::path cache_dir = build_dir(detected_kind);

    // ── Warm cache — nothing to download ─────────────────────────────────
    if (cache_is_warm(cache_dir)) {
        return cache_dir;
    }

    std::filesystem::create_directories(cache_dir);
    std::cout << "hf://" << repo_id;
    if (subset) std::cout << ":" << *subset;
    std::cout << "  [" << detected_kind << "]\n"
              << "  -> " << cache_dir << "\n";

    // ── Subset filter (datasets only) ─────────────────────────────────────
    if (subset && detected == RepoType::Dataset) {
        std::vector<std::string> filtered;
        for (const auto& f : all_files) {
            if (f.starts_with(*subset + "/") ||
                f.find("/" + *subset + "/") != std::string::npos)
                filtered.push_back(f);
        }
        if (filtered.empty())
            throw ResolveError(
                "Subset '" + *subset + "' not found in '" + repo_id + "'.\n"
                "Available files: use hf-fetch list hf://" + repo_id);
        all_files = std::move(filtered);
    }

    // ── Include / exclude pattern filter ─────────────────────────────────
    std::vector<std::string> files;
    for (const auto& f : all_files) {
        std::string fname = std::filesystem::path(f).filename().string();

        if (!options.include_patterns.empty() &&
            !matches_any(options.include_patterns, fname))
            continue;

        if (!options.exclude_patterns.empty() &&
            matches_any(options.exclude_patterns, fname))
            continue;

        files.push_back(f);
    }

    if (files.empty())
        throw ResolveError(
            "No files matched the given filters in '" + repo_id + "'.\n"
            "Relax --include / --exclude patterns or omit them to download everything.");

    // ── Download loop ─────────────────────────────────────────────────────
    std::string auth_header;
    if (token && !token->empty())
        auth_header = "Authorization: Bearer " + *token;

    std::size_t idx   = 0;
    std::size_t total = files.size();

    for (const auto& file : files) {
        ++idx;
        std::filesystem::path out_file =
            cache_dir / std::filesystem::path(file).filename();

        // Per-file prefix  "[2/14] config.json"
        std::cout << "[" << idx << "/" << total << "] "
                  << std::filesystem::path(file).filename().string();

        if (std::filesystem::exists(out_file) &&
            std::filesystem::file_size(out_file) > 0) {
            std::cout << "  (cached)\n";
            continue;
        }

        if (options.show_progress) std::cout << "\n";
        else                       std::cout << "  " << std::flush;

        // ── Fix: seed concatenation with std::string so the ternary
        //   (const char* vs const char*) doesn't produce a raw pointer add.
        std::string hf_url =
            std::string("https://huggingface.co/") +
            (detected == RepoType::Dataset ? "datasets/" : "") +
            repo_id + "/resolve/main/" + file;

        try {
            auto meta = fetch_metadata(hf_url, auth_header);
            download_file(meta.download_url, out_file, meta.size, options.show_progress);

        } catch (const std::exception& e) {
            std::filesystem::remove(out_file);
            std::string err = e.what();

            if (err == "HTTP_AUTH") {
                std::filesystem::remove_all(cache_dir);
                throw ResolveError(
                    "Access denied to '" + repo_id + "'.\n"
                    "  1. Accept the license at https://huggingface.co/" +
                    (detected == RepoType::Dataset ? "datasets/" : "") + repo_id + "\n"
                    "  2. Generate a token at https://huggingface.co/settings/tokens\n"
                    "  3. Re-run with: HF_TOKEN=<your_token> hf-fetch fetch hf://" + repo_id);
            }

            if (err.find("HTTP 404") != std::string::npos ||
                err.find("metadata HTTP 404") != std::string::npos) {
                std::cout << "  skipped (404)\n";
                continue;
            }

            throw ResolveError("Failed to download " + file + ": " + err);
        }

        if (options.show_progress) std::cout << "\n";
        else                       std::cout << "  done\n";
    }

    return cache_dir;
}

} // namespace tensor::resolve