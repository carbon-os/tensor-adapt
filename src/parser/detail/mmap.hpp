#pragma once

// Internal header — not part of the installed API.
// Provides a RAII memory-mapped file that works on
// POSIX (Linux, macOS) and Windows.

#include <tensor/parser/errors.hpp>

#include <cstddef>
#include <string>
#include <utility>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <windows.h>
#else
#   include <errno.h>
#   include <fcntl.h>
#   include <string.h>
#   include <sys/mman.h>
#   include <sys/stat.h>
#   include <unistd.h>
#endif

namespace tensor::parser::detail {

// ─────────────────────────────────────────────────────────────
//  MmapFile — read-only memory-mapped file, move-only.
//
//  After construction:
//    data()  — const void* to the beginning of the mapping
//    size()  — file size in bytes
//
//  The mapping is released in the destructor (or on move-from).
// ─────────────────────────────────────────────────────────────

class MmapFile {
public:
    MmapFile() noexcept = default;

    explicit MmapFile(const std::string& path) {
        open(path);
    }

    ~MmapFile() noexcept {
        close();
    }

    // Move — transfer ownership, leave source in empty state.
    MmapFile(MmapFile&& other) noexcept { steal(std::move(other)); }
    MmapFile& operator=(MmapFile&& other) noexcept {
        if (this != &other) {
            close();
            steal(std::move(other));
        }
        return *this;
    }

    MmapFile(const MmapFile&)            = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool        empty() const noexcept { return size_ == 0; }

private:

#ifdef _WIN32
    // ── Windows ──────────────────────────────────────────────
    HANDLE file_handle_    = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
    const void* data_      = nullptr;
    std::size_t size_      = 0;

    void open(const std::string& path) {
        file_handle_ = ::CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw ParseError("cannot open file: " + path);
        }

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(file_handle_, &file_size)) {
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw ParseError("cannot get file size: " + path);
        }
        size_ = static_cast<std::size_t>(file_size.QuadPart);

        if (size_ == 0) {
            // Empty file — valid but nothing to map.
            return;
        }

        mapping_handle_ = ::CreateFileMappingA(
            file_handle_,
            nullptr,
            PAGE_READONLY,
            0, 0,
            nullptr
        );
        if (!mapping_handle_) {
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw ParseError("cannot create file mapping: " + path);
        }

        data_ = ::MapViewOfFile(
            mapping_handle_,
            FILE_MAP_READ,
            0, 0, 0
        );
        if (!data_) {
            ::CloseHandle(mapping_handle_);
            ::CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_    = INVALID_HANDLE_VALUE;
            throw ParseError("cannot map view of file: " + path);
        }
    }

    void close() noexcept {
        if (data_) {
            ::UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_handle_) {
            ::CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
        size_ = 0;
    }

    void steal(MmapFile&& other) noexcept {
        file_handle_    = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        data_           = other.data_;
        size_           = other.size_;

        other.file_handle_    = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = nullptr;
        other.data_           = nullptr;
        other.size_           = 0;
    }

#else
    // ── POSIX ────────────────────────────────────────────────
    const void* data_ = nullptr;
    std::size_t size_ = 0;

    void open(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw ParseError(
                std::string("cannot open file: ") + path +
                " (" + ::strerror(errno) + ")"
            );
        }

        struct stat st{};
        if (::fstat(fd, &st) < 0) {
            ::close(fd);
            throw ParseError(
                std::string("cannot stat file: ") + path +
                " (" + ::strerror(errno) + ")"
            );
        }
        size_ = static_cast<std::size_t>(st.st_size);

        if (size_ == 0) {
            ::close(fd);
            data_ = nullptr;
            return;
        }

        void* addr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd); // fd no longer needed after mmap

        if (addr == MAP_FAILED) {
            size_ = 0;
            throw ParseError(
                std::string("mmap failed for: ") + path +
                " (" + ::strerror(errno) + ")"
            );
        }

        // Hint the kernel: sequential access pattern, read-only.
        ::madvise(addr, size_, MADV_SEQUENTIAL);

        data_ = addr;
    }

    void close() noexcept {
        if (data_) {
            ::munmap(const_cast<void*>(data_), size_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    void steal(MmapFile&& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }

#endif // _WIN32
};

} // namespace tensor::parser::detail