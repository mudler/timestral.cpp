#pragma once
// Backend initialisation and read-only weight mapping (docs/PLAN.md, M0b).
#include "ggml-backend.h"
#include <cstddef>
#include <string>

namespace ts {

// A file mapped read-only. The mapping stays valid until destruction.
class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);   // throws ts::Error
    MappedFile(MappedFile&& o) noexcept;
    MappedFile& operator=(MappedFile&& o) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();
    const unsigned char* data() const { return static_cast<const unsigned char*>(base_); }
    size_t size() const { return size_; }
private:
    void release();
    void* base_ = nullptr;
    size_t size_ = 0;
    bool heap_ = false;   // fallback: file read into aligned memory
};

// CPU backend (the only backend in M0b).
ggml_backend_t backend_init_cpu();

// Wrap the mapped memory as a backend buffer without copying, so tensors can point into it.
// The returned buffer does not own the memory.
ggml_backend_buffer_t backend_buffer_from_mapping(MappedFile& file);

} // namespace ts
