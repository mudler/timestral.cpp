#include "backend.hpp"
#include "common.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ts {

MappedFile::MappedFile(const std::string& path) {
#if !defined(_WIN32)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw Error("cannot open " + path);
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); throw Error("cannot stat or empty file: " + path); }
    void* p = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) throw Error("mmap failed for " + path);
    base_ = p;
    size_ = (size_t)st.st_size;
    heap_ = false;
#else
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw Error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); throw Error("empty file: " + path); }
    size_t sz = ((size_t)n + 31) & ~(size_t)31;
    void* p = _aligned_malloc(sz, 64);
    if (!p || std::fread(p, 1, (size_t)n, f) != (size_t)n) { std::fclose(f); throw Error("cannot read " + path); }
    std::fclose(f);
    base_ = p; size_ = (size_t)n; heap_ = true;
#endif
}

void MappedFile::release() {
    if (!base_) return;
#if !defined(_WIN32)
    ::munmap(base_, size_);
#else
    _aligned_free(base_);
#endif
    base_ = nullptr; size_ = 0;
}

MappedFile::MappedFile(MappedFile&& o) noexcept : base_(o.base_), size_(o.size_), heap_(o.heap_) {
    o.base_ = nullptr; o.size_ = 0;
}
MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
    if (this != &o) { release(); base_ = o.base_; size_ = o.size_; heap_ = o.heap_; o.base_ = nullptr; o.size_ = 0; }
    return *this;
}
MappedFile::~MappedFile() { release(); }

ggml_backend_t backend_init_cpu() {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) throw Error("cannot initialise the CPU backend");
    return be;
}

ggml_backend_buffer_t backend_buffer_from_mapping(MappedFile& file) {
    if (!file.data()) throw Error("no mapping");
    ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(const_cast<unsigned char*>(file.data()), file.size());
    if (!b) throw Error("cannot wrap the mapping as a backend buffer");
    return b;
}

} // namespace ts
