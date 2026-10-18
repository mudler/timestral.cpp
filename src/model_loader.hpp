#pragma once
// GGUF loader (docs/PLAN.md, B.1 to B.3). Reads the config from KV, builds the tensor schema
// from it, and refuses any file whose tensors do not match that schema exactly: a missing,
// extra, differently typed or differently shaped tensor is an error, and so is a missing or
// wrongly typed KV. No dimension is hardcoded.
#include "backend.hpp"
#include "config.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gguf.h"
#include <map>
#include <string>
#include <vector>

namespace ts {

struct TensorSpec {
    std::string name;
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};   // ggml order, fastest first
    int n_dims = 1;
    ggml_type type = GGML_TYPE_F32;
};

// Read and validate the config of an open GGUF context. Throws ts::Error.
Config read_config(const gguf_context* g);

// The tensors a file with this config must contain (B.2), in file order.
std::vector<TensorSpec> expected_schema(const Config& c);

// Tensor count of B.2 for this config.
size_t expected_tensor_count(const Config& c);

class Model {
public:
    // Load `path`. Weights are mapped read-only and used in place (no copy).
    static Model load(const std::string& path);   // throws ts::Error

    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model();

    const Config& config() const { return cfg_; }
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    ggml_tensor* tensor(const std::string& name) const;   // throws ts::Error when absent
    const std::vector<std::string>& tensor_names() const { return names_; }   // file order
    size_t mapped_bytes() const { return file_.size(); }
    // True if the tensor data points into the file mapping (zero copy).
    bool in_mapping(const ggml_tensor* t) const;

private:
    Model() = default;
    void reset();
    Config cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    MappedFile file_;
    ggml_backend_buffer_t buf_ = nullptr;
    std::map<std::string, ggml_tensor*> tensors_;
    std::vector<std::string> names_;
};

} // namespace ts
