#pragma once
#include "layers.hpp"
#include <map>
namespace ts {
struct TimeMask {
    // Empty means causal. Otherwise [rows, queries, keys], true = may attend.
    std::vector<int8_t> allowed;
    size_t enc_len = 0; // optional TwoStreamMask split; skipped quadrant must be closed
};
using LayerValues = std::map<std::string, std::vector<float>>;

struct Embedding { std::vector<float> input, hidden, output; };
// Model must outlive Encoder. Input rows are contiguous [rows, length].
class Encoder {
public:
    explicit Encoder(const Model& model) : model_(model) {}
    Embedding embed(const std::vector<float>& values, const std::vector<int8_t>& observed, size_t rows, size_t length);
    Embedding project(const std::vector<float>& features, size_t rows, size_t patches);
    // Time-only replay: x is [rows, sequence, d_model]. Returned vectors own data.
    // Teacher keys replace downstream inputs, not reported outputs. Unknown keys fail.
    // Norm/q/k/v, q_rope/k_rope, cos/sin, sdpa/gate/gated/out and MLP boundaries
    // permit independent local gates. Fused qkv diagnostics use reference order;
    // converted weights themselves remain head-major and read-only.
    LayerValues run_layer(size_t layer, const std::vector<float>& x, size_t rows,
        size_t sequence, const TimeMask& mask, const std::vector<int64_t>& positions,
        const LayerValues& teacher = {});
private:
    const Model& model_;
    GraphRunner runner_;
};
}

namespace ts {
// Convenience wrapper; Encoder::run_layer reuses its persistent graph allocator.
LayerValues replay_time_layer(const Model& model, size_t layer, const std::vector<float>& x,
    size_t rows, size_t sequence, const TimeMask& mask, const std::vector<int64_t>& positions,
    const LayerValues& teacher = {});
}
