#pragma once
#include "layers.hpp"
namespace ts {
struct RopeTables { std::vector<float> cos,sin; };
class Rope {
public:
    explicit Rope(const Model& model) : model_(model) {}
    std::vector<float> inv_freq() const;
    RopeTables gather(const std::vector<int64_t>& positions) const;
    // q flattened [rows, sequence, heads, head_dim], tables from dumped inputs.
    std::vector<float> rotate(const std::vector<float>& q, const RopeTables& tables, size_t rows, size_t sequence, size_t heads);
private:
    const Model& model_;
    GraphRunner runner_;
};
}
