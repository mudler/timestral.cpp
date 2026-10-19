#pragma once
#include "layers.hpp"
namespace ts {
struct Embedding { std::vector<float> input, hidden, output; };
// Model must outlive Encoder. Input rows are contiguous [rows, length].
class Encoder {
public:
    explicit Encoder(const Model& model) : model_(model) {}
    Embedding embed(const std::vector<float>& values, const std::vector<int8_t>& observed, size_t rows, size_t length);
    Embedding project(const std::vector<float>& features, size_t rows, size_t patches);
private:
    const Model& model_;
    GraphRunner runner_;
};
}
