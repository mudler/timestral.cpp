#pragma once
#include "layers.hpp"
namespace ts {
// Packed [rows, decoder patches, patch_size * Q], quantiles fastest.
std::vector<float> monotone_quantiles(const Config&, const std::vector<float>& raw);
// Model outlives Head. Persistent allocator; not reentrant.
class Head {
public:
 explicit Head(const Model& m) : model_(m) {}
 std::vector<float> project(const std::vector<float>& decoder, size_t rows, size_t patches);
private:
 const Model& model_;
 GraphRunner runner_;
};
}

namespace ts {
std::vector<float> context_patch_projection(const Model&,const std::vector<float>&,size_t rows,size_t patches);
}
