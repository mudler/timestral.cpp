#pragma once
#include "scaler.hpp"
namespace ts {
// Row-major (rows, patches); double work and float32 frame boundary.
struct BlockFrames { std::vector<double> loc, scale; };
BlockFrames advancing_block_frames(const Config&, const std::vector<float>& context,
 const std::vector<int8_t>& mask, const std::vector<double>& sinh_median,
 size_t rows, size_t length, size_t patches);
// Quantiles are (rows, patches * patch_size, num_quantiles). Empty target_rows means all.
std::vector<float> refine_block(const Config&, const std::vector<float>& quantiles,
 const std::vector<float>& context, const std::vector<int8_t>& mask,
 size_t rows, size_t length, size_t patches, const std::vector<int8_t>& target_rows = {});
}
