#pragma once
#include "config.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ts {
// Packed row-major (rows, length), with prepared values and boolean observation masks.
// Missing values must already be replaced by zero; NaN inputs are not sanitized here.
struct ScaleResult { std::vector<float> scaled, loc, scale; };
struct InstanceStats { std::vector<float> loc, scale; std::vector<int8_t> is_binary; };
ScaleResult scale_rows(const Config&, const std::vector<float>&, const std::vector<int8_t>&,
                       size_t rows, size_t length);
InstanceStats instance_norm_stats(const Config&, const std::vector<float>&,
                                  const std::vector<int8_t>&, size_t rows, size_t length);
void overwrite_known_rows(const Config&, const std::vector<float>&, const std::vector<int8_t>&,
                          const std::vector<int8_t>& known_rows, size_t length, ScaleResult&);
// loc and scale contain one anchor per row; quantiles are the last axis.
std::vector<float> denormalize_quantiles(const std::vector<float>&, const std::vector<float>& loc,
                                       const std::vector<float>& scale, size_t quantiles, bool use_asinh);
std::vector<float> clamp_nonfinite(const std::vector<float>&, size_t last_axis);
}
