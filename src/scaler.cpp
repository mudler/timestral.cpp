#include "scaler.hpp"
#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ts {
namespace {
void validate(const Config& c, const std::vector<float>& x, const std::vector<int8_t>& mask,
              size_t rows, size_t length) {
    if (!rows || !length || rows > std::numeric_limits<size_t>::max() / length ||
        x.size() != rows * length || mask.size() != x.size())
        throw Error("scaler: invalid packed row shape");
    if (!(c.minimum_scale > 0) || !std::isfinite(c.minimum_scale) || c.stabilize_with_global)
        throw Error("scaler: invalid minimum scale or unsupported global stabilization");
    for (auto v : mask) if (v != 0 && v != 1) throw Error("scaler: mask must be boolean");
}
float affine(float x, float loc, float scale, bool observed, bool squash) {
    // Separate float32 subtraction/division and masking precede asinh in the reference.
    float y = (x - loc) / std::max(scale, std::numeric_limits<float>::epsilon());
    if (!observed) return 0.0f;
    return squash ? std::asinh(y) : y;
}
}

ScaleResult scale_rows(const Config& c, const std::vector<float>& x, const std::vector<int8_t>& mask,
                       size_t rows, size_t length) {
    validate(c, x, mask, rows, length);
    if (!c.patch_size || length % c.patch_size) throw Error("scaler: incomplete patch");
    ScaleResult out;
    out.loc.resize(x.size()); out.scale.resize(x.size()); out.scaled.resize(x.size());
    for (size_t r = 0; r < rows; ++r) {
        double cum_w = 0, cum_x = 0, prev_mean = 0, m2 = 0;
        for (size_t t = 0; t < length; ++t) {
            size_t i = r * length + t;
            double v = x[i], w = mask[i];
            cum_w += w;
            cum_x += w * v;
            double den = std::max(cum_w, 1.0);
            double mean = cum_x / den;
            // Literal shifted-means prefix formula, not the algebraically equivalent
            // online mean update: that changes the rounding at the float32 boundary.
            double increment = (v - prev_mean) * (v - mean) * w;
            m2 += increment;
            prev_mean = mean;
            if ((t + 1) % c.patch_size == 0) {
                float loc = static_cast<float>(mean);
                float scale = static_cast<float>(std::max(std::sqrt(m2 / std::max(den - 1, 1.0)), c.minimum_scale));
                for (size_t j = i + 1 - c.patch_size; j <= i; ++j) {
                    out.loc[j] = loc; out.scale[j] = scale;
                    out.scaled[j] = affine(x[j], loc, scale, mask[j], c.use_asinh);
                }
            }
        }
    }
    return out;
}

InstanceStats instance_norm_stats(const Config& c, const std::vector<float>& x,
                                  const std::vector<int8_t>& mask, size_t rows, size_t length) {
    validate(c, x, mask, rows, length);
    InstanceStats out;
    out.loc.resize(rows); out.scale.resize(rows); out.is_binary.resize(rows);
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0, count = 0;
        bool binary = true, any = false;
        for (size_t t = 0; t < length; ++t) {
            size_t i = r * length + t;
            float weighted = x[i] * static_cast<float>(mask[i]);
            sum += weighted; count += mask[i];
            any |= mask[i] != 0;
            binary &= !mask[i] || x[i] == 0 || x[i] == 1;
        }
        float n = static_cast<float>(std::max(count, 1.0));
        float loc = static_cast<float>(sum) / n;
        double squares = 0;
        for (size_t t = 0; t < length; ++t) {
            size_t i = r * length + t;
            float centered = (x[i] - loc) * static_cast<float>(mask[i]);
            float square = centered * centered;
            squares += square;
        }
        float var = static_cast<float>(squares) / n;
        float scale = std::max(std::sqrt(var), static_cast<float>(c.minimum_scale));
        out.is_binary[r] = binary && any;
        out.loc[r] = out.is_binary[r] ? 0.0f : loc;
        out.scale[r] = out.is_binary[r] ? 1.0f : scale;
    }
    return out;
}

void overwrite_known_rows(const Config& c, const std::vector<float>& x, const std::vector<int8_t>& mask,
                          const std::vector<int8_t>& known_rows, size_t length, ScaleResult& out) {
    validate(c, x, mask, known_rows.size(), length);
    if (out.scaled.size() != x.size() || out.loc.size() != x.size() || out.scale.size() != x.size())
        throw Error("scaler: invalid output shape");
    for (auto k : known_rows) if (k != 0 && k != 1) throw Error("scaler: known rows must be boolean");
    if (std::none_of(known_rows.begin(), known_rows.end(), [](int8_t k) { return k != 0; })) return;
    auto stats = instance_norm_stats(c, x, mask, known_rows.size(), length);
    for (size_t r = 0; r < known_rows.size(); ++r) if (known_rows[r]) {
        for (size_t t = 0; t < length; ++t) {
            size_t i = r * length + t;
            out.loc[i] = stats.loc[r]; out.scale[i] = stats.scale[r];
            out.scaled[i] = affine(x[i], stats.loc[r], stats.scale[r], mask[i], c.use_asinh && !stats.is_binary[r]);
        }
    }
}

std::vector<float> clamp_nonfinite(const std::vector<float>& x, size_t last_axis) {
    if (!last_axis || x.empty() || x.size() % last_axis) throw Error("scaler: invalid last axis");
    std::vector<float> out = x;
    for (size_t i = 0; i < x.size(); i += last_axis) {
        float lo = INFINITY, hi = -INFINITY;
        for (size_t j = i; j < i + last_axis; ++j) if (std::isfinite(x[j])) {
            lo = std::min(lo, x[j]); hi = std::max(hi, x[j]);
        }
        for (size_t j = i; j < i + last_axis; ++j) {
            if (x[j] == INFINITY) out[j] = std::isfinite(hi) ? hi : 0.0f;
            else if (x[j] == -INFINITY) out[j] = std::isfinite(lo) ? lo : 0.0f;
        }
    }
    return out;
}

std::vector<float> denormalize_quantiles(const std::vector<float>& x, const std::vector<float>& loc,
                                       const std::vector<float>& scale, size_t quantiles, bool use_asinh) {
    if (x.empty() || loc.empty() || scale.size() != loc.size() || x.size() % loc.size() ||
        !quantiles || (x.size() / loc.size()) % quantiles) throw Error("scaler: invalid denormalization shape");
    size_t stride = x.size() / loc.size();
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = use_asinh ? std::sinh(x[i]) : x[i];
        float product = v * scale[i / stride];
        out[i] = product + loc[i / stride];
    }
    return clamp_nonfinite(out, quantiles);
}
} // namespace ts
