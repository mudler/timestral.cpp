// Scalar exact-order CPU reference for the default packed F32 linear.
// One output at a time. Ascending k inside a panel of 192, one fused
// multiply-add per k, panel sums added in ascending panel order starting
// from +0, bias last. For k < 192: one FMA chain, no panel boundary.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
inline void linear_exact_ref(const float* w, const float* x, const float* b, float* y,
                             int64_t k, int64_t n, int64_t rows) {
    for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < n; ++c) {
        const float* wr = w + c * k; const float* xr = x + r * k;
        float sum = 0;
        if (k >= 192) {
            for (int64_t s = 0; s < k; s += 192) {
                float panel = 0;
                const int64_t e = std::min<int64_t>(k, s + 192);
                for (int64_t j = s; j < e; ++j) panel = std::fma(wr[j], xr[j], panel);
                sum += panel;
            }
        } else {
            for (int64_t j = 0; j < k; ++j) sum = std::fma(wr[j], xr[j], sum);
        }
        y[r * n + c] = b ? sum + b[c] : sum;
    }
}
