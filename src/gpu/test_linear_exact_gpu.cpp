// Bitwise test: device (or emulated) exact-order linear vs the CPU reference.
// Inputs match the CPU proof test: seed 11, same shapes and generation order.
// Exit code is nonzero on any mismatch.
#include "linear_exact_gpu.hpp"
#include "linear_exact_ref.hpp"
#include <cstring>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0, 1);
    long total = 0, bad = 0, cases = 0;
    for (int k : {768, 2048, 2304, 4096})
    for (int n : {1, 7, 8, 9, 15, 17, 768, 2047, 2048, 2304, 4096})
    for (int rows : {1, 3, 31, 32, 37})
    for (int outl = 0; outl < 2; ++outl) {
        std::vector<float> w((size_t)k * n), x((size_t)k * rows), b(n);
        std::vector<float> y1((size_t)n * rows), y2(y1);
        for (auto& v : w) v = g(rng) * 0.05f;
        for (auto& v : b) v = g(rng);
        for (size_t i = 0; i < x.size(); ++i) { x[i] = g(rng); if (outl && i % k == 5) x[i] *= 100.f; }
        for (int bias = 0; bias < 2; ++bias) {
            const float* bp = bias ? b.data() : nullptr;
            linear_exact_gpu(w.data(), x.data(), bp, y1.data(), k, n, rows);
            linear_exact_ref(w.data(), x.data(), bp, y2.data(), k, n, rows);
            long m = 0;
            for (size_t i = 0; i < y1.size(); ++i) if (std::memcmp(&y1[i], &y2[i], 4)) ++m;
            std::printf("k=%d n=%d rows=%d outlier=%d bias=%d mismatches=%ld\n", k, n, rows, outl, bias, m);
            bad += m; total += (long)y1.size(); ++cases;
        }
    }
    std::printf("cases=%ld elements=%ld mismatches=%ld\n", cases, total, bad);
    return bad != 0;
}
