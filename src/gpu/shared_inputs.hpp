// Platform-independent input generator shared by the CUDA/HIP and Metal exact-order tests.
#pragma once
#include <cstdint>
#include <cstddef>

// Platform-independent generator: splitmix64, seed 11. Float = ((h >> 40) - 2^23) / 2^23
// is exact: uniform in [-1, 1) on a 2^-23 grid. Identical on every platform.
struct Gen {
    uint64_t s;
    explicit Gen(uint64_t seed) : s(seed) {}
    float next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        return (float)((int64_t)(z >> 40) - 8388608) / 8388608.0f;
    }
};

// Checksum of the generated inputs: sequential float sum of the first 64 values. Print with %a.
inline float input_checksum(const float* v) {
    float s = 0.0f;
    for (int i = 0; i < 64; ++i) s += v[i];
    return s;
}
