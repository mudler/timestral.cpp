// Exact-order F32 linear on a GPU: one thread per output (row, column).
// One source for CUDA and HIP. Define TS_GPU_EMULATE to run the same
// per-output function on the CPU instead (loop-structure check only).
//
// Order per output (identical to linear_exact_ref):
//   k >= 192: for each 192-wide panel, ascending k, panel = fmaf(w, x, panel)
//             starting at +0; then sum += panel (panels ascending, sum from +0).
//   k <  192: one chain, sum = fmaf(w, x, sum).
//   bias is added last. No chain is split or reordered across threads.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(TS_GPU_EMULATE)
  #define TS_DEV_FN inline
  #define TS_FMAF(a, b, c) std::fmaf((a), (b), (c))
#else
  #if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    #include <hip/hip_runtime.h>
    #define TS_RT(x) hip##x
  #else
    #include <cuda_runtime.h>
    #define TS_RT(x) cuda##x
  #endif
  #define TS_DEV_FN __host__ __device__ inline
  // Explicit fused multiply-add; never left to the compiler's contraction.
  #define TS_FMAF(a, b, c) fmaf((a), (b), (c))
#endif

constexpr int64_t TS_PANEL = 192;

// The per-output chain. Shared by the device kernel and the CPU emulation.
TS_DEV_FN float ts_exact_output(const float* w, const float* x, const float* b,
                                int64_t k, int64_t n, int64_t row, int64_t col) {
    const float* wr = w + col * k;
    const float* xr = x + row * k;
    float sum = 0.0f;
    if (k >= TS_PANEL) {
        for (int64_t s = 0; s < k; s += TS_PANEL) {
            float panel = 0.0f;
            const int64_t e = (s + TS_PANEL < k) ? (s + TS_PANEL) : k;
            for (int64_t j = s; j < e; ++j) panel = TS_FMAF(wr[j], xr[j], panel);
            sum += panel;
        }
    } else {
        for (int64_t j = 0; j < k; ++j) sum = TS_FMAF(wr[j], xr[j], sum);
    }
    return b ? sum + b[col] : sum;
}

#if defined(TS_GPU_EMULATE)

// Host emulation of the launch: visit every (row, col) thread once.
inline void linear_exact_gpu(const float* w, const float* x, const float* b, float* y,
                             int64_t k, int64_t n, int64_t rows) {
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t col = 0; col < n; ++col)
            y[row * n + col] = ts_exact_output(w, x, b, k, n, row, col);
}

#else

__global__ void ts_linear_exact_kernel(const float* w, const float* x, const float* b,
                                       float* y, int64_t k, int64_t n, int64_t rows) {
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (col >= n || row >= rows) return;
    y[row * n + col] = ts_exact_output(w, x, b, k, n, row, col);
}

#define TS_CHECK(call) do { auto e_ = (call); if (e_ != TS_RT(Success)) { \
    std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, \
                 TS_RT(GetErrorString)(e_)); std::exit(2); } } while (0)

// Same signature as linear_exact_ref. Synchronous: allocate, copy, launch, copy back, free.
inline void linear_exact_gpu(const float* w, const float* x, const float* b, float* y,
                             int64_t k, int64_t n, int64_t rows) {
    float *dw = nullptr, *dx = nullptr, *db = nullptr, *dy = nullptr;
    const size_t bw = (size_t)n * k * 4, bx = (size_t)rows * k * 4;
    const size_t bb = (size_t)n * 4, by = (size_t)rows * n * 4;
    TS_CHECK(TS_RT(Malloc)((void**)&dw, bw));
    TS_CHECK(TS_RT(Malloc)((void**)&dx, bx));
    TS_CHECK(TS_RT(Malloc)((void**)&dy, by));
    TS_CHECK(TS_RT(Memcpy)(dw, w, bw, TS_RT(MemcpyHostToDevice)));
    TS_CHECK(TS_RT(Memcpy)(dx, x, bx, TS_RT(MemcpyHostToDevice)));
    if (b) {
        TS_CHECK(TS_RT(Malloc)((void**)&db, bb));
        TS_CHECK(TS_RT(Memcpy)(db, b, bb, TS_RT(MemcpyHostToDevice)));
    }
    const int threads = 128;
    dim3 grid((unsigned)((n + threads - 1) / threads), (unsigned)rows);
    ts_linear_exact_kernel<<<grid, threads>>>(dw, dx, db, dy, k, n, rows);
    TS_CHECK(TS_RT(GetLastError)());
    TS_CHECK(TS_RT(DeviceSynchronize)());
    TS_CHECK(TS_RT(Memcpy)(y, dy, by, TS_RT(MemcpyDeviceToHost)));
    TS_RT(Free)(dw); TS_RT(Free)(dx); TS_RT(Free)(dy); if (db) TS_RT(Free)(db);
}

#endif
