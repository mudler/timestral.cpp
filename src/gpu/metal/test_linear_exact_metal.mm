// Exact-order F32 linear on Metal vs the CPU reference. See README.md.
// Build: clang++ -std=c++17 -O2 -ffp-contract=off -Wno-deprecated-declarations -ObjC++ -fobjc-arc \
//        -framework Metal -framework Foundation test_linear_exact_metal.mm -o test_linear_exact_metal
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../linear_exact_ref.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Same order as ts_exact_output in linear_exact_gpu.hpp. One thread per (row, col).
static NSString* const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct P { uint k; uint n; uint rows; uint has_bias; };
kernel void ts_linear_exact(device const float* w [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device const float* b [[buffer(2)]],
                            device float* y       [[buffer(3)]],
                            constant P& p         [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x, row = gid.y;
    if (col >= p.n || row >= p.rows) return;
    device const float* wr = w + (ulong)col * p.k;
    device const float* xr = x + (ulong)row * p.k;
    float sum = 0.0f;
    if (p.k >= 192u) {
        for (uint s = 0; s < p.k; s += 192u) {
            float panel = 0.0f;
            const uint e = (s + 192u < p.k) ? (s + 192u) : p.k;
            for (uint j = s; j < e; ++j) panel = fma(wr[j], xr[j], panel);
            sum += panel;
        }
    } else {
        for (uint j = 0; j < p.k; ++j) sum = fma(wr[j], xr[j], sum);
    }
    y[(ulong)row * p.n + col] = (p.has_bias != 0u) ? (sum + b[col]) : sum;
}
// Probe: r[0] = a*b + c (must be two roundings), r[1] = fma(a,b,c),
// r[2] = subnormal product a2*b2 (must stay subnormal, not flushed).
kernel void ts_probe(device const float* in [[buffer(0)]], device float* r [[buffer(1)]],
                     uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    volatile float m = in[0] * in[1];
    r[0] = m + in[2];
    r[1] = fma(in[0], in[1], in[2]);
    r[2] = in[3] * in[4];
}
)MSL";

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

static id<MTLDevice> dev; static id<MTLCommandQueue> q; static id<MTLComputePipelineState> pso;

static id<MTLBuffer> mk(const void* p, size_t n) {
    id<MTLBuffer> b = [dev newBufferWithBytes:p length:n ? n : 4 options:MTLResourceStorageModeShared];
    if (!b) { fprintf(stderr, "buffer alloc failed (%zu)\n", n); exit(2); }
    return b;
}

static void run_metal(const float* w, const float* x, const float* b, float* y, int k, int n, int rows) {
    @autoreleasepool {
        uint32_t p[4] = {(uint32_t)k, (uint32_t)n, (uint32_t)rows, b ? 1u : 0u};
        float dummy = 0;
        id<MTLBuffer> bw = mk(w, (size_t)n * k * 4), bx = mk(x, (size_t)rows * k * 4);
        id<MTLBuffer> bb = mk(b ? b : &dummy, b ? (size_t)n * 4 : 4);
        id<MTLBuffer> by = [dev newBufferWithLength:(size_t)rows * n * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
        [e setBuffer:bb offset:0 atIndex:2]; [e setBuffer:by offset:0 atIndex:3];
        [e setBytes:p length:sizeof p atIndex:4];
        [e dispatchThreads:MTLSizeMake(n, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); exit(2); }
        memcpy(y, by.contents, (size_t)rows * n * 4);
    }
}

static bool probe() {
    // a*b needs 48 bits: fused and unfused differ for c = -round(a*b).
    float a = 1.0f + 0x1p-12f, bq = 1.0f + 0x1p-12f;
    float prod = a * bq;                    // host, rounded (contract off)
    float c = -prod;
    float hostUnfused = prod + c;           // 0
    float hostFused = std::fma(a, bq, c);   // 2^-24 residual
    float in[8] = {a, bq, c, 0x1p-70f, 0x1p-70f, 0, 0, 0};
    in[3] = 0x1p-70f; in[4] = 0x1p-70f;     // product 2^-140: subnormal
    in[3] = 0x1p-70f; in[4] = 0x1p-64f;     // 2^-134 subnormal, exactly representable
    float out[4] = {0};
    @autoreleasepool {
        id<MTLBuffer> bi = mk(in, sizeof in), bo = mk(out, sizeof out);
        id<MTLLibrary> lib = nil;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        NSError* err = nil;
        MTLCompileOptions* o = [MTLCompileOptions new]; o.fastMathEnabled = NO;
        lib = [dev newLibraryWithSource:kSrc options:o error:&err];
        id<MTLComputePipelineState> pp = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ts_probe"] error:&err];
        [e setComputePipelineState:pp]; [e setBuffer:bi offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bo.contents, sizeof out);
    }
    bool okU = std::memcmp(&out[0], &hostUnfused, 4) == 0;
    bool okF = std::memcmp(&out[1], &hostFused, 4) == 0;
    float sub = 0x1p-70f * 0x1p-64f;
    bool okS = std::memcmp(&out[2], &sub, 4) == 0 && sub != 0.0f;
    printf("probe: gpu a*b+c=%a (host unfused %a) %s | gpu fma=%a (host fused %a) %s | gpu subnormal prod=%a (host %a) %s\n",
           out[0], hostUnfused, okU ? "OK" : "FAIL", out[1], hostFused, okF ? "OK" : "FAIL", out[2], sub, okS ? "OK" : "FAIL");
    return okU && okF && okS;
}

int main() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
    printf("device: %s\n", dev.name.UTF8String);
    q = [dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.fastMathEnabled = NO;
    if (@available(macOS 15.0, *)) { o.mathMode = MTLMathModeSafe; o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise; }
    printf("compile options: fastMathEnabled=%d\n", (int)o.fastMathEnabled);
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:o error:&err];
    if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 2; }
    pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ts_linear_exact"] error:&err];
    if (!pso) { fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String); return 2; }
    bool probeOk = probe();

    Gen g(11);
    long total = 0, bad = 0, cases = 0, bad0 = 0, cases0 = 0, tot0 = 0;
    // pass 0: normal scale. pass 1: tiny scale 2^-66 on w and x (products ~2^-132, subnormal).
    for (int pass = 0; pass < 2; ++pass)
    for (int k : {768, 2048, 2304, 4096})
    for (int n : {1, 7, 8, 9, 15, 17, 768, 2047, 2048, 2304, 4096})
    for (int rows : {1, 3, 31, 32, 37})
    for (int outl = 0; outl < 2; ++outl) {
        if (pass == 1 && !(n == 17 || n == 2047)) continue;       // subnormal pass: a subset
        if (pass == 1 && !(rows == 3 || rows == 32)) continue;
        std::vector<float> w((size_t)k * n), x((size_t)k * rows), b(n);
        std::vector<float> y1((size_t)n * rows), y2(y1);
        const float sc = pass ? 0x1p-66f : 1.0f;
        for (auto& v : w) v = g.next() * 0.05f * sc;
        for (auto& v : b) v = g.next() * (pass ? 0x1p-100f : 1.0f);
        for (size_t i = 0; i < x.size(); ++i) { x[i] = g.next() * sc; if (outl && i % k == 5) x[i] *= 100.f; }
        for (int bias = 0; bias < 2; ++bias) {
            const float* bp = bias ? b.data() : nullptr;
            run_metal(w.data(), x.data(), bp, y1.data(), k, n, rows);
            linear_exact_ref(w.data(), x.data(), bp, y2.data(), k, n, rows);
            long m = 0;
            for (size_t i = 0; i < y1.size(); ++i) if (std::memcmp(&y1[i], &y2[i], 4)) ++m;
            printf("pass=%d k=%d n=%d rows=%d outlier=%d bias=%d mismatches=%ld\n", pass, k, n, rows, outl, bias, m);
            bad += m; total += (long)y1.size(); ++cases;
            if (pass == 0) { bad0 += m; tot0 += (long)y1.size(); ++cases0; }
        }
    }
    printf("normal-range: cases=%ld elements=%ld mismatches=%ld\n", cases0, tot0, bad0);
    printf("probe=%s cases=%ld elements=%ld mismatches=%ld\n", probeOk ? "ok" : "FAIL", cases, total, bad);
    return (bad != 0 || !probeOk);
}
