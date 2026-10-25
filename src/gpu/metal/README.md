# Exact-order F32 linear on Metal

`test_linear_exact_metal.mm`: MSL kernel (one thread per output, same order as `ts_exact_output`) plus a host test that compares bitwise (`memcmp`) with `../linear_exact_ref.hpp`. Not part of the default build.

Build (macOS, Apple clang):

    clang++ -std=c++17 -O2 -ffp-contract=off -Wno-deprecated-declarations -ObjC++ -fobjc-arc -framework Metal -framework Foundation test_linear_exact_metal.mm -o test_linear_exact_metal

Run: `./test_linear_exact_metal` (exit code nonzero on any mismatch or failed probe).

## Metal compile options
`MTLCompileOptions.fastMathEnabled = NO`; on macOS 15+ also `mathMode = MTLMathModeSafe`, `mathFloatingPointFunctions = Precise`. The source has `#pragma clang fp contract(off)` and every multiply-add is an explicit `fma()`. The test runs a probe kernel to show the options take effect: `a*b+c` must round twice (differ from `fma`), and a subnormal product must not be flushed.

## Platform-independent generator
libc++ and libstdc++ differ for `std::normal_distribution`, so this test does not use it. splitmix64 with seed 11 (`s += 0x9E3779B97F4A7C15`, then the two xor-shift-multiply rounds with `0xBF58476D1CE4E5B9` and `0x94D049BB133111EB`, then `z ^= z >> 31`). Float = `((int64)(z >> 40) - 2^23) / 2^23`, exact, uniform in [-1, 1). Order: w (x 0.05), b, x; the outlier rule is `x[i] *= 100` when `i % k == 5`. The CPU reference runs on the same Mac with the same generator.
These inputs are not Gaussian and differ from the CUDA test's inputs, so the case count is the same shape set but not the same data.

## Extra cases
Pass 1 scales w and x by 2^-66 (products near 2^-132, subnormal) on n in {17, 2047}, rows in {3, 32}.

## Result on Apple M4 (macOS 26.5.2)
- Normal-range cases: 880 cases, 18,836,480 outputs, 0 mismatches.
- Probe: unfused `a*b+c` and `fma` behave as intended (options take effect). A subnormal product (2^-134) is flushed to zero on the GPU. The 64 subnormal-scale cases mismatch (581,510 outputs); 10 of them match only because the outputs stay zero or normal. So this GPU does not keep subnormals, unlike the CPU reference. The exact-order result holds only when no operand, product or partial sum is subnormal.

## Not verifiable here
Metal gives no easy view of the GPU ISA. The Metal command-line compiler (`xcrun metal`) is not installed on the test Mac, so the AIR was not inspected. Checked instead: the MSL has explicit `fma()`, `fp contract(off)`, `fastMathEnabled = NO`, and the probe shows separate rounding for `a*b+c` and a single rounding for `fma`.
