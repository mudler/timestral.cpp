# Exact-order F32 linear on a GPU

Files:
- `linear_exact_gpu.hpp`: one source for CUDA and HIP. One thread per output (row, column). The kernel and the host wrapper `linear_exact_gpu` (same signature as the CPU reference; allocate, copy in, launch, synchronize, copy out, free).
- `linear_exact_ref.hpp`: scalar CPU reference with the exact order.
- `test_linear_exact_gpu.cpp`: bitwise test over 880 cases (K in {768, 2048, 2304, 4096}, N in {1, 7, 8, 9, 15, 17, 768, 2047, 2048, 2304, 4096}, rows in {1, 3, 31, 32, 37}, with and without bias, with and without an outlier channel). Prints the mismatch count per case and a total. Exit code is nonzero on any mismatch.

The default build does not use this directory.

## Order of operations per output
For K >= 192: for each 192-wide panel, ascending k, `panel = fmaf(w, x, panel)` from +0; then `sum += panel`, panels in ascending order; bias added last. For K < 192: one chain, no panel add. A chain is never split or reordered across threads.

## Compile commands
CUDA:

    nvcc -std=c++17 -O2 -fmad=false -x cu test_linear_exact_gpu.cpp -o test_linear_exact_gpu

HIP:

    hipcc -std=c++17 -O2 -ffp-contract=off test_linear_exact_gpu.cpp -o test_linear_exact_gpu

CPU emulation (no GPU toolchain needed):

    g++ -std=c++17 -O2 -ffp-contract=off -DTS_GPU_EMULATE test_linear_exact_gpu.cpp -o test_linear_exact_emul

Add the right `-arch` / `--offload-arch` flag for the target device.

## Flags and why
- Every multiply-add in the chain is an explicit `fmaf`, which is a single-rounding fused operation on both CUDA and HIP, so bit equality does not depend on the contraction flag.
- We pick `-fmad=false` (CUDA) and `-ffp-contract=off` (HIP) anyway. The only other float operations are `sum += panel` and `sum + bias`, which have no multiply to fuse. Turning contraction off removes any chance that the compiler fuses or reorders something we did not write. `-fmad=true` would also give the same result here, but it adds nothing and leaves room for surprise.
- No `--use_fast_math`, no `-ffast-math`, no flush-to-zero flags.

## Test command
Run the built binary with no arguments: `./test_linear_exact_gpu`. It needs about 2 GB of host memory for the largest case.

## Verified so far
- No `nvcc` or `hipcc` was available on the machine used to write this. The CUDA and HIP paths are NOT compiled and NOT run. The device arithmetic is unverified.
- The CPU emulation build runs the same `ts_exact_output` function (with `std::fmaf`) in a host loop over all (row, column) pairs. It matched `linear_exact_ref` bit for bit on all 880 cases (18,836,480 outputs, 0 mismatches). This proves the loop structure and index math only. It does not prove anything about device fused-multiply-add behaviour.
- The CPU reference itself was checked earlier against the default packed linear: 0 mismatches over the same 880 cases.

## Remaining device-side checks
1. Build with both toolchains (the macro layer has never been compiled).
2. Run the test on hardware and record the real mismatch count (must be 0).
3. Confirm `fmaf` compiles to a single fused instruction (inspect PTX `fma.rn.f32` or the ISA `v_fma_f32`), with no split multiply and add.
4. Check subnormal handling: the CPU keeps subnormals, so the device must not flush them (CUDA without `-ftz=true`, HIP without flush-denormals options). Add a test case with subnormal inputs and products.
5. Check that the rounding mode is round-to-nearest-even for the fused operation and the adds.
