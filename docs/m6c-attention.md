# Bounded SDPA value-contraction experiment

Parent: `2adffc0addce66e4702598a4c62670be2c83e43f`.
The [full-shape profile](m6c-linear.md#rebuilt-full-shape-profile) identifies
SDPA as 62.24% of sampled self time after the linear optimization.
This experiment targets only the value contraction, not the entire callback.

## Arithmetic and dispatch

Eight AVX2 lanes compute independent output head dimensions using contiguous
value loads. Each lane traverses keys in ascending order with sequential F32
FMA, then multiplies by the unchanged reciprocal in a separate instruction.
QK, softmax, masking, normalization, narrow heads below 16, and scalar tails
retain their original arithmetic. Graph task counts remain unchanged.

GCC/Clang x86 builds use the existing guarded AVX2/FMA target attribute and
runtime feature checks. Other targets use the scalar callback.
`TS_LINEAR_DISABLE_AVX2` disables both SIMD helpers for fallback verification.
Release builds retain `-O3 -DNDEBUG` and `-ffp-contract=off` for `layers.cpp`.
No fast-math, reference, tolerance, floor, or baseline changes occur.

## Checks

- Frozen old scalar callback versus candidate: **2,464 byte-exact comparisons,
  3,105,024 output values**, with output sentinels intact.
- Forced scalar fallback: the same exact comparison count passes.
- Head dimensions 1 through 65 include narrow heads and SIMD tails.
  Sequences include 1, 3, 7, 8, 9, 17, and 33; split boundaries include zero,
  one, halfway, and the full sequence. Two heads and two batch rows execute.
- Inputs cover random finite values, cancellation, wide value exponents,
  signed zeros, and subnormals. Causal negative-infinity masks retain one
  visible key. Fully masked rows use finite `-1e30` masks, so outputs remain
  finite; fully negative-infinity rows are not part of this finite-output test.
- Callback partitions of one and six match the oracle byte-for-byte.
- Serial CTest: **35/35 pass**, excluding real-model tests and the loader.
  Loader separately passes **1/1** with its required fixture environment.
- Full tiny/tiny-bias forecast parity: **28 cases, 164 teacher blocks,
  164 free blocks, 28 finals, 684 comparisons; zero failed cases**.
  Worst normalized tolerance ratio: `0.618082`.
- All 28 tiny forecasts pass actual byte comparisons with thread sequence
  1, 6, 6, 1 against an initial one-thread forecast.
- Real 209M 512/32 passes the same bytewise thread/repeat check.

No full real M6b or repeat-five 32k acceptance replay runs for this candidate.

## End-to-end measurements

CPU: AMD Ryzen 5 7430U. Model: pinned 209M F32. Input and forecast settings
follow the [measurement contract](benchmarks.md#measurement-contract).
Builds use `-j2`; heavy runs execute serially under a 12 GiB address-space limit.
Each timed process is isolated, with a 20-second cooldown and no overlapping build.

| Runtime | Context / horizon | Threads | Warmup / repeats | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|
| Prior linear candidate, earlier median | 512 / 32 | 1 | 1 / 5 | 1.541307288 | 849358848 |
| Value SIMD candidate, median | 512 / 32 | 1 | 1 / 5 | 1.514672722 | 849580032 |
| Prior linear candidate, earlier median | 512 / 32 | 6 | 1 / 5 | 0.405898183 | 847519744 |
| Value SIMD candidate, median | 512 / 32 | 6 | 1 / 5 | 0.432533806 | 847683584 |
| Prior linear candidate, fresh cold sample | 32768 / 1024 | 1 | 0 / 1 | 335.340281806 | 943607808 |
| Value SIMD candidate, cold sample | 32768 / 1024 | 1 | 0 / 1 | 224.083898057 | 943169536 |

The fresh matched cold 32k sample improves **1.50x**, approximately 33% less
latency. Candidate runs precede baseline runs, so thermal and frequency
uncertainty remains. A single cold sample is not an acceptance median.
The retained baseline executable passes before/after SHA-256 checks.
All 32k input/output sums match between these probes; byte tests, not sums,
establish the focused numerical evidence.

Warm 512 results show little single-thread movement and a noisy six-thread
regression. No short-shape speedup is claimed. The full-shape movement supports
retaining this bounded candidate for review, but does not satisfy M6c.
The previous optimized full M6c latency failure remains the acceptance record.
No six-thread 32k result exists for this candidate.

## Reproduction and evidence

Scratch directory relative to the work root: `timestral-native/m6c-attention`.

```sh
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_TESTS=ON
cmake --build "$BUILD" -j2
"$BUILD/tests/test_attention"
TIMESTRAL_TEST_TINY="$SUITE" ctest --test-dir "$BUILD" -j1 \
  -E '209m|^test_model_loader$' --output-on-failure
env -u TIMESTRAL_TEST_TINY ctest --test-dir "$BUILD" \
  -R '^test_model_loader$' --output-on-failure
TIMESTRAL_TEST_TINY="$SUITE" "$BUILD/tests/test_forecast_determinism"
TIMESTRAL_TEST_GGUF="$MODEL" "$BUILD/tests/test_forecast_determinism" --real-512
"$BUILD/examples/cli/timestral-cli" bench --model "$MODEL" \
  --context-len 512 --horizon 32 --threads "$N" --warmup 1 --repeat 5
"$BUILD/examples/cli/timestral-cli" bench --model "$MODEL" \
  --context-len 32768 --horizon 1024 --threads 1 --warmup 0 --repeat 1
```

| Artifact | Meaning |
|---|---|
| `configure.log`, `build.log` | Release configuration and build |
| `oracle.log`, `oracle-fallback.log`, `fallback-build.log` | Exact SIMD and fallback checks |
| `tiny-gates.log`, `loader.log` | Passing component/tiny suite and separate loader |
| `forecast-tiny.log` | Complete tiny forecast counts and negative controls |
| `determinism-tiny.log`, `determinism-512.log` | Actual bytewise forecast comparisons |
| `native-512-1.json`, `native-512-6.json` | Warm five-sample diagnostic measurements |
| `native-32k-1.json`, `baseline-32k-1.json` | Matched single cold 32k probes |
| `baseline.sha256`, `baseline-after.log` | Retained pre-change executable checks |
| `candidate-binaries.sha256`, `source-commit.txt` | Candidate hashes and source pin |

Keep the bounded candidate pending root review. Reprofile before another
optimization; linear cache packing is not authorized by this experiment.
Full numerical acceptance and full M6c measurements require separate replay.
