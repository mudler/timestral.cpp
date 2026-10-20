# Bounded independent-key QK experiment

Parent: `eed56885fb87881f0f7ea2efc2067034fdac112c`.
The [post-value-SIMD profile](m6c-attention.md#post-candidate-full-shape-profile)
attributes 87.26% of timed attention phases to QK.
This experiment targets only the wide-sequence sequential-FMA QK branch.

## Arithmetic contract

Eight AVX2 lanes compute independent key dots with strided key loads.
Each lane retains ascending head-dimension traversal and sequential F32 FMA.
The caller consumes completed dots in ascending key order, retaining separate
scaling, mask addition, and maximum updates in that order.
No reduction across SIMD lanes occurs. No packing or threading policy changes.

The existing small-sequence branch and split-tail branch remain scalar.
The scalar remainder handles incomplete groups of eight keys.
Softmax, value contraction, reciprocal boundaries, and graph task counts remain
unchanged. Runtime dispatch requires AVX2 and FMA on supported GCC/Clang x86
builds; other targets use the scalar implementation.
`TS_LINEAR_DISABLE_AVX2` compiles the forced scalar fallback.
Builds retain Release `-O3 -DNDEBUG` and `-ffp-contract=off` for `layers.cpp`.
No fast-math, reference, floor, tolerance, or baseline changes occur.

## Exact coverage and gates

The frozen pre-value-SIMD scalar callback remains the independent oracle.
Its implementation does not change; this experiment extends only its inputs.
Both SIMD and forced scalar fallback pass **7,920 byte-exact comparisons,
18,451,008 output values**, with output sentinels intact.

Selected head dimensions span 1–65, including narrow heads and SIMD tails.
Sequences are 1, 3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, and 65.
Split choices include zero, one, halfway, sequence minus three, sequence minus
four, and the whole sequence. These explicitly exercise the wide branch at
8/16/24/32-key boundaries, incomplete key groups, and both sides of the
split-tail branch condition. Two heads and two batch rows execute.
Inputs include random finite values, cancellation, wide value exponents,
signed zeros, subnormals, causal negative-infinity masks with visible keys,
and fully masked finite `-1e30` rows. Callback partitions of one and six pass.

Further checks:

- Serial component/tiny CTest: **35/35 pass**; loader separately **1/1 pass**.
- Full tiny/tiny-bias parity: **28 cases, 164 teacher blocks, 164 free blocks,
  28 finals, 684 comparisons, zero failures**; worst ratio `0.618082`.
- All 28 tiny forecasts pass bytewise repeats against an initial one-thread
  forecast, with subsequent thread sequence 1, 6, 6, 1.
- Real 209M 512/32 passes the same bytewise forecast determinism check.

Full real M6b replay and full M6c repeat-five measurements remain deferred.
These focused checks are not complete real-model acceptance for this candidate.

## End-to-end measurements

CPU: AMD Ryzen 5 7430U. Pinned 209M F32 model and the unchanged
[benchmark contract](benchmarks.md#measurement-contract).
Builds use `-j2`. Heavy processes execute serially under a 12 GiB address-space
limit, with 20-second cooldowns and no overlapping build.

| Runtime | Context / horizon | Threads | Warmup / repeats | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|
| Value SIMD, earlier median | 512 / 32 | 1 | 1 / 5 | 1.514672722 | 849580032 |
| QK SIMD, median | 512 / 32 | 1 | 1 / 5 | 1.487167610 | 849801216 |
| Value SIMD, earlier median | 512 / 32 | 6 | 1 / 5 | 0.432533806 | 847683584 |
| QK SIMD, median | 512 / 32 | 6 | 1 / 5 | 0.379120556 | 847630336 |
| Value SIMD, archived cold sample | 32768 / 1024 | 1 | 0 / 1 | 224.083898057 | 943169536 |
| QK SIMD, cold sample | 32768 / 1024 | 1 | 0 / 1 | 115.247745859 | 943775744 |

The matched-contract cold 32k comparison improves **1.94x**, approximately 49%
less latency. The baseline is the archived prior candidate, not a simultaneous
or freshly interleaved run; frequency and thermal uncertainty remain.
The baseline and candidate sums match. Exact tests, not sums, establish parity.
Warm 512 movement is small at one thread and noisy at six; the primary evidence
for retaining the candidate is the full-shape movement.

These cold single samples are not acceptance medians. No six-thread 32k probe
or full M6c acceptance measurement runs for this candidate.
The overall 1.5x latency gate remains unmet by the last complete measurement.

## Evidence and reproduction

Scratch directory relative to the work root: `timestral-native/m6c-qk`.
The archived baseline measurement resides in
`timestral-native/m6c-attention/native-32k-1.json`.

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
| `oracle.log`, `oracle-fallback.log`, `fallback-build.log` | SIMD and forced-fallback exact checks |
| `tiny-gates.log`, `loader.log` | Component/tiny and loader gates |
| `forecast-tiny.log` | Full tiny parity counts and negative controls |
| `determinism-tiny.log`, `determinism-512.log` | Actual forecast byte comparisons |
| `native-512-1.json`, `native-512-6.json`, `native-32k-1.json` | Completed end-to-end measurements |
| `candidate-binaries.sha256`, `source-commit.txt` | Candidate executable hashes and source pin |

Keep this bounded candidate pending root review. Reprofile the aggregate before
proposing another optimization. Linear cache packing remains a separate scope;
no packing or additional runtime change is included here.
