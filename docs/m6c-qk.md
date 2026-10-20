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

## Aggregate profile and next proposed linear experiment

A profile-only follow-up rebuilds implementation
`ef752a609d53dcbac38d9ce7e744ddf9bbcfd246` in Release mode with `-pg`
for C/C++ and executable linking. Builds use `-j2`; the isolated CPU process
runs under the 12 GiB guard. One 32768/1024 forecast at one thread, zero
warmups, and one repeat completes in 127.241399 seconds, exiting zero.
Peak RSS is 944173056 bytes; the output sum remains `125.490760441`.
Source pins match before/after; executable hash checks pass.
This cold instrumented run is diagnostic, not an acceptance timing.

| Symbol | Sampled self seconds | Share |
|---|---:|---:|
| `linear_columns` | 92.46 | 76.85% |
| `attention_values` | 9.81 | 8.15% |
| `attention_keys` | 9.37 | 7.79% |
| Remaining `detail::sdpa_f32` body | 4.98 | 4.14% |
| `detail::linear_f32` | 1.39 | 1.16% |

Linear contraction is now the dominant measured bottleneck.
Remaining attention totals approximately 20.08% of sampled self time.
Sampling, instrumentation, and uninstrumented library calls limit exact
wall-time attribution. The six-thread breakdown remains unmeasured.

Evidence under `timestral-native/m6c-qk/profile-32k`:
`configure.log`, `build.log`, `source-commit.txt`, `source-after.txt`,
`binary.sha256`, `binary-after.log`, `native-32k.json`, `native-32k.log`,
`native-32k.gprof`, and `gmon.out`.

### Ownership findings

`Model` owns the read-only mapped weights and outlives its encoders.
The public tensor lookup returns pointers into that mapping; packing must not
replace those pointers or mutate their bytes.
`Encoder` owns a persistent `GraphRunner`, but `GraphRunner::begin()` destroys
and recreates graph metadata on every invocation. The allocator persists;
graph tensor pointers do not. A custom callback's raw userdata has no automatic
C++ ownership or destruction hook.
`Forecaster` retains a predictor/encoder across symmetric passes and blocks,
but standalone component APIs also create short-lived encoders.

Consequently, a process-global cache keyed only by tensor address is unsafe:
model unload/reload can reuse addresses. A graph-owned packed-weight cache
requires explicit owner lifetime and clearing before metadata destruction.
Persistent all-model packing duplicates roughly the linear weight footprint
and changes ownership beyond a focused callback experiment.

### Proposed bounded tile packing, pending root approval

Start with callback-local scratch rather than a persistent per-model cache:
pack one eight-output-column tile as `packed[j*8+lane] = w[(col+lane)*K+j]`.
Only the memory layout changes; values copy bit-for-bit.
Use contiguous eight-float loads instead of eight strided scalar weight loads.
Reuse that tile across all input rows before packing the next output tile.
Process four independent input rows together to reuse each weight vector and
hide sequential-FMA latency without reducing across K or SIMD lanes.
Do not change graph task counts, model ownership, or attention in this experiment.

Each output retains its own panel accumulator, initialized to positive zero
for each 192-wide K panel. Traverse K in ascending order within each panel.
Add each completed panel to its own F32 running sum in ascending panel order.
Apply bias once, after the final panel. Preserve the small-dot path and all
row/column tails with the existing scalar/SIMD behavior.
A candidate threshold of at least 32 input rows limits packing overhead;
this is a tunable execution-shape threshold, not a model-specific dimension.
Benchmark threshold choices separately rather than bundling other changes.

Scratch allocation is `8*K*sizeof(float)` bytes per callback, allocated once
and reused across output tiles, then released when that callback returns.
At K=2048 this is 65536 bytes, independent of sequence length and layer count.
The general implementation must check multiplication overflow and allocation
failure; dimensions remain metadata-driven. No packed pointers survive compute.
With production linear nodes still single-task, scratch has no shared writers.

Packing performs one extra read and write per complete weight element on each
callback, not on every input row. At 1056 rows, that preparation is amortized
over 1056 uses of each weight; four-row blocks share weight-vector loads.
The packed tile remains small enough for cache reuse, though cache-hit behavior
requires measurement. Symmetric passes repack independently; no cross-pass
amortization is claimed. Short workloads below the threshold avoid packing.
A separate persistent-cache proposal is unnecessary unless measured packing
cost or repeated invocations justify its extra ownership and memory complexity.

Before retaining this candidate, require exact old-scalar oracle comparisons
at 191/192/193 K boundaries, row thresholds and four-row tails, output tails,
bias/no-bias, random and adversarial finite inputs, and forced ISA fallback.
Then run components, all tiny parity, one-/six-thread forecast determinism,
warm isolated 512 measurements, and one matched cold 32k forecast.
Reprofile only after a bounded win. Full acceptance remains a separate gate.
No packing or runtime source change is included in this profile-only report.
