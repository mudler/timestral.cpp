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
- Selected head dimensions spanning 1–65 include narrow heads and SIMD tails.
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


## Post-candidate full-shape profile

This read/profile-only follow-up measures implementation
`9aa38d10ffd5900681e712110c1f3f07729d0535` after independent bounded review.
No accepted runtime or test source changes accompany these measurements.
The previous full M6c latency failure remains the acceptance record.

### Current optimized gprof result

The profiling executable is rebuilt from the current revision in Release mode,
with `-pg` for C/C++ and linking, `-j2`, and unchanged floating-point flags.
One isolated forecast runs at context 32768, horizon 1024, one thread,
zero warmups, and one repeat under the 12 GiB guard.
It completes in 236.577170 seconds with peak RSS 944091136 bytes.
The output sum remains `125.490760441`. Executable hash checks pass.

| Symbol | Sampled self seconds | Share |
|---|---:|---:|
| `detail::sdpa_f32`, excluding value helper | 105.43 | 49.05% |
| `linear_columns` | 91.26 | 42.46% |
| `attention_values` | 11.90 | 5.54% |
| `_init` | 2.87 | 1.34% |

The remaining SDPA body is slightly larger than linear contraction.
This profile alone does not separate QK and softmax.
Sample shares are diagnostic, not exact wall-time attribution.

### Scratch-only phase attribution

A separate scratch copy of `layers.cpp` adds steady-clock timers around QK,
softmax, and value contraction for each independent query/head job.
It preserves all arithmetic expressions and output paths.
Only the scratch object replaces `layers.cpp` when linking the diagnostic CLI;
accepted runtime files remain unchanged. This build does not use `-pg`.
The timed callback passes the frozen scalar oracle: 2,464 exact comparisons,
3,105,024 output values. No acceptance output is substituted or regenerated.

A second isolated cold 32768/1024 forecast completes in 235.382933 seconds,
with peak RSS 943636480 bytes and the unchanged output sum.
Summing timers across all 60 SDPA callbacks gives:

| Phase | Timed seconds | Share of timed SDPA phases |
|---|---:|---:|
| QK, including dot selection, mask addition, and maximum | 126.677731 | 87.26% |
| Softmax, including exp, sum, and reciprocal | 6.743052 | 4.64% |
| Value contraction and reciprocal multiplication | 11.756458 | 8.10% |

QK is the dominant remaining attention phase; softmax is not the first target.
Clock calls perturb code layout and add overhead. These phase times do not
combine with gprof samples into a single exact wall-time breakdown.
Both runs are cold single forecasts, not acceptance medians.
Neither run measures the six-thread breakdown.

### Evidence and proposed next experiment

Evidence under `timestral-native/m6c-attention/profile-32k`:

| Artifact | Meaning |
|---|---|
| `source-commit.txt`, `source-after.txt` | Matching implementation pins around gprof run |
| `configure.log`, `build.log`, `binary.sha256`, `binary-after.log` | Rebuilt Release profile and hash verification |
| `native-32k.json`, `native-32k.log`, `native-32k.gprof`, `gmon.out` | Completed current-runtime profile |
| `layers-timed.cpp`, `layers-timed.o`, `timed-build.log` | Scratch-only phase instrumentation |
| `timed-oracle.log`, `timed-oracle-phases.log` | Exact oracle verification of scratch callback |
| `timed-32k.json`, `timed-32k.log`, `phase-summary.json` | Completed phase run and 60-callback totals |

Propose one bounded QK experiment, subject to root approval: AVX2/FMA across
independent key columns on the existing wide-sequence sequential-FMA branch.
Each lane must retain ascending head-dimension traversal and the original dot
FMA chain. Preserve the subsequent scale, mask addition, and maximum ordering.
Values and key rows are head-dimension-contiguous, so key-column SIMD requires
strided loads or scratch packing. Start with bounded strided loads; do not
combine packing or threading changes into the first experiment.
Keep the small-sequence and narrow split-tail lane-reduction branches unchanged.
Keep softmax, value contraction, and scalar key tails unchanged.

Require exact old/new attention oracle checks over every execution branch,
then components, all tiny parity, actual one-/six-thread determinism, warm 512,
and one matched 32k probe before retaining a candidate.
Linear contraction remains a second major bottleneck. Even a successful QK
experiment cannot establish the overall 1.5x gate without full measurements.
Linear packing requires a separate measured proposal and approval.
No QK optimization is implemented or authorized by this profile-only report.
