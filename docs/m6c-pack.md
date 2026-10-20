# Bounded callback-local linear packing experiment

Parent: `8c45ce24061fddfeeb7eb6573dbcb723b5f398c3`.
The [aggregate profile](m6c-qk.md#aggregate-profile-and-next-proposed-linear-experiment)
attributes 76.85% of sampled self time to linear column contraction.

## Implementation and ownership

For K at least 192, eight or more output columns, and at least 32 input rows,
the callback allocates one `[K,8]` F32 tile. It copies eight weight rows into
contiguous key-major vectors, then reuses those vectors across four independent
input rows. Each output retains its original ascending K traversal, sequential
FMA within each 192-wide panel, separate F32 panel additions, and final bias.
Small dots and incomplete output-column groups use the prior implementation.
Incomplete four-row groups retain the same per-output arithmetic.

The scratch buffer belongs to the callback through `unique_ptr` and releases
on return. Its checked size is `8*K*sizeof(float)`, or 65536 bytes at K=2048.
An overflow guard or failed nothrow allocation selects the previous path.
No packed data survives compute; mapped weights remain unmodified.
Each callback partition owns its allocation and writes only its assigned rows.
Production graph nodes still request one task. No shared cache or threading
policy changes accompany this experiment.

Runtime dispatch requires AVX2/FMA on supported GCC/Clang x86 builds.
Other targets and `TS_LINEAR_DISABLE_AVX2` retain the scalar implementation.
Release floating-point options remain unchanged; no fast-math, reference,
tolerance, floor, or baseline changes occur. Attention code remains unchanged.

## Verification

Both the optimized callback and forced scalar fallback pass **10,080 byte-exact
comparisons / 32,848,416 output values** against the original scalar oracle.
Coverage includes row counts 1, 3, 17, 31, 32, 33, 34, 35, 36, and 39;
K boundaries 191/192/193 and 383/384/385; K up to 2048; output boundaries
7/8/9 and other tails; bias/no-bias; random finite values; cancellation;
wide exponents; signed zeros; and subnormals.
One- and six-partition callback runs match exactly. Output sentinels and
byte comparisons of weights, inputs, and bias verify no unintended writes.
The callback partition test invokes partitions serially; actual multithreaded
forecast checks below cover the existing graph policy, not parallel packed nodes.
Allocation-failure injection and sanitizer checks are not part of this run.

- Serial component/tiny CTest: **35/35 pass**; separate loader **1/1 pass**.
- Full tiny parity: **28 cases, 164 teacher blocks, 164 free blocks, 28 finals,
  684 comparisons, zero failures**, worst normalized ratio `0.618082`.
- All 28 tiny forecasts pass bytewise repeats against an initial one-thread
  forecast with subsequent thread sequence 1, 6, 6, 1.
- Real 209M 512/32 passes the same bytewise determinism check.

Full real M6b and repeat-five full M6c remain deferred for this aggregate.

## End-to-end results and trade-off

CPU: AMD Ryzen 5 7430U. Model and input follow the unchanged
[measurement contract](benchmarks.md#measurement-contract).
Heavy runs execute serially under a 12 GiB guard, with 20-second cooldowns.
Builds use `-j2` and do not overlap measurements.

| Runtime | Context / horizon | Threads | Warmup / repeats | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|
| Prior QK candidate, earlier median | 512 / 32 | 1 | 1 / 5 | 1.487167610 | 849801216 |
| Packing candidate, median | 512 / 32 | 1 | 1 / 5 | 1.824304844 | 849199104 |
| Prior QK candidate, earlier median | 512 / 32 | 6 | 1 / 5 | 0.379120556 | 847630336 |
| Packing candidate, median | 512 / 32 | 6 | 1 / 5 | 0.594551865 | 847536128 |
| Prior QK candidate, archived cold sample | 32768 / 1024 | 1 | 0 / 1 | 115.247745859 | 943775744 |
| Packing candidate, cold sample | 32768 / 1024 | 1 | 0 / 1 | 56.128879830 | 943620096 |

The matched-contract cold 32k sample improves **2.05x**, approximately 51% less
latency. The baseline is archived, not contemporaneous; frequency and thermal
uncertainty remains. These cold single samples are not acceptance medians.
Input/output sums match between the probes; exact tests, not sums, establish
focused parity evidence. No six-thread 32k result exists for this candidate.

Warm 512 latency regresses approximately **23% at one thread and 57% at six**
relative to earlier medians. This candidate is not an across-shape speed win.
A scratch-only shape logger checks the activation hypothesis: every linear
callback at 512 has 1, 16, or 17 input rows. Therefore **packing never activates
for this workload**. The query stream contributes one patch, not 32 rows.
Raising the packing threshold cannot fix this measured short-shape regression.

A fresh isolated one-thread prior-QK run gives median 1.684203624 seconds
(one warmup, five samples), versus the candidate's 1.824304844 seconds.
That narrows the observed difference to approximately 8.3%, suggesting the older
baseline overstates the regression. It does not prove the remaining difference
is noise. The nonpacking callback now includes dispatch to a helper that rejects
short rows; compiler layout or dispatch overhead may also contribute.
Neither hardware counters nor instruction-level attribution establishes a cause.
No fresh six-thread control runs, and no thermal explanation is assumed.

Keep this as a bounded full-shape win for root review, not an unconditional
cross-shape acceptance. Root must weigh the remaining short-shape cost.
A separate experiment can place the eligibility guard in the caller or isolate
the unchanged short path, then remeasure both shapes. No threshold sweep or
additional runtime change occurs in this follow-up.

The cold 56.13-second result still exceeds the full one-thread gate's
21.57-second limit by a substantial margin. Cold versus warm comparison is
not an acceptance ratio, and full M6c remains on hold.
Reprofile this aggregate before proposing further packing or register changes.

## Reproduction and artifacts

Scratch directory relative to the work root: `timestral-native/m6c-pack`.
The prior cold measurement resides in `timestral-native/m6c-qk/native-32k-1.json`.

```sh
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_TESTS=ON
cmake --build "$BUILD" -j2
"$BUILD/tests/test_linear"
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
| `oracle.log`, `oracle-fallback.log`, `fallback-build.log` | Exact optimized/fallback checks |
| `tiny-gates.log`, `loader.log`, `forecast-tiny.log` | Component and full tiny parity |
| `determinism-tiny.log`, `determinism-512.log` | Actual forecast byte comparisons |
| `native-512-1.json`, `native-512-6.json`, `native-32k-1.json` | Completed measurements |
| `candidate-binaries.sha256`, `source-commit.txt` | Candidate hashes and source pin |
| `layers-shapes.cpp`, `shape-build.log`, `shapes-512.log` | Scratch-only activation audit; all rows below 32 |
| `fresh-baseline-512-1.json`, matching `.log` | Fresh prior-QK one-thread warm control |
