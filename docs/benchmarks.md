# CPU benchmarks

## Current standalone result: latency fails, memory passes

The qualified standalone-ON runtime `68a5a53` records formal medians
**36.392652770s (one thread)** and **18.281547354s (six threads)**.
Both exceed the unchanged 1.5× limits; both RSS measurements pass.
See [the complete formal result](#qualified-standalone-on-68a5a53-measured-latency-failure)
for raw samples, resource qualifications, and source identities.
A profile built from public sources is diagnostic attribution,
not an additional formal performance pass.

An isolated two-row wide-linear candidate
passes a full-shape byte comparison. Its performance remains **UNDECIDED**;
it is not adopted. Numerical identity under contention does not establish speed.

## Historical optimized-native result at b03d674

The optimized runtime at `b03d674953008e7c557f4148892656eba171e700`
passes full real-model numerical parity, but fails the full latency gate.
The initial measurements below remain historical evidence, not current status.

| Implementation | Context / horizon | Threads | Warmup / repeats | Median seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|
| Reference 209M | 32768 / 1024 | 1 | 1 / 5 | 14.379144 | 1919361024 |
| Optimized native 209M | 32768 / 1024 | 1 | 1 / 5 | 330.000421947 | 951521280 |
| Reference 209M | 32768 / 1024 | 6 | 1 / 5 | 4.443924 | 1919823872 |
| Optimized native 209M | 32768 / 1024 | 6 | 1 / 5 | 79.884071984 | 951058432 |

Native/reference latency ratios are **22.95x** and **17.98x**.
The corresponding 1.5x limits are 21.568716 and 6.665886 seconds: both fail.
Both native RSS measurements pass reference RSS plus 1 GB.
These processes use matching inputs, one warmup, and five samples, under the
measurement contract below. Input sums match at `-8.5625`. Native output sum is `125.490760441`;
the reference sum is `125.490904057544`. Sums do not replace parity gates.

A historical rebuilt optimized Release profile identifies `sdpa_f32` as the dominant
full-shape kernel: 62.24% sampled self time, versus 32.07% in `linear_columns`.
This supersedes the short-shape bottleneck assumption for 32k work.
No further runtime change accompanies this profile or documentation update.

## Measurement contract

Build in Release mode, with the CPU backend only:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_CLI=ON
cmake --build build --target timestral-cli -j2
```

Run each configuration in a separate process on the same machine. Run heavy
processes serially and allow a cooldown between configurations. Set OpenMP,
MKL and OpenBLAS thread counts to the requested count before starting Python.
Do not benchmark while compiling. Use the pinned reference checkout
`a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` and its reference environment.

```sh
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N
build/examples/cli/timestral-cli bench --model "$TIMESTRAL_TEST_GGUF" \
  --context-len 32768 --horizon 1024 --variates 1 --threads "$N" --repeat 5
"$VENV/bin/python" scripts/bench_reference.py --model mistralai/Timestral-209M \
  --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 \
  --context-len 32768 --horizon 1024 --variates 1 --threads "$N" --repeat 5
```

Use `N=1` and the physical core count, not the logical CPU count. A local
reference model directory must contain both `config.json` and
`model.safetensors`; specifying `--revision` does not pin a local directory.
Use the converted F32 GGUF from the same model revision.

Both tools use batch size one and the exact F32 input formula
`(((17*t + 13*v) % 257) - 128) / 64`, named `dyadic-v1`. Division by 64 is exact;
no random generator or transcendental function differs between implementations.
There are no covariates or missing observations. The reference uses its shipped
CPU F32 attention, without baseline instrumentation or numerical patches.
The native tool uses the accepted Forecaster API and the GGUF forecaster
settings. For this checkpoint these match the reference defaults: symmetric
averaging and nonnegative clipping enabled, 32 patches per shot, cap `1e4`, and
the model's context limit. These are not generic CLI overrides.

Each tool defaults to one discarded warmup and five timed forecasts. Loading,
input generation and output checksums are outside the wall-time interval.
JSON on stdout includes all samples, median seconds, CPU model, input checksum,
output checksum and peak resident bytes. Progress goes to stderr. Peak RSS is
the **whole process high-water mark**, including loading and warmup, not a
per-forecast allocation delta. Linux `ru_maxrss` is converted from KiB to bytes;
macOS already reports bytes. CPU identification uses `/proc/cpuinfo` where
available. The native measurement tool currently requires a POSIX platform.
Output checksums diagnose gross mistakes; they are not a numerical parity test.

`--warmup 0 --repeat 1` is useful for a bounded cold estimate, but is not an
acceptance measurement. Use a process timeout and the deployment's memory limit
when estimating expensive native runs. Initial runs used an address-space memory limit
for both implementations.

## Initial measurements (historical, before optimization)

CPU: AMD Ryzen 5 7430U with Radeon Graphics, six physical cores and twelve
logical CPUs. Reference: PyTorch `2.14.1+cpu`, pinned source/model revisions
above. Native: Release build of the accepted runtime, without optimizations.
The full reference rows below were measured without an overlapping build, in
separate processes, with 20 seconds between configurations.

| Implementation | Context / horizon | Threads | Warmup / repeats | Median seconds | Peak RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference 209M | 32768 / 1024 | 1 | 1 / 5 | 14.379144 | 1,919,361,024 |
| Reference 209M | 32768 / 1024 | 6 | 1 / 5 | 4.443924 | 1,919,823,872 |
| Native 209M, diagnostic | 512 / 32 | 1 | 1 / 1 | 17.188338 | 848,171,008 |
| Reference 209M, diagnostic | 512 / 32 | 1 | 1 / 1 | 0.419565 | 1,919,471,616 |
| Native 209M, cold diagnostic | 32 / 32 | 1 | 0 / 1 | 2.055269 | 845,737,984 |
| Native tiny, tooling smoke | 32 / 8 | 1 | 1 / 5 | 0.002065 | 7,843,840 |
| Reference tiny, tooling smoke | 32 / 8 | 1 | 1 / 5 | 0.004265 | 251,068,416 |

Full-shape reference samples (seconds):

- One thread: 14.344952, 14.378912, 14.379144, 14.385326, 14.406284.
- Six threads: 4.361762, 4.443924, 4.471497, 4.437134, 4.494231.

Both full reference runs have input sum `-8.5625` and output sum
`125.490904057544`. At 512/32 both tools have input sum `-3.234375`;
native/reference output sums are `99.59357074` / `99.593632422388`.
Tiny inputs also match (`-0.515625`). The preserved full-model evidence remains the
numerical acceptance record; these sums do not replace it.

### Initial gate status: incomplete, likely latency failure

The latency gate requires native 32768/1024 medians at most 1.5 times the reference
and native RSS at most reference RSS plus 1 GB, at **both** thread counts.
Native full-shape medians and RSS are **not measured** here. Neither gate passes
from this table. Do not compare a short native shape with the full reference
shape to claim a speed ratio or a memory gate result.

At the matched 512/32 shape, the native single warm sample is approximately
41 times the reference single warm sample. This diagnoses a serious gap; it is
not a repeat-five acceptance ratio. Scaling only by sequence length suggests
minutes per full native forecast and hours for both repeat-five configurations.
That is a planning estimate, not a measured full-shape latency. Full native
runs are deferred pending review of an optimization step.

## Initial bounded profiling and proposed next step

Hardware profiling was unavailable (`perf_event_paranoid=4`). A separate
Release build with `-pg` for C, C++ and executable linking ran one cold native
209M forecast at 512/32, one thread. The wall time was 20.284500 seconds;
profiling overhead makes this unsuitable for the latency table. `gprof` reported
9.59 seconds of sampled CPU time:

| Symbol | Sampled self time | Share |
| --- | ---: | ---: |
| `ts::linear_f32` | 8.45 s | 88.11% |
| `_init` (unattributed) | 1.08 s | 11.26% |
| `ts::sdpa_f32` | 0.02 s | 0.21% |

These are approximate short-shape samples. Unattributed/library time and gprof's
sampling limitations prevent treating them as a complete wall-time breakdown.
Attention may become more important at 32k; this run does not prove otherwise.

The first proposed optimization scope is the measured linear contraction:

1. Preserve the existing 192-wide FMA panels, K traversal, F32 panel addition,
   bias boundaries and small-dot behavior exactly.
2. Parallelize independent output elements and vectorize across independent
   outputs, not across the reduction dimension. Current custom linear nodes
   request one task, so more backend threads do not parallelize those loops.
3. Change one aspect at a time, run the established component parity gates and
   full real-model parity, then remeasure the same short workload and full latency workloads.
4. Reject any numerical regression. Do not substitute an ordinary GEMM merely
   because it is faster, and do not relax tolerances.

Only after that evidence should a later step target attention, symmetric
pass batching or cache reuse. No runtime optimization is part of this tooling
change. General CLI, ABI and quantization work remains separate.

## Standalone-ON formal attempt: not accepted

After trained replay acceptance (`67ee694`), a separately authorized attempt
uses the existing unprofiled Release ON/shared CLI from runtime `68a5a53`.
LD_PRELOAD is empty and explicitly unset; no interposition is used.
Effective -O3/-DNDEBUG/contraction flags and expected shared loader closure pass.
The wrapper then stops **before admission or any forecast** because its reviewed
per-file pin lookup does not contain `cmake/StandaloneMKL.cmake`.

This is a setup inventory limitation, not a numerical or performance failure.
The prior `after-matrix.json` tracks files present in its original per-file
inventory; seven newly added optional-backend files are instead preserved in
the reviewed `candidate-matrix.tar`. A read-only diagnosis verifies all seven
current source files byte-for-byte against that archive: StandaloneMKL.cmake,
standalone_mkl.cpp/.hpp, standalone_mkl_algorithm.hpp, callback_context.cpp/.hpp,
and graph_lifecycle.hpp. No manifest is silently rewritten or runtime changed.

No timed samples, model runs, formal latency result, or retention verdict exist for this
attempt. No automatic retry follows; owner approval is required for a reconciled archive-based
pin lookup and a fresh output directory. Historical results remain unchanged.
Evidence: `effective-build-flags.json`, `loader-before.json`, and
`pin-inventory-diagnosis.json`. Heavy job exits1 at preflight; no child forecast
remains running. Formal latency remains unmeasured for this qualified production profile.

## Qualified standalone-ON 68a5a53: measured latency failure

After trained replay acceptance, a second formal
attempt was authorized. It uses the same unprofiled Release ON/shared binary and original model.
The setup-only first attempt, which was not accepted, is preserved. Attempt2 reconciles only the seven
missing per-file entries against exact members of reviewed source archive
SHA256 `082ef042a9e44304f0d3c75870437bdd803050c1e4fd765018b24b4358db2b38`
and committed68a5a53. It does not rebaseline arbitrary files.

Both configurations complete in separate processes with one discarded warmup,
five timed 32768/1024 univariate dyadic-v1 forecasts, 60-second cooldown plus
quiet admission, a hard memory limit, and no build/instrumentation overlap. MKL is the
qualified sequential profile, one vendor thread; ggml requests one/six threads.
LD_PRELOAD is unset. Actual process maps confirm the reviewed timestral library
and qualified LP64/sequential/core/ISA SDK DSOs, with no Torch or threaded-MKL
library. Backend JSON is `standalone-mkl-lp64-sequential-2024.0.2-20240605`.

| Threads | Native median seconds | Reference median seconds | Native/reference | Allowed seconds | Native RSS bytes | Allowed RSS bytes |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 36.392652770 | 14.379144 | 2.531x | 21.568715892 | 977420288 | 2919361024 |
| 6 | 18.281547354 | 4.443924 | 4.114x | 6.665886145 | 972148736 | 2919823872 |

**Both latency checks fail; both RSS checks pass.** Exit0 means measurements
complete, not performance acceptance. Final resource qualification is still pending.
No fast-math, numerical-tolerance, reference, or performance-gate change occurs.

All timed samples in order, seconds:
- One thread: **36.236992258,36.527033130,36.337611389,36.392652770,36.803461885**.
- Six threads: **18.746205724,17.988207285,17.755263101,18.916108938,18.281547354**.

Every metadata field, five-sample count/recomputed median, finite output checksum,
inputsum-8.5625, CPU model, and reference contract is checked. Output sums remain
diagnostic rather than parity gates. All **2219 immutable source/vendor/SDK/model/
archive/binary/reference entries** match at initial, per-configuration before/
after, and final checks. No samples are excluded or automatic retry performed.

### Resource interpretation

One-thread process records217.21user+0.37system/218.50wall seconds (0.996CPU).
Across103 monitoring snapshots, load1 is5.23–10.22 and pressureavg10 is4.37–7.56%.
Sampled thread schedstat deltas record216.62s CPU and0.90s runqueue wait.
This supports available single-core execution, not an externally starved run.

Six-thread process records315.28user+0.63system/110.60wall seconds (2.856CPU).
Across52 snapshots, load1 is6.80–15.34 and pressureavg10 is4.39–18.36%.
Six actual threads appear, with up to six runnable. Sampled scheduler deltas
sum315.22s CPU and30.62s runqueue wait across workers; these per-thread sums
are not directly additive wall time. Main threadCPU105.37s versus approximately
42s per other worker is consistent with serial regions/sequential vendor work,
but does not isolate the exact limiting kernel. No universal600% utilization
requirement or global-pressure-only invalidation is applied.

Both visible cgroup throttling deltas are zero. Other resident agent/service
processes consume CPU, but monitoring does not establish a competing heavy
Timestral run or justify removing samples. Available all-core frequency snapshots
span1111–3518MHz (one-thread configuration) and2550–3418MHz (six-thread);
these are not per-thread residency or thermal-causality measurements.
Recovery load/pressure declines after each process; six-thread final recovery
sample is9.32/5.62%. Evidence supports reporting the measured failure with
resource caveats, not asserting that contention alone explains the gap.

Evidence: numeric-summary.json,
analysis-{1,6}.json, native-{1,6}.json/.time/.stderr, monitor/recovery JSONL,
admission records, loader/maps records, archive-reconciliation.json,
effective-build-flags.json, pins-before/per-configuration/final JSON.
Acceptance and any next profiling or optimization step are a separate decision. No quantization
run or further formal replay follows automatically. Current formal latency remains
unmet despite the passing trained numerical profile.
