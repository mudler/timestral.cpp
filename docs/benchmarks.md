# CPU benchmarks (M6c)

## Current result: latency fails, memory passes

The optimized runtime at `b03d674953008e7c557f4148892656eba171e700`
passes full real M6b numerical parity, but fails the full M6c latency gate.
The initial measurements below remain historical evidence, not current status.
See [full-shape validation and profile](m6c-linear.md#full-shape-validation-and-profile)
for the accepted numerical replay, profile method, and next proposed experiment.

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
measurement contract below. Native raw samples reside in
`timestral-native/m6c-opt/native-32k-{1,6}.json` under the work directory.
The reference samples remain in `timestral-native/m6c/reference-isolated-{1,6}.json`.
Input sums match at `-8.5625`. Native output sum is `125.490760441`;
the reference sum is `125.490904057544`. Sums do not replace parity gates.

A rebuilt optimized Release profile now identifies `sdpa_f32` as the dominant
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
when estimating expensive native runs. Initial runs used a 12 GiB address-space
limit for both implementations.

## Initial measurements (historical, before optimization)

CPU: AMD Ryzen 5 7430U with Radeon Graphics, six physical cores and twelve
logical CPUs. Reference: PyTorch `2.14.1+cpu`, pinned source/model revisions
above. Native: Release build of the accepted M6b runtime, without optimizations.
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
Tiny inputs also match (`-0.515625`). The preserved M6b evidence remains the
numerical acceptance record; these sums do not replace it.

### Initial gate status: incomplete, likely latency failure

The M6c gate requires native 32768/1024 medians at most 1.5 times the reference
and native RSS at most reference RSS plus 1 GB, at **both** thread counts.
Native full-shape medians and RSS are **not measured** here. Neither gate passes
from this table. Do not compare a short native shape with the full reference
shape to claim a speed ratio or a memory gate result.

At the matched 512/32 shape, the native single warm sample is approximately
41 times the reference single warm sample. This diagnoses a serious gap; it is
not a repeat-five acceptance ratio. Scaling only by sequence length suggests
minutes per full native forecast and hours for both repeat-five configurations.
That is a planning estimate, not a measured full-shape latency. Full native
runs are deferred pending review of an optimization milestone.

## Initial bounded profiling and proposed next milestone

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
   full M6b, then remeasure the same short workload and full M6c workloads.
4. Reject any numerical regression. Do not substitute an ordinary GEMM merely
   because it is faster, and do not relax tolerances.

Only after that evidence should another milestone target attention, symmetric
pass batching or cache reuse. No runtime optimization is part of this tooling
change. M7/M8 general CLI, ABI and quantization work remains separate.
