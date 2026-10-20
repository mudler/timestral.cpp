# Approved bounded lazy key-transposition design

Root approval: callback-local key transposition only, following the measured
23.48% QK share. No query batching, job reordering, task-policy, linear, value,
softmax, masking, or reference changes.

For the existing wide sequential-FMA branch, require S>=128 and at least 32
owned eligible queries per head/batch row. Pack complete eight-key groups as
`packed[h,group,dimension,lane] = K[(r*S+group*8+lane)*D+h*hd+dimension]`.
Each dot keeps ascending dimension order and sequential F32 FMA; scaling,
mask addition, and maximum consumption retain scalar ascending-key order.
Small-sequence and split-tail reduction branches and incomplete key groups stay.

One RAII buffer per callback stores H*floor(S/8)*hd*8 floats, bounded to 16 MiB.
Check every size product before multiplication; overflow, budget excess, or
nothrow allocation failure uses the existing path. Lazily pack each eligible
head, invalidate head flags when batch row changes, and release on return.
Production graph nodes remain single-task. No pointer survives compute.
At S=1056,H=12,hd=64 scratch is 3244032 bytes (approximately 3.09 MiB).

Tests keep the frozen scalar oracle unchanged. Add focused 127/128/129 and
key-tail cases, owned-query thresholds 31/32/33, multiple heads/batches,
split tails of three/four, immutable inputs and sentinels, and deterministic
internal allocation/budget/overflow fallback probes. Run optimized and forced
scalar checks before components, all tiny parity, and actual forecast repeats
at threads 1,6,6,1. Pin source archives and tested binaries before/after checks.

Performance retention requires an isolated host window. Capture CPU pressure,
load, cgroup counters, and process CPU/wall time around measurements. If severe
contention persists, finish correctness as a local WIP with performance
UNDECIDED; do not benchmark retention or infer regression from noisy samples.
Otherwise run cooled isolated warm 512 at one/six threads and matched 32k at
one thread, warmup one/repeat three, as diagnostics—not formal acceptance.
Full real replay and formal full-shape repeat-five remain deferred.

## Implemented correctness WIP: performance UNDECIDED

Parent: `24cfe18b4aed25a1bc3ef54b3ccaafa38af4e41a`.
The approved implementation is complete, but has **no performance acceptance**.
Severe host contention blocks retention benchmarks. This local WIP is not a
measured optimization win and is not approved for publication or full M6c.

The existing ascending job traversal remains unchanged. Head state has three
values: unknown, packed, and ineligible. On the first eligible query for a
head/batch row, count owned eligible queries under the existing ith/nth policy.
Pack only when the count reaches 32. Reset all head states when batch row r
changes. Queries with fewer than eight keys do not trigger packing.
The packed key helper uses contiguous vector loads with the same per-key
ascending-dimension FMA chain. Existing scalar consumption retains scaling,
mask addition, and maximum ordering. Original key tails and narrow split
branches remain unchanged.

The allocation happens once per eligible-shape callback before job traversal;
head contents are filled lazily. Both packed floats and head-state bytes use
RAII/nothrow allocation. Size checking bounds float storage to 16 MiB;
head-state storage adds H bytes. If either allocation fails, release both and
use the original path. No state survives the callback or mutates model data.
An internal source-header probe supports deterministic failure/coverage tests;
no installed public header or public API changes.

### Completed checks

- Frozen scalar oracle implementation unchanged: **9,360 byte-exact comparisons,
  22,248,000 values** pass on optimized and forced-scalar builds.
- Focused key-packing suite: **432 exact comparisons** on both builds.
  Optimized counters: **648 packed head instances, 297 injected allocation
  fallbacks, 594 size/budget/overflow fallbacks**. Forced scalar counters are zero.
- Focused cases cover S=127/128/129/132; head dimensions 7/33/65; two heads and
  two batch rows; split tails of three/four; callback partitions 1/4/6;
  original-key tails; immutable Q/K/V/mask bytes; and output sentinels.
  Ownership-derived pack counts verify head/batch invalidation and thresholds.
  With two heads and four partitions, the selected split/query counts exercise
  owned-query counts 31/32/33 around activation.
- Direct checked-size tests cover the exact 3244032-byte full-shape calculation,
  overflow at SIZE_MAX heads, and insufficient budget without allocation.
  Internal injected allocation, overflow, and budget failures preserve outputs.
- Serial component/tiny CTest: **35/35 pass**, loader separately **1/1 pass**.
- Full tiny parity: **28 cases, 164 teacher blocks, 164 free blocks, 28 finals,
  684 comparisons, zero failed cases**, worst ratio `0.618082`.
- Actual forecast bytewise determinism: all 28 tiny cases and real 512/32 pass
  thread sequence 1,6,6,1 against an initial one-thread forecast.
- Candidate base archive/patch and tested CLI/oracle/forecast executable hashes
  match before/after the final gates. Frozen source oracle remains unchanged.

The callback partitions execute serially in focused tests. Existing real/tiny
forecast tests exercise actual graph threading, but production SDPA nodes
still request one task. Concurrent packed-callback stress and sanitizer checks
remain limitations, not claimed coverage.

### Performance blocked by measured host contention

No warm 512 or 32k retention benchmark launches for this WIP.
Read-only host snapshots before/after gates record high load and CPU pressure.
The final snapshot shows load averages **43.26 / 55.39 / 41.43** on 12 logical
CPUs; CPU pressure `some` averages **51.77% / 43.96% / 54.64%** over
10/60/300 seconds. Other CPU-consuming processes are present.
The visible service cgroup uses `cpu.max = max 100000`; `nr_throttled` and
`throttled_usec` remain zero. These observations support current host contention,
not quota throttling or a retrospective causal explanation of prior runs.

The serial CTest process records **226.22 s user + 16.45 s system** versus
**415.84 s wall**, approximately **58% CPU utilization**.
Real 512 determinism records **20.52 s user + 0.69 s system** versus
**24.68 s wall**, approximately **85% CPU utilization**.
These are correctness-run resource diagnostics, not latency benchmarks.
No unrelated process is killed and no host setting changes.

Root review must decide when an isolated window is available for the approved
cooled matched diagnostics. Until then, performance stays **UNDECIDED** and
full real replay/formal full-shape repeat-five remain deferred.

### Evidence

Scratch directory relative to the work root: `timestral-native/m6c-keypack`.

| Artifact | Meaning |
|---|---|
| `configure.log`, `build.log`, `final-build.log`, `fallback-build.log` | Release/fallback builds |
| `oracle-final.log`, `oracle-fallback-final.log` | Final exact oracle and focused checks |
| `tiny-gates.log`, `loader.log`, `forecast-tiny.log` | Component and full tiny parity |
| `determinism-tiny.log`, `determinism-512.log` | Actual forecast byte comparisons |
| `host-before-gates.log`, `host-after-gates.log` | Read-only load/pressure/cgroup/process snapshots |
| `source-base.tar`, `source.patch`, `before.sha256`, `after.log` | Reconstructable tested source and before/after executable hashes |
| `source-commit.txt` | Final local WIP source pin |

## Test-only review follow-up

This follow-up changes only `tests/test_attention.cpp` and this document.
Runtime code, the frozen scalar oracle, eager shape-eligible allocation, and
all thresholds remain unchanged. No internal runtime capability API is needed:
the test checks the same compiler/architecture guard and runtime AVX2/FMA CPU
features. The manually linked forced-scalar executable declares its build using
`--expect-scalar`; unsupported hosts automatically expect no activation.

Activation assertions now require the exact expected packed-head count for
every case, including zero. They no longer skip validation when the observed
count is zero. Injected fallback counters likewise must equal expected values.
An AVX2/FMA host therefore cannot silently pass if eligible packing disappears.
A negative control runs the forced-scalar executable without its scalar
expectation: it fails nonzero with the dispatch assertion, confirming the test
rejects missing activation on this supported host. This is an intentional
negative control, not an oracle failure.

The focused S>=128 inputs now add cancellation/wide-exponent Q/K/V samples
and signed-zero/subnormal samples. Queries bound the exponent range so oracle
outputs remain finite; every focused oracle output is checked for finiteness.
Existing masks, ownership thresholds, head/batch resets, immutability checks,
and output sentinels apply to these patterns as well.

Completed exact checks:

- Optimized: **1,296 focused comparisons**, **1,944 packed head instances**,
  **891 injected allocation fallbacks**, **1,782 size/budget/overflow fallbacks**.
- Forced scalar: **1,296 focused comparisons**, all packing/fallback counters
  zero as required by the disabled dispatch expectation.
- Both builds retain **9,360 exact comparisons / 22,248,000 values** from the
  unchanged existing oracle matrix.
- Updated test binaries and reconstructable source archive/patch pass all four
  before/after SHA-256 checks. No redundant component/full suite runs here.

Build/test execution is serial under 12 GiB, with `-j2` compilation.
Read-only host snapshot still shows contention: load **23.26 / 34.15 / 35.83**
on 12 logical CPUs and CPU pressure `some` **21.74% / 20.69% / 33.62%**.
Visible cgroup CPU quota remains unlimited with zero recorded throttling.
No benchmark is authorized or launched. Performance remains **UNDECIDED**.
Concurrent packed-callback stress and sanitizers remain unclaimed coverage.

Evidence under `timestral-native/m6c-keypack/test-followup`:
`build.log`, `fallback-link.log`, `oracle.log`, `oracle-fallback.log`,
`negative-missing-scalar-expectation.log`, `host.log`, `source-base.tar`,
`source.patch`, `before.sha256`, `after.log`, and the two archived test binaries.
