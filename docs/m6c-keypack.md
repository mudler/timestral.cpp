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

## Retention-window screening: blocked

The benchmark-only follow-up monitors the host for the approved ten-minute
maximum, from 07:30:13 to 07:40:13 UTC on 20 October 2026.
It records **31 read-only snapshots**, approximately 20 seconds apart.
Eligibility requires both CPU pressure avg10 below 10% and load1 below 12,
for three consecutive samples. These resource-screening thresholds do not
change numerical tolerances or the M6c performance gate.

Only sample 19 qualifies: load1 **11.17**, pressure avg10 **8.79%**.
The longest eligible streak is **one**, not three.
Across the window, load1 ranges **10.83–21.10** and pressure avg10 ranges
**6.16–26.86%**. The last snapshot has load1 **12.72** and pressure **6.16%**,
still failing the load condition. No qualifying window occurs before timeout.

**No retention benchmark launches. Performance remains UNDECIDED.**
No builds, runtime changes, unrelated process termination, or configuration
changes occur. Source/binary benchmark pins are not created because execution
never passes screening; prior correctness pins remain the applicable evidence.
No samples are selected retrospectively to relax the three-snapshot condition.

Evidence under `timestral-native/m6c-keypack/retention`:
`screening.jsonl`, `host-00.log` through `host-30.log`, and `blocked.txt`.
Each host record includes load, CPU pressure, cgroup quota/counters, and process
CPU snapshots. The screening job exits zero after completing its bounded wait;
that exit status indicates monitoring completion, not benchmark acceptance.
Root owns authorization of any later measurement window or next scope.

## Second retention attempt: invalidated by CPU availability

Root authorizes one further attempt in separate scratch `retention-2`.
Initial screening qualifies after three samples at 20-second intervals:
load/pressure pairs **8.72/3.40%, 8.49/4.46%, 9.37/5.44%**.
Preflight then stops on a historical test-binary hash mismatch.
Only `build-j2/tests/test_attention` differs from the original correctness
manifest: the approved test-only follow-up intentionally rebuilt it.
Its current SHA-256 matches the archived follow-up optimized test exactly:
`28c06940e378e9233c99cf2d683285a7dc01e251f89b2599cdbe0f2e9c4f5d51`.
All four follow-up pins pass. The original candidate CLI, source base/patch,
forecast, and determinism hashes pass; runtime sources match `35fdc68`.
The original failed hash log remains preserved, not rewritten.

The first continuation job is explicitly stopped before any benchmark child
launches, honoring root's stop-and-diagnose request. A process inspection finds
no leftover benchmark child. A benchmark-specific manifest pins baseline and
candidate CLI binaries and archived sources; no builds occur.
Because the initial window is stale, continuation rescreens into new files.
No initial screening or partial measurement files are overwritten.

Rescreening qualifies at samples 8/9/10, with load/pressure pairs
**11.92/6.84%, 10.47/4.36%, 8.63/3.29%**.
After cooldown, only the baseline 512/32 one-thread configuration runs:
one warmup, five samples, median **1.675918730 seconds**, exit zero.
However, process accounting records **10.14 s user + 0.39 s system** versus
**13.81 s wall**, only **76% CPU**. Before/after host snapshots remain below
the screening bounds, demonstrating that these snapshots alone cannot establish
uninterrupted CPU availability. No particular contention mechanism is proven.

**This measurement is invalid for retention. No candidate, six-thread, or 32k
configuration runs. Performance remains UNDECIDED.**
The control script conservatively stops when one-thread process CPU utilization
falls below 95%, a resource-quality screen—not a numerical/performance-gate
relaxation. No useful baseline/candidate performance comparison exists here.
All four benchmark-specific source/archive/CLI pins match before/after.
No new attempt, runtime change, unrelated process termination, or configuration
change follows this decision. Root owns further measurement authorization.

Evidence under `timestral-native/m6c-keypack/retention-2`:
`screening.jsonl`, `qualified.json`, `screen-*.log`,
`m6c-keypack-prior-hashes.log`, `m6c-row4-prior-hashes.log`,
`followup-hashes.log`, `pins-before.json`, `pins-after.json`,
`baseline-source.tar`, `candidate-source.tar`, `rescreening.jsonl`,
`rescreen-*.log`, `baseline-512-1.json`, `baseline-512-1.log`,
`baseline-512-1-host.json`, matching before/after snapshots, and `interrupted.txt`.

## Actual callback ownership audit: task-policy correction

This section explicitly supersedes earlier inferences that custom nodes declared
with task count one necessarily run their callbacks serially. Historical timing
records remain unchanged. At pinned ggml e705c5fe, requested node task counts
contribute to graph planning, but custom callbacks receive graph-wide ith/nth.
Other nodes can raise the graph's worker count. Changing only the custom task
argument therefore is not an established parallelization opportunity.

A scratch-only audit from committed base `0cae4e3` adds local integer counters
and mutex-protected logging. No production source, floating-point expression,
job ownership, packing condition, or task-policy changes occur. Exact original
linear oracle passes **10080 comparisons / 32848416 values**. Attention passes
**1296 focused comparisons** with expected activation/fallback counters and
**9360 original comparisons / 22248000 values**. These checks precede model runs.

Four actual single-cold-forecast diagnostics complete: context/horizon 512/32
and 32768/1024, each with one and six requested threads. Full-shape runs follow
three eligible load/pressure samples. All runs/builds execute serially under
the hard 12 GiB guard with -j2 compilation. Instrumentation overhead means none
of their wall times establishes performance, retention, or M6c acceptance.

| Shape | Requested threads | Actual callback nth | Linear callback invocations | SDPA invocations | Total owned linear rows | Total attention jobs |
|---|---:|---:|---:|---:|---:|---:|
| 512/32 | 1 | 1 | 290 | 60 | 4826 | 12240 |
| 512/32 | 6 | 6 | 1740 | 360 | 4826 | 12240 |
| 32768/1024 | 1 | 1 | 290 | 60 | 299840 | 760320 |
| 32768/1024 | 6 | 6 | 1740 | 360 | 299840 | 760320 |

Every node has exactly one invocation per worker, with complete ith sets.
Counters match the cyclic partition's disjoint/exhaustive row/job sets, so each
output owner occurs once. This verifies partition arithmetic and observed worker
coverage, not a sanitizer-level proof against every possible memory race.
Full-shape time attention has S=1056,H=12,hd=64. At six workers, worker i owns
all 1056 queries for heads i and i+6. Keypack therefore retains its >=32 owned
query eligibility; heads are not split six ways at this geometry.

Both thread counts record **576 packed-head events** and **608256 packed
queries**. Full-shape packed scratch per time-attention callback is **3244044
bytes** including 12 head-state bytes. Across all invocations its allocation
sum is **155714112 bytes at one thread** and **934284672 bytes at six**.
These are repeated-allocation sums, **not simultaneous peak RSS**. Six concurrent
time-attention callbacks reserve approximately 18.56 MiB in total while filling
only their assigned heads. Linear packed scratch allocation sums likewise grow
from **9437184** to **56623104 bytes**; each worker owns independent scratch.
At 512 neither linear packing nor key packing activates. Both thread counts
retain the same output sums; scalar-oracle checks, not sums, establish the
instrumentation's focused numerical equivalence.

Tested source archive and CLI/oracle binary hashes match before/after all runs.
The post-audit model hash matches the existing pre-audit canonical-run model
pin; an immediately-before-audit model hash was not captured, so this is not
claimed as a fresh adjacent before/after model check. The archived vendor's
1962 blobs match e705c5fed490514458bdd2eaddc43bd098fcce9b; outer-repository
archive Git metadata is not the vendor identity.

Evidence under `timestral-native/m6c-ownership-audit`: `base.tar`, `ggml.tar`,
`instrumentation.patch`, `tested-source.tar`, `before.json`, `after.json`,
`oracle-linear.log`, `oracle-attention.log`, `512-{1,6}.log/json`,
`32k-{1,6}.log/json`, parsed summaries, `coverage-verification.json`,
`model-pin-verification.json`, `vendor-verification.json`, screening and host
snapshots. Parsed counters are measured execution evidence; allocation totals
must not be reported as memory peaks. No task-policy optimization is justified
by the old serial assumption. Root owns the next uninstrumented retention task.

## Third uninstrumented retention attempt: partial one-thread evidence

The separately authorized attempt uses the original pinned pre-reduced-format
baseline CLI (runtime 8a83cb8) and keypack CLI (runtime 35fdc68), without builds
or code changes. Three eligible host samples precede execution. Configurations
run serially under 12 GiB, with 60-second cooldowns, one warmup, and three timed
32k/1024 forecasts. This is diagnostic, not the formal repeat-five gate.

| One-thread runtime | Samples, seconds | Median seconds | Peak RSS bytes |
|---|---|---:|---:|
| Baseline | 45.035663283, 47.820362552, 47.949021372 | 47.820362552 | 951500800 |
| Keypack | 42.246256123, 41.804174760, 42.659341805 | 42.246256123 | 951595008 |

The clean one-thread pair gives **1.132x**, or **11.66% less median latency**.
Both processes receive 99% CPU; every before/during/after snapshot remains below
the approved load/pressure thresholds. Baseline process CPU/wall is
184.99 s user + 0.17 s system / 185.79 s wall. Candidate is
168.81 + 0.24 / 169.66 s. All shapes/sample counts/medians and output/input sums
match the contract. Sums are diagnostic, not substitutes for exact parity.

The following six-thread baseline completes with samples 14.045278096,
13.437964014,13.203582084 seconds (median 13.437964014), but fails the resource
screen: pressure avg10 rises to 18.00–20.61%, and load1 reaches 13.82.
Its 536% CPU utilization is recorded, not a passing isolation guarantee.
Six active benchmark workers can themselves contribute to host pressure;
these snapshots do not prove unrelated contention or an intrinsic regression.
Per the fixed screening rule, this six-thread measurement is **invalid for
retention**. The script stops without a candidate-six run or any 512 reruns.
No slow sample is discarded and no alternative median is selected.

**Overall keypack retention remains UNDECIDED.** The one-thread improvement is
bounded useful evidence, but does not fill the missing valid six-thread pair,
formal M6c run, or aggregate numerical acceptance. In particular 42.25 seconds
still exceeds the one-thread formal limit; no tolerance or performance gate is
relaxed. Root owns any subsequent measurement-policy/retention decision.

Both CLI binaries and the model receive immediate before/after SHA256 checks
for every completed configuration; all nine comparisons match. Original source
archives match the earlier reconciled benchmark-specific source pins. The known
old attention-test manifest mismatch is irrelevant to the pinned CLI files;
no historical manifest is overwritten.

Evidence under `timestral-native/m6c-keypack/retention-3`: `run.py`,
`screening.jsonl`, host snapshots including available CPU frequencies/cgroup
counters, `source-pins.json`, `baseline-32768-1.json`,
`candidate-32768-1.json`, `baseline-32768-6.json`, corresponding process logs,
per-configuration before/after hash JSON and host JSON, `completed-summary.json`,
and `interrupted.txt`. No new attempt or optimization follows this report.
