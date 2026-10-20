# Public-source-only standalone-ON attribution

Runtime68a5a53, isolated archive plus scratch-only project instrumentation.
No production runtime changes, proprietary sampling/disassembly/reverse engineering,
interposition, binary modification, or vendor internal investigation occur.
Documented cblas_sgemm calls are opaque wall-time boundaries only.
This diagnoses the measured M6c latency failure; it is not an optimization or gate.

## Controls and numerical identity

Linear oracle passes **43200 exact comparisons /94053960 values**. Head checks
pass **48 production cases /640 direct binary64 cases**, with concurrent
partitions1/2/3/6. Qualified standalone boundary/route checks pass **37 cases**.
Actual tiny forecast output bytes match the reviewed ON/shared library.

After quiet admission and cooldowns, one uninstrumented control and one
instrumented N1 cold32768/1024 dyadic-v1 forecast execute serially. Both use
matching68a5a53 headers, qualified sequential SDK, unchanged arithmetic and
GEMM geometry/beta, hard12GiB, and no build overlap. The control links the
reviewed ON/shared library; the instrumented harness links the isolated library.
All **21504 F32 outputs /86016 bytes match exactly**, not merely a checksum.

Control forecast wall33.5196s, instrumented36.8741s (approximately10% higher).
Both processes record99% CPU; whole-process wall33.56/36.92s, peakRSS
944596/944656KiB. These are cold diagnostics, not the formal warm medians.
Clock/counter/layout effects and run-to-run variation prevent treating their
ratio as a pure overhead estimate or claiming a speedup.

## Hierarchical public-source timing

Timers surround whole callbacks, graph phases, and public GEMM calls; no per-FMA
clocks or hot-loop logging. Invocation/thread-local stats merge after execution.
At one thread nested exclusive totals conserve outer forecast wall to clock
precision; parent/child times must not be added twice. The table lists major
exclusive categories plus selected inclusive parents.

| Category | Calls | Seconds | Accounting |
|---|---:|---:|---|
| Forecast | 1 | 36.874034 | inclusive outer wall |
| Encoder execute | 2 | 35.807091 | inclusive, overlaps graph/callback rows |
| Backend graph compute | 8 | 36.811096 | inclusive, includes callbacks |
| Native global linear | 282 | 18.697395 | exclusive, approximately50.71% outer wall |
| Public-source softmax/rescale | 158400 | 8.993364 | exclusive, approximately24.39% |
| SiLU | 48 | 3.316661 | exclusive, approximately8.99% |
| Sigmoid | 60 | 1.544011 | exclusive, approximately4.19% |
| Opaque public QK SGEMM interval | 158400 | 1.243334 | exclusive, approximately3.37% |
| Opaque public value SGEMM interval | 158400 | 1.231929 | exclusive, approximately3.34% |
| Head binary64 | 6 | 0.276978 | exclusive |
| Patch callback | 6 | 0.180020 | inclusive |
| Opaque patch SGEMM interval | 6 | 0.142560 | child of patch callback |
| Graph allocation/traversal | 8 | 0.004610 | inclusive allocation phase |
| Graph uploads | 16 | 0.006384 | exclusive |
| Graph reads | 10 | 0.024593 | exclusive |
| Graph begin/reset | 8 | 0.000151 | exclusive |

Remaining graph-compute exclusive residual is0.888877s; this includes ordinary
unmeasured graph work and framework overhead, not a named kernel. SDPA callback
inclusive11.848828s contains the public QK/softmax/value/final-copy phases;
its own exclusive0.346042s includes checks, scratch setup, and unseparated work.
Softmax/rescale combines scale/mask/max, exp/reduction, and output rescaling;
the result does not isolate scalar exp alone. Graph construction/preparation
outside measured methods is parent residual, not independently phase-separated.
These limitations matter: do not infer exact cycle counts from clock scopes.

## Causal next-step proposal, not implementation

The largest measured project component is native global linear, unchanged by
standalone ON. A useful next bounded experiment must target its own arithmetic
contract, not change vendor GEMM internals or silently widen backend scope.
The reverted explicit-row4 and unmeasured pack16 candidates remain historical
non-wins; this profile does not retroactively approve either.

First consider a separately reviewed **two-row/16-output packed microtile**
versus current four-row/8-output layout: share input broadcasts across two output
vectors while limiting register pressure. This changes a single independent-
output reuse geometry, retaining every output's ascending K FMA chain,192-wide
panel boundaries, panel additions, bias, fallbacks, and thread ownership.
It is distinct from merely naming four existing accumulators and from retaining
unmeasured pack16/four-row code. Benefit is a hypothesis requiring exact oracles,
current assembly inspection of project code only, bounded parity, and matched
uninstrumented end-to-end diagnostics. No candidate is implemented here.

An independent second opportunity is the project-owned softmax/rescale loop.
Any vectorization must preserve its existing lane-reduction, exp approximation,
scalar tails and online rescaling exactly; it cannot substitute a mathematically
similar vector exp or change key/query tile geometry without new correctness
approval. Measure separate public-source phases before selecting a specific
softmax optimization. Do not speculate about proprietary library algorithms.

## Pins, evidence, and limits

**362 entries** retain before/after hashes: all155 reference artifacts,199 SDK
header/runtime files, source/vendor/instrumented archives, model, control and
instrumented harnesses, and reviewed shared library. No reference, model, SDK,
or production source change occurs. Archives preserve actual instrumentation;
configuration Git metadata is not a substitute for pinned vendor source.
The model hash is checked immediately before the paired stage and after it;
there is no claim of a separate per-subrun hash manifest.

Evidence under `timestral-m6c-on-public-profile-prep` relative to work root:
`base.tar`, `vendor.tar`, `instrument.py`, instrumented source, harness source,
`configure.log`, `build.log`, `harness-build.log`, oracle/route logs,
`tiny-control.bin`, `tiny-instrumented.bin`, `full-control.bin`,
`full-instrumented.bin`, associated logs and timing CSV, `output-identity.json`,
`before.json`, `after.json`, admission/host snapshots. No vendor symbol profile,
proprietary disassembly, or LD_PRELOAD is involved.

Jobs2919 and2929 complete successfully. Heavy slot is released after the paired
runs; no additional model/profile/optimization or quantization task follows.
Current M6c latency remains failed as documented in benchmarks.md; this
attribution neither relaxes its gate nor establishes a new performance result.
