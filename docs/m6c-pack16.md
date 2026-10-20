# Sixteen-column packing experiment: not retained

Root approved a bounded output-width experiment based on the measured packed
linear share, not a repeat of the explicit-row4 naming/unrolling experiment.
Two adjacent eight-output groups share each input broadcast; up to four rows
retain separate ascending-K sequential FMA chains, positive-zero 192-wide
panels, F32 panel additions, and final bias. The prior eight-column remainder,
scalar columns, eligibility, ownership, attention, and thread policy stay.
Checked nothrow RAII scratch is at most 16*K floats (128 KiB at K=2048).
Allocation failure falls back before output writes.

The candidate is built from isolated `74a2893` plus only kernel/test changes,
with pinned ggml and without concurrent producer edits. Release, hard 12 GiB,
serial checks, and -j2 compilation apply throughout.

## Correctness results

Optimized and forced-scalar oracle runs each pass **43200 exact comparisons /
94053960 output values**. The frozen scalar oracle is unchanged. Extensions
cover output widths 15/16/17 and31/32/33; K-panel boundaries; row thresholds and
four-row tails; bias, adversarial finite inputs, zeros/subnormals; partitions
1/2/3/4/6; input immutability and output sentinels. A modest actual concurrent
subset at K193,N33,rows35 exercises each worker count and bias choice.
This is not sanitizer or comprehensive concurrent stress coverage.

F32 component/tiny CTest passes **40/40**, with loader separately **1/1**.
The isolated baseline includes registered reduced-format synthetic tests; their
presence does not constitute a new real quantized-quality gate. Full tiny
forecast replay passes **28 cases / 684 comparisons**, zero failures, worst
ratio `0.618082`. Actual forecast bytewise determinism passes all 28 tiny cases
and real512/32 at thread sequence 1,6,6,1. Archive/test/CLI/model hashes match
before and after correctness gates. Candidate disassembly is retained for
inspection; eight panel chains increase register pressure and may spill, but
no performance cause is established from assembly alone.

## Performance comparison interrupted

After three quiet admission samples and a 60-second cooldown, only the pinned
baseline runs at 32768/1024, one thread, warmup1/repeat3:

- Samples: **50.972198454,51.029269957,54.578844743 seconds**.
- Median: **51.029269957 seconds**; peak RSS **951476224 bytes**.
- CPU accounting: **201.57 user +0.31 system /204.40 wall seconds**, 98% CPU.

During this one-thread run, load reaches **13.29** and pressure avg10 spikes
to **29.60%**. The configured one-thread qualification fails. This does not
prove unrelated interference or an intrinsic kernel regression; it prevents a
qualified comparison under the approved protocol. All samples are preserved.
**No candidate or six-thread configurations run. Performance is UNDECIDED.**
Immediate baseline/candidate CLI and model before/after hashes for the completed
baseline configuration match. No 512 performance run is needed to infer a win:
the packing branch does not activate there, and no such win is claimed.
No full real aggregate replay or formal repeat-five gate runs.

## Decision and artifacts

The approved retention criterion requires a demonstrated end-to-end win.
Because the matched comparison never completes, revert only the experimental
runtime change and retain the expanded exact tests. This is a conservative
non-retention decision, **not proof that sixteen-column packing is slower**.
No follow-up optimization or new measurement attempt is launched.

Scratch evidence under `timestral-native/m6c-pack16`: `base.tar`, `ggml.tar`,
`candidate.patch`, `source/`, `tested-source.tar`, `before.json`,
`after-gates.json`, `oracle.log`, `oracle-fallback.log`, build/fallback logs,
`disassembly.txt`, `component-tiny.log`, `loader.log`, `forecast-tiny.log`,
`determinism-tiny.log`, `determinism-512.log`, screening/host/thread snapshots,
`baseline-1.json`, `baseline-1.log`, `baseline-1-before.json`,
`baseline-1-after.json`, `baseline-1-host.json`, and `interrupted.txt`.
The tested candidate binary remains in scratch, not the retained runtime.
