# Explicit four-row linear microkernel: not retained

Parent and retained runtime: `7ecd7a3991225ca8d0da63a5409e230751a10659`
(documentation HEAD; implementation `8a83cb8564ce7e4c58b489d567272c834891f9f6`).
The experiment changes only the full-four-row case inside packed linear.
Four named panel accumulators and running sums replace dynamic row-count loops.
Incomplete groups, packing, thresholds, ownership, graph policy, and attention
remain unchanged. No numerical contracts or reference artifacts change.

**Decision: revert the runtime experiment.** Exact parity passes, but matched
warm end-to-end measurements do not demonstrate the required performance win.
The noisy results do not prove the microkernel is intrinsically slower.

## Hypothesis and disassembly

The prior profile attributes 39.78% of sampled self time to packed linear.
Baseline disassembly includes row-count comparisons around FMA updates.
Candidate disassembly shows a separate full-four-row loop with four register
FMA chains and no row-count test inside that loop. This supports the structural
hypothesis, not a measured speedup or a claim of eliminated cache misses.
The dynamic incomplete-row path remains in the candidate binary.
Baseline disassembly comes from the prior profiling object; candidate disassembly
comes from the uninstrumented Release object. Their instruction addresses and
whole-function layouts are not directly comparable performance measurements.

Every output retains ascending K traversal, positive-zero initialization for
192-wide panels, sequential FMA, separate F32 panel additions, and final bias.

## Completed checks on the experimental candidate

- Original scalar oracle: **10,080 byte-exact comparisons, 32,848,416 values**.
- Forced scalar fallback: the same exact comparison count passes.
- Existing coverage includes row thresholds and four-row tails, K-panel and
  output-column boundaries, bias/no-bias, signed zeros, adversarial finite
  inputs, one-/six-partition callback runs, immutable inputs, and sentinels.
- Serial component/tiny CTest **35/35 pass**; separate loader **1/1 pass**.
- Full tiny parity: **28 cases, 164 teacher blocks, 164 free blocks, 28 finals,
  684 comparisons, zero failures**, worst normalized ratio `0.618082`.
- Actual forecast bytewise repeat/thread checks pass for all 28 tiny cases and
  real 512/32, with thread sequence 1, 6, 6, 1 against an initial one-thread run.
- Baseline archive/CLI and candidate base archive/patch/tested executable hashes
  pass their before/after checks. Source patch and binaries remain in scratch.

No full real aggregate replay or formal full-shape repeat-five gate runs.

## Matched warm diagnostic measurements

CPU: AMD Ryzen 5 7430U; pinned 209M F32 and unchanged benchmark input/settings.
All builds use Release and `-j2`. Heavy work runs serially under 12 GiB.
Each configuration uses a separate process with 20 seconds of cooldown and
no overlapping compilation or heavy test. Baseline runs precede candidate runs.
The 512 configurations use one warmup and five samples; 32k uses one warmup
and three samples, explicitly diagnostic rather than formal acceptance.

| Shape | Threads | Baseline median seconds | Candidate median seconds |
|---|---:|---:|---:|
| 512 / 32 | 1 | 2.393737794 | 2.386710391 |
| 512 / 32 | 6 | 1.398275376 | 1.088651652 |
| 32768 / 1024 | 1 | 67.567859548 | 81.612443247 |

Full-shape timed samples, seconds:

- Baseline: **86.392296779, 65.802881503, 67.567859548**.
- Candidate: **60.699248800, 81.612443247, 182.461716081**.

The candidate median is approximately **20.8% slower**. The wide sample range
and changing process CPU utilization expose substantial measurement variability.
The evidence does not establish its cause or isolate intrinsic kernel cost.
Do not discard the slow sample or select the fastest sample to claim a win.
The short-shape path does not activate packing; its noisy six-thread change
cannot establish benefit from the new microkernel.

Full-shape peak RSS: baseline **950906880 bytes**, candidate **950308864 bytes**.
Input/output sums match, but exact checks—not sums—establish focused parity.
No performance gate is relaxed. The runtime patch is reverted because no
end-to-end win is demonstrated, not because of a numerical failure.

## Artifacts and pins

Scratch directory relative to the work root: `timestral-native/m6c-row4`.

| Artifact | Meaning |
|---|---|
| `baseline-source-commit.txt`, `baseline-source.tar`, `baseline-cli` | Retained baseline source and executable |
| `candidate-source-base.tar`, `candidate-source.patch` | Reconstructable experimental source |
| `baseline-before.sha256`, `baseline-after.log` | Baseline archive and executable hash checks |
| `candidate-before.sha256`, `candidate-after.log` | Candidate source and tested executable hash checks |
| `baseline-disassembly.txt`, `candidate-disassembly.txt` | Structural disassembly inspection |
| `configure.log`, `build.log`, `fallback-build.log` | Release and fallback builds |
| `oracle.log`, `oracle-fallback.log` | Exact oracle results |
| `tiny-gates.log`, `loader.log`, `forecast-tiny.log` | Component and full tiny parity |
| `determinism-tiny.log`, `determinism-512.log` | Actual forecast byte comparisons |
| `baseline-512-{1,6}.json`, `candidate-512-{1,6}.json` | Warm diagnostic samples |
| `baseline-32k-1.json`, `candidate-32k-1.json` | Matched warm repeat-three samples |

```sh
# Each configuration runs separately with cooldowns and the memory guard.
"$BIN" bench --model "$MODEL" --context-len 512 --horizon 32 \
  --threads "$N" --warmup 1 --repeat 5
"$BIN" bench --model "$MODEL" --context-len 32768 --horizon 1024 \
  --threads 1 --warmup 1 --repeat 3
```

No next optimization, extra benchmark, or further profile runs after this decision.
Root review selects the next measured experiment. Full M6c remains on hold.
