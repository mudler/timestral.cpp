# 5M all-missing final parity diagnostic

## Status and scope

The 13-case reference corpus exists, but its existence does not establish second-model acceptance.
The current `all_missing` case passes eight teacher/free prediction/refinement comparisons and fails its final comparison.
No runtime, reference, floor, tolerance, or precision policy changes accompany this diagnostic.
All observations use the corrected reviewed library, immutable five-mode case artifacts, and the existing strict metric.
The full tensor retains both rows and its original positive anchor/RMS/derivative normalization.

## Final failure and postprocessing

Final normalized error is `2.08006087e-5`, tolerance `1.04900727e-5`, ratio `1.9828851`.
The worst element is fully missing row 1, step 2, quantile index 7.
Native value is `-8.532032371e-5`; reference is `-8.531101048e-5`.
Observed row 0 has maximum normalized error `2.063434635e-7`, below the unchanged tolerance.
The missing row anchor is location zero and scale `1e-6`.

The case has one patch and one block, so refinement is a bypass and no autoregressive feedback occurs.
Averaging stored positive and negative raw passes reproduces final reference bytes exactly: 0/1344 differences.
Reconstructing native caps reproduces native raw-pass bytes exactly for both signs.
The missing-row cap is `[-0.009999999776, 0.009999999776]`, matching the native formula.
Averaging native raw passes reproduces native final bytes exactly: 0/1344 differences.
Neither row is eligible for nonnegative clipping; the missing row has no observations.
These results do not support a cap, averaging, clipping, row-map, or quantile-flip fix.

Mixed stored/native sign passes amplify the discrepancy to ratio 93.0428931 near an exactly cancelling median.
That is cancellation evidence, not an alternative gate or a supported arithmetic fix.

## Upstream causal isolation

The source labels below apply independently to both signs before the same cap/average stages.

| Source | Native head input | Combined final ratio |
|---|---|---:|
| 0 | Stored F32 encoder output | 1.9828851 FAIL |
| 1 | Native encoder on stored `tf.in` | 1.9828851 FAIL |
| 2 | Native encoder on native preparation | 1.9828851 FAIL |
| 3 | Stored raw head output through native monotone/unscale | 0.0026314985 PASS |

Thus native raw-head rounding on exact stored encoder inputs is sufficient to reproduce the final failure.
It is not necessary to change preparation, encoder, SDPA, or final postprocessing to explain this witness.
The stored missing-row head inputs are byte-identical between signs.
For source 0, native raw head differs on missing row 1 in 583/672 values for each sign, maximum `7.62939e-5`.
Observed-row raw differences are 628/672 positive and 626/672 negative, with maxima `4.52995e-6` and `5.24521e-6`.

## Independent head arithmetic witness

Fresh pinned Torch `linear -> SiLU -> linear + residual` calls on exact stored head inputs reproduce all stored raw head bytes.
Both signs match 1344/1344 values.
A separate scalar C++ reconstruction retains current ascending F32 FMA within 192-wide panels, separate panel additions, and existing activation boundaries.
It reproduces the native `Head::project` output exactly: 1344/1344 values for each sign.
This establishes the discrepancy as the current contraction arithmetic versus the pinned reference on this execution shape, not an indexing or SIMD-only bug.

| Teacher-forced projection | Positive differing / total; max absolute | Negative differing / total; max absolute |
|---|---|---|
| Hidden, K192 | 1371/1536; `4.76837e-6` | 1349/1536; `4.76837e-6` |
| Output on stored Torch activation, K768 | 1203/1344; `8.39233e-5` | 1217/1344; `8.39233e-5` |
| Residual, K192 | 1118/1344; `4.29153e-6` | 1122/1344; `4.29153e-6` |

The complete scalar-policy head differs from stored raw output in 1211/1344 positive and 1209/1344 negative values, maximum `7.62939e-5`.
The reference's exact internal GEMM accumulation order is not established by these results.
No model-specific branch, proprietary kernel copy, best-match reduction heuristic, or new precision policy is proposed as proven.
A future correction requires a separately approved shape-generic arithmetic contract and regression plan.

## Evidence integrity and limitations

The before/after manifest covers 68 files: reviewed library, forecast executable, model, and all 65 reference artifacts.
All hashes match; no golden artifact or floor is rewritten.
Scratch witness sources/executables and actual logs remain hash-pinned separately.
Execution is serial under a 12 GiB address-space cap; only small diagnostic witnesses compile.

Two harness-only compile errors and one missing-key abort remain retained.
The first mixed `auto` declaration combined float/int8 vectors; the later label call supplied a string to a C-string parameter.
The abort requested `pos.b0.cap.in`; header inventory established the actual pass-level key `pos.cap.in` (and negative equivalent).
Corrected runs preserve the production assertions and complete the requested boundary comparisons.
These harness failures are not numerical gate results.

The diagnostic stops at the supported raw-head arithmetic boundary.
The 5M prerequisite and full 209M aggregate replay remain HOLD pending root review.
No broader inference, remaining-case generation, model conversion, performance benchmark, or production fix follows automatically.
