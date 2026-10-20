# Private BLAS multi prediction: patch-projection boundary

## Scope

This diagnostic links the retained private standalone-SDPA library and, separately, accepted e70af1d on the same immutable 5M multi positive block0 inputs.
It changes no runtime or precision contract and reads no reference array in production.
Reference-boundary interventions are scratch causal tests only.
All65 reference artifacts, model, candidate/accepted libraries, and forecast binaries retain matching before/after hashes.

## Failure versus first divergence

The actual private prediction reproduces normalized error `5.05560754e-6`, tolerance `4.55173739e-6`, ratio `1.11069842`.
Worst coordinate is row3 (batch1,variate0), step23, quantile1.
The six location and scale anchors are byte-exact.
Reference raw head through native monotone/unscale passes at ratio0.0479011572.

The first observed discrepancy is patch projection, not scaler feature formation:
- patch features: 0/12288 differences;
- native patch projection on those exact features:13066/36864 differences, max `1.52588e-5`;
- stored patch features through native projection produce the identical difference;
- assembled encoder input differs in13066/39168 values (query tokens remain unchanged).

## Encoder propagation

All eight local layers on each layer's stored input are byte-exact under the private SDPA callback.
Running the entire encoder from stored tf.in also yields byte-exact final out_norm.
Running from native patch projection introduces inherited drift from layer0 onward:
layer maxima are `4.57764e-5`, `6.10352e-5`, `0.000122070`, `0.000366211`, `0.000305176`, `0.000610352`, `0.000488281`, `0.00766754`.
These are each layer's worst coordinate, not one monotone signed-error trajectory.
Native final out_norm differs in38686/39168 values, maximum `2.52724e-5`.
The selected decoder head input differs in2220/2304 values, maximum `1.12504e-6`.
Thus passing local encoder gates does not remove error inherited from native projection.

## Head and prediction causal controls

| Source | Private candidate prediction ratio | Accepted mixed-policy ratio |
|---|---:|---:|
| Stored F32 encoder output -> native head64 | 0.283733375 PASS | 0.283733375 PASS |
| Native encoder on stored tf.in -> head64 | 0.283733375 PASS | 0.498050108 PASS |
| Native patch preparation/encoder -> head64 | 1.11069842 FAIL | 0.805987289 PASS |

Stored-encoder native head64 raw output differs in7348/8064 values, max `6.4373e-6`, but its complete prediction passes.
Legacy head on stored encoder input is raw-byte-exact and predicts at ratio0.0479011572.
Legacy head on native candidate encoder input still fails at ratio1.04127977.
Therefore head rounding contributes but changing head policy alone is not a supported solution.
The exact stored-input encoder control shows patch-projection drift is sufficient to create the private candidate's failed composition.
The accepted runtime's SDPA rounding partly cancels or redirects that inherited error; it is not evidence of universal superiority.

## Outcome and next dependency

The diagnostic stops at the first supported projection boundary.
It does not isolate hidden projection, activation, output projection, or patch residual contraction internally.
A next separately approved task can compare those operations on exact stored features with pinned reference arithmetic and an independent native contraction witness.
No generic projection correction, backend extension, arithmetic sweep, head switch, tolerance/floor change, case exclusion, or reference rewrite follows automatically.
The private integration and aggregate acceptance remain HOLD.

Evidence remains under the private prediction-diagnostic scratch directory, with actual source/executable pins and raw logs.
Execution is serial under12GiB; only small observational executables compile.
No full suite, 209M/API/performance/quantization task, proprietary inspection, redistribution, or push runs.
