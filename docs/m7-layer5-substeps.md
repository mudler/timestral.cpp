# Layer-5 SDPA substep diagnostic addendum

## Boundary and reference validation

This diagnostic uses immutable 209M `uni_one_patch`, positive block 0, layer 5 q/k after RoPE, v, and masks.
Context calls have 8 queries x 8 keys; decoder calls have 1 query x 9 keys.
Both use 12 heads and head dimension 64; token leading dimension is 768.
The decoder query leading dimension normalizes to 64 for its single query.
Fresh pinned Torch flash calls reproduce all stored outputs exactly: context 6144/6144, decoder 768/768.
Torch is 2.14.1+cpu at `5c4886908584029761b579af026dcfb627c84070`, AVX2.
Actual supported verbosity identifies oneMKL SGEMM with Dyn0, NThr1, FastMM1.
No generic `torch.softmax` or forced math SDPA result is treated as shipped flash output.

## Source operation order

Pinned CPU flash calls QK `cpublas::gemm(Transpose,NoTranspose)` with `(keys,queries,head_dim)` and alpha1/beta0.
It applies the F32 scale after the dot, then the additive mask; this head dimension gives exact scale 0.125.
It computes row maximum, F32 shift, vector `exp_u20` over complete eight-key chunks, and scalar exp on the remaining key.
The vector sum reduces with the AVX2 shuffle tree; scalar tails join afterward.
Values use `gemm(NoTranspose,NoTranspose)` with `(head_dim,queries,keys)` before final reciprocal multiplication.
One key chunk covers these shapes, so online inter-chunk max/rescaling does not participate.
Native score scaling/masking and these high-level boundaries follow the same order.

The first known numerical divergence is QK accumulation: native ascending binary64 versus reference F32 backend accumulation.
Context dots differ in 404/768; decoder dots differ in 51/108, maximum absolute difference `0.0001220703125` in each region.
The backend witness calls exported `sgemm_` with actual observed flags/dimensions/strides, not a replacement production dependency.
The exact proprietary microkernel's accumulation order is not derived or copied.

## One-variable causal controls

All modes retain original masks, scale, max, exp/sum, and reciprocal stages unless explicitly stated.

| Mode | QK source | Value contraction | Context output differences | Decoder output differences |
|---|---|---|---|---|
| 0 | Native ascending binary64 | Native F32 sequential FMA | 2581/6144; max `1.3113e-5` | 456/768; max `1.57356e-5` |
| 1 | Actual reference SGEMM | Native F32 sequential FMA | 0/6144 | 136/768; max `4.76837e-7` |
| 2 | Actual reference SGEMM | Actual reference SGEMM | 0/6144 | 3/768; max `1.19209e-7` |

Mode0 matches the production SDPA discrepancy exactly, including all 3037 differing values.
Replacing QK alone is sufficient to make context outputs exact and to pass the unchanged composed layer gate.
The decoder contains a separate value-contraction rounding contribution: mode2 changes only that contraction relative to mode1.
The remaining three decoder differences are not assigned to a unique internal operation by this evidence.

Installed matching Torch headers compile directly for AVX2 to evaluate `exp_u20` and `vec_reduce_all` on the reconstructed scaled logits.
That source-derived softmax witness leaves the same three decoder residuals when used with both backend GEMMs.
A separate scalar-tail promotion control also leaves those residuals unchanged.
These controls reject a simple polynomial/reduction or float-versus-double-tail explanation for this witness.
They do not expose internal flash buffers or prove all normalization intermediates are byte-identical.
Scale/mask/max/shift/exp/sum values remain reconstructed from exact stored inputs and observed GEMM outputs, not injected golden intermediates.

## Composition, accuracy, and compatibility are distinct

| Mode | Standalone SDPA ratio | Composed local-layer ratio |
|---|---:|---:|
| 0 | 0.693811075728 PASS | 1.04752799085 FAIL |
| 1 | 0.0210245780524 PASS | 0.130940998856 PASS |
| 2 | 0.00525614451309 PASS | 0.032735249714 PASS |

Composition executes native gate/output/MLP/residual operations after the supplied observational SDPA boundary.
It does not alter production or relax either metric.
The closer reference-backend path can satisfy both metrics on this witness; that is not proof for all model shapes.
The current binary64 implementation matches its independent precision-policy oracle, yet this layer still fails reference composition.
Higher mathematical accuracy and reference compatibility are different contracts.
No further uniform-higher-precision candidate runs here, so no claim is made that it would resolve both gates.

## Next generic proposal and unresolved work

A reference-compatible CPU flash proposal needs a shape-generic, independently derived contraction implementation or a separately approved backend strategy.
It must establish QK and value contraction rules across context/decoder sizes, strides, tails, and architectures, without model/layer/case selection or proprietary code copying.
The three decoder residuals need captured/instrumented reference normalization intermediates if exact reference matching becomes the requirement.
Alternatively, a coherent higher-precision contract needs a separately approved experiment and independent accuracy oracle, then the unchanged local/composed gates.
This diagnostic does not authorize either implementation, a backend dependency, or another blanket precision change.

## Integrity and outcome

All 90 canonical/extended artifact hashes, model, reviewed library, and encoder executable match before/after.
Source/executable/libtorch/header witness hashes remain retained separately.
Only small scratch witnesses compile; execution is serial under 12 GiB with no full replay.
No runtime, reference, floor, tolerance, or public API changes occur.
The complete M6b15/159 pass remains separate; aggregate acceptance remains HOLD at the strict local M3 blocker.
