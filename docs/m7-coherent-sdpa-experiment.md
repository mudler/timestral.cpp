# Scratch coherent binary64 SDPA experiment

## Fixed candidate contract

This is one authorized scratch accuracy experiment, not production adoption or a candidate sweep.
QK products and ascending sums, scale multiplication, additive mask, maximum, subtraction, exp, weight sum, value products/sums, and normalization use binary64.
Each output rounds once to F32 after normalization.
The scale is the reference F32 `float(1/sqrt(double(head_dim)))` promoted to binary64, not an exact higher-precision inverse square root.
Normalization uses direct binary64 division by the ascending binary64 weight sum, not multiplication by a rounded reciprocal.
Compilation uses `-ffp-contract=off`; products and additions remain separate, without reassociation or FMA substitution.
No model/layer/case/geometry branch chooses candidate arithmetic.
No F32 intermediate exists after the input/scale loads and before the final output cast.

Tests use finite Q/K/V, finite additive masks or negative infinity, and at least one allowed key per query.
Fully masked/nonfinite inputs are outside this observational callback's tested contract, consistent with production graph validation rejecting fully masked queries.
Called directly with a fully masked row, this scratch callback produces NaN from negative-infinity subtraction; it does not define a zero-output fallback.
The callback assumes the current contiguous canonical ggml q/k/v layout `[head_dim,heads,sequence,rows]` and matching contiguous mask.
It does not claim arbitrary-stride support; independent tests exercise multiple batches and head counts within that layout.
The experiment makes no new exceptional-input policy claim.

## Cross-precision accuracy witness

The witness recomputes the full formula with long double and the same promoted F32 scale.
Its coordinate/indexing and traversal largely mirror the candidate; it is a cross-precision check, not strong independent indexing/reduction validation.
A compile-time check proves long-double mantissa width exceeds double: 64 versus 53 bits on this host.
720 adversarial/random geometries cover 1/3/7/8/17/33/64/65 dimensions, vector/tail lengths, batches, three/twelve heads, and context splits.
Patterns cover causal and finite additive masks, equal logits, extreme separation, value cancellation, subnormals, signed zeros, and finite F32 operands whose products exceed F32 range.
All 1,218,888 output comparisons satisfy the diagnostic accuracy bound `4*F32_epsilon*max(1,abs(oracle_output))`.
Maximum observed bound fraction is 0.222223.
This is not a parity-tolerance change, universal correct-rounding proof, or bitwise-long-double requirement.
The max-with-one bound is loose near zero, cancellation, and subnormal outputs; it is insufficient as a production accuracy contract.
No absolute/ULP distribution or condition-scaled error claim is inferred from the retained aggregate bound.
Before production acceptance, use independently structured coordinate access/materialized head rows and dot/reduction helpers, plus absolute and ULP reporting near cancellation/subnormal cases.
Input immutability, output sentinels, and finite output checks pass.

## Same failing layer: retained red and coherent green

The original production local layer witness remains red at ratio 1.04752799085.
The scratch coherent SDPA on exact stored layer-5 inputs passes standalone at ratio 0.500647764872.
Supplying that computed output to native downstream layer operations passes composition at ratio 0.130940998856.
The old diagnostic label `DIRECT-reference-input-new` denotes the coherent scratch callback in this run, not the production binary64-QK-only implementation.
No reference output is chosen as a candidate result.

## Bounded runtime controls

An isolated archived source tree replaces only the production SDPA callback dispatch with the same generic scratch callback.
It links neither Torch nor MKL; all other model arithmetic remains unchanged.
Both signs execute through unchanged strict tests.

| Control | M3 | M6b |
|---|---|---|
| 5M uni_one_patch | 16 local layers, 34 comparisons, missing0 | 1 case, 2 teacher/free blocks, 1 final, 9 comparisons; max ratio 0.739586; failed0 |
| 209M uni_one_patch + uni_short | 96 local layers, 196 comparisons, missing0; max ratio 0.978681 | 2 cases, 4 teacher/free blocks, 2 finals, 18 comparisons; max ratio 0.700757; failed0 |

Amplification diagnostics remain visible; these are explicit subsets, not a full replay or full 5M acceptance.
No golden inputs, floors, tolerances, or reference artifacts change.

## Pins, limits, and next decision

The scratch runtime source archive SHA-256 is `789617effcd06539a8914313d805248f4ae0369c94ed237d038ccfd0f0fe8107`.
Its inference library SHA-256 is `97a23e72fed4d439f455676071019af88ebdd9b37ac328acaed85554bb398c22`.
Its forecast executable SHA-256 is `e3a63b2c13700add93f29973c60ba006f0c395c5d4e43e08d2bfce2e283f9084`.
All 1962 ggml blobs match the pin.
All 90 canonical/extended reference identities, original runtime/library/model pins remain unchanged before/after.
Scratch witness sources/executables are separately hash-pinned.
Builds use at most two compiler jobs; execution is serial under 12 GiB.

The coherent contract is promising on these witnesses and satisfies both local and composition metrics here.
It does not establish full-model parity, performance, exceptional-input semantics, or a production precision policy.
Root review must choose whether to authorize stronger independent oracle validation, broader unchanged gates, and a production TDD contract.
No production runtime file, dependency, public API, reference, floor, or tolerance is modified by this experiment.
