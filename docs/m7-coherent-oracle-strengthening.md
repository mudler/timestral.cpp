# Coherent SDPA oracle strengthening (scratch only)

## Independent structure and frozen candidate

The candidate remains the same scratch coherent binary64 callback; production dispatch is unchanged.
The new reference first materializes separate `[batch,head][token][channel]` long-double query/key/value matrices using explicit coordinate loops.
It uses separate dot helpers with pairwise long-double product summation, independently reduced softmax weights, and independently materialized value columns.
This differs from the candidate's flattened job indexing and ascending binary64 reductions.
Both use the same promoted reference F32 scale; the candidate retains separate multiply/add, ascending sums, direct division, and one final F32 rounding.
The reference intentionally uses a different, declared higher-precision reduction strategy; it is not a bitwise implementation-policy oracle.
Compile-time verification establishes 64 long-double mantissa bits versus 53 double bits.

The prior 720-case witness and logs remain retained unchanged.
This strengthening is new scratch evidence, not a replacement result assigned to those earlier runs.

## Coverage and finiteness

216 deterministic geometries cover multiple batches, three/twelve heads, odd dimensions, sequence tails, splits, causal masks, and finite additive masks.
Patterns include random operands, equal logits, extreme separation, value cancellation, nonzero subnormals, signed zero, and finite large operands.
Analytic controls verify one visible key, equal-logit means, exact signed cancellation, and nonzero subnormal outputs.
The oracle explicitly checks finite maximum, positive finite weight sum, finite unrounded long-double output, finite conditioning scale, finite difference, and finite bound.
NaN cannot silently evade an error comparison.
All candidate outputs are finite within the supported domain.

Actual concurrent worker calls at ith/nth partitions 1/2/3/6 reproduce serial candidate bytes exactly.
Bytewise memcmp checks preserve query/key/value/mask arrays including signed zeros; output sentinels remain intact.
The test compares current production mixed-boundary output separately and finds 62,250 differing output bits/values.
An assertion requires this mismatch, establishing that the old mixed binary64-QK/F32-rest implementation does not satisfy the coherent exact-output contract.
It is not a production-route red test for a future dispatch change, and it does not prove mixed arithmetic violates an accuracy bound.
Legacy F32, scalar binary64-QK/F32-rest, and coherent binary64 remain distinct policies.

## Measured errors

All 294,840 outputs match the F32-rounded materialized long-double reference: maximum ULP distance is zero on this corpus.
Maximum absolute error against the unrounded long-double result is `5.95930870201e-8`.

| Subset | Count | Maximum absolute error to unrounded oracle | Maximum F32 ULP distance |
|---|---:|---:|---:|
| Near zero, absolute oracle value <1e-6 | 32,797 | `4.67099488108e-46` | 0 |
| Rounded subnormal output | 24,498 | 0 | 0 |
| Strong value cancellation, absolute output <1e-6 of weighted absolute value | 4,374 | 0 | Not separately aggregated |

The conditioning scale is `sum(abs(weight*value))/sum(weight)` for each output, not max-with-one.
The checked diagnostic bound is final F32 rounding error plus `4096*(DBL_EPSILON+LDBL_EPSILON)*conditioning_scale` plus one smallest F32 subnormal.
This bound is not a reference-parity tolerance or a universal forward-error proof.
Maximum observed error/conditioning-scale is 1/3; tiny values and unavoidable F32 quantization make this ratio large even when rounded-oracle ULP distance is zero.
No universal correct-rounding or all-input bit equality is claimed from these finite cases.

## Input domain and native-contract qualification

This callback assumes contiguous canonical ggml `[head_dim,heads,sequence,rows]` tensors and the corresponding contiguous mask.
Supported witness inputs are finite Q/K/V, finite additive mask values or negative infinity, and at least one unmasked allowed key per query after the split.
Production encoder graph construction rejects fully masked boolean query rows; `attention_time` itself validates types/shapes/contiguity but does not independently reject all nonfinite tensor values.
Therefore the callback's witness domain is narrower than every possible direct internal API invocation.
A direct fully masked scratch call produces NaN through negative-infinity subtraction; no zero-output policy is invented.
Nonfinite Q/K/V and positive-infinity/NaN masks remain outside validated evidence and require an explicit policy before adoption.

## Integrity and outcome

Candidate header SHA-256 is `adfa225507a82df3e95b1d048b9a7a95d1b7dfa3e2ebb43ea92a7ae4fe23b8a9`.
Independent witness source SHA-256 is `ec3ad70b667925245dbb390fa9e0a4c20340bb373ecb80fbdfca0f51b281039b`.
Witness executable SHA-256 is `3052be89ccc5237e3726ef2ef0f10ad638ce9cc2214e914395dee88b9c871652`.
Original reviewed library, encoder executable, and model hashes remain unchanged.
All execution is serial under a 12 GiB address-space cap; no full forecast replay runs.
No production files, dispatch, public API, reference tensors, floors, tolerances, or dependencies change.
This strengthens the scratch accuracy evidence; production adoption and a production-route red/green test remain separate decisions.
