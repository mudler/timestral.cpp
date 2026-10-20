# Four-key binary64 QK SIMD slice

## Scope

The candidate computes four independent key dots in AVX2 binary64 lanes without packing.
Each lane traverses head dimensions in ascending order using explicit multiply then add, followed by one F32 conversion.
There is no horizontal reduction, second accumulator per dot, eight-key unroll, persistent cache, or task-policy change.
Scalar tails and forced-scalar dispatch implement the same precision policy.
Runtime CPU checks select implementation only; non-AVX2 builds retain scalar behavior.
Scale/mask/max consume the resulting dots in original key order.
Softmax, attention values, reciprocal scaling, and frozen legacy callbacks remain unchanged.

The existing `-ffp-contract=off` applies to both scalar and SIMD production code.
F32 inputs convert before multiplication; exact-product representability does not authorize reassociation or FMA substitution.
Direct-dot tests distinguish ascending-double semantics from correctly rounded exact summation with order-sensitive cancellation.
They also use nonzero subnormal products, signed zeros, odd dimensions, strided rows, and 3/4/5 plus 7/8/9 key boundaries.
Full SDPA tests compare independent scalar oracles with default and forced-scalar dispatch, actual concurrent tasks 1/2/3/6, and bytewise input immutability.
Head counts include three and twelve; the complete frozen legacy oracle remains intact.

## Execution boundary

Code preparation does not establish compiled correctness or performance.
Build/test execution waits for the serialized heavy slot after the independent 5M corpus work.
Later screened benchmarks start with 512/32 before any long component or 32k execution.
## Correctness-verified WIP; performance UNDECIDED

The implementation remains retained pending isolated end-to-end evidence, not as a measured optimization win.
The legacy test passes 1296 packing checks and 9360 frozen-oracle comparisons unchanged.
The new precision test passes 240 direct-dot cases and 10,368 full-SDPA cases over 22,650,624 values.
Both default SIMD and forced-scalar paths match the independent oracle with concurrent task counts 1/2/3/6.
Direct tests check binary64 accumulator bits as well as final F32 bits, including nonzero subnormal products and order-sensitive cancellation.

A 34/34 component selection passes; the attention executables run separately.
Full tiny forecasting passes 28 cases and 684 comparisons, maximum ratio 0.618082.
The existing 5M pilot/multi/long_h artifacts pass strict M3 with 64 local layers and 136 comparisons, no missing layers.
Their strict M6b run passes three cases and 43 comparisons, maximum ratio 0.891812.
The 209M two-case subset passes 18 comparisons, maximum ratio 0.703131.
Tiny and 209M 512/32 thread/repeat determinism pass.
These remain explicit subsets, not full-model acceptance.

The final rebuild adds optional internal accumulator capture for testing and stronger tests; null capture leaves production arithmetic unchanged.
The independent full oracle reruns successfully after that addition.
The broader parity runs precede this observational test-seam addition; no repeated full suite is claimed afterward.

The 512/32 one-thread screen uses one warmup and five samples, ordered scalar/SIMD/scalar with cooldowns.
Medians are 3.767, 4.145, and 4.547 seconds: the candidate lies inside baseline drift.
The long QK component checks 13,381,632 dots byte-exact, but its timing is invalid as performance evidence.
It receives only 40% CPU amid load 64–68 and CPU-pressure averages around 85–87%.
No 32k or six-thread performance run follows this failed host screen.
No component speedup or end-to-end win is claimed.

Before-screen hashes exist for both executables and the model.
A later test-seam rebuild changes the candidate executable, so the final hash is not a matching after-screen pin.
The original screened candidate was not copied before that rebuild; do not claim a verified unchanged benchmark binary afterward.
The scalar executable and model hashes still match; retained timing records remain inconclusive regardless.

The final source archive SHA-256 is `2c190f1b25ac04474c0084b20912ad11386a2c3d10f68aabe409b0bbe0913646`.
The final inference archive SHA-256 is `2449a34a66977c0f75806134f9da48326b28e10d55a434b04de9dcb20b18133b`.
The final precision test SHA-256 is `969704e4ee4b995609eb96fffd2d70794004695b2485947ad67a5a0e2ffd634a`.
All execution stays serial under 12 GiB, builds at most two compiler jobs.
No performance win, M6c acceptance, full 5M gate, or full 209M gate is claimed.
