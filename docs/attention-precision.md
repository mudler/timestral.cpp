# QK precision policy

The default attention QK dot sums binary64 products in ascending head-dimension order, then rounds once to F32.
This is an explicit arithmetic-contract change, not a bitwise optimization or a claim of universal MKL agreement.
All shapes, heads, models, context queries, and decoder queries use the same policy.
The first implementation is scalar and does not allocate or transpose key-packing buffers.
There is no persistent cache or model-specific branch.

Scale, mask, maximum, exponential, sum reduction, value contraction, and reciprocal multiplication retain their F32 boundaries.
Linear projections and attention value arithmetic remain unchanged.
Parity floors, tolerances, and reference artifacts remain unchanged.
Binary64 ascending summation is not universally correctly rounded; agreement with high-accuracy fsum is bounded evidence only.

The internal `sdpa_f32_legacy` callback and legacy key-packing probe preserve the previous F32 arithmetic for compatibility tests.
They are not public runtime precision options.
The complete frozen `test_attention` oracle remains unchanged and runs against that legacy callback/probe.
A separate `test_attention_precision` contains a full independent scalar oracle with only QK accumulation changed.
It checks masks, splits, tails, head dimensions, batches, multiple heads, sentinels, input immutability, and actual concurrent task partitions.
Cancellation, wide exponents, signed zeros, and subnormal inputs remain in coverage.
The separate 2190-dot fsum witness is accuracy evidence, not a replacement for full SDPA testing.

## Bounded implementation checks

The complete legacy test passes 1296 focused packing comparisons and 9360 exact attention comparisons over 22,248,000 values.
Its frozen oracle source differs only in the callback name used for the legacy test seam.
The independent precision test passes 4032 cases over 3,948,672 values with actual concurrent task counts 1, 2, 3, and 6.
These executables run separately from a 34/34 component selection that excludes both attention tests.
The 5M pilot passes all nine comparisons, both signs, refinement/free/final checks, and invariants; maximum ratio is 0.891812.
Full tiny forecasting passes 28 cases and 684 comparisons, maximum ratio 0.618082.
An initial tiny run accidentally retained the pilot case selection; its two-case output is not full coverage evidence.
The corrected run explicitly selects `all` and retains the partial log separately.
The 209M two-case subset passes 18 comparisons, maximum ratio 0.703131.
Tiny 28-case and 209M 512/32 determinism pass repeated thread counts 1 and 6.
No floors, tolerances, or reference tensors change.

The tested source archive SHA-256 is `17694ecf6c8d0f1bf415bb57a6e45476e0aec27722545692d1b8833f22c64ba1`.
The inference library SHA-256 is `01f058da5427e93a86d42ad90ea6c14929fe3c0843b5534d36a3ae52a15aa4c1`.
The forecast executable SHA-256 is `2b18ef4a3144cd775ca36063ceede73c45dc7c41eacd5ba0ec36ad84a87dee45`.
All 1962 vendor blobs match ggml `e705c5fed490514458bdd2eaddc43bd098fcce9b`.
Model/reference artifacts and forecast binary hashes remain unchanged before and after parity runs.
Builds use at most two compiler jobs; tests execute serially under a 12 GiB address-space cap.

SIMD, broader real-model acceptance, and performance measurements require separate review.
Passing the bounded 5M pilot and 209M subsets does not complete either model's full milestone gate.
