# Known-row reduction precision

`instance_norm_stats` follows the pinned Torch contiguous F32 row-sum operation order.
Weighted values, mask counts, and centered squares use the four-level cascade helper in `src/torch_sum.hpp`.
The helper adapts `multi_row_sum`, `row_sum`, and `vectorized_inner_sum` from Torch `5c4886908584029761b579af026dcfb627c84070`.
Its exact file coverage and retained licence are in `third_party/notices/README.md` and `PyTorch.txt`.
No NumPy float64 pairwise reducer is used for these Torch operations.

This correction replaces widened sequential sums, not the surrounding F32 operation boundaries.
Centering, masking, squaring, division, sqrt, minimum scale, binary detection, and affine/asinh remain unchanged.
Positive 0/1 indicators retain their bypass; mirrored 0/-1 rows correctly use normal instance statistics.
Causal float64 prefix scaling, QK/value arithmetic, reference artifacts, floors, and tolerances remain unchanged.

The regression first fails against the pre-correction library on the length-864 mirrored indicator scale.
Independent actual-Torch sum witnesses cover zero/short/vector/tail/cascade-promotion lengths, mixed exponents, cancellation, signed zeros, subnormals, and masks.
No length-specific model branch or reference-file dependency is introduced into production.
## Verified bounded correction

The first standalone regression mistakenly paired a contiguous -1 block with the expected scale of the canonical alternating five-step indicator.
Actual Torch on that different layout gives scale bits `0x3effff4d`; equal counts do not imply equal F32 reduction order.
The corrected fixture uses the canonical five-step pattern and still fails against the old library before the production patch.
Both failed logs remain retained; equality is not relaxed.
The corrected regression passes after the patch.

The actual pinned Torch cross-check passes 126 witnesses over 2,524,182 values.
Lengths cover 0/1/7/8/9, 31/32/33, vector/tail boundaries, 511/512/513, and larger cascade promotions through 131,073.
These are bounded AVX2 contiguous-row witnesses, not a universal all-platform claim.
The source-derived operation order and full upstream pin are documented in the helper and notice file.

The unchanged `known_cov` artifacts pass M3: 16 local layers, 34 comparisons, no missing layers.
Its complete M6b passes nine comparisons with maximum ratio 0.954023, without reference-scaled injection.
The five other existing 5M cases pass M3 with 96 local layers and 204 comparisons, and M6b with 61 comparisons.
Amplification warnings remain visible; these are six-case subsets, not the complete 13-case prerequisite.
The component selection passes 35/35; full tiny forecasting passes 28 cases and 684 comparisons.
The 209M known_cov/uni_short/uni_one_patch subset passes 27 comparisons, maximum ratio 0.703131.
Tiny and real 512/32 repeated/threaded determinism pass.
Separate legacy and new attention oracles pass unchanged.
Causal-prefix source is byte-identical; QK/value sources are unchanged.

Final tested source archive SHA-256: `a9bde69ff91c57465adaf2e4a746338b8f2468d04b58d8897432505f5d9f7976`.
Inference library SHA-256: `a19bc76277604c36618059f376f86c9c6e881f0662c203560c389b0dad9ee6bd`.
Forecast executable SHA-256: `0cf65b261bd51f70398409851a8c7181c455ca5f1eeca735e4747540d36d703f`.
All 1962 ggml vendor blobs match the pin; model/reference artifacts and forecast executable hashes remain unchanged before/after gates.
Builds use at most two compiler jobs; tests run serially under 12 GiB.
Broader model gates and performance remain separate from this operation-order correction.
