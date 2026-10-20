# Private standalone SDPA integration: stopped at 5M prediction parity

## Isolation and runtime

The study archives product `35f2fcd` and pinned ggml, then modifies only that private source tree for standalone callback dispatch/linkage and explicit retained mixed-policy test routing.
The callback bytes match approved SHA-256 `7e291781b7f3cc7efd2d4d9664bf1adcfc3a5bfc008245bca2f234bb6b8dcb19`.
Canonical production shape/type/contiguity/mask validation remains in the archived graph builder.
Private linkage fails explicitly when unavailable; no numerical fallback is selected.
No product runtime implementation, default backend, reference, floor, or tolerance changes.
The adapted public-source callback retains scoped attribution and the complete pinned PyTorch licence in the private source tree.

## Completed checks

The final-built callback passes 88 concurrent mechanics checks over266,044 values, including S1/head_dim1 and dimension-degenerate shapes.
Six malformed preallocation cases reject; sixteen masked-chunk partition checks pass.
Query/key tiling boundaries, aligned guards, original ldc, masks, byte immutability, and actual ith/nth1/2/3/6 partitions remain covered.
Legacy F32 and mixed-QK64 scalar/SIMD oracles pass through explicit retained internal seams; they are not mislabeled as active standalone arithmetic tests.
The component selection passes36/36.
Full tiny forecasting passes28 cases/684 comparisons, failed0, maximum ratio0.932042.
Tiny28-case repeated/threaded determinism passes.

The full approved13-case 5M M3 profile passes30 selections,240 local layers,510 comparisons,missing0.
Maximum ratio is0.649274; amplification diagnostics remain visible.
This establishes the layer gates for this private integration, not full end-to-end parity.

## Genuine M6b failure and stop

The next13-case 5M forecast command fails `multi`, positive block0, teacher-forced prediction `G6.2 floor_tf predict pos.b0`.
It is a numerical comparison failure, not an input exception or missing artifact.
The block has six packed rows, context1024, two forecast patches.
The unchanged normalized-v1 floor is `2.2758687e-6`, error `5.05560754e-6`, tolerance `4.55173739e-6`, ratio `1.11069842`.
No causal head-versus-encoder attribution is made from this gate alone.

The test executable catches per-case failures and continues its own remaining cases, then exits1.
Its final summary is12 successful cases,30 teacher/free blocks,12 finals,132 successful comparisons,failed1 with the full13-name selection.
The reported maximum0.845242 covers successful comparisons and must not be presented as the overall maximum below1.
The failing multi comparison is not counted as a successful comparison; remaining comparisons inside that case do not execute.
The outer script stops at exit1 and launches no further model command.
Supplemental209M eight-case M3, bounded209M forecasts, and real512 determinism do not run.
No full209M replay, API, quantization-quality, or performance task runs.

## Pins and outcome

Private source archive SHA-256: `56822c9bbeab348e185e06cc2a82a8ef9a9563c7c784945d161e43b35d543b7b`.
Private inference library SHA-256: `f15e32bab244160626eda374ad99cd3b6ff79d3e43c0d8961f1d66b9bfa593f7`.
Private forecast executable SHA-256: `e010f8eb5848afc0be738301036020486fc2e406743646974585236a17d957b6`.
All1962 ggml blobs match the pin; the complete private backend/header inventory remains unchanged.
Loaded-object mechanics enumeration contains private LP64/sequential/core/default/VML-default plus system libraries, not Python/libtorch/OpenMP/TBB.
All90 canonical/extended reference hashes, all65 5M references, and both model hashes match before/after.
Execution remains serial under12GiB, builds at most two compiler jobs.

The private integration remains HOLD at end-to-end prediction parity.
No retry, arithmetic tuning, case exclusion, output selection, production adoption, proprietary inspection, redistribution, or push follows.
A future task requires a separately approved causal boundary diagnosis using this retained failing witness.
