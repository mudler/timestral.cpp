# Private context-patch addmm experiment

## Scope

This isolated experiment extends the diagnosis in [the bias report](m7-patch-bias-diagnostic.md).
It does not adopt a backend or change the product runtime.
The private source applies one policy uniformly to all three F32 contractions in `context_patch_projection`.
It pre-fills bias and calls documented SGEMM with beta1; bias-free calls use beta0.
It preserves SiLU, the final residual addition, global affine operations, binary64 heads, and reduced-weight routes.
The default native product backend remains unchanged.

The backend uses MKL LP64 sequential APIs without Python or libtorch at inference.
Its reported numeric version is 2024.0 update2, build20240605.
The experiment does not inspect, interpose on, modify, or redistribute proprietary implementation code.

## Qualification

The retained actual-route RED returns1: 13066/36864 final values differ from Torch.
The corrected actual route returns0 with 0/36864 differing values.
At the same aligned C address, beta0 plus separate bias differs at16716/36864 output values.
Bias prefill plus beta1 differs at0/36864 values; external scratch guards remain intact.
This closes the previous allocation-address confound for this witness.

The policy corpus passes2160 configurations and rejects five invalid dimensions or overflowing products.
It covers both model widths, K63/64/65/191/192/193, model contractions, M/N tails, and singleton dimensions.
It covers bias/no bias, cancellation, signed zero, subnormals, guards, and immutable inputs.
The row-major control is another API formulation of the same backend, not an independent numerical oracle.
The independent long-double check establishes an error bound, not correct rounding.

The actual route matches36864 reference values with graph thread counts1,2,3,6.
Each contraction retains its whole GEMM geometry; only worker0 writes its output.
Scratch guards pass within each contraction.

The witness retains legacy reconstruction labels such as `complete-native-vs-Torch`.
Those labels describe the frozen old chain, not the corrected actual route.
Their13066 differences are expected controls, not failed acceptance comparisons.

## Completed gates

Every row below passes. Model cases run serially, one invocation per case, to stop at the first genuine failure.

| Gate | Actual coverage | Maximum error/tolerance |
| --- | --- | --- |
| Original 5M multi forecast | 1 case, 2 teacher blocks, 2 free blocks, 1 final, 9 comparisons | 0.283733 |
| Component CTest selection | 34/34 tests | Not aggregated |
| Tiny forecasts | 28 cases, 164 teacher blocks, 164 free blocks, 28 finals, 684 comparisons | 0.932042 |
| Tiny determinism | 28 forecasts, repeated threads1,6,6,1 | Byte-exact |
| Full 5M M3 | 13 cases, 30 selections, 240 local layers, 510 comparisons, no missing layers | 0.649274 |
| Full 5M M6b | 13 cases, 32 teacher blocks, 32 free blocks, 13 finals, 141 comparisons | 0.552672 |
| Supplemental 209M M3 | 8 cases, 16 selections, 384 local layers, 784 comparisons, no missing layers | 0.5 |
| Bounded 209M M6b | 3 cases, 6 teacher blocks, 6 free blocks, 3 finals, 27 comparisons | 0.49984 |
| Real 512/32 determinism | Repeated threads1,6,6,1 | Byte-exact |

The supplemental canonical cases are `uni_short`, `uni_one_patch`, `nonneg`, `const`, and `all_missing`.
The extended cases are `multi`, `known_cov`, and `nans`.
The bounded forecast cases are `uni_short`, `known_cov`, and `all_missing`.
The34 component tests exclude loader tests, real tests, forecast parity, determinism, C API, anchors, and standalone precision/oracle executables.
Thus34 does not restate the earlier36-test selection or claim fresh standalone legacy-oracle coverage.
Tiny parity and determinism run separately, as listed above.
No negative-control diagnostic differences enter these acceptance counts.

## Execution corrections and identity

Two compile failures precede qualification: missing ggml declarations and missing `<algorithm>` in the witness.
Their failed logs remain available privately.
The first forecast invocation stops before parity at its threads2 invariant with `patch partition`.
Public ggml source passes graph-wide `nth` to custom callbacks despite a one-task request.
The correction accepts valid workers but executes the entire contraction only on worker0.
It changes no arithmetic policy or matrix geometry.

The concurrent witness source changes while the correction builds, before its witness compilation begins.
The resulting log explicitly passes threads1,2,3,6 before model gates start.
A separate post-gate qualification rerun reproduces those passes with identical source and executable hashes.
No runtime source changes during the completed model gates.
A raw qualification-manifest comparison fails because one manifest uses absolute paths and the other uses relative paths.
Normalized path-to-hash maps match for all four entries; no artifact changes.

The candidate archive SHA-256 is:
`da86cb1d100481d8b15ed67a0d44a1076656b0650d49225a67b74a9c1f30b1cc`.
The base archive SHA-256 is:
`56822c9bbeab348e185e06cc2a82a8ef9a9563c7c784945d161e43b35d543b7b`.
The unchanged SDPA callback SHA-256 is:
`7e291781b7f3cc7efd2d4d9664bf1adcfc3a5bfc008245bca2f234bb6b8dcb19`.
The tested forecast executable SHA-256 is:
`e4eb8d985ab217a0e96a311620c60a98a3f90ef0333fe02c5b687bc45e9830b7`.
The tested encoder executable SHA-256 is:
`cafcfb1f246aff487f739f59450c83729ac287e3902f440fce43b9a0ba0fb4f8`.

Before/after manifests match2533 entries, including all90+65 reference artifacts, both models, source/vendor files, backend files, and test executables.
The source inventory includes1962 vendor files within2146 source files.
The backend inventory includes41 runtime files and158 header files.
The manifest also includes30 built library/test files.
The candidate archive and four qualification source/executable pins also pass closing checks.
The observed maximum resident set across timed gates is1990928KiB.
The runs use a hard12GiB virtual-memory ceiling and builds use `-j2`.
No reference, floor, or tolerance changes occur.

## Limits and next decision

The private callback relies on its sole valid loader/graph caller.
It does not independently validate every bias dtype/shape/layout/data pointer, input K, weight rank, output extent/type, positive row dimension, or null destination.
Production integration or generic reuse requires a separate validation review before adoption.
This evidence does not establish full209M15 forecast acceptance, API acceptance, performance, quantization quality, or a redistribution decision.
The owner retains the integration decision; the private experiment alone changes no release status.
