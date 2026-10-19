# Parity notes

Measured facts about the parity gates. Tolerances are in `docs/PLAN.md`, section D; this file
records what was measured. It does not change a gate.

## G0.3: stored rope tables vs the tables the reference uses at run time

Preview made in M0b by `tests/python/check_convert.py` (the full gate is M0d). It runs the
reference `SinCosRoPE.forward` on run-time-like layouts and compares every value with the
stored table row of its position.

- Layouts: R rows in {1, 2, 3, 6, 17}; N patches in {1, 2, 3, 5, 8, 33, 100, and the largest
  position count the model can reach, capped at 1056}; three random `first_obs` draws per
  pair; positions `p - first_obs`, clamped to the table range.
- Timestral-209M (head_dim 64): 13,452,288 values compared, 0 differ in value (max 0 ulp),
  1,786,880 differ only in the sign of a zero.
- Tiny checkpoint (head_dim 8): 573,504 values, 0 differ in value, 88,500 differ only in the
  sign of a zero.
- An independent review scan found 0 value differences in 176,748,160 values; every
  difference was a signed zero.

The sign-of-zero differences are in the dims with `inv_freq = 0` (the identity rotation), where
`sin` is `+0` or `-0` depending on the sign of the position and on the layout. They are
counted apart from value differences, as the plan's D.1 requires.

An earlier, unconventional scan of the tiny checkpoint (one position per row, very few patches)
showed many bit differences. They were counted by raw bit pattern, so they were these same
signed zeros, not value differences. The gate is unchanged.

## Floor computation and publication

`compute_floors.py --dir DIR` processes every discovered f32 case in that directory.
It does not prove that the canonical suite is complete. Use `check_baseline.py` for
full milestone acceptance. Repeated `--case NAME` options select a partial run;
an unknown or missing selected case is an error. A partial run replaces the summary
with the selected cases only.

Every selected numerical case requires all five modes. The shared contract checks
case/model/source metadata, required records, shapes, precision, statistics,
reproducibility, and the f64 rerun self-check before publication. Exception artifacts
cannot supply numerical floors. Companion source identifiers must be present and agree;
this validates recorded provenance, not the authenticity of the source installation.

The C.4 denominators are always f64 and f64-rerun respectively. Both companion
numerators are mandatory. Required rerun outputs always receive `floor_tf` metadata.
For refined blocks, the G6.2 floor uses `block_refined`, not isolated `refined`.
A zero reference RMS produces zero only for zero error; a mismatch is an error.
Nonfinite evidence, statistics, floors, and caps are errors. Raw context copies may
retain NaNs only at matching comparison positions. Infinities are not accepted.

Caps remain warning-only. The gate tolerance remains `max(2 * floor, 16 * u)`.
Missing required statistics are errors, never `n/a`. Summary entries count cap
applicability separately as `bounded`, `copy`, or `na`; a zero bounded cap is not a copy.
The existing explicit exceptions in `EXTRA_NOCAP` cover raw copies (`input`, `args`,
`assemble`, `scale.in`, `refine.context`, `cap.rows`) and host/intermediate values
without a declared bound (normalization inputs, scaling, frames, refinement helpers,
masks, positions, rotary tables, and monotone outputs). Unmatched float names fail.

Final raw caps resolve through the matching `pos.b0` or `neg.b0` prefix.
For single-block `final.quantiles`, the bound is the larger of the two mirrored
single-block bounds. Final outputs with two or more blocks have no free-running cap.
Per-block `predict` and `refined` after block zero also have no free-running cap.
SDPA uses the stored key tensor's sequence axis for `S`, including variate attention.
For split time attention, the full key sequence bounds both the encoder prefix and
the decoder's full-key call; concatenated query length is not the definition of `S`.

The command preflights every selected case before writing any f32 metadata.
It reads cases again during publication to avoid retaining the whole suite in memory.
Do not run concurrent generators or writers against the directory. Concurrent input
changes between preflight and publication are outside this guarantee.

Each GGUF and JSON write uses a sibling temporary file, closes or flushes it,
fsyncs its contents, and replaces the destination atomically. Failure before replacement
preserves the original bytes and removes the temporary file. Metadata updates preserve
all stored tensor values and dtypes. Tests use generated synthetic files only.

This is not a directory transaction. An I/O failure or crash can leave some cases
updated and others unchanged, or leave an older summary. Abrupt process termination
can leave sibling temporary files. The code does not fsync the parent directory, so
power-loss durability of the rename depends on the filesystem. Repeat the command
after resolving the failure; no partially written destination file is published.

## Prescribed math probe

`scripts/probe_math.py` compares torch CPU operations at one thread against real
libm float functions loaded through `ctypes.util.find_library('m')`. Missing
libraries or symbols cause an error; there is no Python or NumPy math fallback.
Every C argument and result uses `c_float`. The probe needs no model or baseline.

```bash
$VENV/bin/python scripts/probe_math.py --n 1000 --out "$TIMESTRAL_WORK/probe-small.json"
# Full measurement, after independent review:
$VENV/bin/python scripts/probe_math.py --n 1000000 --out "$TIMESTRAL_WORK/probe.json"
```

`--n` accepts integers from 8 through 10,000,000, inclusive, with a default of
1,000,000 per function. These limits retain endpoints and prevent accidental
unbounded allocation. The output directory must exist. The shared atomic JSON
helper preserves an existing report if serialization or replacement fails.

### Sampling and comparison contract

Each function starts a NumPy `PCG64` generator with seed `20261020`.
The probe constructs one float32 input array and feeds its exact values to both
backends. There are no additional special inputs in the measured ranges.

- `asinh`: `k = n // 2` signed logarithmic samples followed by `n - k` uniform
  samples in `[-10, 10)`. The negative logarithmic count is `k // 2`; the positive
  count is `k - k // 2`. Each sign uses `geomspace(1e-30, 1e7, count)`, including
  both magnitude endpoints. Negative samples precede positive samples.
  Before the float32 cast, the first three uniform samples become `-10`, `0`, and `10`.
- `sinh`: uniform samples in `[-80, 80)`.
- `sin` and `cos`: uniform samples in `[-1100, 1100)`.
- `softplus`: uniform samples in `[-100, 30)`.

For the last four functions, the probe casts uniform samples to float32, then
replaces the first three with the lower endpoint, zero, and upper endpoint.
Endpoint replacement does not increase the sample count. Float32 rounding can
produce the upper endpoint from a uniform draw.

Torch softplus uses its default `beta=1` and `threshold=20`: it returns `x` when
`x > 20`, otherwise its softplus implementation. Libm uses literal
`log1pf(expf(x))`, with float32 rounding after each C call. The probe does not
replace this formula with the torch threshold shortcut. Formula and implementation
differences both contribute to the measured histogram.

Finite ULP distance uses ordered float32 bits with both zeros at one position.
Negative numbers, subnormals, and distances across zero retain their numerical
order. `finite_histogram` contains only pairs with two finite outputs.
`special_counts` partitions all other pairs into matching NaNs, same-sign
infinities, and nonfinite mismatches. Histogram counts plus special counts equal
`total_count`, which equals `n`. Matching NaNs are an explicit reporting convention,
not evidence of finite numerical parity. `mismatch_count` includes positive finite
ULP distances and nonfinite mismatches.

`max_finite_ulp` excludes nonfinite pairs and is null when no finite pairs exist.
The report includes the first maximum example and the first eight mismatches.
Examples contain the index, input, both outputs, and their hexadecimal float32
bits. Nonfinite values use strings so the JSON remains strict. Environment fields
record Python, NumPy, torch, the CPU architecture and torch capability, and libc.
Reports omit host names, local paths, and timings; repeated runs in the same
environment produce identical reports.

### Small-sample smoke measurement

The 1,000-input CLI smoke run used Python 3.12.15, NumPy 2.5.4,
torch 2.14.1+cpu, x86_64 with AVX2, glibc 2.43, and one torch thread.
All special counts were zero; each histogram contains exactly 1,000 pairs.

| Function | Finite ULP histogram (distance: count) | Maximum |
|---|---|---|
| asinh | 0: 1000 | 0 |
| sinh | 0: 1000 | 0 |
| sin | 0: 951, 1: 49 | 1 |
| cos | 0: 948, 1: 52 | 1 |
| softplus | 0: 865, 1: 135 | 1 |

This smoke run is not the prescribed million-input acceptance measurement.

### Full million-input measurement

After independent review of the probe implementation, the prescribed full run
completed successfully. The `probe-million.json` report records 1,000,000 float32
inputs per function (5,000,000 compared pairs overall).

The environment was Python 3.12.15, NumPy 2.5.4, torch 2.14.1+cpu,
x86_64 with torch CPU capability AVX2, glibc 2.43, and one torch thread.
Each function used NumPy `PCG64` with seed `20261020`, as specified above.

For `asinh`, 500,000 inputs were signed logarithmic samples: 250,000 per sign,
with magnitude endpoints `1e-30` and `1e7`. Another 500,000 were uniform samples
in `[-10, 10)`, with the first three replaced by `-10`, `0`, and `10`.
The overall `asinh` range was `[-1e7, 1e7]`.
The other ranges were `[-80, 80)` for `sinh`, `[-1100, 1100)` for `sin`
and `cos`, and `[-100, 30)` for `softplus`.
Each included the lower endpoint, zero, and upper endpoint by replacement.
The float32 rounding and softplus formulas follow the sampling contract above.

| Function | Finite ULP histogram (distance: count) | Maximum |
|---|---|---|
| asinh | 0: 1000000 | 0 |
| sinh | 0: 1000000 | 0 |
| sin | 0: 949690, 1: 50310 | 1 |
| cos | 0: 950147, 1: 49853 | 1 |
| softplus | 0: 883800, 1: 116160, 2: 40 | 2 |

For every function, `finite_count` and `total_count` were 1,000,000.
All three special counts (`both_nan`, `same_infinity`, and
`nonfinite_mismatch`) were zero for every function.
The histogram and special counts sum to the total for each function.

These results are sampled evidence, not universal bounds.
In particular, the zero-distance `asinh` and `sinh` samples do not prove universal ULP(0).
No gate tolerance changes. This measurement alone does not complete M0d:
full-suite floors, caps, and runtime RoPE evidence remain separate acceptance requirements.

## M1 native scaler measurement

The native host scaler passes all 14 canonical tiny cases for each of `tiny`
and `tiny-bias` (28 fixtures). The test checks the first and last block of
both positive and mirrored passes, counting a single-block pass once:
64 pass/block selections in total. Configurations come from the converted
GGUF models, not their JSON sidecars. No Python runs in the native tests.

| Gate | Comparisons | Measured maximum | Required bound |
|---|---:|---:|---:|
| G1.1 causal loc | 64 | 0 ULP | 0 ULP |
| G1.1 causal scale | 64 | 0 ULP | 0 ULP |
| G1.2 causal scaled | 64 | 0 ULP | 2 ULP |
| G1.2 known scaled | 16 | 0.9494 units of `u*M` | 4 |
| G1.3 known loc | 16 | 1.195 units of `u*M` | 32 |
| G1.3 known scale | 16 | 0.9638 units of `u*M` | 32 |
| G1.3 binary | 16 | exact | exact |
| G1.4 denormalization | 64 | 0 units of `u*M` | 6 |
| Dumped clamp inputs | 52 | 0 ULP | exact |

These measurements support the sequential double-prefix assumption P1 on
these fixtures; they do not establish universal bit equality. Tolerances
remain unchanged. Instance reductions sum float32 terms in double, round
the sum once, then retain the reference's float32 arithmetic boundaries.
Scaler compilation disables multiply/add contraction, not IEEE semantics.

Unit tests also cover no observations, constants, indicator and non-indicator
known rows, mixed known/nonknown replacement, prepared missing values,
unprepared NaN propagation, observed and masked signed zeros, the separate
affine epsilon clamp, sinh overflow, finite extrema, entirely nonfinite
quantile axes, and invalid shapes/masks/settings. Smoke, parity-helper
negative tests, tiny model loader, and real model loader pass separately.

A concurrent real-reference run supplied only a subset during this check:
11 of 15 manifest cases, 22 pass/block selections, all available gates passing.
The absent cases were `long_h`, `long_h_known`, `ctx_cap`, and `bench_32k`.
The old command omitted `bench_32k` and returned **77**, not a full-suite pass. This does not complete real
M1 acceptance, later milestones, or native end-to-end forecasting.
The root acceptance run must recheck the complete real suite.

### Reproduce

`test_scaler_units` needs only the tiny GGUF models. `test_scaler` additionally
uses `TIMESTRAL_TEST_TINY` as a suite root containing `tiny/` and `tiny-bias/`
baseline directories. `TIMESTRAL_TEST_MODEL_ROOT` independently selects the
model directory, defaulting to `tests/fixtures/tiny`. This does not change the
existing model-loader test's `TIMESTRAL_TEST_TINY` convention: that test expects
a model directory, so unset that variable when running loader regressions
against the bundled fixtures.

```sh
cmake -S . -B "$TIMESTRAL_WORK/build" -DTIMESTRAL_BUILD_TESTS=ON
cmake --build "$TIMESTRAL_WORK/build" -j2
# Set TIMESTRAL_TEST_TINY to the canonical suite root for this command.
ctest --test-dir "$TIMESTRAL_WORK/build" \
  -R '^(test_scaler|test_scaler_units)$' --output-on-failure
# Use the bundled tiny models for existing loader tests.
env -u TIMESTRAL_TEST_TINY ctest --test-dir "$TIMESTRAL_WORK/build" \
  -R '^(test_smoke|test_parity_helpers|test_model_loader|test_model_loader_209m)$' \
  --output-on-failure
```

For real scaler gates, set `TIMESTRAL_TEST_REAL_BASELINES` to the suite root
containing `real/`, and `TIMESTRAL_TEST_GGUF` to the converted real model.
Then run `ctest --test-dir "$TIMESTRAL_WORK/build" -R '^test_scaler_209m$'
--output-on-failure`. An entirely absent suite or incompatible host metadata
returns 77. A partially present suite, malformed fixture, or numerical mismatch
fails with 1.

### Coverage checks after review

The scaler derives case selection from `tests/cases.json`: 14 tiny cases per
variant and 15 real cases, including `bench_32k`. It checks file presence before
opening any baseline. Block selection comes from the manifest horizon and GGUF
patch/block sizes, not from tensors present in the dump. Tiny horizons use
`max(1, H / 4)`, matching `baseline_contract.scaled_case`.

For each selected block in both passes, all scaler and denormalization tensors
are mandatory. Covariate-enabled cases require known-row tensors; cases with
known covariates also require their statistics. Clamp inputs and outputs are
mandatory when the block contains more than one horizon patch. These conditions
match `baseline_contract.expected`; its implementation remains unchanged.

The negative CLI fixtures copy or rename tensor records only in scratch files.
They reject a missing real benchmark, a missing terminal block, and missing
known/clamp tensors with exit 1. Run them without a Python environment or packages:

```sh
python3 tests/test_scaler_coverage.py "$TIMESTRAL_WORK/build/tests/test_scaler" \
  "$TIMESTRAL_TEST_TINY" "$TIMESTRAL_WORK"
```

After this coverage fix, all six negative scenarios pass, alongside the units,
28 tiny fixtures (64 block/pass selections), and four loader/helper/smoke
regressions. None skip. Production arithmetic and tolerances remain unchanged.
The full real suite remains pending; this review does not test files while the
reference generator is writing them.

## M2a native graphs: measured component acceptance

The embedding-only encoder and stored-table RoPE now execute native CPU ggml
graphs. No Python runs during inference. `GraphRunner` retains its gallocr and
backend across calls, rebuilds graph metadata per invocation, uploads owned inputs
only after allocation, and copies outputs before graph destruction. The loaded
model must outlive its encoder/RoPE objects; these objects are not reentrant.
Weights remain F32 in the read-only model mapping. The default metadata capacity
is 4096 nodes (not a model dimension).

Patch input order is all values, then all missing flags, per patch. The graph
computes hidden affine, SiLU, output affine, and residual affine plus addition.
A CPU custom affine graph node uses a double accumulator with F32 weights,
activations, bias, and output. This is deliberate: the initial native F32
`ggml_mul_mat` path failed a tiny patch output at error `1.209e-6` versus the
unchanged measured tolerance `1.205e-6`. Replacing only SiLU with libm did not
change that failure. Widening the affine accumulation passes without changing
floors or tolerance formulas. This scalar parity kernel prioritizes accuracy,
not throughput; GPU support and GEMM optimization are not claimed here.

RoPE gathers stored rows using checked integer positions, including valid negative
positions. Extreme and out-of-table positions fail before index arithmetic can
overflow. Rotation pairs features `i` and `i + head_dim/2`, not adjacent features
or `i + rope_dim/2`. It uses graph view/neg/concat/mul/add with dumped tables as
teacher-forced inputs. Partial-frequency padding leaves the unrotated tail intact.
The stored inverse frequencies are exposed without recomputation.

Gate labels in the original D.3 table differ from the M2a review numbering:
this measurement covers inverse frequencies, EXACT row gathering, ULP(2) rotation
on dumped inputs, and teacher-forced NORM patch hidden/output. It does not revise
the original ULP(0) prediction: **all measured rotations actually achieve ULP(0)**.
NORM still uses `max(2 * floor_tf, 16u)`, with missing floor metadata a hard error.
Caps remain warning-only. EXACT treats signed zeros as equal and reports their
differences separately.

Measured on the completed acceptance suite:

- All **28 tiny cases** (14 each for tiny and tiny-bias), **64 pass/block
  boundaries** per component, including positive/negative passes and the first
  and terminal dumped blocks. Patch: 128 NORM comparisons and 64 EXACT feature
  boundaries. Maximum patch normalized error: `1.137e-6`; maximum error/tolerance
  ratio: `0.777`. RoPE: **384 q/k rotations**, all ULP(0); every gathered table
  compares EXACT, with **zero signed-zero differences**.
- Real **partial subset only**: `uni_one_patch,multi,past_cov`, six pass/block
  boundaries, 12 patch NORM comparisons, six EXACT feature boundaries and
  **128 q/k rotations**. Maximum patch normalized error: `1.035e-5`; maximum
  error/tolerance ratio: `0.622`. Rotations are ULP(0), gathered rows EXACT,
  zero signed-zero differences. Floors were computed with the existing script
  on scratch copies of all five already-complete modes for these three cases.
  Original reference files were not modified. The original real f32 files did
  not yet contain teacher-forced floors; that run correctly failed rather than
  substituting a tolerance. No incomplete `ctx_cap` or `bench_32k` input was read.
- Native regressions: nine CTest entries pass together, then the tiny model
  loader passes separately (ten total), with zero skips. The split is necessary
  because the loader uses `TIMESTRAL_TEST_TINY` for model files while component
  tests use it for baseline directories. The first combined invocation failed
  that loader because it received the baseline root; rerunning with the model
  fixture root passed. Scaler still covers all 28 cases / 64 boundaries; both
  existing scaler coverage-negative tests pass. Full real scaler and full real
  M2a acceptance remain pending, not counted as passes or skips.

Unit tests additionally reject incomplete/zero patch shapes, nonboolean masks,
invalid feature layouts and bias dimensions, extreme/out-of-range positions,
invalid RoPE table layouts, invalid graph capacity, upload size/state mistakes,
and reads before compute. Repeated same shapes, growth/shrink, owned outputs,
valid negative positions, empty table gathers, and paired-half/tail semantics pass.
Coverage follows `cases.json`, not discovered tensor names: absent selected files,
missing required tensors/floors, and disagreement with expected layer schedules
fail. Real subsets require an explicit case list; an unset list means all cases.

Reproduction (all commands must propagate nonzero status):

```sh
set -eu
cmake -S . -B "$BUILD" -DTIMESTRAL_BUILD_TESTS=ON
cmake --build "$BUILD" -j2
export TIMESTRAL_TEST_TINY="$SUITE"
export TIMESTRAL_TEST_GGUF="$REAL_MODEL"
ctest --test-dir "$BUILD" -E '^(test_scaler_209m|test_model_loader)$' -V
TIMESTRAL_TEST_TINY="$PWD/tests/fixtures/tiny" \
  ctest --test-dir "$BUILD" -R '^test_model_loader$' -V
python3 tests/test_scaler_coverage.py \
  "$BUILD/tests/test_scaler" "$SUITE" "$TIMESTRAL_WORK"
# REAL_FLOORS contains scratch copies with measured floors, under real/.
export TIMESTRAL_TEST_REAL_BASELINES="$REAL_FLOORS"
export TIMESTRAL_TEST_REAL_CASES=uni_one_patch,multi,past_cov
"$BUILD/tests/test_patch_embed" --real
"$BUILD/tests/test_rope" --real
```

This completes the native M2a implementation and tiny gates, with real subset
coverage only. It is not a full-real milestone acceptance or end-to-end forecaster.

Four additional scratch-copy coverage-negative checks passed: each component
rejects a missing selected `ctx_cap` file; patch rejects removed floor metadata;
RoPE rejects a removed required q-rotation tensor. An initial assertion expected
"missing tensor" after replacing every occurrence of a patch name, but this also
removed its floor key. The gate correctly failed earlier at "missing metadata";
the negative-check assertion was corrected, not the production gate.

### Native time-layer replay (M2b)

`Encoder::run_layer` replays only metadata-scheduled `time` layers. Inputs include
explicit boolean masks and positions; empty masks select causal attention. A
TwoStream split validates the closed context-to-horizon quadrant and evaluates
its mathematically equivalent full mask. This is not covariate preparation, a
full encoder, or decoding. Converted head-major QKV weights remain read-only;
only the returned fused-QKV diagnostic is reordered to reference layout.

The CPU parity path uses custom ggml graph kernels for widened affine dots,
normalization and non-flash stable SDPA. Activations and weights remain F32;
normalization rounds its normalized activation before the learned scale.
The initial rounded F32 normalization exceeded the `cat_uni` norm floor
(1.367e-6 versus 1.152e-6); independent recomputation identified reduction/inverse
rounding. Widening that evaluation passes the existing floor without changing
it. Scalar widened kernels are correctness-first, not an inference-speed claim;
quantized and GPU kernels remain future work.

Tests require each available scheduled time layer in both tiny variants, all 14
canonical tiny cases, both sign passes, and first/last blocks. Independent local
replays consume dumped norm, q/k/v, tables, rotated q/k, SDPA, gate, gated output,
attention output and MLP norm. The whole layer separately consumes its dumped
layer input and uses that layer's local `floor_tf`; no global tolerance or
upstream `floor_fr` substitutes for a local gate. Required numel/rms, applicable
kappa/max_logit/rho, floors and applicable caps are validated. Missing required
outputs or metadata fail rather than skip.

Measured native results:

| selection | layer replays | NORM comparisons | max normalized error | max error/tolerance |
|---|---:|---:|---:|---:|
| tiny and tiny-bias, 28 case/variant pairs, 64 sign/block selections | 64 | 896 | 2.24021e-6 | 0.748127 |
| real `uni_short`, layers 0 and 22, both signs | 4 | 56 | 4.42796e-4 | 0.711712 |
| real `known_cov`, additive layer-22 extension, both signs | 4 | 56 | 7.08346e-4 | 0.647168 |

Every comparison uses `max(2 * floor_tf, 16 * 2^-24)`; the fixed minimum is
9.5367431640625e-7. Large absolute normalized errors on real tensors are compared
to their own published floors, never the tiny limits. Thirty tiny cap warnings
occur on layer-0 residual outputs across the selected cases; the existing cap
sanity warnings do not replace the floor bound and all those comparisons pass.
These are the same published floor/cap disagreements, not relaxed tolerances.

Negative controls cover head interleave (including a comparator rejection),
mask changes, skipped gates, non-time/out-of-range indices, malformed teacher
layouts, malformed weight/bias dimensions, empty shapes, fully masked queries,
nonboolean masks, forbidden split quadrants and positions outside stored tables.
A persistent Encoder is replayed across repeated/grown/shrunk graphs.

Verification (suite/model paths are caller supplied):

```sh
cmake --build "$BUILD" -j2
# TIMESTRAL_TEST_TINY is the baseline suite root for component tests, but the
# older loader test interprets it as a model directory. Run that test separately.
TIMESTRAL_TEST_REAL_CASES=uni_short ctest --test-dir "$BUILD" \
  -E '^test_model_loader$' --output-on-failure
env -u TIMESTRAL_TEST_TINY "$BUILD/tests/test_model_loader"
TIMESTRAL_TEST_REAL_CASES=uni_short "$BUILD/tests/test_layer_time" --real
TIMESTRAL_TEST_REAL_CASES=known_cov "$BUILD/tests/test_layer_time" --real
```

The first CTest command passes 13/13 with zero skips; the standalone loader also
passes. The original `known_cov` command fails at missing `pos.b0.layer.22.in`.
That failure remains valid evidence: the original suite contains no layer-22
inputs. An authorized additive extension resolves this reference gap without
changing the native implementation, tolerances, or original suite.

The [additive regeneration recipe](baselines.md#additive-layer-22-reference-for-m2b)
uses the same pinned model, case settings, and five modes, with layers
`[0,3,7,22,23]`. Exact metadata and coverage checks pass before and after strict
floor calculation: 353 records, 23 call counters, 301 float tensors, 250 local
floors, and 183 caps. No tensor has `2 * floor > cap` in this extension.
Both checker runs explicitly report partial coverage, not full-suite acceptance.

All 319 original f32 arrays are bit-identical in the extension, including two
inputs and three final outputs. Its 34 additional arrays belong only to layer 22.
Case settings and model provenance match, apart from the declared layer superset.
G0.3 passes for f32 and f32-math: 3,240 table rows, 414,720 values, zero value
differences, zero signed-zero differences, and zero out-of-range positions.

Native `known_cov` replay passes with the baseline environment pointing only to
the extension: one case, two sign/block selections, four layer replays, and
56 NORM comparisons. Missing tensors: zero. Cap warnings: zero.
Together with the original `uni_short` result above, this resolves the M2b
reference blocker. It does not claim native end-to-end forecasting or acceptance
of later milestones.

All extension commands exit 0 under a 12 GiB address-space limit, one thread,
and 1800-second timeouts. Peak generation RSS is 3,201,940 KiB (`f64-rerun`);
its wall time is 26.92 seconds. Native replay takes 7.61 seconds with peak RSS
228,924 KiB. Host available memory stays above the 6 GiB guard.
The five files total 852,354,336 bytes after floor metadata. This exceeds the
plan's per-case artifact estimate; the separate artifact-budget issue remains.
No size budget or numerical gate is waived.

## M2c: variate and composite layer replay

`Encoder::run_layer` now accepts every metadata-scheduled layer kind. The
existing time-only convenience wrapper keeps its API and rejects other kinds.
Variate attention uses `[S,R,D]` inputs and `[S,R,R]` boolean masks, without
RoPE. Its gated output returns to `[R,S,D]` before the output projection.
Composite layers apply each axis's norm and residual in schedule order, then
one MLP with SiLU on the second fused half. Weights remain head-major and
read-only. The persistent graph allocator is reused; no Python inference is
introduced. No single-row optimization is enabled.

The red test compiled with the added mask argument and failed at the existing
`not a time layer` rejection before the generalized implementation. During
implementation, a variate-only graph exposed uploads to unused time-mask
inputs. Creating graph inputs only for present axes resolved that failure.
No reference tensors or tolerance settings changed.

Final canonical tiny coverage is 14 cases times both model variants, both
signs, and first/last forecast blocks: 64 layer replays per new kind.
All subcomponents are teacher-forced on their dumped inputs, followed by an
independent whole-layer replay from its dumped input. Required statistics,
applicable caps, and local floors remain mandatory.

| selection | layer replays | comparisons | max normalized error | max error/tolerance |
|---|---:|---:|---:|---:|
| tiny variate, layer 1 | 64 | 768 | 2.47989e-6 | 0.735703 |
| tiny time_variate, layer 2 | 64 | 1472 | 2.95799e-6 | 0.832238 |
| tiny variate_time, layer 3 | 64 | 1472 | 1.65786e-6 | 0.804203 |
| real multi,past_cov, layer 3 time_variate | 4 | 92 | 9.52688e-5 | 0.864109 |

The real test requires the layer's type, not just index 3. All comparisons use
unchanged `max(2 * floor_tf, 16u)`. Cap warnings retain their existing sanity
check meaning. The tiny tests include bitwise `R=1` comparisons of `v` and
full-path SDPA, both in the pinned reference and native replay. Synthetic tests
also cover singletons, changed masks, per-slot mask indexing, wrong
head layouts, absent variate RoPE, invalid indices/shapes/types/strides,
nonboolean and fully closed masks, malformed teacher inputs, gate-off controls,
and allocator reuse across graph size changes. Both bias/gate configurations run.

Final regression CTest passes 19/19 with zero skips, excluding the older loader
and real composite test for separate invocations. The standalone loader passes.
Real M2b `uni_short` and the separate additive `known_cov` extension pass with
56 comparisons each, retaining ratios 0.711712 and 0.647168. The real composite
gate passes separately on canonical `multi,past_cov`. CLI negative tests reject
unknown/duplicate selections, missing cases, and missing tensor/floor/stat/cap
families. The scaler CLI coverage regression also passes.

Reproduce with caller-supplied paths (`SUITE` is the finalized canonical A1
suite; `EXTENSION` is its separate M2b extension; `TIMESTRAL_TEST_GGUF` is the
real F32 model):

```sh
export TIMESTRAL_TEST_TINY="$SUITE"
export TIMESTRAL_TEST_REAL_BASELINES="$SUITE"
cmake --build "$BUILD" -j2
TIMESTRAL_TEST_REAL_CASES=uni_short ctest --test-dir "$BUILD" \
  -E '^test_model_loader$|^test_layer_time_variate_209m$' --output-on-failure
env -u TIMESTRAL_TEST_TINY "$BUILD/tests/test_model_loader"
TIMESTRAL_TEST_REAL_CASES=multi,past_cov \
  "$BUILD/tests/test_layer_time_variate" --real
TIMESTRAL_TEST_REAL_CASES=uni_short "$BUILD/tests/test_layer_time" --real
TIMESTRAL_TEST_REAL_CASES=known_cov TIMESTRAL_TEST_REAL_BASELINES="$EXTENSION" \
  "$BUILD/tests/test_layer_time" --real
python3 tests/test_layer_coverage.py "$BUILD/tests/test_layer_time_variate" \
  . "$SUITE" "$SCRATCH"
python3 tests/test_scaler_coverage.py "$BUILD/tests/test_scaler" "$SUITE" "$SCRATCH"
```

Default real M2b coverage is intentionally not redirected to the extension:
the canonical known_cov fixture lacks layer 22. Explicit selections above avoid
mistaking that known reference gap for a skip or overwriting canonical data.
These results establish layer replay only, not full-encoder or forecasting
acceptance. No performance, GPU, or quantized parity claim is made.

## M3 native full encoder: code complete, real acceptance pending

`Encoder::run` builds one CPU F32 graph from the dumped `tf.in`, runs every
metadata-scheduled block, and applies `out_norm`. RoPE is gathered once. The
persistent gallocr is reused across invocations; only requested layer outputs
and the final norm are read back. The existing independent `run_layer` teacher
API shares the graph construction. Graph metadata capacity is bounded before
construction. There is no Python inference, dimension/schedule override, or
change to `max(2 * floor, 16u)`.

A throwing full-encoder placeholder produced a meaningful red unit failure;
the implementation then passed the complete tiny suite. Both bias/gate
configurations and all four layer kinds run, with both signs and first/last
forecast blocks:

| selection | blocks | local layer replays | comparisons | max normalized error | max error/tolerance |
|---|---:|---:|---:|---:|---:|
| 14 tiny cases × 2 variants | 64 | 256 | 576 | 1.53930e-6 | 0.580189 |
| real multi, sparse diagnostic | 2 | 8 | 18 | 2.35221e-5 | 0.572035 |
| real known_cov, sparse diagnostic | 2 | 8 | 18 | 1.13394e-5 | 0.701453 |
| real nans, sparse diagnostic | 2 | 6 | 14 | 2.27916e-5 | 0.691452 |

Tiny maximum adjacent-layer `err_l / err_previous` was 2.00871; no two
consecutive ratios exceeded 1.5. Sparse real diagnostics cannot establish
adjacent-layer amplification and explicitly print that limitation. The tests
report repeated amplification without relaxing tolerances.

**Real M3 is not accepted.** Actual canonical tensor inspection found all 24
layers for `uni_short` (not the initially assumed sparse coverage), layers
`[0,3,7,23]` for `multi` and `known_cov`, and `[0,3,23]` for `nans`.
The separate existing known_cov extension adds layer 22 only. Strict tests for
multi, known_cov, and nans exit 1 at missing layer 1, as required. Across both
signs these lack 40, 40, and 42 layer references respectively. No reference was
modified or generated for M3.

There is also an actual numerical failure, **not a missing-reference failure**:
strict canonical `uni_short` passes all 24 positive-pass local/free-running
layers and out_norm, but fails the negative-pass teacher-forced layer 6 output:
normalized error 3.277e-6 versus tolerance 3.178e-6 (approximately 1.031 times
tolerance). The positive pass has a single amplification spike of 4.78753 at
layer 15, not a repeated streak. These results require numerical localization
before all-real acceptance; additional references alone do not resolve it.
The combined real diagnostic stopped at this same failure, exit 1. Individual
sparse diagnostics above exited 0 but are explicitly **not G3 acceptance**.
Measured combined diagnostic peak RSS was 924604 KiB, below 12 GiB.

Regression CTest: 21/21 pass, zero skips, excluding the old loader and the two
real composite/encoder gates for separate execution. The standalone loader,
real M2c multi/past_cov and real M2b extended known_cov all exited 0. Encoder
units cover invalid embeddings, masks, positions, selection, malformed teacher
boundaries, bounded graph capacity, teacher isolation, repeated calls, and
shape growth/shrinkage. The CLI negative suite passed 3 tests, including nine
missing tensor/floor/stat/cap mutations on temporary copies. No original
baseline writes occur.

Reproduce with caller-supplied `BUILD`, `SUITE`, `EXTENSION`, `SCRATCH`, and real
`TIMESTRAL_TEST_GGUF`:

```sh
export TIMESTRAL_TEST_TINY="$SUITE" TIMESTRAL_TEST_REAL_BASELINES="$SUITE"
cmake --build "$BUILD" -j2
TIMESTRAL_TEST_REAL_CASES=uni_short ctest --test-dir "$BUILD" \
  -E '^test_model_loader$|^test_layer_time_variate_209m$|^test_encoder_209m$' \
  --output-on-failure
env -u TIMESTRAL_TEST_TINY "$BUILD/tests/test_model_loader"
TIMESTRAL_TEST_REAL_CASES=multi,past_cov "$BUILD/tests/test_layer_time_variate" --real
TIMESTRAL_TEST_REAL_CASES=known_cov TIMESTRAL_TEST_REAL_BASELINES="$EXTENSION" \
  "$BUILD/tests/test_layer_time" --real
python3 tests/test_encoder_coverage.py "$BUILD/tests/test_encoder" . "$SUITE" "$SCRATCH"
# Strict acceptance commands below currently fail; retain their actual exit status.
TIMESTRAL_TEST_REAL_CASES=uni_short "$BUILD/tests/test_encoder" --real
TIMESTRAL_TEST_REAL_CASES=multi "$BUILD/tests/test_encoder" --real
# Diagnostic only, never a replacement for the strict gates:
TIMESTRAL_TEST_REAL_CASES=multi,known_cov,nans \
  "$BUILD/tests/test_encoder" --real --selected-reference
```

M4–M8, the C API, and Python-free forecasting end-to-end acceptance remain
outside this encoder milestone.

### M3 numerical repair: strict real encoder gates

The negative `uni_short` layer-6 failure is repaired without changing any
comparison tolerance, floor, reference tensor, or model weight. Strict replay
now passes both signs of all 24 layers and `out_norm`: 48 local layers, 98
comparisons, maximum error/tolerance 0.5, no missing layers. The separate
all-layer extension for `multi`, `known_cov`, and `nans` passes 144 local
layers and 294 comparisons, maximum error/tolerance 0.670333, no missing
layers. These are explicit real subsets, not all-manifest acceptance.

Localization on reference inputs found individually acceptable component
rounding whose combination exceeded the local layer floor. Widened RMSNorm
removed the reference's F32 square/mean/rsqrt/multiply boundaries. Matching
the pinned CPU cascade made the layer-6 regression pass, but exposed another
local failure at negative layer 20. Independent contraction probes identified
192-wide FMA panels: all 155,648 tested MLP projection values matched Torch
exactly, unlike widened or single-running F32 sums. The small SDPA GEMM path
uses non-fused lane reductions, unlike the large sequential-FMA path. The
implementation chooses by execution dimensions, never by layer or case name.

CPU flash attention also normalizes after the value contraction, uses a
vector exponential polynomial on full eight-value chunks, and libm on tails.
Context queries use the actual context key count, not padded masked horizon
keys. SiLU/sigmoid require the reference's vector exponential rounding:
independent binary probes matched 100,000 exponential values and all 417,792
sampled SiLU activations. The scalar implementation emulates this AVX2 F32
order on every CPU; other reference CPU capabilities are not claimed bitwise
identical. The reference Torch source/binary pin is
`5c4886908584029761b579af026dcfb627c84070`; model source remains
`a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`.

Repeated amplification remains visible, not suppressed: one report for
`uni_short`, five for the three extended cases; maximum adjacent ratios are
2.03883 and 56.9861. Near-zero predecessor errors can produce large ratios.
All resulting strict floor comparisons pass, but these reports still warrant
review under G3.2; this is not a claim that amplification disappeared.

Teacher-boundary diagnosis also found a reuse crash inside graph allocation:
same-size graphs with changed leaf assignments reused an invalid allocation
plan. Refreshing the allocator reservation before allocation repairs the
crashing sequence. Encoder units now replay each teacher boundary in order.
RMS units cover small/non-multiple dimensions and rounded F32 boundaries.
The dedicated `test_encoder_negative_layer6_209m` fails on the original
implementation and checks that a corrupted output is rejected.

Verification (caller supplies `BUILD`, `SUITE`, `EXTENSION`, and the real
`TIMESTRAL_TEST_GGUF`; no private paths are required):

```sh
export TIMESTRAL_TEST_TINY="$SUITE" TIMESTRAL_TEST_REAL_BASELINES="$SUITE"
cmake --build "$BUILD" -j2
TIMESTRAL_TEST_REAL_CASES=uni_short "$BUILD/tests/test_encoder" --real
TIMESTRAL_TEST_REAL_CASES=multi,known_cov,nans \
  TIMESTRAL_TEST_REAL_BASELINES="$EXTENSION" "$BUILD/tests/test_encoder" --real
TIMESTRAL_TEST_REAL_CASES=uni_short ctest --test-dir "$BUILD" \
  -E '^test_model_loader$|^test_layer_time_variate_209m$|^test_encoder_209m$' \
  --output-on-failure
env -u TIMESTRAL_TEST_TINY "$BUILD/tests/test_model_loader"
TIMESTRAL_TEST_REAL_CASES=multi,past_cov \
  "$BUILD/tests/test_layer_time_variate" --real
```

The regression selection passes 22/22, zero skips, including the 28 tiny
cases, bias/gate variants, scaler/patch/RoPE gates, and negative-layer-6 test.
Separate loader, real M2c (`multi,past_cov`), and extended real M2b
(`known_cov`, layers 0/22) checks also pass. M4 and later milestones remain
outside this repair.

### M4 normalized floor contract addendum

G4.3 uses `normalized-v1`, with the dumped float32 anchor held fixed across
both companions. The existing `floor_fr.*.predict` metadata uses raw output
space (PLAN C.4); it is not a normalized-space floor (PLAN D.3 G4.3).
`tests/predict_metric.hpp` therefore derives a separately named in-memory
floor from the existing f64 and f32-math companion predictions. It never
rewrites or reinterprets stored floor keys. G3 and G4.1 retain their existing
floors and comparison helpers.

All stored values are promoted to double before arithmetic. For each element,
`z64 = asinh((y64 - loc32) / scale32)` and
`w = scale32 * cosh(z64)`. The floor is the larger maximum of
`abs(y32-y64)/w` and `abs(y32-ymath)/w`, divided by `rms(z64)`.
The native error uses `abs(native-y32)/w / rms(z64)` in the same frame.
The acceptance rule remains `max(2*floor, 16*2^-24)`.
Both companions are mandatory; invalid frames, nonfinite values, missing
companions and zero normalized RMS are errors, never fallback paths.
`test_predict_metric` covers large offsets with small normalized values,
multiple row frames, both companions, zero RMS, invalid scales and corrupt
native output. Reference generation and all baseline files remain unchanged.

M4 native validation covers all context-only manifest cases, both signs, block
zero: tiny/tiny-bias 16 cases, 32 selections, 96 comparisons; real 9 cases,
18 selections, 54 comparisons, including `ctx_cap` and `bench_32k`.
All three gates pass with no missing companions. Maximum G4.1 ratios are
0.632549 (tiny variants) and 0.627645 (real). The strict extended M3 replay
remains mandatory after the shared small-linear and narrow SDPA changes.
Native kernels read weights and inputs only, never reference tensors.

The anchor regression uses distinct rows and multiple context patches:
native scaler frames are expanded per time step, so the last context frame
is at `row*T + T-1`, not `row*L + L-1`. Small linear contractions retain
float32 FMA rounding; narrow attention value contractions retain separate
float32 multiply/add. The head uses the existing SLEEF-derived exponential
and a double prefix accumulator with each cumsum prefix cast to float32.
The retained component and end-to-end gates reject the pre-fix arithmetic.

M4 companion admission checks are performed before flattening predictions:
`mode` must declare `f64` or `f32-math`, respectively, and the prediction dtype
must be F64 or F32. Every padded dimension must equal `[Q,h*patch_size,R,1]`.
The reference must declare `f32`. Case, model configuration, source revision,
HF revision and environment provenance must match across the three files;
only the recorded SDPA kernel may differ between modes. Every file must
match the current CPU/glibc host. Filenames alone do not establish identity.
Scratch-GGUF boundary tests reject swapped modes, wrong dtypes, equal-size
wrong shapes and provenance mismatches. Explicit real case selections reject
unknown, duplicate and non-context cases rather than ignoring them.
NaN raw head values are errors, never saturated into finite quantiles.
