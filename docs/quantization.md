# Quantization evaluation design

This document defines the approved M7 evaluation contracts, not an implemented evaluator or completed acceptance run.
The 64-series composition has conditional approval; its explicit mappings, options, and anchor extraction design await final review.
No measurements or model artifacts accompany this document.

## Preparatory Python implementation

`scripts/quant_metrics.py` implements model-free row metrics, equal-row/equal-item summaries, relative changes, coverage HOLD, and strict-order checks.
Scores encode infinities and undefined values as JSON-safe tagged objects with `value=null`, a `kind`, and a `status`.
Summaries preserve item identities, row counts, and row-status counts without finite-only filtering.
The caller supplies frozen native anchors; this module cannot extract or recompute them.

`scripts/quant_fixtures.py` loads the 79 case mappings and constructs independent future truth.
It isolates the unchanged `make_inputs` function through Python AST compilation instead of importing the reference module.
Before AST parsing or execution, the helper checks the complete generator source against the accepted SHA-256 pin.
Only that function executes, with NumPy and math bindings; torch imports and module-level determinism changes do not execute.
The accepted source at `65933ae` has SHA-256 `596a45ed8dd14faae0b04718d79308c5fc37af41fb2599256fdf5bd0c185425f`.
Any source change fails closed, including changes outside the function.
Pin updates require explicit review of the source diff, input-identity checks, and fixture semantics before a code commit.
Do not derive acceptance pins from current files at runtime or provide a runtime override.
Fixture provenance records Python/NumPy versions, source/specification hashes, case hash, options, and input/truth hashes.
These helpers produce in-memory arrays only; they do not write golden artifacts or load models.
Model-weight hashes remain the responsibility of the future evaluation runner.

Focused tests cover small synthetic shapes and full-shape input identity for all 79 mappings, including missingness and covariate paths.
The full-shape checks include canonical `nans` patch 12 and verify input bytes, shapes, dtypes, options, and provenance hashes.
All metric operands validate before undefined propagation; unknown score statuses and NaN/negative loss operands fail in either order.
Signed relative-change outputs remain valid JSON report values.
Empty coverage fails; the future runner must require exactly the 79 frozen identities, not an arbitrary nonempty subset.
They compare input bytes with a separately extracted original generator body and check future independence.
They also check known-bad metrics, zero denominators, aggregation statuses, and error-match coverage HOLD.
These tests are preparation, not M7 numerical acceptance.
The model runner, CLI transport, and native anchor extraction remain unimplemented.
`eval_quant.py` now orchestrates recorded-output validation, per-row scoring, separate suite summaries, and numerical checks.
Its unit tests use explicitly mocked small fixtures, not model outputs or milestone evidence.

## Preparatory F16 implementation slice

The F16 slice adds storage policy and projection dispatch, not real-model acceptance.
Keep the existing F32 callback branches and their arithmetic unchanged.
Reduced weights use GGML matmul with F32 graph inputs and outputs, followed by F32 bias addition.
GGML may internally round activations to F16; M7 quality metrics cover this difference, not strict reference parity.
Cover `affine`, the residual patch projection, and the small-input head projection before the loader accepts F16.
Only B.4 weights permit F16; all other tensors remain F32.
Reject Q8, K-quants, BF16, and every unsupported type until their implementation slices.
The converter permutes QKV before F16 casting and records output-type provenance without replacing licence/source metadata.
Use synthetic fixtures and existing F32 regressions for this preparatory slice.
M6c, the 5M F32 prerequisite, real-model quality evaluation, and milestone acceptance remain separate gates.

## Scope and invariants

Compare reduced-precision forecasts with F32 C++ forecasts on identical inputs.
Score both forecasts against independent held-out truth.
Use all 15 C.2 cases, including both slow cases, plus 64 synthetic evaluation items.
Keep the existing reference generators, parity helpers, floors, and canonical case specification unchanged.
These quantization metrics do not replace the M3 or M6b parity gates.

Use Timestral-5M only for the F32 prerequisite described in [the plan](PLAN.md).
Use Timestral-209M for quantization evaluation, including K-quants.
Reject rows incompatible with the requested quantization block size.
Do not pad dimensions, change runtime dimensions, or silently retain incompatible allowlisted weights.
The B.4 allowlist and its explicit F32 exclusions still apply.

## Recorded-output scoring protocol: `quant-record-v1`

`eval_quant.py --manifest FILE --report FILE` reads records only; it never launches a model or public forecast command.
Exit codes are 0 for recorded numerical checks passing, 1 for FAIL/HOLD, and 2 for invalid input or arguments.
Container and record validation precedes lookups, including preliminary overwrite checks.
Malformed JSON structures exit with a diagnostic and no report, not a traceback.
The command rejects manifest/NPZ overwrite aliases, including symlink aliases; recorded NPZ symlinks are invalid.
A passing report is not independent proof of model execution or milestone acceptance.
The future native driver remains responsible for truthful model/source identities and exact anchor extraction.

The JSON manifest has exactly these fields:

- `version`: `quant-record-v1`.
- `fixture_version`: `c2-truth-v1`.
- `options`: exactly `quant_fixtures.OPTIONS`, including explicit resolved forecasting options.
- `source_sha256`: the current five source/specification hashes emitted by `make_fixture`.
- `driver`: exactly `source_sha256` (64 lowercase hex characters) and `runtime_commit` (40 lowercase hex characters).
- `models`: exactly the six format names `f32`, `f16`, `q8_0`, `q6_k`, `q5_k`, `q4_k`.
  Each model has `sha256`, `hf_revision`, and `reference_sha`; the latter two match the pinned 209M identities.
- `items`: exactly the 79 frozen case identities, once each, in any order.

Each item has exactly `id`, `case_sha256`, `input_sha256`, `truth_sha256`, `npz`, `arrays`, `anchor_model_sha256`, and `outputs`.
The item provenance matches a freshly constructed in-memory fixture; the scorer checks every declared hash against that fixture.
`npz` is a sibling basename, never a path outside the manifest directory.
Load it with `allow_pickle=False` and reject undeclared arrays.
Each `arrays` entry describes one NPZ key with exactly `dtype`, `shape`, and `sha256`.
Hash dtype, shape, and contiguous bytes with `quant_fixtures.array_hash`.
Use little-endian F32 arrays and little-endian F64 quantile levels.

Every item stores `context` `(B,V,T)`, `truth` `(B,V,H)`, and `levels` `(Q)`.
Context and truth match the fixture byte for byte, including context NaN locations.
Levels equal the pinned 209M levels: 0.01, 0.05, 0.1 through 0.9 by 0.05, 0.95, 0.99.
Each `outputs` map has all six formats.
A forecast record is exactly `{"status":"forecast"}` and requires `forecast_FORMAT` `(B,V,H,Q)`.
Forecasts must be finite and nondecreasing along quantiles.
An error record has exactly `status`, `type`, `message`, and `stage`; status is `expected-error` or `error`.
These error details are driver assertions, not independent verification of the reference's expected error.
A match requires every field to equal the F32 expected-error record.
Matching errors remain coverage gaps and HOLD numerical acceptance.

When F32 forecasts exist, store only one pair of F32 `anchor_loc` and `anchor_scale` arrays, both `(B,V)`.
Associate them through `anchor_model_sha256`, which must equal the F32 model hash, never a quantized model hash.
No quantized-anchor keys are permitted.
When F32 has no forecast, use null anchor association and omit both anchor arrays.
Do not score any format without its F32 forecast and shared anchors.

The report preserves per-row metrics, item identities, named coverage gaps, and all three suite summaries: C.2, synthetic, and overall.
Missing required identities invalidate the manifest; declared error records produce HOLD rather than invented numeric values.
Use the approved F16/Q8 bounds and overall strict normalized-mean ordering; MASE has no numerical bound.
Reports distinguish checked array/fixture hashes from asserted driver/model provenance.
Write reports through a temporary file in the destination directory, then atomically replace the destination.
Serialize tagged nonfinite results with strict JSON; never emit NaN or Infinity literals.

## Held-out truth: `c2-truth-v1`

The existing generator defines target context, not target future truth.
Its shared RNG also generates missingness and covariates.
Changing its target draw length from `T` to `T+H` changes later inputs.
Do not make that change.

### Construction

1. Generate inference inputs with the existing `make_inputs` behavior, unchanged.
2. Independently replay its target stage with the original seed and exactly `T` normal draws per target row.
3. Retain those latent increments before positive transforms, constant overrides, F32 conversion, and missingness.
4. Create each row's future RNG as `Generator(PCG64(SeedSequence([seed, 7001, b, v])))`.
5. Draw `H` normal increments with the case's `noise_std`, defaulting to 1.0.
6. Concatenate the original and future increments; compute their float64 cumulative sum in chronological order.
7. Evaluate the existing target formula at absolute times `T` through `T+H-1`.
8. Apply the positive transform or constant override, as applicable, then cast truth to F32.
9. Store truth separately; never supply it as a target input.

For batch index `b`, variate index `v`, and absolute time `t`, the latent formula is:

```text
walk[t] = cumsum(context_increments ++ future_increments)[t]
a[b,v,t] = level + trend*t
           + 10*sin(2*pi*t / (period*(1 + 0.25*v)) + 0.3*b)
           + walk[t]
```

Use the same float64 operation order as the existing target formula.
Defaults are `level=0`, `trend=0`, `period=24`, and `noise_std=1`.
For positive series, take `abs(a)` before the F32 cast.
For constant series, replace every value with the specified constant before that cast.
Future truth remains fully observed, including cases with missing context.
This defines a new deterministic continuation, not a claim about an existing canonical future RNG stream.

Check every inference input against canonical generation before accepting artifacts.
Require bitwise equality of finite values and identical missing-value locations.
This includes numeric and categorical covariates throughout their original lengths.
Record the specification version, RNG, dependency versions, input hashes, truth hashes, and model hashes.

### Rules for all C.2 cases

`P`, `L`, `trend`, and `sigma` below refer to the formula above.
Retain each case's canonical `B`, `V`, `T`, `H`, seed, and forecasting options.
Retain variate-dependent periods and batch-dependent phases.

| Case | Future truth and input constraints |
|---|---|
| `uni_short` | Continue `L=100`, `trend=0.05`, `P=24`, `sigma=1` for both batch rows. |
| `uni_one_patch` | Continue `L=-20`, `trend=-0.02`, `P=16`, `sigma=0.7`. |
| `multi` | Continue all six rows with `L=50`, `trend=0.01`, `P=30`, `sigma=1`. |
| `past_cov` | Continue `L=10`, `trend=0`, `P=24`, `sigma=1`; preserve both past numeric rows. |
| `known_cov` | Continue `L=30`, `trend=0.02`, `P=24`, `sigma=1`; preserve numeric and indicator inputs through `T+H`. |
| `cat_uni` | Continue `L=5`, `trend=0`, `P=12`, `sigma=0.8`; preserve categorical NaNs and the unseen horizon code. |
| `cat_multi` | Continue both variates with `L=5`, `trend=0`, `P=12`, `sigma=0.8`; preserve categorical past inputs. |
| `long_h` | Continue `L=200`, `trend=0.01`, `P=48`, `sigma=1` through all 2500 future steps without block resets. |
| `long_h_known` | Continue `L=20`, `trend=0`, `P=48`, `sigma=1` through 1100 steps; preserve the complete known numeric row. |
| `nans` | Continue all four latent rows with `L=15`, `trend=0`, `P=24`, `sigma=1`; preserve context and past-covariate missingness. |
| `nonneg` | Continue the signed latent process with `L=1e6`, `trend=0`, `P=24`, `sigma=50`, then take absolute values. |
| `const` | Truth equals F32 `5.0` at every future step. |
| `all_missing` | Continue both latent rows with `L=3`, `trend=0`, `P=24`, `sigma=1`; keep the second context row entirely missing. |
| `ctx_cap` | Continue `L=40`, `trend=0`, `P=96`, `sigma=0.5` from absolute time 40000 and the full latent walk. |
| `bench_32k` | Continue `L=40`, `trend=0`, `P=96`, `sigma=0.5` from time 32768 through 1024 future steps. |

Do not restart truth at a context-cap boundary or decode-block boundary.
Do not invent future past-covariate inputs.
Keep existing known-future arrays, including indicators and categorical codes, exactly as generated.
The existing covariates do not causally generate the target; retain that property.

## Metrics and degenerate values

Compute metrics in float64 over the stored F32 forecasts and truth.
Reject nonfinite forecasts, nonfinite truth, invalid shapes, or invalid anchors as evaluation failures.
Do not replace denominators with epsilons or silently omit rows.

### Normalized deviation

Use the original positive-pass, initial-context F32 C++ target anchor defined below.
Freeze its F32 location `l` and scale `s` across every horizon step and both symmetric passes.
Use that same stored anchor for every precision; quantized evaluation must never recompute the metric anchor.

#### Exact anchor source and extraction design

The anchor is `BlockPrediction.loc` and `BlockPrediction.scale` from the first positive call to `PredictBlock::predict`.
Capture these values before `refine_block`, autoregressive feedback, mirroring, output capping, or nonnegative clipping.
Use final F32 forecasts after all those operations as `f`; do not replace them with positive-pass predictions.

`Forecaster::forecast` first calls `assemble_rows` on the original request.
That path applies observation masking, target preparation, categorical encoding, context capping, and left padding.
The cap keeps the last `min(request.length, 32768)` steps; padding adds zero values with false masks on the left.
Use this prepared context, not raw input arrays or a hand-written sample standard deviation.

For a context-only prediction, `context_only` calls `scale_rows` on that prepared context of length `T0`.
It selects `scaled.loc[r*T0 + T0-1]` and the corresponding scale.
For covariate predictions, `predict` constructs length `W=T0+h*patch_size`, including the first block's prepared future and masks.
It calls `scale_rows`, then `overwrite_known_rows`, before selecting anchors.
With `L=T0/patch_size`, it selects `scaled.loc[r*W + (L-1)*patch_size]` and the corresponding scale.
These are the exact existing selections in `src/predict_block.cpp`, not alternative metric formulas.

Packed rows are contiguous by batch, with targets before past and known covariate rows.
Let `all=rows_count/batch`; target `(b,v)` maps to packed row `r=b*all+v`.
Store its anchor at metric row `b*targets+v`.
Final forecasts have layout `(B,V,H,Q)`; all `H*Q` elements of that metric row share this anchor.
This matches the final-output mapping in `tests/test_forecast.cpp` from `pos.b0.anchor.loc` and `pos.b0.anchor.scale`.
Those baseline fields come from the reference denormalization hook; they are corroborating parity artifacts, not substitutes for native F32 anchors.

The approved extraction strategy captures the actual first positive
`BlockPrediction.loc/scale` directly, immediately after predict returns and before
refinement. It supersedes the earlier shared preparation/scaling-helper proposal;
no second preparation, scaling, or prediction is part of production extraction.

`ForecastTrace.capture_initial_anchors` opts in; `capture_blocks=false` avoids
retaining full block tensors. Defaults preserve existing trace behavior.
The owned optional `initial_anchors` contains batch, targets, and F32 [B,V]
location/scale vectors using the mapping above. Capture clears on entry and is
published only after the complete forecast succeeds. Negative passes and later
blocks cannot overwrite it. Progress callbacks remain active in anchor-only mode.
No C ABI or public CLI extraction endpoint is added by this internal extension.

Tests must compare captured bytes with actual independently replayed native
`PredictBlock` anchors, and compare final forecasts with no/default/full/anchor-only
traces. Cover target mapping, covariates, categorical inputs, cap/padding,
multiblock symmetry, stale reuse/failure clearing, and real uni_short/known_cov.
The record driver remains responsible for successful F32 execution and truthful
model association; no driver or metric formula changes accompany this capture.

Verified capture checks use an isolated `751f69d` pre-F16 source archive plus
only the anchor plumbing/test registration. Tiny capture passes **6 cases,
414 checks**; actual real uni_short/known_cov pass **2 cases, 68 checks**.
Captured bytes match independently replayed native BlockPrediction values;
no/default/full/anchor-only forecast bytes agree. Tests cover distinct B>1/V>1
target frames, covariates/categorical inputs, first-future slices, cap/padding,
multiblock symmetry, progress, owned reuse, and transactional failure clearing.
F32 component/tiny CTest passes **37/37**, loader separately **1/1**; complete
tiny forecast replay passes **28 cases / 684 comparisons**, zero failures,
worst ratio `0.618082`. No full real aggregate parity replay runs here.

Evidence under `timestral-native/m7-anchors`: `base-commit.txt`, `base.tar`,
`anchor-runtime.patch`, `configure.log`, `final-build.log`,
`anchors-final-tiny.log`, `anchors-real.log`, `f32-component-tiny.log`,
`loader.log`, `forecast-tiny.log`, `tested-source.tar`, `before.json`,
`after.json`, `vendor-verification.json`, and `source-commit.txt`.
Archive/library/anchor-test/forecast-test hashes match before/after checks.
The archive configure log reports surrounding-repository commit `be955f52d`
through Git discovery; all **1962 vendor blobs** in the actual vendor archive
and pinned tested-source archive match ggml
`e705c5fed490514458bdd2eaddc43bd098fcce9b` exactly.
Builds/tests use `-j2`, serial CPU execution, and the hard 12 GiB guard.
Concurrent F16/Q8 runtime edits are excluded from this validation.
This internal extraction does not implement the quant-record driver or accept
M6c, quantized quality, or the complete M7 milestone.

Let `f[i]` denote F32 C++ forecasts and `q[i]` denote reduced-precision forecasts.
Include all horizon steps and quantiles within each row.

```text
z[i] = asinh((f[i] - l) / s)
w[i] = s * cosh(z[i])
R = sqrt(mean(z[i]^2))
e[i] = abs(q[i] - f[i]) / (w[i] * R)       when R > 0
```

Require finite `l`, finite positive `s`, and finite positive `w[i]`.
Report each row's mean and maximum error.
If `R=0` and every prediction equals F32, assign zero error and status `zero-rms-identical`.
If `R=0` and any prediction differs, assign infinite row mean and maximum errors with status `zero-rms-different`.
This F32-reference definition does not change the F64-companion metric in the parity helper.

### Weighted quantile loss

For each target row, use every model quantile level `tau` and every held-out step `h`.
Let `y[h]` denote truth and `p[h,tau]` denote the forecast.

```text
rho(tau, u) = u * (tau - indicator(u < 0))
N = (2 / Q) * sum_tau sum_h rho(tau, y[h] - p[h,tau])
D = sum_h abs(y[h])
WQL = N / D                              when D > 0
```

If `D=0` and `N=0`, assign zero loss with status `zero-truth-exact`.
If `D=0` and `N>0`, assign infinite loss with status `zero-truth-inexact`.

### MASE

Use the median forecast and lag 1.
Use only original finite target observations within the context window available to inference after context capping.
Do not forward-fill observations or use hidden latent context for scaling.
Let `A` contain adjacent original time pairs with both observations finite.

```text
d = mean_(t in A) abs(x[t] - x[t-1])
a = mean_h abs(y[h] - p[h,0.5])
MASE = a / d                             when A is nonempty and d > 0
```

If `A` is nonempty, `d=0`, and `a=0`, assign zero MASE with status `zero-scale-exact`.
If `A` is nonempty, `d=0`, and `a>0`, assign infinite MASE with status `zero-scale-inexact`.
If `A` is empty, report undefined MASE with status `no-observed-adjacent-pairs`.
The constant case can have infinite MASE; the fully missing row has undefined MASE.
An undefined full aggregate is an explicit limitation, not a finite-only replacement metric.
MASE has no acceptance bound in the plan.

### Relative loss changes

Compute changes from the aggregated losses, not from an average of per-row percentage changes.
For finite positive F32 loss `Lf`, report both:

```text
signed_change_percent = 100 * (Lq - Lf) / Lf
absolute_change_percent = abs(signed_change_percent)
```

If both losses equal zero, report zero change with status `zero-baseline-exact`.
If `Lf=0` and `Lq>0`, report positive infinite change with status `zero-baseline-degraded`.
Undefined aggregate losses produce undefined relative changes.
An infinite F32 baseline produces an indeterminate relative change, including when both losses are infinite.
Always report the underlying losses and statuses.

## Aggregation, coverage, and acceptance

Average equally across target rows within each evaluation item.
Then average equally across all 79 items: 15 C.2 cases and 64 synthetic series.
For normalized error, average each row's elementwise mean before these two aggregation steps.
Also report the global maximum normalized error.
Use the same equal-row, equal-item hierarchy for WQL and MASE.
Report separate C.2 and synthetic summaries with the same weighting rules.

An undefined item makes the corresponding full aggregate undefined.
Otherwise, an infinite item makes that aggregate infinite.
Retain each item's status, row count, and identity in the report.
Any finite-only summary is supplementary and must show its denominator.
It cannot replace the full aggregate.

Match expected canonical errors for every precision when F32 returns an expected error.
Error matching is not a numerical pass.
Report the missing forecast as a coverage gap and HOLD numerical acceptance when any required forecast is missing.
Never score an error as zero forecast deviation.

The existing acceptance targets remain:

| Format | Aggregate mean normalized error | Absolute relative change of aggregate WQL |
|---|---:|---:|
| F16 | <= 1e-3 | <= 0.5% |
| Q8_0 | <= 5e-3 | <= 1% |

Report signed WQL changes too; an improvement beyond the absolute bound does not pass that bound.
Apply strict monotonicity only to aggregate mean normalized deviation:

```text
0 = E_F32 < E_F16 < E_Q8_0 < E_Q6_K < E_Q5_K < E_Q4_K
```

Do not require per-case, maximum-error, WQL, or MASE monotonicity.
A tie or reversal fails the strict-order target; report measured results without sorting or adding slack.
Any target change requires a separate measured decision under the plan.

## Conditionally approved: frozen 64-series specification

The family composition has conditional approval, subject to review of the explicit inputs and options below.
Implementation and acceptance execution remain on HOLD.
The [generator-shaped case mappings](synthetic64-v1.cases.json) contain all 64 inputs, not golden baselines or forecast results.
The JSON uses the existing `version` and `cases` shape; it does not embed forecast options unsupported by that schema.
Apply the following resolved options to every item instead of relying on evolving defaults.

### Frozen forecast options and provenance

Use the pinned 209M configuration at Hub revision `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`.
Its configuration values below come from PLAN A.0; constructor defaults come from the converter, not Hub configuration fields.
The upstream algorithm pin is `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (`NOTICE` and converter `CODE_SHA`).
The inspected native source snapshot is `8017889c84c5540ad852e1c57dbc9b6185a88f9d`.

| Resolved option | Value | Source |
|---|---|---|
| Symmetric averaging | `true` | Converter `DEFAULT_FORECASTER`; native `Forecaster::forecast`. |
| Nonnegative clipping | `true` | Same sources; native eligibility uses original observed target context. |
| Maximum context | `32768` steps | PLAN A.0 `context_length`; native zero request resolves to that value. Set it explicitly for evaluation. |
| Per-shot horizon | `32` patches | Converter default; native decode limits it to configuration `block_size=32`. |
| Patch size | `32` | PLAN A.0 configuration. |
| Quantile real cap | `10000.0` | Converter default; native `Forecaster::raw`. |
| Extreme feedback threshold | `1e30` | Converter default; native `Forecaster::decode`. |
| Refinement | Call `refine_block` for every block; bypass only when that block has <=1 patch. | Native decode and `src/refine.cpp`; no request-level refine switch exists. |
| Refinement targets | Target rows only when covariate roles exist; otherwise all packed rows. | Native decode creates `role==0` selection. |
| Scaling | `minimum_scale=1e-6`, `use_asinh=true`, global stabilization `false` | PLAN A.0 configuration. |
| User observation mask | Absent | Missingness comes only from each case's generated values. |

Use the converter's stored metadata for non-request options and check these resolved values before evaluation.
Do not change a mismatching model's metadata to force compatibility.
Evaluate complete free-running forecasts, including refinement, both symmetric passes, caps, and eligible clipping.

Source blobs at the inspected snapshot:

| Path | Git blob |
|---|---|
| `scripts/gen_reference_baseline.py` | `d52f28677fa238222e9d4bdd231e2744cae19a5f` |
| `scripts/convert_timestral_to_gguf.py` | `91e298b85b61b62f5e1254d2ca8387fc88b70f6c` |
| `src/forecaster.cpp` | `15670621fd63332669eede77084f950bb23093a6` |
| `src/predict_block.cpp` | `af43189bf93128e47cec35281b20fe87ea3899df` |
| `src/covariates.cpp` | `5bbf3e89600c4ef8600ff3f780b223e9cc1ab963` |
| `src/refine.cpp` | `912cf46c32a778d9df4b2b7423ec2adb49de4760` |

### Input mappings

Use eight families with eight seeds each, giving exactly 64 evaluation items.
Each item has `B=1`; `V` specifies its target variates.
For family `f` in `0..7` and replicate `j` in `0..7`, use seed `20000 + 8*f + j`.
Name each item `synth-f{f}-r{j}`.
Use the existing generator's formulas, draw order, F32 conversions, and default forecasting options.
Use `c2-truth-v1` for the future, with this item's seed.

| f | Family | V | T | H | Target parameters | Missingness and covariates |
|---|---|---:|---:|---:|---|---|
| 0 | Seasonal random walk | 1 | 512 | 96 | `period=24, level=100, trend=0, noise_std=1` | None. |
| 1 | Negative trend | 1 | 500 | 65 | `period=16, level=-20, trend=-0.02, noise_std=0.7` | None. |
| 2 | Multivariate | 3 | 1000 | 64 | `period=30, level=50, trend=0.01, noise_std=1` | None; retain variate-dependent periods. |
| 3 | Missing targets and past covariate | 1 | 900 | 64 | `period=24, level=15, trend=0, noise_std=1` | Leading 100 missing target steps, missing patch 12, scattered fraction 0.03; one past numeric row with NaN fraction 0.1. |
| 4 | Known numeric and indicator | 1 | 800 | 1100 | `period=48, level=30, trend=0.02, noise_std=1` | One known numeric row and one known indicator row, no NaNs. |
| 5 | Categorical covariates | 1 | 600 | 40 | `period=12, level=5, trend=0, noise_std=0.8` | One past and one known categorical row, five codes each, NaN fraction 0.05 each; unseen horizon code enabled only for the known row. |
| 6 | Positive large level | 1 | 700 | 64 | `period=24, level=1e6, trend=0, noise_std=50, positive=true` | None. |
| 7 | Constant | 1 | 512 | 64 | `period=24, constant=5` | None; preserve the generator's default latent draws before the constant override. |

Every unspecified missingness or covariate field is absent.
Use patch size 32 for missing-patch placement.
The known numeric row follows `5*sin(2*pi*t/(period+7)) + 0.5*normal()` through `T+H`.
The indicator equals `((t // 5) % 2 == 0)` through `T+H`.
Categorical rows use the generator's integer draws and NaN masking.
Set the known categorical code at `T + H//2` to `n_codes + 7` before its NaN masking, as the generator does.
That masking can hide the inserted code; do not alter canonical generator behavior to force visibility.

Generate context and covariates before drawing any future target innovations.
Never expose held-out target values through inputs, target encoding, MASE scaling, or covariate construction.
Generate categorical target encodings from observed context only.
Evaluate complete horizons without truth-driven feedback or teacher forcing.
These series test deterministic synthetic quality, not accuracy on an external forecasting dataset.
