# Quantization evaluation design

This document defines the approved M7 evaluation contracts, not an implemented evaluator or completed acceptance run.
The 64-series specification below remains pending review.
No measurements or model artifacts accompany this document.

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

Use the fixed F32 C++ anchor location `l` and scale `s` for each target row.
Use that same anchor for every precision.
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

## Pending review: frozen 64-series specification

This section proposes `synthetic64-v1`; it is not yet approved for execution or acceptance.
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
