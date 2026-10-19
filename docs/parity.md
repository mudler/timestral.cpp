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
11 of 14 cases, 22 pass/block selections, all available gates passing.
The absent cases were `long_h`, `long_h_known`, and `ctx_cap`; the real scaler
command returned **77**, not a full-suite pass. This does not complete real
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
--output-on-failure`. Missing cases or incompatible host metadata return 77;
a malformed available fixture or numerical mismatch fails with 1.
