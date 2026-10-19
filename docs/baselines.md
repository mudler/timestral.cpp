# Reference baselines

`scripts/gen_reference_baseline.py` runs the Python reference and writes one GGUF file per case
and mode. This page records what is in a file and how the inputs are made. The plan
(`docs/PLAN.md`, section C) defines the modes and the instrumentation; this page fills in the
details that the plan leaves open.

## Commands

```bash
for m in f32 f32-math f64 f64-rerun f32-math-rerun; do
  $VENV/bin/python scripts/gen_reference_baseline.py --model mistralai/Timestral-209M \
      --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 --cases tests/cases.json \
      --mode $m --output-dir $TIMESTRAL_TEST_BASELINES
done
scripts/regen_tiny_baselines.sh
$VENV/bin/python scripts/compute_floors.py --dir $TIMESTRAL_TEST_BASELINES
$VENV/bin/python tests/python/check_baseline.py --dir $TIMESTRAL_TEST_BASELINES
$VENV/bin/python tests/python/check_rope_tables.py --gguf $TIMESTRAL_TEST_GGUF --dir $TIMESTRAL_TEST_BASELINES
```

Cases labelled `slow` (`ctx_cap`, `bench_32k`) run only with `--include-slow` or `--case`.
`--tiny` selects the cases with `"tiny": true` and scales them (`T / 4`, `H / 4`, plan C.2).
For a tiny checkpoint every layer is dumped; for the 209M model the `layers` field of the case
lists the dumped layers (the last layer is always included).

## Input generator

All inputs come from `numpy.random.default_rng(seed)` and are stored in the file
(`input.context`, `input.past`, `input.known`, `input.cat_past`, `input.cat_known`), so the C++
tests never regenerate them. A missing field of a case means zero or none.

- Target `x[b, v, t] = level + trend * t + 10 * sin(2 pi t / (period * (1 + 0.25 v)) + 0.3 b)
  + cumsum(normal(0, noise_std))`; `positive` takes the absolute value; `constant` replaces the
  series by that value. The values are rounded to float32.
- `missing`: `leading` NaN steps and the NaN patches in `missing_patches` apply to the series
  with flat index 0 (`b = 0, v = 0`); `scattered_frac` puts NaN at random positions of every
  series; `all_missing_series` lists flat indices (`b * V + v`) that are all NaN.
- Numeric past and known covariates are sine waves with a different period per row plus noise;
  `nan_frac` puts NaN at random positions. Indicator rows are 0/1 square waves.
- Categorical rows are random integer codes in `[0, n_codes)` as float64; `nan_frac` puts NaN
  codes; `unseen_in_horizon` sets one horizon code to `n_codes + 7`, which never occurs in the
  context.

## Record names

Names follow C.3. A prefix `<pass>.b<k>.` is added to everything that belongs to decode block `k`
of a pass (`pos` = the forward pass, `neg` = the mirrored pass). Pass-level records use `<pass>.`
(`assemble.*`, `cap.*`); the final outputs are `final.quantiles`, `final.raw.pos`,
`final.raw.neg`; the inputs are `input.*`. Records outside any pass (`enc_cat.<n>.*`,
`target_encode.<n>.*`) have no prefix.

Per block (`<pass>.b<k>.`): `args.*` (the arguments of `predict_block`), `scale.in`,
`scale.observed`, `scaled`, `loc`, `scale`, `known.*`, `patch.in`, `patch.hidden_out`,
`patch.out`, `tf.in`, `tf.positions`, `tf.enc_len`, `layer.<l>.in`, `layer.<l>.attn.<j>.{norm,
qkv, q, k, v, cos, sin, q_rope, k_rope, sdpa, gate, gated, out}`, `layer.<l>.mlp_norm`,
`layer.<l>.mlp`, `layer.<l>.out`, `out_norm`, `head.in`, `head.raw`, `head.monotone`, `denorm.in`,
`anchor.loc`, `anchor.scale`, `predict`, `refine.*`, `frames.*`, `refined`, `mask.*`, `positions`,
`first_obs.group`, `group.inverse`.

- `qkv` is the output of `wQKV` in torch row order (undo B.3 before comparing); `q`, `k`, `v` are the
  split tensors `(batch, seq, heads, head_dim)` before RoPE; `gate` is the pre-sigmoid output of
  `_gate`; `gated` is the input of `wO`; `out` is the output of `wO`.
- `sdpa` is the concatenation of the two `_sdpa` calls of a time attention along the sequence axis,
  in call order.
- Booleans are I8, integers I32, floats F32 (F64 in the `f64` file; `frames.*` are F64 in every file).
- A wrapped function that returns `None` records nothing; a tensor with no elements is not stored
  (a KV `empty.<name>` records its shape).

## Statistics and KV

Per float tensor: `stat.<name>.rms`, `stat.<name>.numel`. For the cap table (D.3):
`stat.<name>.kappa` (and `kappa_res` for `patch.out` and `head.raw`) with the definition
`max_i (|x| @ |W|^T + |b|)_i / rms(y)`, `max_logit` for `sdpa` (the largest `|q.k| * scale` over
the unmasked entries of all calls of that attention), and `rho` for sublayer outputs
(`rms(sublayer out) / rms(residual stream at the sublayer input)`). `kappa` of a layer `mlp` is the
larger of the two linears.

Other KV: `env.*`, `model.*` (dimensions and the config as JSON), `case.*`, `layers.dumped`,
`calls.<module>.<function>` (how often each wrapped function ran), `repro.bitwise_equal`,
`truth.patches` (`f64` only), `selfcheck.*` (`f64-rerun` only), `expect.error.*` (when the
reference raised an exception).

`env.cpu_capability` and `env.glibc` are what the host gates compare (plan D.4).

## Strict acceptance checks

Run the checker without `--case` for a full suite:

```bash
$VENV/bin/python tests/python/check_baseline.py --dir "$TIMESTRAL_TEST_BASELINES"
$VENV/bin/python tests/python/check_baseline.py --dir "$TIMESTRAL_TEST_TINY/tiny" --tiny
$VENV/bin/python tests/python/check_baseline.py --dir "$TIMESTRAL_TEST_TINY/tiny-bias" --tiny
```

The full real suite requires all 15 canonical cases, including both slow cases.
The full tiny suite requires all 14 eligible cases, with no `bench_32k` case.
The checker validates tiny scaling against `case.json` and `case.tiny`.
It never infers tiny selection from missing files.
Generate slow cases explicitly; the generator's default selection does not satisfy the full gate.

Use `--case NAME` for a partial check:

```bash
$VENV/bin/python tests/python/check_baseline.py --dir "$TIMESTRAL_TEST_TINY/tiny" \
    --tiny --case uni_one_patch
```

A successful subset reports `PARTIAL (not a milestone gate)`, not a full milestone PASS.
Unknown or ineligible case names fail.
A custom case specification requires explicit `--case` selection.

Each successful case requires all five modes.
The checker derives both passes, every decode block, and required components from the requested case and model configuration.
Deleting a complete pass or later block does not reduce the required coverage.
The checker requires condition statistics for each relevant tensor, not merely one statistic somewhere in the file.
The numeric selfcheck must remain finite, nonnegative, and at most `1e-12`.
The stored selfcheck limit must equal `1e-12`.
These checks do not change the plan's tolerance formulas.

Exception artifacts currently fail closed, including those for `const` and `all_missing`.
Acceptance is blocked pending generator provenance for a supported reference refusal and its exact execution step.
Matching exception type, message, and step across modes does not prove that the reference raised the error.
The synthetic error fixture is deliberately unproven and must fail; it is not an approved exception identity.

At the pinned reference commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`,
`forecaster.py:forecast` accepts missing context values through `_as_context` and does not explicitly reject constant series.
`scaler.py:compute_causal_statistics` clamps denominators and scales for missing or constant observations.
No legitimate pre-block refusal for the canonical `const` or `all_missing` inputs is established by these paths.
The degenerate-output `ValueError` in `_forecast_raw` follows `_block_ar_decode` and cannot justify pre-block omissions.
Older artifacts may record arbitrary caught exceptions and the last wrapped call as `expect.error.step`;
that record alone cannot distinguish a reference refusal from an instrumentation failure.
The generator now propagates every unsupported reference exception and instrumentation failure.
It does not write an exception artifact; the checker continues to reject existing unproven exception artifacts.

Exception validation still requires mandatory stored inputs with valid shapes, dtypes, and statistics in all three full modes.
A claimed pre-block error requires zero recorded blocks and no rerun artifacts.
Any block-prefixed tensor, block statistics or empty-record metadata, or block execution counter contradicts that claim.
Missing steps and inconsistent exception identities or steps also fail validation.
Post-block errors remain blocked pending a separate step/coverage contract.
Successful regular and tiny cases still require complete output and component coverage.

### Known follow-up requirements

Small read-only probes of existing `uni_one_patch` artifacts fail the stricter statistics contract.
The real artifacts lack `rho` for 36 single-attention MLP outputs across both passes.
The tiny artifacts lack `rho` for four single-attention MLP outputs across both passes.
Repair that instrumentation separately; do not remove the required statistics to accept these artifacts.

The checker validates input/output shapes and shape consistency between modes.
It does not yet independently validate every intermediate shape or every integer/boolean tensor's semantic dtype.
The full error-step contract for exceptions after block execution also needs a separate design.
The scalar selfcheck validates the generator's reported maximum; it does not independently rerun components.
These limitations do not authorize missing ordinary modes, passes, blocks, or component statistics.

Run the fast synthetic regressions without inference:

```bash
PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python -m unittest discover \
    -s tests/python -p test_check_baseline.py -v
```

## Teacher-forced rerun contracts (G2)

These contracts apply to both rerun modes. Each local operation reads the specified
source-file tensors, upcast to the rerun dtype. It must not read another local rerun's
output. The ordinary gold forward and its attention implementation are unchanged.

For each `layer.<l>.attn.<j>.` prefix:

| Output | Exact input boundary |
|---|---|
| `norm` | Dumped layer residual plus preceding dumped attention outputs, summed in source dtype |
| `qkv`, `q`, `k`, `v` | Projection on dumped `norm`; variate attention first permutes `(rows,N,D)` to `(N,rows,D)` |
| `cos`, `sin` (time only) | RoPE table gather from dumped `tf.positions`; independent of rotary application |
| `q_rope`, `k_rope` (time only) | Dumped `q`, `k`, `cos`, `sin`, not newly projected q/k or newly gathered tables |
| `sdpa` | Dumped post-RoPE q/k for time, raw q/k for variate, dumped `v`, and the axis-specific dumped mask |
| `gate` (when configured) | Gate linear projection on dumped `norm` in attention-axis layout; pre-sigmoid |
| `gated` | Dumped axis-layout `sdpa` times sigmoid of dumped pre-sigmoid `gate`, or just dumped `sdpa` without a gate; restore physical row layout |
| `out` | `wO` on dumped restored-layout `gated`, not the newly computed gated result |

The fused projection reshapes `(batch,seq,3,head_dim,num_heads)`, then permutes
`(2,0,1,4,3)` and splits, matching the pinned reference's interleaving.
SDPA receives `(B,S,H,d)` and transposes sequence/heads in the reference helper.
With `TwoStreamMask`, context queries and keys/values use `[:enc_len]` and the
context mask quadrant; horizon queries use all keys/values and the horizon mask rows.
The two results concatenate along sequence. Without a time mask, time attention is
causal. Variate attention is not implicitly causal. Its `sdpa` and `gate` remain
axis-layout tensors, whereas `gated` is always the physical-row-layout input of `wO`.

Existing larger boundaries remain separate: a whole layer uses dumped `layer.in`,
masks and positions; MLP norm uses the source-dtype reconstructed residual; MLP uses
dumped `mlp_norm`; output norm uses dumped last-layer output; head projection uses
dumped `head.in`; monotone conversion uses dumped `head.raw`. Patch embedding and
scaler retain their existing local contracts.

### Isolated refine versus combined block refine

- `predict`: `predict_block` on dumped `args.*`.
- `refined`: isolated `_refine_block` on dumped `predict`, `refine.context`,
  `refine.mask`, and `refine.target_rows` (G6.1).
- `frames.*`: isolated frame computation during that dumped-prediction refine.
- **`block_refined`**: combined `predict_block` plus `_refine_block` (G6.2).
  Refine receives the **newly computed prediction**, with dumped refine context,
  mask and target rows. This name exists only in rerun files; its gold comparison
  target is `refined`.

`scripts/rerun_contracts.py` defines the shared output-to-gold mapping. The checker
requires both refine contracts, validates the alias shape against gold `refined`,
and rejects older ambiguous rerun artifacts missing `block_refined`.
`compute_floors.py` selects `block_refined` when computing `floor_tf.<prefix>refined`,
so the block floor includes prediction error. It fails rather than falling back to
isolated refine when the combined contract is absent. Isolated `refined` remains
available in rerun files for G6.1 consumers; no tolerance or unrelated floor formula
changes accompany this selection repair.

The numeric selfcheck derives its complete required set from source contracts,
not returned outputs. Missing/extra outputs, shape mismatches, nonfinite values,
and nonfloat mismatches fail. Both isolated and combined refinement compare to
free-running f64 `refined`. The production limit remains **`1e-12 * rms`**.
Rerun coverage errors and hook/harness exceptions abort generation, and rerun context
is reset on exit even after failure. Source coverage itself still requires the
independent case/configuration checks in `check_baseline.py`.

Fast bounded regressions (including actual f64 truth-patched CPU math attention):

```bash
PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python -m unittest discover \
    -s tests/python -p 'test_*.py' -v
```

The separate artifact-budget follow-up must reconcile C.4's size estimates with
reported existing suite totals: about 11.5 GB real and 1.34 GB per tiny suite.
This repair does not regenerate those suites or change their artifact budget.
