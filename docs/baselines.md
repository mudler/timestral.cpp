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
