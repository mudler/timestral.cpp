# timestral.cpp: porting plan

This is the work order for a C++17/ggml port of Timestral-209M, the time-series
forecaster from Mistral AI. The port runs one self-contained GGUF file on CPU, with no
Python at inference, and its forecasts are numerically equal to the reference.

- Reference model code: <https://github.com/mistralai/timestral>, pinned at commit
  `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` ("Timestral release", 2026-10-15). In this
  document "the reference" means the `timestral` Python package at that SHA.
- Weights: <https://huggingface.co/mistralai/Timestral-209M>, pinned at revision
  `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`. One `model.safetensors` (F32,
  837,271,648 bytes, 203 tensors, 209,312,256 parameters) and one `config.json`.
- Sibling port for layout, CLI, C ABI and tests: <https://github.com/mudler/parakeet.cpp>.
- ggml: the `ggml-org/ggml` repository as a git submodule, pinned at tag `v0.13.0`
  (the same pin as parakeet.cpp; it has `ggml_soft_max_ext` with mask broadcast,
  `ggml_swiglu_split`, `ggml_softplus` and `ggml_cumsum`).

Every claim below was checked in the reference code at the pinned SHA, unless it is
marked **UNVERIFIED**. Each UNVERIFIED item says what would verify it.

Rules for every milestone:

1. Parity first. A milestone is done only when its gate passes with the tolerances in
   section D. Do not loosen a tolerance to make a test pass (section D.5).
2. Metadata-driven. Every dimension and every setting comes from GGUF KV. No hardcoded
   dims, no sidecar config, no `if d_model == 768`.
3. One milestone is one commit series. Run the gate command, then commit.
4. Scratch output (models, baselines, venvs, benchmark logs) goes under `$TIMESTRAL_WORK`
   (default `./work`, gitignored). Never commit model weights or real-model baselines.

---

## A. Architecture trace

### A.0 Released configuration (from `config.json` at the pinned revision)

| key | value | derived |
|---|---|---|
| `patch_size` | 32 | |
| `d_model` | 768 | `head_dim = 64` |
| `num_heads` | 12 | `rope_dim = int(64 * 0.5) = 32` |
| `mlp_hidden_dim` | 2048 | |
| `block_schedule` | 24 entries: `time, time, time, time_variate` repeated 6 times | 18 `time` + 6 `time_variate` layers, 30 attention sublayers |
| `quantile_knots` | 0.01, 0.05, 0.1, ..., 0.9, 0.95, 0.99 (21 levels) | `median_index = 10` |
| `block_size` | 32 patches (1024 steps) | |
| `context_length` | 32768 steps (1024 patches) | |
| `rope_theta` | 10000.0 | |
| `rope_partial_factor` | 0.5 | |
| `norm_eps` | 1e-6 | |
| `attn_bias`, `mlp_bias` | false, false | |
| `time_attn_gate`, `variate_attn_gate` | `sdpa_sigmoid`, `sdpa_sigmoid` | |
| `minimum_scale` | 1e-6 | |
| `use_asinh` | true | |
| `stabilize_with_global` | false | the reference rejects `true` |
| `covariate_enabled` | true | |
| `split_context_horizon_attn` | true | |
| `attend_missing_patches` | true | |
| `quantile_width_factor` | **3.0** | the dataclass default is 2.0; the JSON value wins |

The 5M, 28M and 123M checkpoints use the same pattern (`time` x3 then `time_variate`) and
the same scalar settings. None of the four released checkpoints uses the `variate` or
`variate_time` layer kinds. The port must still implement them (the config allows them),
and gates them with a tiny random checkpoint (milestone M0b, `scripts/make_tiny_checkpoint.py`).

### A.1 Data layout

The reference packs every variate of every series into rows: `R = B * V_all`, where
`V_all = V_target + C_past + C_known (+ encoded categorical rows)`. Each row is one
sequence of `N` patches with width `D`: torch shape `(R, N, D)`, ggml `ne = [D, N, R]`.
`group_ids[r]` is the series index. Rows of one series are contiguous, in the order
targets, past covariates, known covariates (`forecaster.py: _assemble_rows`).

### A.2 Input preparation (`forecaster.py`)

| step | function | semantics | parity note |
|---|---|---|---|
| A.2.1 | `TimestralForecaster._as_context` | cast to float32; `(T,)` -> `(1,1,T)`, `(B,T)` -> `(B,1,T)`; `mask = isfinite(x) & user_mask`; unobserved values set to `0.0` | host, exact |
| A.2.2 | `_as_covariates` / `_as_categorical` | numeric covariates float32; categorical codes kept **float64** on CPU; `inf` in codes is an error | host, exact |
| A.2.3 | `_encoding_target` | only when `V_target == 1`: forward-filled target from the first observation on, NaN before the start and outside the last `max_context` steps | host, exact |
| A.2.4 | `_encode_categorical` + `covariate_encoding.target_encode` | per series and per categorical row, smoothed mean target encoding in float64 (A.9); multivariate targets get the raw codes cast to float32 | host float64; see A.9 |
| A.2.5 | `_cat_rows` | encoded rows are appended after the numeric rows of the same kind | host, exact |
| A.2.6 | `ffill_with_mask` | targets only: forward fill; leading unobserved become 0; returns the input unchanged when the mask is all true | host, exact |
| A.2.7 | `_assemble_rows` | covariate rows: `mask = ~isnan`, values `nan_to_num(0)`; known covariates split into `[:-prediction_length]` and `[-prediction_length:]`; context capped to the last `max_context` steps (default `context_length`), **then** left-padded to a multiple of `patch_size` with value 0 and mask false (`pad_to_patch`); horizon rounded up to whole patches, the known future zero-padded with mask false | host, exact |
| A.2.8 | `role_ids` | `0` target, `1` past covariate, `2` known covariate, repeated per series; `None` when there are no covariates | host, exact |

### A.3 Causal patch scaling (`scaler.py`)

`TimestralModel.scale` calls `causal_patch_loc_scale` then `apply_affine_asinh_masked`.

- `compute_causal_statistics(data, mask, minimum_scale, stabilize_with_global=False)`:
  all in **float64**. `w = mask` (float32 then float64), `x = data` (float64).
  `cum_w = cumsum(w)`, `cum_x = cumsum(w*x)`, `den = max(cum_w, 1)`, `means = cum_x/den`.
  `shifted[t] = means[t-1]` (`shifted[0] = 0`), `inc = (x - shifted) * (x - means) * w`,
  `m2 = cumsum(inc)`, `var = m2 / max(den - 1, 1)`, `scale = max(sqrt(var), minimum_scale)`.
  This is the streaming (Welford) M2 update written with prefix sums. Results are cast to
  float32 at the end.
- `causal_patch_loc_scale`: per patch, take the statistic at the **last** step of the
  patch (`unfold(...)[..., -1]`) and repeat it over the patch. Output `(R, T)` float32.
- `apply_affine_asinh_masked(data, loc, scale, mask, use_asinh)`: work dtype float32
  (`_fp32_io`): `y = (data - loc) / max(scale, FLT_EPSILON)`; `y = 0` where unobserved;
  `y = asinh(y)` where observed. `FLT_EPSILON = 1.1920929e-07` (`torch.finfo(float32).eps`).
- Known-future covariate rows (`TimestralModel._known_cov_scaling`, only on the covariate
  path): **non-causal** whole-row statistics from `instance_norm_stats` in float32:
  `n = max(sum(mask), 1)`, `loc = sum(x*mask)/n`, `var = sum(((x-loc)*mask)^2)/n`
  (population variance, no Bessel), `scale = max(sqrt(var), minimum_scale)`. A row whose
  observed values are all 0 or 1, with at least one observed value, is an indicator:
  `loc = 0`, `scale = 1`, and no asinh. The other known rows get `asinh` in float32 on
  observed positions. The result replaces `scaled`, `loc`, `scale` on known rows only.

Parity flags:

| # | place | why it matters | how the port reproduces it |
|---|---|---|---|
| P1 | float64 prefix sums | rounding of a long cumsum depends on order | host C++ `double`, strictly sequential loop over time, the same operation order as written above. Expected bit-exact after the float32 cast. **UNVERIFIED:** torch CPU `cumsum` on float64 is a sequential scan. Verify with gate G1.1 (it is bit-exact if and only if this holds) |
| P2 | `max(sqrt(var), minimum_scale)` in float64, `minimum_scale` as a double | the clamp happens before the float32 cast | store `minimum_scale` as F64 KV; clamp in double |
| P3 | `max(scale, FLT_EPSILON)` in float32 | second, different clamp | reproduce literally in float32 |
| P4 | float32 `asinh` | torch may use a vectorized (SLEEF) or libm implementation | host `asinhf`. Gate G1.2 at ULP(2). **UNVERIFIED:** which implementation torch 2.14.1 uses on CPU; the M0d math probe measures it |
| P5 | `instance_norm_stats` float32 reductions over up to ~33k steps | torch float32 `sum` uses a vectorized cascade order that depends on the CPU ISA | host: sum in double, round to float32 once (error <= 0.5 ulp from the exact sum). Gate G1.3 uses `MAG(32, max \|x\|)` because the mean can be near 0 |
| P6 | indicator test `(x == 0) \| (x == 1) \| ~mask` | must use the raw values, before scaling | reproduce literally |
| P7 | mirrored pass negates the context | `-0.0` appears at unobserved positions | negate every element, including zeros, exactly as `-inputs` does |

### A.4 Patch representation (`model.py: PatchRepresentation`)

- Input per patch: `concat([values(32), missing(32)])`, 64 features. `values` are the
  scaled values; `missing = 1.0` where unobserved, `0.0` where observed. Order: the 32
  values first, then the 32 flags.
- `patch_proj = ResidualBlock(64, 3072, 768, bias=True)` (`layers.py: ResidualBlock`):
  `y = W_out @ silu(W_hid @ x + b_hid) + b_out + (W_res @ x + b_res)`.
- `forecast_query`: one learned `(768,)` vector, broadcast to every decoder token.

Shapes: `(R, N, 64)` -> `(R, N, 768)`. In ggml: `ne = [64, N, R]` -> `[768, N, R]`.

### A.5 Transformer (`layers.py`)

`Transformer.forward(x, time_mask, variate_mask, positions)`: compute RoPE tables once,
run every `TransformerBlock`, then `out_norm` (RMSNorm) on every token.

**A.5.1 RMSNorm** (`RMSNorm.forward`): `x * rsqrt(mean(x^2) + eps)`, computed in float32,
then multiplied by `scale`. In ggml: `ggml_mul(ggml_rms_norm(x, eps), scale)`. The same
order: normalize, then multiply by the gain.

**A.5.2 TransformerBlock** (`TransformerBlock.forward`): for each attention axis of the
layer kind, in order (`config.py: LAYER_KIND_AXES`):
`time = (time)`, `variate = (variate)`, `time_variate = (time, variate)`,
`variate_time = (variate, time)`. Each axis is a pre-norm residual
`x = x + Attn_j(RMSNorm_j(x))`. Then one MLP: `x = x + MLP(RMSNorm_mlp(x))`.

**A.5.3 Attention** (`Attention.forward`, `Attention._qkv`, `_sdpa`):

- Time attention runs per row over the patch axis: sequence `S`, batch `R`.
  Variate attention runs per patch slot over the rows: input permuted to `(S, R, D)`,
  sequence `R`, batch `S`.
- `qkv = W_qkv @ x` (`2304 x 768`, no bias). **The weight rows are not head-major.** The
  output is reshaped to `(batch, seq, 3, head_dim, n_heads)` and permuted to
  `(3, batch, seq, n_heads, head_dim)`. So the output feature index is
  `o = qkv*D + d*n_heads + h`. The converter permutes the rows to head-major order
  `o' = qkv*D + h*head_dim + d` (section B.3). A row permutation of a weight matrix is exact.
- RoPE on q and k, time attention only (A.5.4).
- SDPA, `scale = 1/sqrt(head_dim) = 0.125` (the default of
  `F.scaled_dot_product_attention`). The boolean mask means "may attend"; masked keys get
  `-inf` before the softmax. Every query row has its diagonal open (A.6), so no softmax
  row is fully masked and no NaN can appear.
- With `split_context_horizon_attn` (`TwoStreamMask`), time attention runs as two query
  passes: context queries over context keys only, horizon queries over all keys. The
  skipped quadrant (context queries to horizon keys) is all false in every mask builder,
  so one pass with the full mask is mathematically identical.
- Output layout after `_sdpa`: `(batch, seq, n_heads*head_dim)`, head-major
  (`h*head_dim + d`).
- Gate (`sdpa_sigmoid`): `out = out * sigmoid(W_gate @ xn)`, where `xn` is the **normed
  input of this sublayer** (permuted for variate attention), and `W_gate` is `768 x 768`,
  no bias. The gate is element-wise over the 768 features, before `W_O`.
- `y = W_O @ out` (`768 x 768`, no bias). Variate output is permuted back to `(R, S, D)`.
- Variate attention with one row (`R == 1`): the mask is `None` and `is_causal` is false,
  so attention over one key returns `v` exactly. The port may skip the softmax there; the
  result is bit-identical.

**A.5.4 Partial RoPE** (`SinCosRoPE`, `_rotate_half`, `_apply_rotary`):

- `inv_freq` has length `head_dim/2 = 32`. Entries `i < rope_dim/2 = 16` are
  `1 / (theta ** ((2i) / rope_dim))`, computed in float32 by torch. Entries `i >= 16` are
  **0**. `rebuild_inv_freq` recomputes the table on the device after load.
- `freqs[r, n, i] = positions[r, n] * inv_freq[i]` (float32; a matmul with K = 1, so it is
  one rounded product). `emb = concat(freqs, freqs)` (length 64); tables `cos(emb)`,
  `sin(emb)`, float32.
- `rotate_half(x) = concat(-x[32:], x[:32])`, over the **whole** head (pairs `i` and
  `i + 32`). `q' = q*cos + rotate_half(q)*sin`. The rotated dims are `0..15` and `32..47`;
  dims `16..31` and `48..63` get `cos = 1`, `sin = 0`, which is an exact identity.
- `positions` are integers, possibly negative (A.6.5), cast to float32.

Parity flags:

| # | place | why it matters | how the port reproduces it |
|---|---|---|---|
| P8 | `ggml_rope_ext` does not match | NEOX mode with `n_dims = 32` pairs `i` with `i + 16`, but the reference pairs `i` with `i + 32`; ggml also builds theta by repeated multiplication, not `theta ** (2i/rope_dim)` | do **not** use `ggml_rope_ext` in the parity path. Gather `cos`/`sin` rows per position from the stored tables (P10) into `[64, S, 1, R]` F32 graph inputs and apply `q*cos + rotate_half(q)*sin` with `ggml_mul`, `ggml_view_*`, `ggml_neg`, `ggml_concat`, `ggml_add`. These are the same IEEE operations as torch, so the result is bit-exact for the same q, cos and sin |
| P9 | `inv_freq` values | torch float32 `pow` vs C `powf` can differ by 1 ulp | the converter computes `inv_freq` with the reference code and stores it as GGUF tensor `rope.inv_freq` (F32 `[32]`). The loader also recomputes it and logs a warning when they differ |
| P10 | `cos`/`sin` | torch may use SLEEF (error <= 1 ulp); libm `cosf`/`sinf` can differ from it | the converter runs the reference `SinCosRoPE.forward` (one thread) on positions `pos_min..pos_max` and stores the results as GGUF tensors `rope.cos` and `rope.sin` (F32 `[head_dim, P]`), with `pos_min = -(context_length/patch_size + block_size)` and `pos_max = -pos_min`: for 209M, positions -1056..1056, `P = 2113`, about 1.08 MB for both. Every reachable position is in range: positions are `p - first_obs` with `p < context_length/patch_size + block_size` and `first_obs < context_length/patch_size` (A.6.5). The runtime looks rows up by position and returns an error for a position out of range (so `max_context > context_length` is refused; the reference allows it, and this is a documented difference). Gate G2.2 is EXACT. Caveat: torch could compute a different last bit for the same position in another tensor layout (a scalar tail loop instead of the vector loop). The M0d check compares the stored table with every table the reference used at run time (dumped from `_apply_rotary`) |

**A.5.5 SwiGLU MLP** (`FusedSwiGLU`): `[g, u] = chunk(W_12 @ x, 2)` (`W_12` is
`4096 x 768`), `y = W_3 @ (g * silu(u))` (`W_3` is `768 x 2048`). **The silu is on the
second half.** In ggml: `ggml_swiglu_split(ctx, u, g)` (it computes `silu(a) * b`), or
`ggml_mul(g, ggml_silu(u))`. Gate both against the dump before choosing.

Parity flags for the transformer:

| # | place | why it matters | how the port reproduces it |
|---|---|---|---|
| P11 | float32 GEMM accumulation order | torch (MKL/oneDNN) and ggml sum in different orders | unavoidable; bounded in D.3. Keep every weight F32 in the parity build, and check that `ggml_mul_mat` with F32 weights keeps F32 activations (the vec-dot type of F32 is F32) |
| P12 | SDPA kernel | torch CPU may pick the flash kernel (online softmax, scale on `q.k`) or the math kernel (scale split as `sqrt(scale)` on q and k) | ggml: `ggml_mul_mat(k, q)` -> `ggml_soft_max_ext(kq, mask, 0.125, 0)` -> `ggml_mul_mat(v^T, p)`, all F32. Do **not** use `ggml_flash_attn_ext` in the parity path (on CPU it can convert K/V to F16; **UNVERIFIED** for F32 K/V at v0.13.0). The dump script records which torch kernel ran and also dumps with the math kernel forced, so the floor (D.2) covers the kernel choice. **UNVERIFIED:** which SDPA kernel torch 2.14.1 picks on CPU for a boolean `attn_mask` |
| P13 | mask fill value | boolean mask -> `-inf` | additive mask with `0` and `-INFINITY`. F16 holds both values exactly, so the mask tensor may be F16 to halve its memory; values do not change |
| P14 | `silu`, `sigmoid`, `exp` | ggml uses its own SIMD exp approximation | ulp-level; bounded in D.3 |
| P15 | RMSNorm reduction | torch float32 mean vs ggml double accumulation | ulp-level |

### A.6 Masks and positions (`masks.py`)

All masks are boolean, true = may attend. The host builds them; the graph gets additive
float masks. Integer and boolean outputs must match the reference **exactly**.

- A.6.1 `compute_enc_key_validity(obs_per_patch, attend_missing)`: `obs_per_patch[r, p]` is
  true if any step of patch `p` is observed. With `attend_missing_patches`:
  `key_valid = obs | cummax(obs)`. Leading all-unobserved patches (left padding, late
  start) are never valid keys. Unobserved patches after the start are valid keys.
- A.6.2 `enc_causal_mask(key_valid)`: `(R, L, L)`, `tril & key_valid[col]`, diagonal
  always true.
- A.6.3 Context-only path, `build_forecast_time_mask(L, h, key_valid)`: `(R, 1, L+h, L+h)`.
  Enc->Enc: A.6.2. Enc->Dec: false. Dec->Enc: `key_valid[col]`. Dec->Dec: lower
  triangular (every decoder token reads itself and the earlier decoder tokens).
- A.6.4 Covariate path, `build_two_stream_time_mask(min_context, n, block_size, key_valid,
  bidirectional_rows)`: `(R, 1, 2n, 2n)`. `compute_block_layout`: `active[p] = p >= s`,
  `anchor[p] = s + block_size * floor((p - s)/block_size)` if active else 0, with `s = L`
  for every row. Enc->Enc: causal over valid keys; for known-covariate rows full
  `key_valid[col] | eye`. Enc->Dec: false. Dec->Enc: `active[q] & (col < anchor[q]) &
  key_valid[col]`. Dec->Dec: `(anchor[q] == anchor[k]) & tril & active[q] & active[k]`,
  then `| eye`.
- A.6.5 Positions. `group_first_observed_patch`: per series (group), the first patch that
  any row of the group observes; 0 if none (`scatter_reduce amax` then `argmax`, which
  returns the first maximum). `torch.unique(group_ids)` is sorted.
  Context-only (`build_forecast_positions`): `pos[r, p] = p - first_obs[r]` for
  `p in 0..L+h-1`. Covariate path (`build_two_stream_positions`): `pos = p - first_obs[r]`
  for `p in 0..n-1`, the same for the encoder and the decoder token of one patch.
  Positions can be negative.
- A.6.6 `build_two_stream_variate_mask(group_ids, enc_key_valid, n_dec, target_rows,
  bidirectional_rows)`: `(slots, 1, R, R)` with `slots = N_enc + N_dec`; `None` when
  `R == 1`. Query row `a` may read key row `b` at slot `p` if
  `group[a] == group[b]` and the key slot is valid (encoder slots: `key_valid[b, p]`;
  decoder slots: always), and not forbidden by `variate_firewall`: a non-target row may
  not read a target row, and a known (bidirectional) row may not read a non-known row.
  The diagonal is always open. Context-only path: `target_rows = bidirectional_rows = None`.

### A.7 Quantile head (`model.py: QuantileHead`, `monotone_quantiles`)

- Input: the decoder tokens after `out_norm`, `(R, N_dec, 768)`.
- `projection = ResidualBlock(768, 3072, 32*21 = 672, bias=False)`.
- Reshape to `(R, N_dec, patch_size, Q)`: feature `f = step*Q + q`.
- `monotone_quantiles(raw, knots, width_factor)`:
  `widths[i] = knots[i] - knots[i-1]` for `i > m`, `knots[i+1] - knots[i]` for `i < m`,
  `widths[m] = 0`, computed in Python float64, then a float32 tensor.
  `inc = (width_factor * widths) * softplus(raw)` (left to right, float32).
  `median = raw[m]`; `up = cumsum(inc[m+1:])`; `down = flip(cumsum(flip(inc[:m])))`;
  result `[median - down, median, median + up]`, clamped to `+-FLT_MAX`.
- Denormalize (`model.py: _predict_context_only` / `_predict_window`, then
  `scaler.py: denormalize_quantiles`): anchor `loc`/`scale` = the frame of the **last
  context patch** (`loc[:, L-1]` per row). `q = sinh(x) * scale + loc` in float32, then
  `clamp_nonfinite`: `+inf` -> the largest finite value along Q, `-inf` -> the smallest,
  a position that is all non-finite -> 0, NaN unchanged.

Parity flags:

| # | place | why it matters | how the port reproduces it |
|---|---|---|---|
| P16 | `widths` | computed in double from the knots, then cast | store knots as an **F64** array in GGUF; compute widths in double; cast to float |
| P17 | `softplus` | torch: `x > 20 ? x : log1p(exp(x))`; `ggml_softplus` uses `logf(1 + expf(x))`, which loses precision for very negative x | do the whole monotone step on the host (21 values per step) with `log1pf(expf(x))` and the threshold 20 |
| P18 | `cumsum` order | sequential over at most 10 values | host, the same order (down: from `m-1` toward 0) |
| P19 | `sinh` amplifies error | `d(sinh x) = cosh(x) dx` | compare quantiles in the normalized space too (D.3, G4.3) |

### A.8 Single block forecast (`model.py: predict_block`)

`predict_block(inputs (R,T), observed, group_ids, h, role_ids, known_future,
known_future_mask)`:

- **Context-only path** `_predict_context_only` when `role_ids is None`, or all rows are
  targets, or `covariate_enabled` is false. `L = T/ps`. Scale (A.3), embed (A.4), decoder
  tokens = `h` copies of `forecast_query`, masks A.6.3 and A.6.6 (with `n_dec = h`),
  positions A.6.5, transformer over `[Enc ; Dec]` (`S = L + h`), head on the last `h`
  tokens, denormalize with the frame of patch `L-1`. Output `(R, h*ps, Q)`.
- **Covariate path** `_predict_window`, with `1 <= h <= block_size`. The window is
  `[context ; known_future]` (`n = L + h` patches; target and past rows have a blank,
  unobserved horizon). Causal scaling over the whole window, then the known-row
  replacement (A.3). `x_dec` = `n` copies of `forecast_query` (**decoder tokens for every
  patch, S = 2n**). Masks A.6.4 and A.6.6 with `target_rows` and `bidirectional_rows =
  known_rows`. Positions A.6.5 (shared). The head runs on all `n` decoder tokens; the
  output keeps patches `L..n-1`. Denormalize with the frame of patch `L-1`.

Decoder tokens of inactive patches (`p < L`) read only themselves in time attention and
only the decoder tokens of the same slot in variate attention, so they never affect the
active tokens. The first port must still compute them (bit-faithful layout). Dropping them
is a later optimization, gated again at M6.

### A.9 Categorical target encoding (`covariate_encoding.py: target_encode`)

Per series and per categorical row, float64 (NumPy):

- `observed = isfinite(target)`; `target = where(observed, target, 0)`;
  `item_count = sum(observed)`; `item_mean = target.sum() / item_count` (0 if no count).
- `categories = sorted unique finite codes`; NaN gets the slot `len(categories)`.
- `sums = bincount(slots, weights=target*observed)`, `counts = bincount(slots,
  weights=observed)`; `lookup[c] = (smooth*item_mean + sums[c]) / (smooth + counts[c])`,
  `smooth = 1`; an extra last entry `item_mean` is the "unseen" slot.
- Past values -> `lookup[slot]`. Future codes (`_slots(..., unseen=nan_slot + 1)`): a NaN
  code -> the NaN slot (`lookup[nan_slot]`, the same value a NaN past code gets); a finite
  code not in `categories` -> the unseen slot (`item_mean`). Cast to float32.
- The caller sets past positions to NaN where the target is NaN **and** the code is NaN
  (padding before the series start).

Parity flags:

| # | place | why it matters | how the port reproduces it |
|---|---|---|---|
| P20 | `target.sum()` | NumPy uses pairwise summation (blocks of 128, 8-way unrolled) for float64 | implement NumPy's pairwise sum in C++ (`src/numpy_sum.hpp`) to match bit for bit; gate G5.3 at exact float32 equality, with <= 1 ulp allowed only with a written reason |
| P21 | `bincount` | sequential accumulation in input order | sequential loop |
| P22 | codes as float64 | codes above 2^24 must stay distinct | the C ABI takes categorical codes as `double` |

### A.10 Decode (`forecaster.py`)

`forecast(...)`:

1. Prepare inputs (A.2).
2. `quantiles = _forecast_raw(inputs, mask, H, covariates)`.
3. If `symmetric_averaging` (default true): `mirrored = _forecast_raw(-inputs, mask, H,
   negated covariates)` (every covariate row negated, including indicators and encodings);
   `quantiles = 0.5 * (quantiles + flip_Q(-mirrored))`. The knots must be symmetric
   (checked with `1e-9`).
4. If `clip_nonnegative` (default true): series whose observed context is all `>= 0`
   (and has at least one observation) are clamped at 0 (`nonnegative_context`).

`_forecast_raw`:

1. `_assemble_rows` (A.2.7).
2. `_block_ar_decode`:
   - `context (R, T)`, `group_ids = arange(B).repeat_interleave(V_all)`.
   - Loop while `remaining > 0` (`remaining = ceil(H/ps)` patches):
     `block = min(per_shot_horizon_patches, remaining)` with `per_shot =
     min(32, block_size)`; slice the known future of this block; `q =
     predict_block(...)`; `q = _refine_block(q, context, mask, target_rows)`; append.
   - Feedback for the next block: targets get the median, with
     `replace_extreme_values(threshold=1e30)` (non-finite or `|x| >= 1e30` -> 0) and mask
     true; known rows get their real future values and mask; past rows get 0 with mask
     false. Append to the context. If the context is longer than `max_context`, keep the
     last `max(ps, floor(max_context/ps)*ps)` steps.
   - Concatenate the blocks, cut to `H`, reshape `(B, V_all, H, Q)`.
3. Keep the target rows.
4. `_cap_quantiles` (if `quantile_real_cap_k = 1e4 > 0`): clamp to
   `[ctx_min - k*s, ctx_max + k*s]` per target row, with `ctx_min`/`ctx_max` over observed
   steps of the assembled rows, and `s` the causal scale at the last context step.
5. Any NaN -> error.

`_refine_block(quantiles, context, mask, target_rows)` (only when `n_pred > 1`):

- `loc0, scale0` = float32 frame of the last context step (`model.scale` on the
  context).
- `median` (float64) -> `sinh_median = (median - loc0) / scale0` (float64).
- `advancing_block_frames(context, mask, sinh_median, minimum_scale)`, all float64:
  per-patch `(cnt, mean, m2)` of the context, with zero states for the `n_pred` horizon
  patches; an **inclusive Hillis-Steele prefix scan** (`cumulative_welford`, `log2(N)`
  rounds of `welford_combine(left, self)` with zero padding); take the state at index
  `-n_pred-1` (the last context patch). Then for each horizon patch `j`:
  `var = max(m2 / max(cnt-1, 1), 0)`, `scale_j = max(sqrt(var), minimum_scale)`,
  `loc_j = mean`; fold the patch's median `folded = clamp(sinh_median[j]*scale_j + mean -
  offset, -clip, clip) + offset` with `offset` = the context mean and
  `clip = FLT_MAX ** 0.5 / 1e3`; merge `(32, mean(folded), sum((folded - mean)^2))` with
  `welford_combine` (`w_safe = max(w, DBL_MIN)`).
- Patch 0 keeps `(loc0, scale0)`. Other patches: `loc_j`, `scale_j` cast to float32.
- `refined = clamp_nonfinite(((q - loc0) / scale0) * scale_j + loc_j)` in float32.
  Covariate rows keep the unrefined quantiles.

Parity flags:

| # | place | why it matters | how the port reproduces it |
|---|---|---|---|
| P23 | Hillis-Steele scan | float64 results depend on the merge tree | implement the same scan literally (same rounds, same argument order, same zero padding), not a sequential Welford |
| P24 | float64 sums over 32 steps in `advancing_block_frames` | torch vectorized reduction order depends on the CPU ISA | sequential double sum; gate the float32-cast frames at <= 1 ulp |
| P25 | `clip` constant | `FLT_MAX ** 0.5` in Python is `pow`, not `sqrt` | `std::pow((double)FLT_MAX, 0.5) / 1e3` |
| P26 | float32 re-normalization `(q - loc0)/scale0` | undoes the anchor frame with float32 rounding, and does not recover the head output exactly | reproduce literally in float32, in this order; do not reuse the pre-denormalization head output |
| P27 | AR feedback | errors in block k's median enter block k+1's context | gate block by block with teacher forcing (D.3 G6.2) and free-running (G6.3) |
| P28 | symmetric averaging is exactly odd | the reference test asserts `forecast(-x) == -flip(forecast(x))` with `atol = 0` | the C++ port must keep this exact (invariant I6.4) |
| P29 | `0.5 * (a + b)` | operation order | `0.5f * (a + (-m_flipped))`, float32 |

### A.11 Known limits of the reference that the port keeps

- With covariates, a decode block must be `<= block_size` (assert in `_predict_window`).
- Known covariates need a schedule that `supports_known_covariates` (`config.py`); the
  loader computes this from `block_schedule` and returns an error like the reference.
- A NaN in the output is an error, not a value.

---

## B. GGUF schema

`general.architecture = "timestral"`. All keys use the prefix `timestral.`. The loader
reads every one of them and fails on a missing key. It also runs the same checks as
`TimestralConfig.__post_init__`.

### B.1 KV keys

| key | type | source | used for |
|---|---|---|---|
| `general.architecture` | str | `"timestral"` | |
| `general.name` | str | e.g. `"Timestral-209M"` | |
| `general.license` | str | the full text of the weights' LICENSE file, including the attribution notice | licence condition 1 (section F.1) |
| `general.license.name` | str | `"Modified MIT"` | |
| `general.license.link` | str | the LICENSE URL on the Hub | |
| `general.source.url` | str | the Hub repo URL | |
| `timestral.source.hf_revision` | str | `9874d252...` | traceability |
| `timestral.source.code_sha` | str | `a534bdbd...` | traceability |
| `timestral.source.weights_sha256` | str | SHA-256 of `model.safetensors` | traceability |
| `timestral.converter_version` | u32 | the converter's schema version (start at 1) | loader compatibility |
| `timestral.patch_size` | u32 | `patch_size` | |
| `timestral.embedding_length` | u32 | `d_model` | |
| `timestral.attention.head_count` | u32 | `num_heads` | |
| `timestral.feed_forward_length` | u32 | `mlp_hidden_dim` | |
| `timestral.block_count` | u32 | `len(block_schedule)` | |
| `timestral.block_schedule` | arr[str] | `block_schedule` | layer kinds |
| `timestral.quantile_knots` | arr[f64] | `quantile_knots` | widths (P16), output levels |
| `timestral.block_size` | u32 | `block_size` | |
| `timestral.context_length` | u32 | `context_length` | default `max_context` |
| `timestral.rope.freq_base` | f64 | `rope_theta` | |
| `timestral.rope.partial_factor` | f64 | `rope_partial_factor` | |
| `timestral.rope.dimension_count` | u32 | `rope_dim` (derived, for a load-time check) | |
| `timestral.rope.table_pos_min` | i32 | first position of `rope.cos`/`rope.sin` (P10) | table lookup |
| `timestral.attention.layer_norm_rms_epsilon` | f32 | `norm_eps` (torch adds it as float32) | |
| `timestral.attention.bias` | bool | `attn_bias` | |
| `timestral.mlp.bias` | bool | `mlp_bias` | |
| `timestral.attention.time_gate` | str | `time_attn_gate` | `none` / `sdpa_sigmoid` |
| `timestral.attention.variate_gate` | str | `variate_attn_gate` | |
| `timestral.attention.split_context_horizon` | bool | `split_context_horizon_attn` | |
| `timestral.attention.qkv_head_major` | bool | `true` (the converter permuted the rows, B.3) | loader refuses `false` |
| `timestral.scaler.minimum_scale` | f64 | `minimum_scale` | float64 clamp (P2) and float32 clamp |
| `timestral.scaler.use_asinh` | bool | `use_asinh` | |
| `timestral.scaler.stabilize_with_global` | bool | must be `false` | loader refuses `true` |
| `timestral.covariate_enabled` | bool | `covariate_enabled` | |
| `timestral.attend_missing_patches` | bool | `attend_missing_patches` | |
| `timestral.quantile_width_factor` | f64 | `quantile_width_factor` | |
| `timestral.forecaster.per_shot_horizon_patches` | u32 | 32 (constructor default) | overridable at run time |
| `timestral.forecaster.quantile_real_cap_k` | f64 | 1e4 | overridable |
| `timestral.forecaster.extreme_value_threshold` | f64 | 1e30 | overridable |
| `timestral.forecaster.symmetric_averaging` | bool | true | overridable |
| `timestral.forecaster.clip_nonnegative` | bool | true | overridable |
| `timestral.target_encoding.smooth` | f64 | 1.0 | |

The converter rejects unknown `config.json` keys, like `TimestralConfig.from_dict`.

### B.2 Tensor map

GGUF/ggml shapes are listed as `ne` (fastest first), the reverse of the torch shape. A
torch `nn.Linear` weight `(out, in)` becomes `ne = [in, out]`, which is what
`ggml_mul_mat(W, x)` takes. **No weight is transposed** and **no weight is tied** (each of
the 203 safetensors entries is separate; the converter asserts that no two entries share
storage).

| safetensors name | torch shape | GGUF name | `ne` |
|---|---|---|---|
| `representation.patch_proj.hidden_layer.weight` | (3072, 64) | `patch.hidden.weight` | [64, 3072] |
| `representation.patch_proj.hidden_layer.bias` | (3072) | `patch.hidden.bias` | [3072] |
| `representation.patch_proj.output_layer.weight` | (768, 3072) | `patch.out.weight` | [3072, 768] |
| `representation.patch_proj.output_layer.bias` | (768) | `patch.out.bias` | [768] |
| `representation.patch_proj.residual_layer.weight` | (768, 64) | `patch.res.weight` | [64, 768] |
| `representation.patch_proj.residual_layer.bias` | (768) | `patch.res.bias` | [768] |
| `representation.forecast_query` | (768) | `patch.forecast_query` | [768] |
| `transformer.layers.{i}.attn_norms.{j}.scale` | (768) | `blk.{i}.attn_norm.{j}.weight` | [768] |
| `transformer.layers.{i}.attns.{j}.wQKV.weight` | (2304, 768) | `blk.{i}.attn.{j}.qkv.weight` (rows permuted, B.3) | [768, 2304] |
| `transformer.layers.{i}.attns.{j}._gate.weight` | (768, 768) | `blk.{i}.attn.{j}.gate.weight` | [768, 768] |
| `transformer.layers.{i}.attns.{j}.wO.weight` | (768, 768) | `blk.{i}.attn.{j}.out.weight` | [768, 768] |
| `transformer.layers.{i}.mlp_norm.scale` | (768) | `blk.{i}.ffn_norm.weight` | [768] |
| `transformer.layers.{i}.mlp.w12.weight` | (4096, 768) | `blk.{i}.ffn_w12.weight` (rows `0..H-1` = the linear half `g`, rows `H..2H-1` = the silu half `u`) | [768, 4096] |
| `transformer.layers.{i}.mlp.w3.weight` | (768, 2048) | `blk.{i}.ffn_w3.weight` | [2048, 768] |
| `transformer.out_norm.scale` | (768) | `out_norm.weight` | [768] |
| `head.projection.hidden_layer.weight` | (3072, 768) | `head.hidden.weight` | [768, 3072] |
| `head.projection.output_layer.weight` | (672, 3072) | `head.out.weight` | [3072, 672] |
| `head.projection.residual_layer.weight` | (672, 768) | `head.res.weight` | [768, 672] |
| (derived) | (32) | `rope.inv_freq` (F32, computed by the reference `SinCosRoPE._build_inv_freq`) | [head_dim/2] |
| (derived) | (P, 64) | `rope.cos`, `rope.sin` (F32, computed by the reference `SinCosRoPE.forward`, P10) | [head_dim, P] |

`j` runs over `0..len(LAYER_KIND_AXES[kind])-1`: one attention for `time`/`variate`, two
for `time_variate`/`variate_time`, in the order of the axes.

Biases (`layers.py: Attention.__init__`, `FusedSwiGLU.__init__`, `model.py:
PatchRepresentation.__init__`, `QuantileHead.__init__`):

- `attn_bias` controls three biases per attention: `wQKV.bias` (GGUF
  `blk.{i}.attn.{j}.qkv.bias`, `[3D]`, permuted with the same B.3 row permutation as the
  weight), `wO.bias` (`.out.bias`, `[D]`) and `_gate.bias` (`.gate.bias`, `[D]`).
- `mlp_bias` controls `w12.bias` (`blk.{i}.ffn_w12.bias`, `[2F]`) and `w3.bias`
  (`blk.{i}.ffn_w3.bias`, `[D]`).
- The patch embedding always has its three biases (`use_bias=True` is hardcoded).
- The head never has biases (`use_bias=False` is hardcoded).

Both flags are false in every released checkpoint, so the converter and loader must
handle the `true` case from the flags, and the tiny checkpoint is built twice (M0b): once
with both flags false and once with both true, so the bias paths are tested.

Shape checks (converter and loader, all from KV):

- `qkv`: `[D, 3D]`; `gate`, `out`: `[D, D]`; `ffn_w12`: `[D, 2F]`; `ffn_w3`: `[F, D]`.
- `patch.hidden`: `[2*ps, 4*D]`; `patch.out`: `[4*D, D]`; `patch.res`: `[2*ps, D]`.
- `head.hidden`: `[D, 4*D]`; `head.out`, `head.res`: `[.., ps*Q]`.
- Tensor count without biases: `11 + sum over layers of (3 + 4 * n_attn) + 3` (the `+3`
  are `rope.inv_freq`, `rope.cos`, `rope.sin`). For 209M: `11 + 18*7 + 6*11 + 3 = 206`.
  The safetensors file has 203 (verified from its header). With `attn_bias`, add
  `3 * n_attn` per layer; with `mlp_bias`, add 2 per layer.
- `rope.cos`/`rope.sin`: `[head_dim, pos_max - pos_min + 1]`, with KV
  `timestral.rope.table_pos_min` (i32).

### B.3 The QKV row permutation

Torch row index `o = s*D + d*H + h` (`s` in q/k/v, `d < head_dim`, `h < n_heads`).
Converter: `W.reshape(3, head_dim, H, D).permute(0, 2, 1, 3).reshape(3*D, D)`, so GGUF row
`o' = s*D + h*head_dim + d`. After this, q, k and v are views of the matmul output
(`ne = [head_dim, H, S, R]` with offsets `0`, `D`, `2D` elements in each token). The
attention output and `W_O`/`W_gate` already use head-major order; they are not permuted.
If `attn_bias` is true, `wQKV.bias` gets the same permutation. The dump stores the `wQKV`
output in torch row order, so the q/k/v comparisons in the tests undo B.3 on the dumped
tensor (or apply it to the C++ tensor) before they compare.
`tests/python/check_convert.py` undoes the permutation and checks the bytes against
safetensors.

### B.4 Quantization allowlist (M7)

Quantize only `ggml_mul_mat` weights: `blk.*.attn.*.{qkv,gate,out}.weight`,
`blk.*.ffn_w12.weight`, `blk.*.ffn_w3.weight`, `patch.out.weight`, `head.hidden.weight`,
`head.out.weight`, `head.res.weight`. Keep F32: norms, biases, `patch.forecast_query`,
`rope.inv_freq`, `rope.cos`, `rope.sin`, and `patch.hidden.weight`/`patch.res.weight` (row length 64: valid for
`q8_0`/`f16` blocks of 32, but not for K-quants with blocks of 256; keep them F32 for every
type, because they are 0.1% of the parameters).

---

## C. Reference dumps (Gate 0)

### C.1 Python environment

`scripts/setup_reference_env.sh` creates a venv at `${TIMESTRAL_REF_VENV:-$TIMESTRAL_WORK/venv-ref}`:

```bash
python3.12 -m venv "$VENV"   # Python 3.12 exactly; numpy depends on it (below)
"$VENV/bin/pip" install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install \
  "timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff" \
  numpy==2.5.4 safetensors==0.8.0 "huggingface_hub>=0.23" gguf==0.19.0
"$VENV/bin/python" -c "import timestral, torch, numpy, gguf; print(timestral.__version__, torch.__version__)"
```

The versions come from the reference's `uv.lock`: `torch==2.14.1` (one entry for every
Python) and `safetensors==0.8.0`. The lock has three numpy entries by Python version:
2.2.6 for Python < 3.11, 2.4.6 for 3.11 and 2.5.4 for >= 3.12. This plan pins Python 3.12
and `numpy==2.5.4`. `gguf==0.19.0` exists on PyPI. Write the Python version and all pins
to `scripts/requirements.txt` and record them in the baseline KV (`env.python`).

### C.2 Test cases

`tests/cases.json` lists every case. Inputs are generated by NumPy
`default_rng(seed)` inside the dump script and **stored in the baseline**, so the C++
tests read the exact inputs and never regenerate them. Generator: `10*sin(2*pi*t/period)
+ cumsum(normal)` plus a per-case level and trend (as in the reference tests).

| case | B x V x T | covariates | H | what it covers |
|---|---|---|---|---|
| `uni_short` | 2 x 1 x 500 | none | 96 | left padding (500 is not a multiple of 32), one block of 3 patches, refine with `n_pred > 1` |
| `uni_one_patch` | 1 x 1 x 256 | none | 20 | `n_pred = 1` (refine skipped), H not a multiple of 32 |
| `multi` | 2 x 3 x 1000 | none | 64 | variate attention with `R = 6`, group mask |
| `past_cov` | 1 x 1 x 800 | 2 numeric past | 48 | covariate path, firewall, past-row feedback |
| `known_cov` | 1 x 1 x 800 | 1 numeric known + 1 indicator (0/1) known | 64 | bidirectional rows, instance-norm scaling, indicator rule |
| `cat_uni` | 1 x 1 x 600 | 1 categorical past + 1 categorical known (one horizon code unseen in the context, NaN codes) | 40 | target encoding, unseen slot, NaN slot |
| `cat_multi` | 1 x 2 x 600 | 1 categorical past | 40 | raw codes path |
| `long_h` | 1 x 1 x 4096 | none | 2500 | 3 blocks (32, 32, 15 patches), feedback, partial last block |
| `long_h_known` | 1 x 1 x 2048 | 1 known | 1100 | 2 blocks on the covariate path, known-future feedback |
| `nans` | 2 x 2 x 900 | 1 past with NaNs | 64 | leading NaN run (late start), scattered NaNs, one fully missing patch inside the series |
| `nonneg` | 1 x 1 x 700 | none | 64 | positive series with level 1e6: clip path, large level |
| `const` | 1 x 1 x 512 | none | 64 | constant series (value 5.0): scale clamps to `minimum_scale` |
| `all_missing` | 2 x 1 x 512 | none | 32 | series 0 normal, series 1 all NaN: no valid key, `first_obs = 0`. If the reference raises, the baseline stores `expect.error` (exception type and message) instead of outputs, and the port must return an error at the same step |
| `ctx_cap` | 1 x 1 x 40000 | none | 32 | context cap at 32768, label `slow` |
| `bench_32k` | 1 x 1 x 32768 | none | 1024 | the benchmark shape (only final outputs and layers 0 and 23 are dumped), label `slow` |

The tiny checkpoint (M0b) runs the same cases scaled down (`T / 4`, `H / 4`, patch 8,
block 4), except `bench_32k`, and has schedule `(time, variate, time_variate,
variate_time)`. Its `context_length` is **2048**, not the 256 of the reference tests:
with 256, most scaled cases would be cut by the context cap and the tests would not
cover what the table says. With 2048, only `ctx_cap` (T = 10000) hits the cap, as in the
209M set. `long_h` scaled (T = 1024, H = 625 -> 79 patches of 8, blocks of 4) covers 20
decode blocks.

The schema of `tests/cases.json` is defined in M0d.

### C.3 The dump script: `scripts/gen_reference_baseline.py`

```
gen_reference_baseline.py --model <hub id or dir> --revision <sha> --cases tests/cases.json
    [--case NAME ...] --mode f32|f32-math|f64|f64-rerun|f32-math-rerun
    --output-dir $TIMESTRAL_WORK/baselines
    [--layers all|0,1,3,11,23]
```

Determinism setup, at the top, before the model loads:

```python
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)          # gold files are single-thread; see the check below
torch.manual_seed(0); np.random.seed(0)
torch.set_default_dtype(torch.float32)
```

The model loads with `timestral.load_model(..., device="cpu", dtype=torch.float32)` and
the forecaster with the constructor defaults. Run each case twice and assert the two
results are **bitwise equal**; fail otherwise.

Modes:

- `f32`: the reference exactly as shipped. These are the gold files.
- `f32-math`: the same, inside `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`. It
  measures how much the reference moves when only the attention kernel changes.
- `f64`: the "truth" run. `model.to(torch.float64)`. Call `_forecast_raw` (and the rest
  of `forecast`) with the float32-prepared inputs upcast to float64, because
  `_as_context` forces float32. Two reference functions do part of their work in float32
  whatever the dtype: `RMSNorm.forward` (`x.float()`) and `SinCosRoPE.forward` (float32
  tables). In this mode only, the script replaces them with float64 copies of the same
  formula. List the replacements in the baseline KV `truth.patches`. This run is
  free-running: its tensor at layer 20 carries the f64 error of layers 0..19 too, so it
  gives the floor only for the free-running gates.
- `f64-rerun`: the floors for the teacher-forced gates. It reads the `f32` baseline and
  reruns each component **alone** in float64 (with the same replacements as `f64`) on the
  dumped f32 input upcast to float64: the patch embedding on the dumped `scaled` and mask;
  each layer `l` on the dumped layer-`l` input, with the dumped masks and positions; each
  sublayer on its dumped normed input; the head on the dumped `out_norm`;
  `predict_block` + `_refine_block` of block `k` on the dumped block-`k` arguments. Tensor
  names are the same as in the `f32` file.
- `f32-math-rerun`: the same component reruns in float32 with the math SDPA kernel.
- Rerun self-check (part of `f64-rerun`): run the same rerun harness on the **f64**
  free-running inputs and compare with the f64 free-running outputs of that component. The
  difference must be at most `1e-12 * rms`. A larger difference means the harness does not
  run the same computation as the model (a wrong mask, position or argument), and the
  script exits with an error. This is the check that the f64 rerun itself is consistent.

Instrumentation (wrapping only; no change to the computation):

- Wrap functions **in the namespace that calls them**. `model.py` and `forecaster.py`
  bind names with `from ... import`, so a wrap of `timestral.scaler.X` or
  `timestral.masks.X` is not seen by them. Patch these names:
  - in `timestral.layers` (called inside that module): `_apply_rotary` (post-RoPE q, k
    and the cos/sin tables), `_sdpa`. With `split_context_horizon_attn`, `_sdpa` runs
    **twice** per time attention (context queries, then horizon queries): record both
    calls and store their concatenation along the sequence axis, in call order, as the
    pre-gate attention output. The wrapper also computes the maximum `|q.k| * scale`
    over the unmasked entries for the cap statistics;
  - in `timestral.model`: `build_forecast_time_mask`, `build_forecast_positions`,
    `build_two_stream_time_mask`, `build_two_stream_variate_mask`,
    `build_two_stream_positions`, `compute_enc_key_validity`,
    `group_first_observed_patch`, `causal_patch_loc_scale`, `apply_affine_asinh_masked`,
    `denormalize_quantiles`, `instance_norm_stats` (its third output is `is_binary`);
  - in `timestral.masks`: `compute_block_layout` (called by `build_two_stream_time_mask`
    inside that module);
  - in `timestral.forecaster`: `advancing_block_frames`, `causal_patch_loc_scale`,
    `clamp_nonfinite`, `target_encode`.
  `check_baseline.py` fails if any wrapped function was never called for a case that
  should call it.
- Forward hooks: `representation.patch_proj` (output), every `transformer.layers[i]`
  (input and output), every `attn_norms[j]`, `attns[j]` (output = after `W_O`),
  `attns[j].wQKV` (output, in torch row order: undo B.3 before comparing), `attns[j]._gate`
  (output, pre-sigmoid),
  `attns[j].wO` (pre-hook: the gated input), `mlp_norm`, `mlp`, `out_norm`,
  `head.projection` (raw), `head` (monotone, normalized space).
- Method wraps: `TimestralModel.predict_block` (all arguments and the return),
  `TimestralModel.scale`, `_known_cov_scaling`, `TimestralForecaster._assemble_rows`,
  `_refine_block`, `_forecast_raw`, `_cap_quantiles`, `_encode_categorical`.
- Every record has a prefix `<pass>.b<k>.` with `pass` in `pos` (the forward pass) and
  `neg` (the mirrored pass) and `k` the block index. Names after the prefix (the cap
  table in D.3 matches on them): `scaled`, `loc`, `scale`, `known.loc`, `known.scale`,
  `known.is_binary`, `patch.in`, `patch.hidden_out`, `patch.out`, `layer.<l>.in`,
  `layer.<l>.attn.<j>.norm`, `.q`, `.k`, `.v`, `.cos`, `.sin`, `.q_rope`, `.k_rope`,
  `.sdpa`, `.gate`, `.out`, `layer.<l>.mlp_norm`, `layer.<l>.mlp`, `layer.<l>.out`,
  `out_norm`, `head.raw`, `head.monotone`, `predict`, `frames.loc`, `frames.scale`,
  `refined`, and the mask, position and argument tensors of `predict_block` under
  `args.*` and `mask.*`.

Per tensor, the script also writes condition statistics as KV, used for the a-priori caps
(D.3): `stat.<name>.rms`, `stat.<name>.numel`; for each linear output
`stat.<name>.kappa = max_i (|x| @ |W|^T)_i / rms(y)` (a max over elements, because the
error measure is a max over elements divided by `rms`); for attention
`stat.<name>.max_logit`; for each sublayer `stat.<name>.rho = rms(sublayer out) /
rms(residual stream)`.

Final outputs per case: `final.quantiles` with the defaults (symmetric averaging and clip
on), and `final.raw.pos` / `final.raw.neg` (`_forecast_raw` of each pass).

### C.4 File format

One baseline GGUF per case and mode: `baseline-<case>-<mode>.gguf`, written with
`gguf.GGUFWriter` (`arch = "timestral-baseline"`). The C++ tests read it with the same
loader code as parakeet.cpp's `tests/parity.hpp` (`load_baseline`, `load_baseline_i32`,
KV readers).

- Model float tensors: F32 for `f32` modes, F64 for `f64` mode, with the
  following categorical producer contracts (also in `f32-math`):

  | Record suffix | f32 / f32-math | f64 |
  | --- | --- | --- |
  | `enc_cat.N.codes` | F64 | F64 |
  | `enc_cat.N.target`, `enc_cat.N.out` | F32 | F64 |
  | `target_encode.N.ctx`, `.past`, `.future` | F64 | F64 |
  | `target_encode.N.out_past`, `.out_future` | F32 | F32 |

  The f64 truth patch upcasts the forecaster encoding result, not the NumPy
  target-encoding outputs. Stored numeric inputs remain F32; stored categorical
  inputs remain F64. NaN is kept in genuine context/code inputs, including
  `enc_cat.N.codes` / `.target` and `target_encode.N.ctx` / `.past` / `.future`.
  Floor comparisons require matching NaN positions. Encoded outputs cannot
  contain NaN or infinity, even when the multivariate raw-code path retains a
  missing code. No input or output may contain infinity.
- Bool tensors (masks, observation masks, `key_valid`, `active`): I8 (0/1).
- Integer tensors (positions, `first_obs`, `anchor`, `group_ids`, `role_ids`, slots): I32.
- Torch row-major arrays are written as they are; GGUF reverses the shape.
- KV: `case.*` (all case parameters), `env.torch`, `env.numpy`, `env.timestral_sha`,
  `env.hf_revision`, `env.cpu_capability` (`torch.backends.cpu.get_cpu_capability()`),
  `env.sdpa_kernel` (the kernel seen in a `torch.profiler` trace of one forward:
  `aten::_scaled_dot_product_flash_attention_for_cpu` or the math op), `env.threads`,
  `env.python`, `env.glibc` (`platform.libc_ver()`).

`scripts/compute_floors.py --dir DIR [--case NAME ...]` reads, per case,
`baseline-<case>-{f32,f32-math,f64,f64-rerun,f32-math-rerun}.gguf` from `DIR` and adds to
the `f32` file, for every float tensor `t`:

- `floor_fr.<t> = max(max|f32 - f64|, max|f32 - f32math|) / rms(f64)` (free-running
  floor);
- `floor_tf.<t> = max(max|f32 - f64rerun|, max|f32 - f32mathrerun|) / rms(f64rerun)`
  (teacher-forced floor; only for tensors that a rerun produces);
- `cap.<t>`, from the cap table in D.3 matched on the tensor name.

The C++ tests read `floor_fr.*`, `floor_tf.*` and `cap.*` from the same file. The same
command is used everywhere in this plan.

Size budget: at most 300 MB per real-model case; `--layers` limits per-layer dumps for the
long cases. Real-model baselines are never committed. The tiny-checkpoint baselines are
small (well under 10 MB in total) and are committed under `tests/fixtures/tiny/`, so CI
runs the graph gates without the real weights. Host math (libm, the torch math kernels,
the CPU ISA of the reductions) differs between machines, so committed baselines are not
valid for the strict host gates everywhere; see D.4.

---

## D. Parity gates and tolerances

### D.1 Comparison kinds

- **EXACT**: integer and boolean tensors, shapes, block counts, token counts, and float
  tensors produced only by copies and selects. Every element equal. EXACT on floats is
  IEEE value equality (+0 == -0); signed-zero differences are counted and reported.
- **ULP(k)**: float32, every element within `k` units in the last place of the reference
  value (the distance between bit patterns, with `+0 == -0`, NaN equal to NaN at the same
  positions). Used only where no cancellation can happen.
- **MAG(k, M)**: float32, `|c_i - r_i| <= k * u * M_i`, `u = 2^-24 = 5.96e-8`, where `M_i`
  is a magnitude stated per gate (for example the largest absolute value in the row, or the
  sum of the magnitudes of the terms). Used where the result can be much smaller than its
  inputs (cancellation), so an ulp of the result is not a meaningful unit.
- **NORM**: float tensors from the ggml graph:
  `err = max_i |c_i - r_i| / rms(r)`, and the gate is `err <= tol` with
  `tol = max(2 * floor, 16 * u)`. `floor` is `floor_tf` for a teacher-forced gate and
  `floor_fr` for a free-running gate (C.4).

Why `2 * floor`: the floor is how far the shipped f32 reference is from a float64
evaluation of the same computation (and from itself with another attention kernel). For a
teacher-forced gate, the float64 evaluation is a rerun of that one component on the same
f32 input, so the floor contains only the rounding of that component, not error carried
in from earlier components. The C++ port is an independent float32 evaluation with the
same kind of rounding. If it is no further from the reference than twice the reference's
own distance from the truth, it is as exact as the reference is (triangle inequality).
This is the meaning of "exact in practice" in this plan. The `16u` minimum avoids a zero
tolerance when the f32 reference happens to equal the f64 result.

The cap is an a-priori bound used as a **sanity check of the floor, not as a pass/fail
limit on the port**. If `2 * floor > cap`, the test prints a warning, still gates with
`tol = max(2 * floor, 16u)`, and the case goes in `docs/parity.md` for review. The floor
is trusted because of the rerun self-check (C.3): if that check fails, the dump script
already stopped. So a correct port cannot fail a gate because of the cap.

The caps use the probabilistic rounding bound for a dot product of length `K`:
`|error| <= lambda * sqrt(K) * u * sum_k |x_k w_k|`, which holds for one element with
probability at least `1 - delta`. For all `N` elements of a tensor at once, take
`lambda(N) = sqrt(2 * ln(2 * N / delta))` with `delta = 1e-6`, where `N` is
`stat.<name>.numel`. Examples: `lambda(1e6) = 7.5`, `lambda(1e8) = 8.1`. Define
`gamma(K, N) = lambda(N) * sqrt(K) * u`. For `N = 1e6`: `gamma(64) = 3.6e-6`,
`gamma(768) = 1.2e-5`, `gamma(2048) = 2.0e-5`, `gamma(3072) = 2.5e-5`. The dumped `kappa`
(a max over elements, C.3) converts the bound on `sum |x_k w_k|` into a bound on
`max |error| / rms(y)`.

### D.2 What the floor covers

The floor includes the GEMM order (P11), the SDPA kernel (P12), the activation functions
(P14) and the reductions (P15). It does not cover a wrong operation, a wrong layout or a
wrong mask: those give errors orders of magnitude above the floor. That is how the gates
find them.

### D.3 Gates per milestone

"Teacher-forced" means that the C++ component gets the reference's own input tensor from
the dump. "Free-running" means it gets the output of the C++ component before it.

| gate | milestone | tensor | kind | bound and why |
|---|---|---|---|---|
| G0.1 | M0b | every GGUF tensor vs safetensors (after undoing B.3) | bytes equal | the converter must not change a value |
| G0.2 | M0b | config read by the loader vs `config.json` | EXACT | |
| G0.3 | M0d | `rope.cos`/`rope.sin` vs every runtime table in the dumps | EXACT | P10 caveat; a mismatch is recorded with its positions and decides G2.2; the output also records the count of signed-zero differences, apart from the value differences |
| G1.1 | M1 | causal `loc`, `scale` | ULP(0) | float64 sequential prefix sums cast to float32 (P1, P2). If it fails with 1-ulp differences only, P1 is false: record it in `docs/parity.md` and use ULP(1) |
| G1.2 | M1 | `scaled` (asinh) on causal rows; known rows: `MAG(4, max \|x\| / scale + 1)` | ULP(2) | the subtraction and the division have bit-equal inputs (G1.1) and are correctly rounded; `asinhf` vs torch's `asinh` are each <= 1 ulp from the true value. The M0d probe gives the measured distance |
| G1.3 | M1 | known rows `loc`, `scale`, `is_binary` | `loc`: MAG(32, `max_t \|x_t\|` over observed steps of the row); `scale`: MAG(32, `max_t (x_t - loc)^2 / scale`); `is_binary`: EXACT | torch sums float32 in a cascade order that the port does not copy (P5). Each order has error <= `(log2 n + 1) * u * sum \|x\|` and `sum \|x\| / n <= max \|x\|`, so two orders differ by <= `2 * 16 * u * max \|x\|` for `n <= 33,792`. For the scale, the variance error is bounded the same way by `max (x - loc)^2`, and `d sqrt(v) = dv / (2 sqrt(v))`. A mean near 0 gives no unbounded ulp error with this measure |
| G1.4 | M1 | `denormalize_quantiles`, `clamp_nonfinite` on dumped inputs | MAG(6, `\|sinh(x) * scale\| + \|loc\|`) | `sinhf` (<= 1 ulp each side, so 2), one multiply, one add; the add can cancel when `loc` and `sinh(x)*scale` have opposite signs, so the bound uses the magnitudes of the terms |
| G2.1 | M2a | `rope.inv_freq` | EXACT | stored by the converter (P9) |
| G2.2 | M2a | `cos`, `sin` rows gathered for the dumped positions | EXACT | stored tables (P10), if G0.3 passed; else ULP(1) at the positions G0.3 listed, with the reason in `docs/parity.md`. The output records the count of signed-zero differences |
| G2.3 | M2a | post-RoPE q, k (given dumped q, k, cos, sin) | ULP(0) | the same IEEE operations (P8) |
| G2.4 | M2a | patch embedding, teacher-forced | NORM with `floor_tf` | |
| G2.5 | M2b, M2c | one layer of each kind, teacher-forced, residual stream out; also each sublayer (q, k, v, pre-gate attention, gate, attention out, MLP out), each teacher-forced on its own dumped input | NORM with `floor_tf` | |
| G3.1 | M3 | every layer output, teacher-forced | NORM with `floor_tf` | localizes a bad layer |
| G3.2 | M3 | every layer output and `out_norm`, free-running | NORM with `floor_fr` | the test also reports `err_l / err_{l-1}`; a ratio above 1.5 at several layers in a row is error amplification, and must be reported even if the gate passes |
| G4.1 | M4 | head raw output, teacher-forced from the dumped `out_norm` | NORM with `floor_tf` | |
| G4.2 | M4 | monotone quantiles (host), teacher-forced from the dumped raw | MAG(8, the largest magnitude in the row) | `log1pf(expf)` vs torch softplus (<= 2 ulp each, measured by the M0d probe), a cumsum of <= 10 terms in the same order |
| G4.3 | M4 | `predict_block` output on the context-only cases, free-running from the inputs | NORM with `floor_fr`, in normalized space: `err = max \|c - r\| / (scale0 * cosh(asinh((r - loc0)/scale0)))` divided by `rms` of the normalized reference | `sinh` multiplies a normalized-space error by `cosh(x)` (P19). Normalized space is where the model computes |
| G5.1 | M5 | every mask, `key_valid`, `first_obs`, positions, `anchor`, `active`, `role_ids`, `group_ids`, assembled masks, padding | EXACT | integer logic |
| G5.2 | M5 | assembled float rows (ffill, `nan_to_num`, cap, padding) | EXACT | copies and selects only |
| G5.3 | M5 | target encodings | ULP(0); ULP(1) only with a written reason | float64 with NumPy's pairwise sum reproduced (P20) |
| G5.4 | M5 | `predict_block` output on the covariate cases | as G4.3 | |
| G6.1 | M6a | `advancing_block_frames` on dumped inputs; `_refine_block` on dumped inputs | frames: ULP(1); refined: MAG(8, `(\|q\| + \|loc0\|) * scale_j / scale0 + \|loc_j\|`) | frames are float64 math cast to float32 with an ISA-dependent float64 sum order (P24). The refine does `q - loc0` (which can cancel), a division, a multiply and an add in float32; the bound uses the term magnitudes |
| G6.2 | M6b | each block, teacher-forced (the C++ `predict_block` + refine gets the dumped context of block k) | as G4.3 per block, with `floor_tf` from the block-`k` rerun | no feedback error, so this is the single-block check for every block |
| G6.3 | M6b | each block and `final.quantiles`, free-running | as G4.3, with `floor_fr` | the f64 run measures the error growth through the feedback (P27); the test prints the error per block |
| I6.4 | M6b | invariant: `forecast(-x)` vs `-flip(forecast(x))` with clip off | bitwise equal | holds for any forward pass that keeps the averaging formula (P28, P29), so it checks the averaging code, **not** parity |
| I6.5 | M6b | invariant: the same forecast with 1 thread and with N threads; two runs in one process | bitwise equal | checks determinism of the port, **not** parity |
| I6.6 | M6b | invariant: quantiles non-decreasing along Q; output shape `(B, V, H, Q)` | EXACT | **not** parity |
| G7.x | M7 | quantized types vs the f32 C++ output | see M7 | not a parity gate |

Cap table (implemented in `scripts/compute_floors.py` as an ordered list of
`(regular expression on the tensor name, formula)`; the first match wins; a float
tensor that matches no entry makes the script fail). `N` is the tensor's `numel`,
`kappa`, `max_logit` and `rho` are the `stat.*` values of that tensor, `S` is the
attention sequence length of the case.

| tensor name pattern | cap |
|---|---|
| `\.patch\.hidden_out$` | `gamma(2*ps, N) * kappa + 2u` |
| `\.patch\.out$` | `(gamma(2*ps, N) + gamma(4*D, N)) * kappa + gamma(2*ps, N) * kappa_res + 4u` |
| `\.layer\.\d+\.attn\.\d+\.(q\|k\|v)$` | `gamma(D, N) * kappa` |
| `\.layer\.\d+\.attn\.\d+\.(q\|k)_rope$` | `gamma(D, N) * kappa + 4u` |
| `\.layer\.\d+\.attn\.\d+\.sdpa$` | `gamma(head_dim, N) * (1 + 2 * max_logit) + gamma(S, N) + 8u` |
| `\.layer\.\d+\.attn\.\d+\.gate$` | `gamma(D, N) * kappa` |
| `\.layer\.\d+\.attn\.\d+\.out$` | `cap(sdpa) + gamma(D, N) * kappa + 4u` |
| `\.layer\.\d+\.mlp$` | `(gamma(D, N) + gamma(F, N)) * kappa + 4u` |
| `\.layer\.\d+\.out$` | `sum over the layer's sublayers of rho * cap(sublayer)` |
| `\.out_norm$` | `cap(last layer out) + 4u` |
| `\.head\.raw$` | `(gamma(D, N) + gamma(4*D, N)) * kappa + gamma(D, N) * kappa_res + 4u` |
| `\.predict$`, `\.refined$`, `^final\.` | `cap(out_norm) + cap(head.raw)` (single block; for block `k >= 1` free-running no cap: printed as `n/a`) |

Expected values for the 209M model: `floor` for a layer output of about 1e-6 to 1e-5,
and for the end-to-end forecast of about 1e-5 to 1e-4 in normalized space.
**UNVERIFIED:** these are estimates. Gate 0 measures the real values; write them to
`docs/parity.md`.

### D.4 How to run a gate

Each C++ test reads `TIMESTRAL_TEST_GGUF` (the model) and `TIMESTRAL_TEST_BASELINES` (the
baseline directory), prints one line per compared tensor
(`[name] n=.. err=.. tol=.. floor=.. cap=.. -> OK|FAIL`), and returns 0 (pass), 1 (fail)
or 77 (skip: missing input). **A skip is not a pass for a milestone gate.** The gate
commands below export both variables, so the test runs.

Machine dependence. The EXACT gates on integer logic are machine-independent. The ULP,
MAG and host EXACT float gates (G1.x, G2.2, G2.3, G4.2, G5.2, G5.3, G6.1) compare against
torch and libm results, which depend on the CPU ISA and the glibc version. Each test
compares `env.cpu_capability` and `env.glibc` in the baseline with the running machine:

- equal: every gate runs;
- different (for example CI with the committed tiny baselines): those host float gates
  return 77 with the message `host gates need baselines from this machine`, and the
  integer EXACT and NORM gates still run.

Milestone gates always run on baselines generated on the machine that runs the gate
(`scripts/regen_tiny_baselines.sh` regenerates the tiny set locally; the 209M baselines
are generated locally anyway).

### D.5 When a gate fails

1. Stop. Do not start the next milestone.
2. Localize: run the teacher-forced variant of the failing gate, then the per-sublayer
   comparisons, and find the **first** tensor whose `err / floor_tf` is above 2.
3. Check that tensor against the parity flags in section A (P1 to P29). Most failures are a
   layout (P8, B.3), a chunk order (A.5.5) or a mask (A.6).
4. Do not change a tolerance to make a test pass. A change to a tolerance needs a written
   reason in `docs/parity.md` (the measured numbers, the operation that causes the
   difference, why it cannot be matched) and the owner's approval in the review.

---

## E. Milestones

The commands assume this environment (set once per shell):

```bash
export TIMESTRAL_WORK=${TIMESTRAL_WORK:-$PWD/work}
export VENV=${TIMESTRAL_REF_VENV:-$TIMESTRAL_WORK/venv-ref}
export TIMESTRAL_TEST_GGUF=$TIMESTRAL_WORK/models/timestral-209m-f32.gguf
export TIMESTRAL_TEST_BASELINES=$TIMESTRAL_WORK/baselines
export TIMESTRAL_TEST_TINY=$PWD/tests/fixtures/tiny
cmake -B build -DTIMESTRAL_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
```

Repository layout (created across M0..M8):

```
CMakeLists.txt  .gitignore  .gitmodules  AGENTS.md  NOTICE  README.md
(LICENSE: not created by any milestone until the owner picks the code licence, F.2)
third_party/ggml/                 submodule, ggml-org/ggml at v0.13.0
include/timestral.h               C++ API
include/timestral_capi.h          flat C ABI
src/common.hpp/.cpp               logging, error type
src/config.hpp                    ts::Config (every KV of B.1)
src/model_loader.hpp/.cpp         GGUF -> ts::Config + name -> tensor map, shape checks
src/backend.hpp/.cpp              backend init, persistent ggml_gallocr, graph run
src/numpy_sum.hpp                 NumPy pairwise float64 sum (P20)
src/scaler.hpp/.cpp               A.3, denormalize, clamp_nonfinite, advancing frames (A.10)
src/rope.hpp/.cpp                 inv_freq check, cos/sin tables (host)
src/layers.hpp/.cpp               graph builders: rms_norm, residual_block, attention, swiglu
src/encoder.hpp/.cpp              patch embedding + transformer stack, per-layer outputs
src/head.hpp/.cpp                 head graph + host monotone quantiles
src/masks.hpp/.cpp                A.6 (host)
src/covariates.hpp/.cpp           A.2, A.9
src/predict_block.hpp/.cpp        A.8
src/forecaster.hpp/.cpp           A.10
src/quantize.cpp                  M7
src/timestral_capi.cpp            M8
examples/cli/main.cpp             timestral-cli
scripts/setup_reference_env.sh  scripts/requirements.txt
scripts/convert_timestral_to_gguf.py  scripts/make_tiny_checkpoint.py
scripts/gen_reference_baseline.py  scripts/compute_floors.py  scripts/probe_math.py
scripts/regen_tiny_baselines.sh
scripts/bench_reference.py  scripts/eval_quant.py
tests/CMakeLists.txt  tests/parity.hpp  tests/cases.json  tests/fixtures/tiny/
tests/test_*.cpp  tests/python/check_convert.py  tests/python/check_baseline.py
docs/PLAN.md  docs/conversion.md  docs/parity.md  docs/quantization.md  docs/benchmarks.md
```

### M0a: scaffold

- **Goal:** an empty library that builds with ggml, a smoke test, and the parity test
  helpers. No model code.
- **Files:** `CMakeLists.txt` (options `TIMESTRAL_BUILD_TESTS`, `TIMESTRAL_BUILD_CLI`,
  `TIMESTRAL_SHARED`, `TIMESTRAL_GGML_CUDA/METAL/VULKAN`; `GGML_NATIVE` and
  `GGML_LLAMAFILE` on by default, as in parakeet.cpp), `.gitmodules` (ggml-org/ggml at
  `v0.13.0`), `.gitignore` (`/build*/`, `/work/`, `*.gguf`, with
  `!tests/fixtures/tiny/*.gguf`), `AGENTS.md`, `NOTICE` (upstream Apache-2.0 attribution,
  F.2), `src/common.*`, `tests/CMakeLists.txt` (a `ts_add_test` function with
  `SKIP_RETURN_CODE 77`), `tests/parity.hpp` (from parakeet.cpp, plus `compare_exact_i32`,
  `compare_exact_i8`, `compare_ulp`, `compare_mag`, `compare_norm` that reads
  `floor_tf.*`/`floor_fr.*`/`cap.*`, and the machine check of D.4), `tests/test_smoke.cpp`,
  `tests/test_parity_helpers.cpp` (each compare function on hand-made vectors, including
  `+0/-0`, NaN and a cancellation case for `compare_mag`).
- **Do not create `LICENSE`.** The code licence is an open owner decision (F.2).
- **Command:** `git submodule update --init && cmake -B build -DTIMESTRAL_BUILD_TESTS=ON && cmake --build build -j && ctest --test-dir build -R 'test_smoke|test_parity_helpers' --output-on-failure`
- **Acceptance:** both tests pass; `ldd` of a test binary shows no external `libggml`.

### M0b: converter, loader, tiny checkpoints

- **Goal:** the reference venv; a converter that writes the B.1/B.2 schema; a loader that
  reads it with no hardcoded value; two tiny checkpoints.
- **Files:** `scripts/setup_reference_env.sh`, `scripts/requirements.txt`,
  `scripts/convert_timestral_to_gguf.py`, `scripts/make_tiny_checkpoint.py`,
  `src/config.hpp`, `src/model_loader.*`, `src/backend.*` (backend init and weight mmap
  only), `tests/test_model_loader.cpp`, `tests/python/check_convert.py`,
  `docs/conversion.md`.
- **Converter:** `convert_timestral_to_gguf.py --model <hub id or dir> [--revision SHA]
  --outtype f32 --output OUT`. It reads `config.json`, `model.safetensors` and `LICENSE`
  (for a local directory without `LICENSE`, `--license-file`); validates the config with
  `timestral.TimestralConfig.from_dict`; writes every KV of B.1; permutes QKV (B.3);
  computes `rope.inv_freq`, `rope.cos` and `rope.sin` with the reference class
  (`torch.set_num_threads(1)`); writes the SHA-256 of the weights; asserts every
  safetensors tensor is used once and no two share storage.
- **Tiny checkpoints:** `make_tiny_checkpoint.py --output-dir DIR [--bias]` writes
  `config.json`, `model.safetensors` and a `LICENSE` stating that the random test weights
  are the project's own, for `patch_size 8, d_model 32, num_heads 4, mlp_hidden_dim 64,
  block_schedule (time, variate, time_variate, variate_time), knots (0.05, 0.1, 0.25,
  0.4, 0.5, 0.6, 0.75, 0.9, 0.95), block_size 4, context_length 2048,
  quantile_width_factor 3.0`, `attn_bias = mlp_bias = --bias`, `torch.manual_seed(0)`,
  every parameter `normal_(std=0.05)` (as the reference tests).
- **Commands:**
  ```bash
  scripts/setup_reference_env.sh
  $VENV/bin/python scripts/convert_timestral_to_gguf.py --model mistralai/Timestral-209M \
      --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 --outtype f32 --output $TIMESTRAL_TEST_GGUF
  $VENV/bin/python tests/python/check_convert.py --gguf $TIMESTRAL_TEST_GGUF \
      --model mistralai/Timestral-209M --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365
  for v in tiny tiny-bias; do
    flag=$([ $v = tiny-bias ] && echo --bias)
    $VENV/bin/python scripts/make_tiny_checkpoint.py --output-dir $TIMESTRAL_WORK/$v $flag
    $VENV/bin/python scripts/convert_timestral_to_gguf.py --model $TIMESTRAL_WORK/$v \
        --outtype f32 --output $TIMESTRAL_TEST_TINY/timestral-$v-f32.gguf
    $VENV/bin/python tests/python/check_convert.py --gguf $TIMESTRAL_TEST_TINY/timestral-$v-f32.gguf \
        --model $TIMESTRAL_WORK/$v
  done
  ctest --test-dir build -R test_model_loader --output-on-failure
  ```
- **Acceptance:** G0.1 and G0.2 pass for 209M, tiny and tiny-bias; `test_model_loader`
  prints the config and the tensor names (206 for 209M), checks the count formula of B.2
  for all three files, and fails when one KV is removed (a test strips a key and expects
  an error).

### M0c: dump script and baselines

- **Goal:** `gen_reference_baseline.py` with all modes and instrumentation of C.3, and
  the baselines of every case for 209M and tiny.
- **Files:** `scripts/gen_reference_baseline.py`, `tests/cases.json` (schema in M0d; M0c
  fills it with the C.2 cases), `tests/python/check_baseline.py`,
  `scripts/regen_tiny_baselines.sh` (the tiny loop below).
- **Commands:**
  ```bash
  for m in f32 f32-math f64 f64-rerun f32-math-rerun; do
    $VENV/bin/python scripts/gen_reference_baseline.py --model mistralai/Timestral-209M \
        --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 --cases tests/cases.json \
        --mode $m --output-dir $TIMESTRAL_TEST_BASELINES
  done
  scripts/regen_tiny_baselines.sh   # the same five modes for tiny and tiny-bias,
                                    # --model $TIMESTRAL_WORK/<v> --output-dir $TIMESTRAL_TEST_TINY/<v>
  $VENV/bin/python tests/python/check_baseline.py --dir $TIMESTRAL_TEST_BASELINES
  $VENV/bin/python tests/python/check_baseline.py --dir $TIMESTRAL_TEST_TINY/tiny
  ```
- **Acceptance:** `check_baseline.py` finds every tensor name of C.3 for every case and
  every wrapped function was called; each case is bitwise reproducible across two runs;
  the rerun self-check passed for every component; `const` and `all_missing` either have
  outputs or an `expect.error` record.

### M0d: floors, caps, math probe, case schema

- **Goal:** `compute_floors.py` with the cap table of D.3; the `cases.json` schema; the
  measured floors; the math probe.
- **`tests/cases.json` schema:** a JSON object `{"version": 1, "cases": [...]}`. Each case
  has: `name` (str), `B`, `V`, `T`, `H` (int), `seed` (int), `series` (`{"period": int,
  "level": float, "trend": float, "noise_std": float, "positive": bool, "constant":
  float|null}`), `missing` (`{"leading": int, "scattered_frac": float, "missing_patches":
  [int], "all_missing_series": [int]}`), `past` (`{"numeric": int, "nan_frac": float}`),
  `known` (`{"numeric": int, "indicator": int, "nan_frac": float}`), `cat_past` and
  `cat_known` (`{"rows": int, "n_codes": int, "nan_frac": float, "unseen_in_horizon":
  bool}`), `labels` (list of str, e.g. `"slow"`), `layers` (`"all"` or a list of layer
  indices), `tiny` (bool: include in the tiny set, scaled by 4 as in C.2). A missing field
  means zero or none. `check_baseline.py` validates the file against this schema.
- **Math probe:** `scripts/probe_math.py --n 1000000 --out $TIMESTRAL_WORK/probe.json`
  draws 10^6 float32 inputs per function over the ranges the model uses (`asinh` on
  [-1e7, 1e7] log-spaced and [-10, 10] uniform; `sinh` on [-80, 80]; `sin`, `cos` on
  [-1100, 1100]; softplus on [-100, 30]), runs torch (one thread) and glibc through
  `ctypes` (`asinhf`, `sinhf`, `sinf`, `cosf`, `log1pf(expf(x))`), and reports the
  histogram of ulp distances per function. The result goes in `docs/parity.md`. It tells
  M1, M2a and M4 which tolerance applies: if torch equals glibc for a function, the matching
  gate can be ULP(0).
- **Files:** `scripts/compute_floors.py`, `scripts/probe_math.py`, `docs/parity.md`.
- **Commands:**
  ```bash
  $VENV/bin/python scripts/compute_floors.py --dir $TIMESTRAL_TEST_BASELINES
  $VENV/bin/python scripts/compute_floors.py --dir $TIMESTRAL_TEST_TINY/tiny
  $VENV/bin/python scripts/compute_floors.py --dir $TIMESTRAL_TEST_TINY/tiny-bias
  $VENV/bin/python scripts/probe_math.py --n 1000000 --out $TIMESTRAL_WORK/probe.json
  ```
- **Acceptance:** every float tensor has `floor_fr`, a `cap`, and `floor_tf` when a rerun
  produces it (the script fails on a tensor that matches no cap pattern); G0.3 is run and
  its result recorded; `docs/parity.md` has the floors per tensor kind, the cases where
  `2 * floor > cap`, and the probe histograms.

### M1: scaler

- **Goal:** A.3 on the host, plus `denormalize_quantiles` and `clamp_nonfinite`.
- **Files:** `src/scaler.hpp/.cpp`, `tests/test_scaler.cpp`.
- **Gates:** G1.1, G1.2, G1.3, G1.4 on every case (block 0 and the last block, both
  passes), and on the tiny fixtures.
- **Command:** `ctest --test-dir build -R test_scaler --output-on-failure`
- **Acceptance:** all gates pass; the test covers rows with no observation, a constant
  row (scale clamps to `minimum_scale`), an indicator row and the mirrored pass (P7).

### M2a: graph infrastructure, patch embedding, RoPE

- **Goal:** graph building with a persistent `ggml_gallocr` and mmapped F32 weights; the
  patch embedding graph; host RoPE tables; the rotation in the graph.
- **Files:** `src/backend.*` (graph run), `src/layers.hpp/.cpp` (`residual_block`,
  `apply_rope`), `src/rope.*`, `src/encoder.hpp/.cpp` (embedding only),
  `tests/test_patch_embed.cpp`, `tests/test_rope.cpp`.
- **Gates:** G2.1, G2.2, G2.3, G2.4 (input: the dumped `scaled` and observation mask).
- **Command:** `ctest --test-dir build -R 'test_patch_embed|test_rope' --output-on-failure`
- **Acceptance:** gates pass for 209M and tiny; the 64-feature order is values then flags.

### M2b: time layer

- **Goal:** `TransformerBlock` of kind `time`: RMSNorm, QKV views, RoPE, masked softmax
  attention, sigmoid gate, `W_O`, SwiGLU MLP (silu on the second half).
- **Files:** `src/layers.*` (`rms_norm`, `attention_time`, `swiglu`), `src/encoder.*`
  (`run_layer(l, x, masks, cos, sin)`), `tests/test_layer_time.cpp`.
- **Gate:** G2.5 on layers 0 and 22 of 209M (`block_schedule[22] = time`; layer 23 is
  `time_variate`), teacher-forced: the input is the
  dumped layer input, the masks and positions are the dumped ones (the native mask builders
  come in M5).
- **Command:** `ctest --test-dir build -R test_layer_time --output-on-failure`
- **Acceptance:** every sublayer tensor passes; both cases `uni_short` (context path) and
  `known_cov` (covariate path, `S = 2n`) are covered.

### M2c: variate, time_variate and variate_time layers

- **Goal:** variate attention (permute to `[D, R, S]`, no RoPE, per-slot mask, the `R == 1`
  shortcut); composite kinds in their axis order.
- **Files:** `src/layers.*` (`attention_variate`), `tests/test_layer_variate.cpp`,
  `tests/test_layer_time_variate.cpp`, `tests/test_layer_variate_time.cpp`.
- **Gate:** G2.5. `time_variate` on 209M layer 3 with `multi` and `past_cov`; `variate` and
  `variate_time` on the tiny checkpoint (layers 1 and 3), the only place they occur.
- **Command:** `ctest --test-dir build -R 'test_layer_variate|test_layer_time_variate|test_layer_variate_time' --output-on-failure`
- **Acceptance:** all pass; a test with `R == 1` checks that the shortcut equals the full
  path bitwise.

### M3: full encoder

- **Goal:** the whole stack plus `out_norm` in one graph, with optional per-layer outputs
  for the tests.
- **Files:** `src/encoder.*`, `tests/test_encoder.cpp`.
- **Gates:** G3.1 (every layer, teacher-forced) and G3.2 (free-running) on `uni_short`,
  `multi`, `known_cov`, `nans`, and on the tiny fixtures.
- **Command:** `ctest --test-dir build -R test_encoder --output-on-failure`
- **Acceptance:** gates pass; the per-layer error ratio report is in the test output and
  copied to `docs/parity.md`.

### M4: quantile head

- **Goal:** the head graph on the decoder tokens; host monotone quantiles; denormalize;
  the context-only `predict_block` (with the dumped masks and positions).
- **Files:** `src/head.hpp/.cpp`, `src/predict_block.*` (context-only path),
  `tests/test_head.cpp`.
- **Gates:** G4.1, G4.2, G4.3 (context-only cases, block 0).
- **Command:** `ctest --test-dir build -R test_head --output-on-failure`
- **Acceptance:** gates pass; the output is non-decreasing along Q.

### M5: masks, positions, covariates, covariate path

- **Goal:** native mask and position builders (A.6); input preparation and row assembly
  (A.2); target encoding (A.9); the covariate `predict_block` (A.8).
- **Files:** `src/masks.*`, `src/covariates.*`, `src/numpy_sum.hpp`, `src/predict_block.*`,
  `tests/test_masks.cpp`, `tests/test_covariates.cpp`, `tests/test_predict_block.cpp`.
- **Gates:** G5.1, G5.2, G5.3, G5.4 on every case (block 0); the masks also on every later
  block.
- **Command:** `ctest --test-dir build -R 'test_masks|test_covariates|test_predict_block' --output-on-failure`
- **Acceptance:** gates pass; `predict_block` now uses only native masks, no dumped ones;
  a known-covariate request on a tiny config whose schedule does not support it returns
  the same error as the reference.

### M6a: refine frames

- **Goal:** `advancing_block_frames` (literal Hillis-Steele scan), `_refine_block`.
- **Files:** `src/scaler.*`, `src/forecaster.*` (refine only), `tests/test_refine.cpp`.
- **Gate:** G6.1 on `uni_short`, `long_h`, `known_cov`, `long_h_known`.
- **Command:** `ctest --test-dir build -R test_refine --output-on-failure`
- **Acceptance:** gate passes; covariate rows keep their quantiles.

### M6b: block AR decode and the end-to-end forecast

- **Goal:** `_block_ar_decode`, feedback, context cap, `_cap_quantiles`, symmetric
  averaging, clip, NaN error; the C++ API `ts::Forecaster::forecast`.
- **Files:** `src/forecaster.*`, `include/timestral.h`, `tests/test_decode.cpp`,
  `tests/test_forecast_invariants.cpp`.
- **Gates:** G6.2, G6.3 on every case (including `ctx_cap`, label `slow`), and the
  invariants I6.4, I6.5, I6.6.
- **Command:** `ctest --test-dir build -R 'test_decode|test_forecast_invariants' --output-on-failure`
  (and `ctest --test-dir build -L slow` for `ctx_cap` and `bench_32k`).
- **Acceptance:** all gates pass on 209M and tiny; `docs/parity.md` has the per-case,
  per-block error table.

### M6c: CPU benchmark gate

- **Goal:** measure the 32k-context latency and memory before any optimization.
- **Files:** `examples/cli/main.cpp` (`bench` subcommand), `scripts/bench_reference.py`,
  `docs/benchmarks.md`.
- **Command:**
  ```bash
  build/examples/cli/timestral-cli bench --model $TIMESTRAL_TEST_GGUF --context-len 32768 \
      --horizon 1024 --variates 1 --threads N --repeat 5
  $VENV/bin/python scripts/bench_reference.py --model mistralai/Timestral-209M \
      --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 --context-len 32768 --horizon 1024 \
      --threads N --repeat 5
  ```
  for `N = 1` and `N = <physical cores>`, on the same machine. Both print the median wall
  time, the peak RSS and the CPU model.
- **Acceptance:** the C++ median wall time is at most 1.5x the reference at the same
  thread count, and the C++ peak RSS is at most the reference peak RSS plus 1 GB. Record
  both, even when the gate passes. If it fails, do not change the parity path: open an
  optimization milestone (profile first; candidates: drop the inactive decoder tokens on
  the covariate path (A.8), batch the two symmetric passes into one graph, attention in
  row chunks, an encoder cache across blocks for causal rows). Every optimization reruns
  M6b.
- Estimate (**UNVERIFIED**): about 0.4 GFLOP per token for the linear layers plus about
  3.4 GFLOP per time-attention layer at `S = 1056`, so about 0.5 TFLOP per pass, 1 TFLOP
  for one univariate forecast with symmetric averaging. On a CPU that sustains 0.3 to 1
  TFLOP/s in F32 this is 1 to 3 s. Attention scores at `S = 1056` are about 54 MB per row
  per layer (F32); on the covariate path (`S = 2112`) about 214 MB per row.

### M7: quantization

- **Goal:** `f16` and `q8_0` from the converter (`--outtype`), and `q6_k`, `q5_k`, `q4_k`
  through `timestral-cli quantize <in.gguf> <out.gguf> <type>`, with the B.4 allowlist.
  First, a second-model check: convert `mistralai/Timestral-5M` at f32 (pin its Hub
  revision in `docs/conversion.md`), dump the C.2 cases without the `slow` label in all
  five modes, compute the floors, and run the M3 and M6b gates on it with
  `TIMESTRAL_TEST_GGUF` pointing at the 5M GGUF. Set `TIMESTRAL_TEST_REAL_BASELINES` to
  the baseline root; the tests read files from its `real/` directory.
  The gates must pass with no runtime or model implementation change.
- **5M fixture profile:** copy the canonical case specification under `$TIMESTRAL_WORK`.
  Change only layer coverage to `all`, covering all eight layers of the selected model.
  Preserve all 13 non-slow cases, their input semantics, seeds, and forecasting options.
  Generate all five modes with this scratch specification, then compute the floors.
  Validate the artifacts against the same scratch specification with explicit case selections.
  Run the existing M3 and M6b tests with those 13 names in `TIMESTRAL_TEST_REAL_CASES`.
  Do not use `--selected-reference`: M3 requires every layer.
  Keep subset and partial diagnostics visible.
  A separate coverage audit must establish completeness for this exact M7 profile.
  This profile does not replace or weaken the canonical 209M specification or gates.
- **Files:** `scripts/convert_timestral_to_gguf.py`, `src/quantize.cpp`,
  `examples/cli/main.cpp`, `scripts/eval_quant.py`, `tests/test_quant.cpp`,
  `docs/quantization.md`.
- **Metrics (vs the f32 C++ output, on all C.2 cases plus 64 synthetic series from
  `eval_quant.py`):** max and mean error in normalized space (G4.3 metric); relative change
  of the mean weighted quantile loss and of MASE computed against held-out ground truth.
- **Command:** `ctest --test-dir build -R test_quant --output-on-failure` and
  `$VENV/bin/python scripts/eval_quant.py --models $TIMESTRAL_WORK/models --cpp build/examples/cli/timestral-cli`
- **Acceptance:** `f16`: mean normalized error <= 1e-3, quantile-loss change <= 0.5%;
  `q8_0`: mean normalized error <= 5e-3, quantile-loss change <= 1%; the error grows
  monotonically f32 < f16 < q8_0 < q6_k < q5_k < q4_k; sizes and numbers in
  `docs/quantization.md`. These bounds are initial targets, not parity gates; adjust them
  only with measured numbers in the doc.

### M8: CLI and flat C ABI

- **Goal:** `timestral-cli` (`info`, `forecast`, `bench`, `quantize`) and
  `include/timestral_capi.h`.
- **CLI `forecast`:** `--model M --context ctx.csv [--past-cov p.csv] [--known-cov k.csv]
  [--cat-past cp.csv] [--cat-known ck.csv] --horizon H [--no-symmetric] [--no-clip]
  [--threads N] [--json]`. CSV: one row per time step, one column per variate, an empty
  cell or `nan` is missing; one series per file, repeat `--context` for a batch. Output:
  CSV `series,variate,step,<one column per quantile level>`, or JSON.
- **C ABI** (C types only, opaque handle, no C++ exception crosses the boundary):
  ```c
  typedef struct timestral_ctx timestral_ctx;
  int            timestral_capi_abi_version(void);
  timestral_ctx* timestral_capi_load(const char* gguf_path, int n_threads);
  void           timestral_capi_free(timestral_ctx* ctx);
  const char*    timestral_capi_last_error(timestral_ctx* ctx);  /* owned by ctx, "" if none */
  int            timestral_capi_num_quantiles(timestral_ctx* ctx);
  double         timestral_capi_quantile_level(timestral_ctx* ctx, int i);

  typedef struct timestral_forecast_request {
      const float*   context;         /* B*V*T, row-major, NaN = missing */
      const uint8_t* observation_mask;/* B*V*T or NULL; 0 = missing */
      int32_t B, V, T, prediction_length;
      const float*   past_cov;  int32_t c_past;   /* B*c_past*T or NULL */
      const float*   known_cov; int32_t c_known;  /* B*c_known*(T+H) or NULL */
      const double*  cat_past;  int32_t cc_past;  /* B*cc_past*T or NULL */
      const double*  cat_known; int32_t cc_known; /* B*cc_known*(T+H) or NULL */
      int32_t symmetric_averaging;    /* -1 = model default, 0, 1 */
      int32_t clip_nonnegative;       /* -1 = model default, 0, 1 */
      int32_t max_context;            /* 0 = model default */
  } timestral_forecast_request;

  /* Returns malloc'd float[B*V*H*Q] (row-major (B, V, H, Q)); NULL on error.
     *out_q receives Q. Free with timestral_capi_free_buffer. */
  float* timestral_capi_forecast(timestral_ctx* ctx, const timestral_forecast_request* req, int32_t* out_q);
  void   timestral_capi_free_buffer(void* p);
  ```
  The request is passed by pointer so that FFI callers (purego) do not need more than a
  few arguments. `abi_version` starts at 1; bump it on any breaking change.
- **Files:** `examples/cli/main.cpp`, `include/timestral_capi.h`, `src/timestral_capi.cpp`,
  `tests/test_capi.cpp`, `tests/test_cli.sh`, `README.md`.
- **Command:**
  ```bash
  cmake -B build-shared -DTIMESTRAL_SHARED=ON -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON && cmake --build build-shared -j
  nm -D build-shared/libtimestral.so | grep -c ' T timestral_capi_'   # expect 8
  ldd build-shared/libtimestral.so | grep -c ggml                      # expect 0
  ctest --test-dir build -R 'test_capi|test_cli' --output-on-failure
  ```
- **Acceptance:** `test_capi` runs `uni_short` and `known_cov` through the C ABI and gets
  the same bytes as `ts::Forecaster` (bitwise); error paths (bad file, NaN output,
  unsupported known covariates) return NULL with a message; the CLI CSV output matches
  the C ABI output for the same input.

### M9: LocalAI backend (outline only; a separate plan specifies it)

Follow the existing `depth-anything-cpp` backend, which also added its own RPC (`Depth`).

- **L0, load and inference:** `backend/go/timestral-cpp/` (`Makefile` with
  `TIMESTRAL_REPO` and `TIMESTRAL_VERSION` pinned to a commit SHA that a tag keeps alive;
  `CMakeLists.txt` that builds `libtimestralcpp-<variant>.so` with static, PIC ggml;
  `main.go` with purego `Dlopen` and `RegisterLibFunc` for the eight `timestral_capi_*`
  symbols; `gotimestralcpp.go` with `Load` and `Forecast`, `base.SingleThread` because the
  C side shares one allocator). `backend/backend.proto`: add
  `rpc Forecast(ForecastRequest) returns (ForecastResponse) {}` with
  `ForecastRequest { repeated float context; repeated bool observation_mask; int32 batch;
  int32 variates; int32 length; int32 prediction_length; repeated float past_covariates;
  int32 num_past; repeated float known_covariates; int32 num_known; repeated double
  categorical_past; int32 num_cat_past; repeated double categorical_known; int32
  num_cat_known; optional bool symmetric_averaging; optional bool clip_nonnegative;
  repeated float quantile_levels; string ModelIdentity; }` and
  `ForecastResponse { repeated float quantiles; repeated float quantile_levels; int32
  batch; int32 variates; int32 horizon; }`. Then `core/backend/forecast.go`,
  `core/http/endpoints/localai/forecast.go`, the route `POST /v1/forecast` in
  `core/http/routes/localai.go`, the request/response schema, and a model usecase flag
  for forecasting.
- **L1:** covariates and categorical covariates through the HTTP API; selection of
  quantile levels in the response.
- **L2:** not applicable (no streaming). Record it as N/A.
- **L3, registration:** `.github/backend-matrix.yml`, `backend/index.yaml`, the root
  `Makefile` (`BACKEND_TIMESTRAL_CPP`), changed-backends detection, the bump-deps entry for
  `TIMESTRAL_VERSION`.
- **L4, gallery:** an importer that detects a GGUF with `general.architecture =
  timestral` (or a Hub repo with `library_name: timestral`), and a gallery entry. **This
  needs the licence decision in F.1 first.**
- **L5:** Go unit tests for the request conversion, an end-to-end test with the tiny
  GGUF, and a docs page.

---

## F. Risks and open questions

### F.1 Weights licence (owner decision)

The weights are under a "Modified MIT License" (read at the pinned revision). It is not
the code licence. It allows copy, modification and distribution, with two conditions:

1. The attribution notice ("2026 - Mistral AI") and the permission notice must be included
   in all copies or substantial portions. The converter embeds the full licence text in
   `general.license`, and any published model card must include it.
2. Nobody may exercise any rights under the licence if their company's (or employer's)
   global consolidated monthly revenue exceeds USD 20 million in the previous month. The
   licence says this restriction applies to derivatives and combined works. A converted
   or quantized GGUF is a derivative.

Must be checked, by the owner (with legal advice if needed): whether publishing converted
and quantized GGUFs on the Hub is acceptable under condition 2, which account publishes
them, and whether a LocalAI gallery entry may point to them (gallery users include
companies above the threshold, who must not use the weights). Until the owner approves,
no converted weights are uploaded and M9 L4 does not start.

### F.2 Code licence

The reference code is Apache-2.0. The C++ port re-implements its algorithms closely, so
treat it as a derived work: keep a `NOTICE` file with the upstream copyright and a
statement that the code was changed, and pick a repository licence that is compatible
with Apache-2.0 (Apache-2.0 itself, or MIT with the Apache attribution kept). **Owner
decision.**

### F.3 Upstream pin

The reference has one public commit (`a534bdbd...`) and does not accept pull requests. A
force-push upstream would orphan the pin. Mitigation: keep a fork or mirror of the
reference at the pinned SHA under the project's own account, and record the SHA in the
GGUF (`timestral.source.code_sha`). The weights are pinned by the Hub revision
`9874d252...`; the converter computes the SHA-256 of `model.safetensors`, writes it to KV `timestral.source.weights_sha256` and
records it in `docs/conversion.md`.

### F.4 CPU latency at context 32,768

Unknown until M6c. The gate in M6c applies. The main costs are 24 layers over up to 1,056
tokens per row (2,112 on the covariate path, where half the decoder tokens are inactive),
two full passes for symmetric averaging, and a full re-encode of the context for every
decode block (the reference has no cache).

### F.5 Ops without a direct ggml equivalent

| op | status |
|---|---|
| `asinh`, `sinh` | not ggml ops; both run on the host (scaling and denormalization are host steps) |
| partial RoPE with the `head_dim/2` pairing | `ggml_rope_ext` does not match (P8); built from `mul`/`view`/`neg`/`concat`/`add` with the stored `rope.cos`/`rope.sin` tables (P10) |
| SDPA with a sigmoid output gate | `ggml_sigmoid` + `ggml_mul`; no fused op needed |
| per-row boolean masks for two attention axes | host builders, then `ggml_soft_max_ext` with an additive mask that broadcasts over heads (`ne12 = 1`) and differs per row or slot (`ne13 = R` or `S`); verified in the ggml v0.13.0 header and CPU kernel |
| monotone quantile head (`softplus`, `cumsum`, `flip`) | host (P17, P18); `ggml_softplus` exists but is less accurate for negative inputs |
| float64 Welford frames and Hillis-Steele scan | host (P23) |
| NumPy pairwise sum | host (P20) |

### F.6 Not verified from the code

- **UNVERIFIED:** which SDPA kernel torch 2.14.1 uses on CPU with a boolean mask. The dump
  records it (C.4), and the `f32-math` mode covers both.
- **UNVERIFIED:** torch CPU float64 `cumsum` is sequential (decides ULP(0) vs ULP(1) in
  G1.1).
- **UNVERIFIED:** torch's float32 `asinh`, `sinh`, `cos`, `sin`, `log1p`, `exp`
  implementations on CPU (SLEEF or libm). The M0d math probe measures them against glibc
  on 10^6 inputs each.
- **UNVERIFIED:** `ggml_flash_attn_ext` precision with F32 K/V on CPU at v0.13.0 (it is
  not used in the parity path).
- **UNVERIFIED:** the reference is bitwise reproducible across runs at one thread. The
  dump script checks it.
- **UNVERIFIED:** behaviour of the reference on a constant series and on a series with no
  observation at all (it may raise the NaN error). The C.2 cases `const` and
  `all_missing` decide it; the port copies the reference behaviour (outputs, or an error
  at the same step).
- **UNVERIFIED:** the stored `rope.cos`/`rope.sin` equal every runtime table of the
  reference (P10 caveat). G0.3 decides it.
- **UNVERIFIED:** the latency and memory estimates in M6c.
- **UNVERIFIED:** the floor estimates in D.3.
- The 5M, 28M and 123M checkpoints were checked only at the config level (same scalar
  settings, schedules with `time` and `time_variate` only). The M7 second-model check runs
  the M3 and M6b gates on the 5M checkpoint.

### F.7 Things that make this port harder than a plain encoder port

- Much of the forecast is host logic in float64 (scaling, refine frames, target encoding,
  caps, symmetric averaging). It must match operation by operation. Plan for it as much
  work as the graph.
- Two attention axes with per-row and per-slot masks, packed rows of several series, and
  two decode paths (context-only, covariate window with `S = 2n`).
- A RoPE layout that `ggml_rope_ext` does not cover.
- `sinh` denormalization: compare in normalized space, or the raw-space errors look large
  on series with large normalized values.
- Feedback across decode blocks: free-running multi-block parity depends on the error
  growth that the f64 run measures. Teacher-forced per-block gates are the reliable check.
- The reference's own float32 reductions and math functions depend on the CPU ISA and
  the glibc version. The tiny baselines are committed, but the strict host gates run
  only when the baseline's `env.cpu_capability` and `env.glibc` match the machine (D.4).
  On CI they skip, and the integer EXACT and NORM gates still run. Milestone gates run on
  baselines generated on the machine that runs them (`scripts/regen_tiny_baselines.sh`).
- The tiny checkpoint uses `context_length = 2048` so that the scaled cases do not hit the
  context cap by accident (C.2); only `ctx_cap` tests the cap.
