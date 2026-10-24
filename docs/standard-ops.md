# Standard-op forecast graph

Status: experimental, milestone (a) of GPU support. Off by default.

The default forecast graph uses `ggml_custom_4d` callbacks. They reproduce the
reference rounding bit for bit on CPU and have no GPU kernel. The standard-op
graph uses only built-in ggml operations, so a GPU backend can schedule every
node. It is not bit-exact. It is held to the NORM gate
(`err <= max(2 * floor, 16u)`, `docs/PLAN.md` D.1).

## Switch

- `ts::standard_ops()` is read when a graph is built.
- Default: off. `-DTIMESTRAL_STANDARD_OPS=ON` makes the default on.
- `TIMESTRAL_STANDARD_OPS=0|1` in the environment overrides the default.
- `ts::set_standard_ops(bool)` changes it at run time.

## Components

| component | standard ops |
|---|---|
| linear layers, patch embedding, head | `ggml_mul_mat`, `ggml_add` (bias) |
| RMS norm | `ggml_rms_norm`, `ggml_mul` |
| SiLU, sigmoid | `ggml_silu`, `ggml_sigmoid` |
| attention | `ggml_mul_mat`, `ggml_soft_max_ext` (additive mask, scale `1/sqrt(head_dim)`), `ggml_mul_mat` |
| RoPE | unchanged: stored tables, `mul`, `neg`, `concat`, `add` (no custom op) |

Host steps stay as they are: scaler, refine, covariates, float64 quantile
steps. The standalone MKL option is CPU only and is not changed.

## Gates

`tests/test_standard_ops.cpp` prints one `GATE` line per comparison. It is not
registered with ctest. Inputs are the same as the real-model ladder:
`TIMESTRAL_TEST_GGUF`, `TIMESTRAL_TEST_REAL_BASELINES`, `TIMESTRAL_TEST_REAL_CASES`.
`TS_GATE_COMPONENTS` selects a comma-separated subset of
`patch,layer,encoder,head,predict_block,forecast`.
