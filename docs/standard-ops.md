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
| linear layers, patch embedding, head | `ggml_mul_mat` per K panel of 192, `ggml_add` (panels in order, then bias) |
| RMS norm | element-wise F32 cascade sum (`ggml_sqr`, views, `ggml_add`), `ggml_div`, `ggml_sqrt`, `ggml_mul` |
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

## Arithmetic choices

Each choice below was found by running one operation on the reference's own
input and comparing it with the stored reference (see the NORM gate in
`tests/test_standard_ops.cpp`).

- Linear layers. The reference F32 GEMM reduces K in panels of 192, with one
  accumulator chain per panel and the panel sums added in order. One
  `ggml_mul_mat` over all of K uses a different rounding path. Layers with
  outlier activations amplify that: the error is a few units in the last place
  of the largest output element, and the gate allows about one. Splitting K
  into panels of 192 and adding the panel results in order removes most of it.
  Dots shorter than 192 stay one `mul_mat`.
- RMS norm. The reference sums x*x in F32 with a fixed cascade (32 lanes, power
  of two merge blocks, fixed fold). `ggml_rms_norm` sums in double. Both are
  accurate, but they round differently, and the last bit of 1/sqrt(mean + eps)
  shows in the output of any row with large channels. The standard-op norm
  rebuilds the cascade from element-wise nodes, which are exact IEEE F32 on
  every backend. Scalars such as D and 1.0 come from `graph_constant`, which the
  graph runner writes when the graph is allocated. If D is not a multiple of 32
  the norm falls back to `ggml_rms_norm`.
- SiLU and sigmoid use `ggml_silu` and `ggml_sigmoid`. Their effect on the gates
  is below 0.05 of the tolerance in the cases checked.

## Known residual

On the 209M F32 model, 33 of 7899 NORM checks still fail (the first version of
this graph failed 122). The worst ratio of error to tolerance is 3.5. The causes
are not rounding bugs.

- Attention scores. The reference accumulates q.k in double and rounds once.
  `ggml_mul_mat` accumulates in F32. With the exact dot products and the standard
  softmax and value product, the layer 0 attention output of `uni_one_patch`
  passes at 0.20 of the tolerance; with F32 dots it fails at 1.95. No standard
  operation computes a double-accumulated dot product.
- Linear layers in layers with outlier activations (largest channel about 100
  times the rms). A different F32 accumulation order moves the largest output by
  one to four units in the last place, which is at or above the gate. A
  compensated panel sum was tried (TwoSum, panels of 64). It made the dot
  products closer to a double-precision sum but did not reduce the number of
  failed gates, so it is not used.
