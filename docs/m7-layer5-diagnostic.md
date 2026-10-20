# 209M local layer-5 precision-policy diagnostic

## Reproduction

The stopped aggregate failure is `uni_one_patch`, positive block 0, layer 5, local teacher-forced residual output.
The observational witness links the exact reviewed e70af1d library and passes stored layer inputs directly to `Encoder::run_layer`.
Geometry is R1, S9, D768, 12 heads, head dimension 64, context split 8.
Input vectors remain unchanged; mask, positions, and layout come from the same immutable fixture fields used by the strict gate.
The witness does not invoke preparation, scaler, or head projection to compute this local layer.

The local failure reproduces at normalized error `3.00365365487e-6`, tolerance `2.86737316913e-6`, ratio `1.04752799085`.
Maximum absolute error is `0.0001220703125`, at row 0, token 2, feature 521.
Native value is `1057.57836914`; reference is `1057.57824707`.
4696/6912 output values differ.
This uses the unchanged `floor_tf` local-layer contract, not `floor_fr`.

## First differing boundary

Within the ordinary local layer chain, norm, QKV, q/k/v, post-RoPE q/k, and gate are byte-exact.
The first differing operation is time SDPA: 3037/6912 values, maximum `1.57356262207e-5`.
Its own teacher-forced normalized ratio is 0.693811075728 and passes.
Downstream gate multiplication inherits that SDPA drift; a chained gated comparison is not an independently teacher-forced gate failure.
Independently teacher-forced comparisons report computed outputs before substituting stored overrides downstream.
Every non-SDPA teacher-forced operation, including output projection, MLP, and final residual, is byte-exact on its stored inputs.
The fully overridden output being exact does not prove native composition exactness.

## Legacy versus current QK causality

The direct attention calls use identical stored q/k-after-RoPE, v, additive masks, split, dimensions, and strides.
The internal legacy callback and current callback retain the same non-QK arithmetic.
No head, scaler, or global linear changes participate.

| Witness | SDPA-local ratio | Composed local-layer ratio |
|---|---:|---:|
| Current binary64 QK | 0.693811075728 PASS | 1.04752799085 FAIL |
| Legacy F32 QK/dispatch | 1.21384087349 FAIL | 0.261881997712 PASS |
| Stored reference SDPA | exact by supplied input | 0, byte-exact output |

The legacy-only layer witness substitutes its computed SDPA output at that boundary; all other layer operations execute natively.
The source inputs to the legacy SDPA are exact because upstream native q/k/v and RoPE outputs already match reference bytes.
The legacy composed output differs in 4998/6912 values but has maximum absolute error `3.0517578125e-5`, within the local gate.
This is direct evidence that the QK policy change alters the downstream composition for this case.
It is not evidence that legacy arithmetic is universally better: the current policy is closer under the separate SDPA metric.
No model-, layer-, or case-specific fallback is justified by these results.

## Stop and preservation

The diagnostic stops at this supported arithmetic-policy compatibility boundary.
No numerical fix, policy reversion, runtime mutation, floor/tolerance change, reference rewrite, or full-suite retry occurs.
A next proposal must explain a shape-generic precision contract and satisfy both local and end-to-end unchanged gates.
Full aggregate acceptance remains HOLD, despite the completed 15-case M6b pass.

Before/after pins match for all 90 canonical/extended references, the reviewed library, encoder executable, and 209M model.
The retained tested-source archive and executable identities are the same as the stopped aggregate run.
The witness source/executable are separately hash-pinned.
All execution is serial under 12 GiB; only one small observational executable compiles.
No public CLI, C ABI, anchor, or remaining aggregate tests run during this diagnosis.
