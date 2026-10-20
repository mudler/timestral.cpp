# Conversion

`scripts/convert_timestral_to_gguf.py` turns a Timestral checkpoint into one GGUF file.
The file holds every setting as a KV, so the C++ loader needs no other file.
The default `--outtype f32` keeps every tensor F32.
`--outtype f16` casts only B.4 projection weights after QKV permutation; all exclusions remain byte-identical F32.
The F16 output records `timestral.conversion.outtype`, `timestral.conversion.type_policy`, and the converter source SHA-256.
Licence and original source provenance remain intact.
`--outtype q8_0` packs only B.4 weights, after QKV permutation, into blocks along the contracted row axis.
Each row must contain a positive multiple of 32 F32 values; each block occupies 34 bytes.
Excluded tensors remain byte-identical F32; metadata records policy `b4-q8_0-v1` and the converter source hash.
K-quants, Q8_1, and BF16 remain unsupported in this preparatory slice.
The schema is in `docs/PLAN.md`, sections B.1 to B.4.
The existing byte-parity checker applies only to F32 outputs; reduced-format tests do not replace that gate.

## Reference environment

`scripts/setup_reference_env.sh` makes a virtual environment with Python 3.12 and the pins
in `scripts/requirements.txt`. Set `PYTHON` to a Python 3.12 interpreter when `python3.12`
is not on the path.

## Convert

```bash
$VENV/bin/python scripts/convert_timestral_to_gguf.py \
    --model mistralai/Timestral-209M \
    --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 \
    --outtype f32 --output "$TIMESTRAL_TEST_GGUF"
```

`--model` is a Hub id or a local directory with `config.json`, `model.safetensors` and
`LICENSE` (use `--license-file` when the directory has no `LICENSE`). The converter:

- validates `config.json` with `TimestralConfig.from_dict` (unknown keys are an error);
- checks that every safetensors tensor is used once, that none shares storage with another,
  that the names match the config exactly and that every shape is the one the config implies;
- permutes the rows of the fused QKV projection and its bias to head-major order (B.3);
- computes `rope.inv_freq`, `rope.cos` and `rope.sin` with the reference `SinCosRoPE` class,
  one thread, for the positions `-(context_length / patch_size + block_size)` to the same
  number with the opposite sign;
- writes the SHA-256 of `model.safetensors` and the licence text into the file;
- copies `config.json` to `<output>.config.json` for the tests (gate G0.2). The GGUF does
  not need this copy.

## Check

```bash
$VENV/bin/python tests/python/check_convert.py --gguf "$TIMESTRAL_TEST_GGUF" \
    --model mistralai/Timestral-209M --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365
```

This is gate G0.1: every GGUF tensor equals the safetensors tensor byte for byte, after the
QKV permutation is undone on the GGUF side. The script maps the names with its own rules,
so a mistake in the converter's name map is found. It also compares the rope tables with a
fresh run of the reference class and prints a preview of gate G0.3: the reference is run at
run-time-like layouts (several row counts and patch counts, positions `p - first_obs`), and
each value is compared with the stored table row of its position. Values that differ only
in the sign of a zero are counted apart from values that differ.

## Second-model checkpoint for M7

Use `mistralai/Timestral-5M` at the immutable Hub revision
`bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` for the M7 prerequisite.
The [pinned configuration](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/config.json)
has `d_model 192`, three heads, `mlp_hidden_dim 512`, and eight layers.
Its schedule repeats `time, time, time, time_variate` twice.
The M7 fixture profile in `docs/PLAN.md` preserves the canonical inputs and records all eight layers.
Local preparation retrieves only `config.json`, `model.safetensors`, and `LICENSE` at that exact revision.
A metadata-only inventory finds 75 F32 tensors and 5,208,960 parameters; names, shapes, byte ranges, and file length match the config-driven schema.
No model instantiation, conversion, reference generation, or parity gate accompanies this preparation.

| File | Bytes | Actual SHA-256 |
|---|---:|---|
| `config.json` | 904 | `b5f27f61513f8fd1a3aa996f3720d3679cb40bce5bcb08eb3dc320ce726e7c2f` |
| `model.safetensors` | 20,843,960 | `7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738` |
| `LICENSE` | 1,695 | `648194727930df730890e9f9affdbcc5c930d406d6829559b25ca3d5a973fc8c` |

The scratch profile contains exactly the 13 non-slow cases, with only `layers` changed to `all`.
A deep comparison preserves every other case field; the canonical specification remains unchanged.
The prepared environment records Python 3.12, NumPy 2.5.4, torch 2.14.1+cpu, safetensors 0.8.0, and gguf 0.19.0.
Installed reference-package provenance identifies `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`.

Later execution requires a separate heavy-slot grant and a pinned source/build snapshot.
Start with F32 conversion and the `uni_one_patch` pilot in all five modes, serially under 12 GiB.
Validate its artifacts, compute floors, and run strict M3/M6b before estimating the remaining profile's cost.
Use the Hub ID plus exact revision with the prepared offline cache to preserve provenance.
Keep partial/subset diagnostics visible; an external audit must establish full 13-case coverage.
Local task authorization does not establish revenue eligibility or publication rights; the owner remains responsible for those licence conditions.

### Executed pilot: HOLD at strict M6b

The isolated reviewed F32 binaries convert the checkpoint to 78 F32 GGUF tensors without model-specific changes.
The byte checker passes all 75 source tensors; the RoPE preview has zero numerical differences across 13,452,288 values.
The converted GGUF SHA-256 is `c320276d8db95fb72fe5f4e043147e29142a1bf197c1b45daf6736780f6736ed`.
Source revision, weights hash, and licence text match the downloaded snapshot.

All five `uni_one_patch` modes complete under 12 GiB, with the all-eight-layer scratch profile.
Their wall times are 4.94, 3.77, 4.30, 4.57, and 3.67 seconds in mode order.
Peak RSS ranges from 356,080 to 439,628 KiB; the five modes total 21.25 seconds.
Artifacts, including the floor summary, occupy 16,782,720 bytes.
The explicit-case checker passes with its PARTIAL label intact.
Floors cover 360 float tensors, including 316 teacher-forced floors and 230 caps; no floor exceeds its cap.

Strict M3 passes 16 local layers and 34 comparisons over both passes, with no missing layers.
It reports one repeated amplification streak and maximum adjacent amplification 21.021; retain that warning.
Strict M6b fails its first positive teacher-forced prediction before completing any block/final comparisons.
Normalized error is `2.24465652e-6`, tolerance `2.09746792e-6`, ratio `1.07017442`.
No tolerance, floor, reference artifact, or runtime changes follow this failure.
The remaining 12 cases do not run.

Boundary diagnostics use source 0 = stored F32 encoder output, source 1 = native encoder on stored `tf.in`,
and source 2 = native encoder on native preparation.
For the positive pass, native head/monotone/unscale ratios are 0.356725, 1.056552, and 1.070174 respectively.
Stored raw-head output through native monotone/unscale passes at ratio 0.102058.
Thus upstream encoder drift is sufficient to reproduce the failure; the head-only threshold is not established as its cause.

Per-layer diagnostics pass stored F32 inputs at each named boundary.
`Encoder::boundary` reports the computed operation output before substituting its stored override downstream.
Across both passes, the only nonzero independently teacher-forced boundaries are the eight time-attention SDPA outputs per pass.
Layer-0 positive SDPA differs in 949/1728 values, maximum absolute difference `9.53674e-7`.
Its norms, QKV, q/k/v, q/k after RoPE, gate, output projection, and MLP are exact on their own stored inputs.
The corresponding negative SDPA differs in 981/1728 values, maximum `1.43051e-6`.
The singleton variate SDPA boundaries are exact.
An exact fully overridden layer output does not prove composition exactness: downstream overrides deliberately remove upstream error.

The recorded reference uses CPU flash SDPA on AVX2.
The native shape is head dimension 64, three heads, eight context keys, and one decoder token.
Context queries take the wide path; the one-token decoder takes the existing narrow contraction branch.
The first arithmetic mismatch is localized to time SDPA.
A scratch substep witness further isolates context-query QK accumulation, with decoder details still unresolved.

The witness loads exact layer-0 stored q/k after RoPE, v, and masks for 5M and 209M, positive and negative passes.
Fresh Torch flash calls match all eight stored attention outputs byte for byte, separately for context and decoder queries.
Torch is `2.14.1+cpu`, source `5c4886908584029761b579af026dcfb627c84070`, reporting AVX2.
The probe computes per-head `torch.mm(q, k.T)` as a separate contraction witness.
That matrix product is not a captured internal flash buffer; its equivalence must not be assumed for every execution shape.

The scratch C++ modes hold scale, mask, exp/reduction, and reciprocal placement fixed:

| Mode | QK source | Value contraction |
|---|---|---|
| 0 | Existing native dot-order reconstruction | Sequential FMA |
| 1 | Per-head Torch matrix product | Sequential FMA |
| 2 | Existing native dot-order reconstruction | Separate multiply/add |
| 3 | Per-head Torch matrix product | Separate multiply/add |

QK exact-count diagnostics always describe the mode-0 dots against Torch matrix products, even in replacement modes.
For 5M context, those counts are 43/192 positive and 42/192 negative; maximum absolute difference is `3.05176e-5`.
Mode 1 makes context outputs exact: 1536/1536 values in each 5M pass and 6144/6144 in each 209M pass.
This demonstrates a sufficient QK-only replacement for context outputs without changing exp, sum, or value FMA.
It does not prove the selected matrix-product API exposes flash's internal arithmetic.

Decoder mode 1 remains inexact: 152/192 positive and 153/192 negative 5M values match.
Mode 3 gives 192/192 positive and 191/192 negative matches; the negative maximum difference is `2.38419e-7`.
The 209M decoder also remains slightly inexact under mode 3.
Thus decoder arithmetic cannot be declared solved by choosing the best probe mode.
A bounded dot-order enumeration finds exact context witnesses for four separate multiply/add lanes with pairwise reduction,
but matching counts alone do not establish backend dispatch or justify a production branch.

The inspected Torch flash source calls `cpublas::gemm(Transpose, NoTranspose)` for QK with `(m,n,k)=(keys,queries,head_dim)`.
It supplies actual Q/K token strides, scales/masks afterward, applies vector `exp_u20` plus scalar tails, then GEMM for values and final reciprocal scaling.
The CPU BLAS wrapper can use MKLDNN or SGEMM before its fallback; this source-level path is not an observed internal microkernel trace.
### Compiled backend witness

Supported MKL verbosity on actual Torch flash calls identifies oneMKL SGEMM, not the fallback or oneDNN path, for these eight controls.
Context QK is `SGEMM(T,N,8,8,64)` and value contraction is `SGEMM(N,N,64,8,8)`.
Decoder QK is `SGEMM(T,N,9,1,64)` and value contraction is `SGEMM(N,N,64,1,9)`.
The Q/K/V token leading dimension is 192 for 5M and 768 for 209M.
The decoder query leading dimension normalizes to 64 when its query count is one.

The float `cpublas::gemm` wrapper is not dynamically exported, but `sgemm_` is.
A scratch compiled witness calls that actual libtorch-exported backend with the observed dimensions, transpose flags, and leading dimensions.
It makes no production link or dependency change.
The witness uses supplied F32 exp/sum intermediates from the earlier reconstruction for the value stage; those are not captured flash internals.
With `MKL_DYNAMIC=FALSE`, verbosity reports `Dyn:0`, `NThr:1`, and `FastMM:1`, matching actual flash execution.
The initial default-Dyn:1 run gives the same numerical results.

| Control | Backend QK versus per-head Torch mm | Backend values plus reconstructed exp/sum versus stored flash |
|---|---|---|
| 5M positive context | 192/192 exact | 1536/1536 exact |
| 5M negative context | 192/192 exact | 1536/1536 exact |
| 5M positive decoder | 27/27 exact | 192/192 exact |
| 5M negative decoder | 27/27 exact | 191/192; maximum `2.38419e-7` |
| 209M positive context | 768/768 exact | 6144/6144 exact |
| 209M negative context | 768/768 exact | 6144/6144 exact |
| 209M positive decoder | 108/108 exact | 766/768; maximum `4.76837e-7` |
| 209M negative decoder | 108/108 exact | 767/768; maximum `2.38419e-7` |

The actual backend now corroborates the independent matrix-product QK witness for every control.
It does not expose the proprietary microkernel's floating-point accumulation order.
Context evidence supports a shape-generic QK arithmetic correction, but a production implementation still requires an independent mathematical derivation and witnesses.
Decoder residuals cannot yet be assigned uniquely to exp, sum, or value normalization because reconstructed intermediates remain in that comparison.
No best-match lane heuristic, MKL production dependency, or proprietary kernel copying is authorized.

A direct experimental `mkl_set_dynamic` symbol call crashes before GEMM; its ABI was not verified and the failure remains a scratch-witness limitation.
The successful bounded rerun uses the documented environment variable instead, without claiming an ABI diagnosis.
The retained `libtorch_cpu.so` SHA-256 is `5acf62dcd9d367db4788122f8fefd506027ed5a7aec70467a7fffd08a97ad64c`.
The successful witness executable SHA-256 is `c531e4937475a3b2de849ca373740b2314203c5c43dac8ab5b9760b60c207fa2`.
Original native binaries and source hashes remain unchanged before and after the compiled probes.
### Subsequent `known_cov` blocker localization

The accepted scalar binary64-QK policy passes M3 but fails negative block-0 teacher-forced prediction for `known_cov`.
Error is `1.34354679e-5`, tolerance `1.30408895e-5`, ratio `1.03025702`.
The diagnostic retains all three packed rows and the original full-tensor RMS/floor; no covariate row is excluded.
The worst row is row 1, the numeric known covariate, at forecast step 32 and quantile index 1.
Target row 0 has normalized error `1.646429046e-6`; indicator row 2 has `5.219704397e-6`.
Both are below the unchanged full-tensor tolerance.

Anchors are not entirely bit-exact in this case.
Numeric-known location differs by `9.31323e-9`; indicator scale differs by `5.96046e-8`.
Known scaled features differ in 896/2592 values, maximum `1.192092896e-7`.
Patch features differ in 896/5184 values by the same maximum; native patch projection differs in 8667/15552 values, maximum `3.051757812e-5`.
These are observed preparation differences, not proof that scaler rounding alone causes the final failure.

On stored F32 encoder output, the native raw head is byte-exact over 54,432 values.
Its complete head/monotone/unscale prediction passes at ratio 0.0272982.
Native encoder on stored `tf.in` passes narrowly at ratio 0.9856133.
Native encoder on native preparation reproduces the failure at ratio 1.0302570.
These comparisons use the same fixed stored frames for evaluation; replacing them does not alter the gate.
Preparation drift is sufficient to cross the threshold on top of the existing encoder drift in this witness.

Independent teacher-boundary enumeration finds ten nonzero boundaries among 146: all eight time SDPA and both variate SDPA outputs.
Every reported non-SDPA teacher operation is byte-exact on its stored inputs, including head and linear projections.
The earliest layer-0 time SDPA differs in 12,361/31,104 values, maximum `4.52995e-6`.
This is compatible with the deliberate higher-accuracy QK policy differing from reference F32 arithmetic; it is not independently a proven kernel bug.
The bounded diagnostic stops here without another arithmetic change or tolerance adjustment.
A future uniform value-contraction accuracy experiment would require a separate root precision decision and independent scalar oracle.
It must preserve all rows, preprocessing, QK policy, and other attention boundaries, and must stop if any existing strict gate fails.
No such experiment or remaining-case generation accompanies this localization.

### Known-covariate preparation root cause

The next bounded diagnostic confirms the complete 3x864 pre-scaling value buffer is byte-identical to stored `scale.in`.
The observation mask is constructed from the unchanged stored context/future masks; no cap, padding, slice, or layout discrepancy is found.
Replaying pinned Torch `instance_norm_stats` on that buffer reproduces stored known-row scaled values exactly.

The first mismatch is the numeric known-row F32 reduction, not causal prefix scaling.
Reference `instance_norm_stats` computes F32 mask counts, weighted sums, centered squares, and square sums before division and sqrt.
Native `instance_norm_stats` currently widens weighted and square sums to sequential binary64, then casts back to F32.
For numeric row 1, Torch weighted sum is `-26.33343505859375`; native widened sum rounds to `-26.33344268798828`.
This yields the observed location difference.
For negative indicator row 2, the weighted sum is exact but Torch square sum is `215.99542236328125`,
versus native widened `215.99537658691406`, producing the observed scale difference.
Negative indicator values are 0/-1, so this mirrored row correctly does not use the 0/1 bypass.

A scratch reduction witness follows `SumKernel.cpp` at Torch `5c4886908584029761b579af026dcfb627c84070`.
The contiguous inner-sum path uses four independent eight-lane F32 partial sums and four cascade levels.
Its level power is `max(4, ceil_log2(vector_count/4)/4)`; completed groups promote through the cascade in source order.
Remaining vectors join partial zero; partial groups combine in ascending order, then scalar tails precede ascending lane accumulation.
For length 864 and element stride four bytes, there are 108 vectors and 27 four-vector groups.
This is a shape-generic operation tree, not a length-864 branch or NumPy pairwise sum.
The scalar C++ witness reproduces Torch sums, locations, square sums, and scales for all three rows exactly.
`numpy_sum.hpp` implements a different NumPy float64 pairwise algorithm and is not the appropriate replacement.

A scratch causal intervention supplies stored reference-scaled patch features only, then uses native projection, encoder, head, and unscale.
The unchanged full-row metric passes at ratio `0.954023177`, versus original `1.03025702`.
Its worst row remains the numeric known covariate, error `1.244131083e-5`; all three rows remain evaluated.
This is a diagnostic intervention, not an allowed production dependency on reference arrays.

The supported correction proposal is to reproduce the pinned contiguous F32 reduction order in `instance_norm_stats` only.
Preserve its F32 subtraction, masking, square, division, sqrt, binary detection, and affine boundaries.
Do not change causal float64 prefix scaling, QK, value contraction, floors, or tolerances.
Implementation still requires separate approval, generic reduction-length/tail/adversarial tests, and 209M/tiny regressions.
No production fix accompanies this root-cause record.

### Scratch-only uniform binary64 QK experiment

A separately authorized accuracy experiment changes only QK accumulation in an isolated `2a687ab` source tree.
Every context and decoder dot sums `double(q)*double(k)` in ascending dimension order, then rounds once to F32.
Scale, mask, max, exp, sum, value contraction, and reciprocal placement remain unchanged.
There is no model-specific or winning-shape branch; the production worktree retains its original kernel.

Across all eight stored controls, all 2190 candidate dots match the F32-rounded independent `math.fsum` of exact binary32 products.
This supports accuracy on those witnesses, not universal exactness of sequential binary64 summation or MKL parity.
Candidate and MKL dots often differ; output agreement is not selected as the mathematical oracle.
All reconstructed SDPA outputs remain finite.

The complete existing 5M `uni_one_patch` M6b pilot passes both signs: two teacher blocks, two free blocks, one final, nine comparisons.
Its maximum error/tolerance ratio is 0.891812, with invariants passing.
The same unchanged floors/artifacts previously failed the retained F32 runtime at ratio 1.07017442.

Conditional bounded regressions pass 34 selected tiny component tests, including F16/Q8 routing.
Full tiny forecasting passes 28 cases and 684 comparisons, maximum ratio 0.618082.
The 209M `uni_one_patch` and `uni_short` subset passes 18 comparisons, maximum ratio 0.703131.
Tiny 28-case and 209M 512/32 determinism checks pass repeated execution at threads 1 and 6.
These subsets are not full 209M or full 5M acceptance.

The unchanged `test_attention` is explicitly excluded from that 34-test selection and fails when run separately.
It aborts at `keypack exact/immutability` because its frozen F32 arithmetic oracle no longer matches the candidate policy.
The original oracle remains intact: this experiment does not weaken or replace it.
The complete test suite and production adoption therefore remain HOLD.
A production proposal requires a separate independent binary64 oracle and an explicit legacy-versus-new precision-policy test separation.
It also requires broader reference parity and performance measurements; neither follows automatically from this pilot.

The scratch source archive SHA-256 is `c509e78a60b67cb3ba29129f89b0479c6b5f148bb0b19ea7693d799fe35d7998`.
Its `libtimestral.a` SHA-256 is `36dbfec5608b833a64718581cdb94fdaf1241e963b33b0017917b6d946e3aad8`.
The candidate forecast executable SHA-256 is `b2e839a8e3d43c20fad93090de5028a674efa9f341c35f28c066227a444ce040`.
All 1962 vendor blobs match ggml `e705c5fed490514458bdd2eaddc43bd098fcce9b` despite misleading outer-repository configure metadata.
Original source/binaries and all 5M model/reference artifact hashes remain unchanged before and after the runs.
All builds/tests use serial execution under 12 GiB, with at most two compiler jobs.
The scratch patch and red/green evidence remain retained; no production arithmetic change accompanies these results.
No production arithmetic, floor, tolerance, or baseline changes follow these probes.
A later correction needs independent shape-generic contraction witnesses, not a model-name branch or tolerance change.
Pilot timings do not justify extrapolating the long multiblock cases; resolving parity precedes the full-profile cost decision.

The [pinned licence](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE)
is titled "Modified MIT License", with attribution "2026 - Mistral AI".
It requires copies or substantial portions of the Model to include the attribution and permission notices.
The Model includes the weights and associated documentation.
The licence restricts rights based on the preceding month's global consolidated monthly revenue of the recipient's company or employer.
The threshold is more than $20 million, or its equivalent in another currency.
The restriction also covers derivatives, modifications, and combined works, including those from third parties.
The licence lists `sales@mistral.ai` for commercial licence requests.
These facts do not establish permission to use or publish converted weights.
The publication question in `docs/PLAN.md`, F.1, remains unresolved.

## Tiny checkpoints

`scripts/make_tiny_checkpoint.py --output-dir DIR [--bias]` writes a random checkpoint with
`d_model 32`, four heads and one layer of each kind. With `--bias`, `attn_bias` and
`mlp_bias` are true. The converted files are in `tests/fixtures/tiny/`. They are random
numbers made for tests and carry no trained information.

## Licence of the weights

The weights of Timestral-209M have their own licence (`docs/PLAN.md`, F.1). The converter
copies its text into `general.license`. Do not publish a converted file before the licence
question is settled.
