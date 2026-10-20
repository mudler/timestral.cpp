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
The first arithmetic mismatch is localized to time SDPA, but its internal dot/softmax/value substep is not yet isolated.
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
