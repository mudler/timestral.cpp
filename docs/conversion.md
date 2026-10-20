# Conversion

`scripts/convert_timestral_to_gguf.py` turns a Timestral checkpoint into one GGUF file.
The file holds every setting as a KV, so the C++ loader needs no other file.
The default `--outtype f32` keeps every tensor F32.
`--outtype f16` casts only B.4 projection weights after QKV permutation; all exclusions remain byte-identical F32.
The F16 output records `timestral.conversion.outtype`, `timestral.conversion.type_policy`, and the converter source SHA-256.
Licence and original source provenance remain intact.
Q8, K-quants, and BF16 remain unsupported in this preparatory slice.
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
This pin records metadata, not a completed conversion or parity gate.

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
