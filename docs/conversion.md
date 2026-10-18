# Conversion

`scripts/convert_timestral_to_gguf.py` turns a Timestral checkpoint into one GGUF file.
The file holds every setting as a KV and every weight as an F32 tensor, so the C++ loader
needs no other file. The schema is in `docs/PLAN.md`, sections B.1 to B.3.

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

## Tiny checkpoints

`scripts/make_tiny_checkpoint.py --output-dir DIR [--bias]` writes a random checkpoint with
`d_model 32`, four heads and one layer of each kind. With `--bias`, `attn_bias` and
`mlp_bias` are true. The converted files are in `tests/fixtures/tiny/`. They are random
numbers made for tests and carry no trained information.

## Licence of the weights

The weights of Timestral-209M have their own licence (`docs/PLAN.md`, F.1). The converter
copies its text into `general.license`. Do not publish a converted file before the licence
question is settled.
