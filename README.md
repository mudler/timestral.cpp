# timestral.cpp

C++17 CPU implementation of Timestral forecasting with self-contained GGUF
models. Inference needs no Python or libtorch. Numerical parity, performance,
quantization quality, and transport acceptance remain separate gates in
[the work plan](docs/PLAN.md).

## Forecast and inspect

```sh
cmake -S . -B build -DTIMESTRAL_BUILD_CLI=ON
cmake --build build --target timestral-cli -j2
build/examples/cli/timestral-cli info --model model.gguf --json
build/examples/cli/timestral-cli forecast --model model.gguf \
  --context context.csv --horizon 64 --json
```

Input CSV is headerless: one time step per row, one variate per column. Empty
cells and `nan` denote missing values. Repeat `--context` for equal-shaped batches.
Repeat each supplied covariate flag once per batch entry, without broadcasting.
See [CLI transport](docs/cli.md) for formats, precision, and errors.

The implemented [flat C ABI](docs/capi.md) provides eight exported functions,
caller-owned forecast buffers, and opaque non-reentrant handles. CLI forecast
uses this ABI. Static/shared trained transport, anchor, and determinism checks
pass on the optional standalone profile.

## Backend and acceptance status

The default build uses native CPU arithmetic without MKL discovery or dependencies.
The optional [standalone MKL profile](docs/standalone-mkl.md) is **OFF by default**.
It requires explicit user-provided Linux x86_64 LP64/sequential/core libraries and
headers reporting 2024.0 update2, build20240605. No vendor download or redistribution
is part of the build.

The [trained standalone replay](docs/standalone-mkl-trained-replay.md) passes
5M strict M3/M6b **510/141 comparisons**, supplemental eight-case 209M M3 **784**,
and full 15-case 209M M6b **159**. All 63 model/transport invocations pass.
This does not establish full 15-case 209M M3 coverage or default-native acceptance
for every standalone result. [Parity notes](docs/parity.md) index scoped evidence.

[Formal standalone measurements](docs/benchmarks.md#qualified-standalone-on-68a5a53-measured-latency-failure)
fail both latency limits; RSS passes. An isolated wide-linear candidate matches
full-shape output bytes but remains performance-UNDECIDED and unadopted.

## Conversion and optional quantization tools

Python is required for conversion/evaluation tooling, not native inference.
Convert an authorized local checkpoint or pinned Hub revision with the Python
converter; preserve the source licence and provenance:

```sh
"$VENV/bin/python" scripts/convert_timestral_to_gguf.py \
  --model mistralai/Timestral-209M \
  --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 \
  --outtype f16 --output "$TIMESTRAL_WORK/timestral-209m-f16.gguf"
```

Use `--outtype q8_0` for Q8_0. K conversion requires the optional tool build and
an F32 GGUF input. Supply the exact source commit and complete preconfigure source
archive hash; the following variables are provenance inputs, not arbitrary labels:

```sh
cmake -S . -B build-quant -DTIMESTRAL_BUILD_QUANTIZE=ON \
  -DTIMESTRAL_QUANTIZE_COMMIT="$RUNTIME_COMMIT" \
  -DTIMESTRAL_QUANTIZE_SOURCE_SHA256="$SOURCE_SHA256"
cmake --build build-quant --target timestral-cli -j2
build-quant/examples/cli/timestral-cli quantize input-f32.gguf \
  "$TIMESTRAL_WORK/output-q6_k.gguf" q6_k
```

The tool requires OpenSSL Crypto development files. Synthetic converter, writer,
and CLI tests pass. The [209M quality matrix](docs/quantization-209m-result.md)
completes **79×6/474 calls with 79 fresh F32 anchors**, but overall quality **FAILS**.
Q8_0 mean normalized error 0.005657737 exceeds 0.005; its absolute WQL change
0.0411646% passes the 1% bound. F16 bounds and strict mean ordering pass.
The final status is execution1/pin0, not M7 completion.
A fresh coherent cohort resolves the historical URI mismatch without changing
validator policy; the original mixed cohort's HOLD remains historical evidence.
See [coherent F32 replay](docs/coherent-f32-replay.md) and
[quantization contracts](docs/quantization.md).
Do not overwrite accepted inputs or infer permission to redistribute weights.
