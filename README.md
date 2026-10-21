# timestral.cpp

A C++17 CPU port of the Timestral-209M quantile forecaster, built on ggml.

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team**, the folks behind LocalAI, the open-source AI engine that runs any model (LLMs, vision, voice, image, video) on any hardware, no GPU required.

## What it is

timestral.cpp runs the [Timestral-209M](https://huggingface.co/mistralai/Timestral-209M) forecaster by Mistral AI. It is a from-scratch port of the reference Python package, built on [ggml](https://github.com/ggml-org/ggml).

The model runs from one self-contained GGUF file. Inference needs no Python, no PyTorch and no libtorch. You use the project through a C++ API, a flat C ABI or the `timestral-cli` command.

## Status

- **Works:** loading a GGUF file, running quantile forecasts on the CPU, the C++ API, the C ABI, and the `info`, `forecast` and `bench` CLI commands. A Python converter makes the GGUF file from the Hugging Face checkpoint.
- **Gated:** numerical parity is checked tensor by tensor against the PyTorch reference, see [docs/parity.md](docs/parity.md). Read that page for the scope of each result.
- **Not proven yet:**
  - Speed. The native build is slower than the PyTorch reference. See [Performance and limits](#performance-and-limits) and [docs/benchmarks.md](docs/benchmarks.md).
  - Q8_0 quality. Q8_0 fails its quality bound. See [Quantization](#quantization).
  - GPU. The forecaster does not run on a GPU yet. See [GPU note](#gpu-note).

## Build

```sh
git clone --recursive https://github.com/mudler/timestral.cpp
cd timestral.cpp
cmake -S . -B build
cmake --build build -j
```

If you cloned without `--recursive`, run `git submodule update --init` before you configure. The default build type is Release.

The CLI is at `build/examples/cli/timestral-cli`. To build a shared library, run:

```sh
cmake -S . -B build-shared -DTIMESTRAL_SHARED=ON
cmake --build build-shared -j
```

On Linux, this writes `build-shared/libtimestral.so`.

CMake options:

| Option | Default | Effect |
| --- | --- | --- |
| `TIMESTRAL_BUILD_CLI` | ON | Build `timestral-cli`. |
| `TIMESTRAL_BUILD_TESTS` | OFF | Build the ctest targets. |
| `TIMESTRAL_SHARED` | OFF | Build `libtimestral` as a shared library. |
| `TIMESTRAL_BUILD_QUANTIZE` | OFF | Build the optional quantizer and the `quantize` command. See [Quantization](#quantization). |
| `TIMESTRAL_GGML_CUDA` | OFF | Builds the ggml CUDA backend only. See [GPU note](#gpu-note). |
| `TIMESTRAL_GGML_METAL` | OFF | Builds the ggml Metal backend only. See [GPU note](#gpu-note). |
| `TIMESTRAL_GGML_VULKAN` | OFF | Builds the ggml Vulkan backend only. See [GPU note](#gpu-note). |

## Get a model

No prebuilt GGUF file is published yet. You make one from the Hugging Face checkpoint with the converter in `scripts/convert_timestral_to_gguf.py`.

The converter needs the Python packages in `scripts/requirements.txt`. That file pins PyTorch, NumPy, safetensors, gguf and the reference `timestral` package. Use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/convert_timestral_to_gguf.py \
  --model mistralai/Timestral-209M \
  --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 \
  --outtype f16 --output model.gguf
```

- `--model` takes a Hub id or a local directory that holds `config.json` and `model.safetensors`.
- `--revision` pins the Hub commit. Use it for reproducible output.
- `--outtype` is `f32` (the default), `f16` or `q8_0`. See [Quantization](#quantization).
- The converter also writes `model.config.json` next to the GGUF file. The GGUF file does not need it.

Run `scripts/convert_timestral_to_gguf.py --help` for all options. The model weights keep their own licence from Mistral AI. Check the model card.

## Quick start (CLI)

Print the model details:

```sh
build/examples/cli/timestral-cli info --model model.gguf
```

Forecast from a context file. This sample has eight time steps and one variate. The file has no header. Each line is one time step:

```text
1.20
1.35
1.10
0.95
1.40
1.72
1.58
1.31
```

Save it as `context.csv`, then run:

```sh
build/examples/cli/timestral-cli forecast --model model.gguf \
  --context context.csv --horizon 64
```

Empty cells and `nan` mark missing values. These flags exist in [docs/cli.md](docs/cli.md):

- `--context FILE`: the target series. Repeat it for a batch of equal-shaped series.
- `--horizon N`: the number of steps to forecast. Required.
- `--past-cov FILE` and `--known-cov FILE`: numeric covariates. Repeat each flag once per batch entry.
- `--cat-past FILE` and `--cat-known FILE`: categorical covariates. Repeat each flag once per batch entry.
- `--threads N`: CPU threads. The default is 1.
- `--json`: JSON output.
- `--no-symmetric` and `--no-clip`: turn off averaging or clipping. The model defaults stay on.

A past covariate file has T rows, where T is the context length. A known covariate file has T+H rows, where H is the horizon.

## Output

By default, `forecast` writes CSV to standard output. The first row is a header: `series,variate,step,q_<level>,...`. The indices start at zero. Each quantile level has one column.

With `--json`, `forecast` writes one object:

```json
{"shape":[B,V,H,Q],"quantiles":[...],"values":[...]}
```

`values` is flat, in B, V, H, Q order.

Exit codes:

- `0`: success.
- `2`: a usage error or an input error, such as a bad argument or a bad CSV file.
- `1`: a load, inference or output error.

Results go to standard output. Diagnostics go to standard error.

## C++ API

Include `timestral.h`, link `timestral`, and use C++17:

```cpp
#include "timestral.h"
auto model = ts::Model::load("timestral-209m-f32.gguf");
ts::Forecaster forecaster(model);
ts::ForecastRequest request;
request.batch = 1;
request.targets = 1;
request.length = 5;
request.horizon = 48;
request.context = {1, 2, 3, 4, 5};
auto forecast = forecaster.forecast(request);
```

The model must outlive its forecaster. A forecaster is not reentrant. Independent instances can run on independent threads.

## C ABI

The flat C ABI is in [include/timestral_capi.h](include/timestral_capi.h):

```c
timestral_ctx* ctx = timestral_capi_load(model_path, 1);
if (!ctx) { /* inspect timestral_capi_last_error(NULL) */ }
timestral_forecast_request r = {0};
r.context = context; r.B = 1; r.V = 1; r.T = length;
r.prediction_length = horizon;
r.symmetric_averaging = -1; r.clip_nonnegative = -1;
int32_t q = 0;
float* output = timestral_capi_forecast(ctx, &r, &q);
/* On failure inspect timestral_capi_last_error(ctx). On success output holds
   B*V*horizon*q floats. It remains owned by the caller across further calls. */
timestral_capi_free_buffer(output);
timestral_capi_free(ctx);
```

The ABI exports eight functions:

- `timestral_capi_abi_version`
- `timestral_capi_load`
- `timestral_capi_free`
- `timestral_capi_last_error`
- `timestral_capi_num_quantiles`
- `timestral_capi_quantile_level`
- `timestral_capi_forecast`
- `timestral_capi_free_buffer`

Free each forecast buffer with `timestral_capi_free_buffer`. Handles are not reentrant. Serialize all calls on one handle. Read [docs/capi.md](docs/capi.md) for the full contract.

## Quantization

| Type | Made by | Input |
| --- | --- | --- |
| `f32` | converter (default) | Hugging Face checkpoint |
| `f16` | converter | Hugging Face checkpoint |
| `q8_0` | converter | Hugging Face checkpoint |
| `q6_k`, `q5_k`, `q4_k` | `timestral-cli quantize` | F32 GGUF file only |

The `quantize` command needs a build with the quantizer. It also needs an exact source commit (40 hexadecimal characters), the source archive SHA-256 (64 hexadecimal characters), Linux and OpenSSL Crypto development files:

```sh
cmake -S . -B build-quant -DTIMESTRAL_BUILD_QUANTIZE=ON \
  -DTIMESTRAL_QUANTIZE_COMMIT=<40-hex-commit> \
  -DTIMESTRAL_QUANTIZE_SOURCE_SHA256=<64-hex-sha256>
cmake --build build-quant -j
build-quant/examples/cli/timestral-cli quantize model-f32.gguf model-q6_k.gguf q6_k
```

The command refuses to overwrite an existing output file.

Quality results, stated plainly:

- **Q8_0 fails its quality bound.** Its mean normalized error is 0.005657737. The bound is 0.005.
- **F16 passes its bounds.**
- **The overall 209M quantization quality matrix fails.** Do not treat the K types as quality-tested.

See [docs/quantization.md](docs/quantization.md) for the contracts and details.

## Verification

```sh
cmake -S . -B build -DTIMESTRAL_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

A test that needs a model or a baseline reads its input from an environment variable, for example `TIMESTRAL_TEST_GGUF`. When that input is missing, the test exits with code 77. ctest reports it as skipped. A skip is not a pass.

Numerical parity is checked tensor by tensor against the PyTorch reference, see [docs/parity.md](docs/parity.md).

## Performance and limits

The native build is slower than the PyTorch reference at the tested settings. The test uses a context of 32768 steps, a horizon of 1024 steps and one variate. The numbers are in [docs/benchmarks.md](docs/benchmarks.md).

- The latest formal runs use the optional standalone MKL build, at runtime `68a5a53`. Both latency limits fail. The limit is 1.5 times the reference.
  - One thread: 36.4 s for timestral.cpp, 14.4 s for the reference.
  - Six threads: 18.3 s for timestral.cpp, 4.4 s for the reference.
- An earlier default native build, at revision `b03d674`, measured 330 s with one thread and 79.9 s with six threads.
- Peak resident memory is lower than the reference. The reference peaked at about 1.9 GB. The native builds peaked at about 0.95 GB to 0.98 GB.

The `timestral-cli bench` command runs the same measurement. See [docs/benchmarks.md](docs/benchmarks.md) for the method.

## GPU note

CPU only. The CMake GPU options compile ggml's CUDA, Metal and Vulkan backends, but the forecaster does not run on a GPU yet.

## Roadmap

- Close the latency gap with the PyTorch reference in the default native build.
- Bring Q8_0 under its 0.005 quality bound. The bound will not change to make the test pass.
- Run the forecaster on the GPU backends.
- Choose a code licence and add a `LICENSE` file.

## Citation

timestral.cpp is a port of the Timestral-209M model by Mistral AI (`mistralai/Timestral-209M`). Cite the Timestral authors and the model card when you use the model. The reference code is at <https://github.com/mistralai/timestral>. No paper is listed for this port.

```bibtex
@software{timestral_cpp,
  title  = {timestral.cpp: a C++/ggml port of the Timestral-209M forecaster},
  author = {Di Giacinto, Ettore},
  url    = {https://github.com/mudler/timestral.cpp},
  year   = {2026}
}
```

## Author

Ettore Di Giacinto (mudler)

## License

The code licence is not chosen yet. Do not reuse this code until a LICENSE file is added.

The Timestral-209M weights are published by Mistral AI under their own licence. [NOTICE](NOTICE) lists the third-party notices. It does not select a code licence.
