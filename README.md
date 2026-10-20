# timestral.cpp

C++17 CPU implementation of Timestral forecasting with self-contained GGUF
models. Numerical parity, benchmark, quantization, and transport acceptance are
tracked separately in [the work plan](docs/PLAN.md).

## Forecast and inspect

```sh
cmake -S . -B build -DTIMESTRAL_BUILD_CLI=ON
cmake --build build --target timestral-cli -j2
build/examples/cli/timestral-cli info --model model.gguf --json
build/examples/cli/timestral-cli forecast --model model.gguf \
  --context context.csv --horizon 64 --json
```

Input CSV is headerless: one time step per row, one variate per column. Empty
cells and nan denote missing values. Repeat --context for equal-shaped batches;
repeat each supplied covariate flag exactly once per batch entry, without
broadcasting. See [CLI transport](docs/cli.md) for formats, precision, and errors.

The [flat C ABI](docs/capi.md) provides caller-owned forecast buffers and opaque
non-reentrant handles. Forecast/info transport is preparatory: M6c performance,
keypack retention, M7 quantization, and full milestone acceptance remain pending.
Quantize is not implemented by this transport change. Bench retains its existing
[measurement contract](docs/benchmarks.md).
