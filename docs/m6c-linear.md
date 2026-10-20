# Bounded M6c linear experiment

## Scope and numerical contract

Parent revision: `519095daf352cd0c4af8d6e75b62e050483881b0`.
Accepted M6b evidence remains at `23fa765`.
The short profile attributes 88.11% of sampled self time to `linear_f32`.
This experiment changes only that contraction and adds focused tests.

Eight AVX2 lanes compute independent output columns. Weights remain
K-contiguous rows; each lane loads the same K coordinate from its own row.
Each output retains ascending K traversal, sequential F32 FMA within each
192-wide panel, separate F32 panel additions, and the final bias addition.
Small dots retain their original single FMA chain without an extra zero add.
Column tails use the scalar implementation. No weight packing or cache persists.

The callback partitions independent input rows. Production linear nodes still
request one task: this experiment does not change graph threading policy.
Six-thread timings include the existing backend execution policy for other
operations. They do not establish six-way parallel execution of linear nodes.
Attention, reference code, tolerances, floors, and benchmark tooling are unchanged.

## Build and dispatch

Both builds use Release `-O3 -DNDEBUG`, C++17, and PIC.
`layers.cpp` retains `-ffp-contract=off`; explicit FMA operations remain fused.
Neither build uses fast-math. `GGML_NATIVE` and `GGML_LLAMAFILE` remain enabled.
The original library does not receive ggml's native ISA flags.

The optimized function uses a local `target("avx2,fma")` attribute on GCC/Clang
x86 builds. Runtime CPU checks require both features before calling it.
Other builds retain the scalar path. `TS_LINEAR_DISABLE_AVX2` disables this
path at compile time for fallback verification; no global ISA flag is required.
Disassembly confirms packed FMA instructions in the optimized function.
Non-x86 hardware and other compilers are not measured here.

## Exact and component checks

- `test_linear`: 3,024 byte-exact oracle comparisons, 2,642,976 output values.
  The oracle independently retains the original scalar indexing and arithmetic.
  Coverage includes K=191/192/193 and 383/384/385, K up to 2048, column tails,
  multiple rows, bias/no-bias, cancellation, signed zero, subnormals, and random
  finite inputs. Callback partitions of one and six produce identical bytes.
- The compile-time scalar fallback passes the same 3,024 comparisons.
- Final serial CTest: 34/34 pass, excluding real-model tests and the loader.
  This includes all tiny component gates, full tiny forecast parity, and the
  new forecast determinism test. No skips occur in this selection.
- Loader: 1/1 pass with `TIMESTRAL_TEST_TINY` unset, its expected fixture layout.
  The first run passed 33/34; the loader alone failed from the suite-root
  environment pointing at a nonexistent model. No fixtures change to fix it.
- All 28 tiny/tiny-bias forecast cases pass bitwise repeats with thread sequence
  1, 6, 6, 1, compared against an initial one-thread forecast.
- The real 209M 512/32 forecast passes the same bitwise thread/repeat check.
  This is determinism evidence, not full real-model parity acceptance.

Builds use `-j2`; heavy runs execute serially under a 12 GiB address-space limit.
No full real M6b or 32k run executes during this experiment.

## Warm isolated end-to-end measurements

CPU: AMD Ryzen 5 7430U, six physical cores. Input and model match
[the benchmark contract](benchmarks.md): 209M F32, context 512, horizon 32,
one variate, `dyadic-v1`, symmetric averaging and clipping enabled.
Each row uses a separate process, one discarded warmup, five timed forecasts,
and a 20-second cooldown. No build overlaps a timed process.
The retained pre-change Release binary runs again after the candidate runs;
its before/after SHA-256 checks pass. This ordering does not remove all thermal
or frequency uncertainty, but the earlier 17.188-second diagnostic agrees.

| Runtime | Threads | Median seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Pre-change | 1 | 17.066421610 | 849743872 |
| Candidate | 1 | 1.541307288 | 849358848 |
| Pre-change | 6 | 3.974667499 | 847941632 |
| Candidate | 6 | 0.405898183 | 847519744 |

The measured end-to-end improvement is 11.07x at one thread and 9.79x at six.
All four output sums are `99.59357074`; exact tests, not sums, establish parity.
Six-thread samples vary substantially; repeat on the target deployment before
using these medians as a performance guarantee. These short-shape results do
not pass the full M6c latency or memory gates.

## Reproduction and evidence

Scratch paths below are relative to `$TIMESTRAL_WORK`, not repository paths.
Evidence directory: `timestral-native/m6c-opt`.

```sh
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_TESTS=ON
cmake --build "$BUILD" -j2
"$BUILD/tests/test_linear"
TIMESTRAL_TEST_TINY="$SUITE" ctest --test-dir "$BUILD" -j1 \
  -E '209m|^test_model_loader$' --output-on-failure
env -u TIMESTRAL_TEST_TINY ctest --test-dir "$BUILD" \
  -R '^test_model_loader$' --output-on-failure
TIMESTRAL_TEST_TINY="$SUITE" "$BUILD/tests/test_forecast_determinism"
TIMESTRAL_TEST_GGUF="$MODEL" "$BUILD/tests/test_forecast_determinism" --real-512
# Run separately for N=1 and N=6, with a cooldown between processes.
"$BUILD/examples/cli/timestral-cli" bench --model "$MODEL" \
  --context-len 512 --horizon 32 --threads "$N" --warmup 1 --repeat 5
```

| Artifact | Meaning |
|---|---|
| `configure.log`, `build.log`, `final-build.log` | Release configuration and builds |
| `oracle.log`, `oracle-final.log`, `oracle-fallback.log` | Exact scalar oracle checks |
| `fallback-build.log`, `layers-scalar.o`, `test-linear-scalar` | Disabled-AVX2 fallback probe |
| `tiny-gates.log`, `loader-correct-env.log` | First run and corrected loader environment |
| `final-tiny-gates.log`, `final-loader.log` | Final 34+1 passing CTest checks |
| `final-forecast-tiny.log` | Full tiny forecast comparisons and negative controls |
| `final-determinism-tiny.log`, `final-determinism-512.log` | Actual forecast byte comparisons |
| `native-512-1.json`, `native-512-6.json` | Candidate five-sample timings and RSS |
| `baseline-512-1.json`, `baseline-512-6.json` | Fresh pre-change five-sample timings and RSS |
| `baseline-binary.sha256`, `baseline-binary-after.log` | Retained baseline binary checks |
| `candidate-binaries.sha256`, `source-commit.txt` | Candidate executable hashes and source pin |

## Decision and remaining gates

Keep this bounded candidate for review: exact focused parity and end-to-end
latency both improve without changing the numerical contract.
Root review and authorization precede full real M6b replay and 32k repeat-five
measurements at one and six threads. Full milestone acceptance remains pending.
