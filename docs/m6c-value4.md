# Bounded four-accumulator attention value experiment

Parent: `168f74c0560a5ae10d5b102f14f294b9eaf5e2eb`.
The [packing aggregate profile](m6c-pack.md#aggregate-packing-profile) identifies
value contraction as 24.48% of sampled self time.

## Change and arithmetic contract

The value helper computes 32 adjacent head dimensions with four independent
eight-lane accumulators in one ascending-key loop. Each probability broadcast
feeds four contiguous value loads. Every output dimension retains the original
key-order FMA chain and separate reciprocal multiplication after contraction.
The previous eight-dimension loop handles remaining complete vectors; scalar
tails and narrow heads remain unchanged. QK, softmax, masks, linear packing,
graph task policy, runtime dispatch, and ownership remain unchanged.
No new storage or persistent cache is introduced.

Release builds retain the established floating-point flags and runtime AVX2/FMA
checks. No fast-math, tolerance, floor, reference, or baseline changes occur.

## Verification

Both optimized and forced-scalar attention callbacks pass **9,360 byte-exact
comparisons / 22,248,000 output values** against the frozen scalar oracle.
The extension adds head dimensions 31 and 33 around the 32-dimension boundary;
63/64/65 already cover a second block and tails. Existing sequence, split,
mask, random/adversarial finite input, signed-zero, and subnormal coverage stays.
One-/six-partition callback checks pass with output sentinels intact.

- Serial component/tiny CTest: **35/35 pass**; loader separately **1/1 pass**.
- Full tiny parity: **28 cases, 164 teacher blocks, 164 free blocks, 28 finals,
  684 comparisons, zero failures**, worst normalized ratio `0.618082`.
- Actual forecast byte comparisons pass for all 28 tiny cases and real 512/32,
  using thread sequence 1, 6, 6, 1 against an initial one-thread forecast.

No full real M6b or full-shape repeat-five replay runs for this candidate.

## End-to-end results

CPU: AMD Ryzen 5 7430U. Pinned 209M F32 and unchanged
[measurement contract](benchmarks.md#measurement-contract).
Builds use `-j2`; heavy processes execute serially under 12 GiB.
Each configuration runs in a separate process, with 20-second cooldowns and
no overlapping compilation.

| Runtime | Context / horizon | Threads | Warmup / repeats | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|
| Prior packing, fresh cold sample | 32768 / 1024 | 1 | 0 / 1 | 63.395587558 | 943316992 |
| Four-accumulator candidate, cold sample | 32768 / 1024 | 1 | 0 / 1 | 49.551510248 | 943292416 |
| Four-accumulator candidate, warm median | 512 / 32 | 1 | 1 / 5 | 1.698840031 | 849539072 |
| Four-accumulator candidate, warm median | 512 / 32 | 6 | 1 / 5 | 0.555982916 | 847478784 |

The fresh matched-contract cold 32k samples improve **1.28x**, approximately 22%
less latency. Candidate precedes baseline; cooldowns do not eliminate frequency
or thermal uncertainty. The prior session's packing sample was 56.128880 seconds;
absolute values vary across sessions. Neither cold single sample is an acceptance
median, and no six-thread 32k result exists for this candidate.
Warm 512 medians are diagnostic only; no fresh warm baseline or short-shape
speedup claim accompanies them. Input/output sums match across the 32k probes;
exact tests, not sums, establish focused numerical evidence.

Keep this bounded candidate for root review because the measured full-shape
latency improves with exact focused parity. The 49.55-second cold result remains
well above the 21.57-second one-thread warm acceptance limit; that mixed-contract
comparison is a planning warning, not a full gate result.
Full M6c remains on hold. Reprofile before selecting the next kernel experiment.

## Commands and artifacts

Scratch directory relative to the work root: `timestral-native/m6c-value4`.

```sh
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_TESTS=ON
cmake --build "$BUILD" -j2
"$BUILD/tests/test_attention"
TIMESTRAL_TEST_TINY="$SUITE" ctest --test-dir "$BUILD" -j1 \
  -E '209m|^test_model_loader$' --output-on-failure
env -u TIMESTRAL_TEST_TINY ctest --test-dir "$BUILD" \
  -R '^test_model_loader$' --output-on-failure
TIMESTRAL_TEST_TINY="$SUITE" "$BUILD/tests/test_forecast_determinism"
TIMESTRAL_TEST_GGUF="$MODEL" "$BUILD/tests/test_forecast_determinism" --real-512
"$BUILD/examples/cli/timestral-cli" bench --model "$MODEL" \
  --context-len 512 --horizon 32 --threads "$N" --warmup 1 --repeat 5
"$BUILD/examples/cli/timestral-cli" bench --model "$MODEL" \
  --context-len 32768 --horizon 1024 --threads 1 --warmup 0 --repeat 1
```

| Artifact | Meaning |
|---|---|
| `configure.log`, `build.log` | Release configuration and build |
| `oracle.log`, `oracle-fallback.log`, `fallback-build.log` | Exact optimized/fallback checks |
| `tiny-gates.log`, `loader.log`, `forecast-tiny.log` | Component and full tiny parity |
| `determinism-tiny.log`, `determinism-512.log` | Actual forecast byte comparisons |
| `native-512-1.json`, `native-512-6.json` | Warm diagnostic medians and samples |
| `native-32k-1.json`, `baseline-32k-1.json` | Matched cold single-forecast probes |
| `candidate-binaries.sha256`, `source-commit.txt` | Executable hashes and source pin |
