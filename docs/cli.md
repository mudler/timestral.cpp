# Approved preparatory CLI design

Forecast and info run alongside the unchanged bench command.
The optional quantize command requires `TIMESTRAL_BUILD_QUANTIZE=ON`.
With that option OFF, the existing unsupported-command behavior remains unchanged.
This transport work does not accept M6c performance, M7 quality, or the whole M8 milestone.

## Optional quantize command

```text
timestral-cli quantize <in.gguf> <out.gguf> q6_k|q5_k|q4_k
```

Only F32 GGUF inputs are accepted; this command does not requantize F16, Q8, or K inputs.
Existing destinations, aliases, and symlinks are rejected without clobbering files.
Exit 2 means invalid argument count or output type; exit 1 means conversion or reporting failure.
Exit 0 prints one JSON object with `input_sha256`, `output_sha256`, `input_bytes`, `output_bytes`,
`converted_tensors`, `copied_tensors`, and `type`.
If publication succeeds but directory durability is unconfirmed, exit 1 explicitly reports that output exists and no rollback occurred.
A stdout failure after publication also reports that output exists.
The optional tool library owns OpenSSL; inference-only builds do not acquire that dependency.
Forecast CSV parsing, info output, and bench contracts remain unchanged.

The isolated Slice B build uses `8141790` plus CLI-only changes.
ON CTest passes 3/3: existing transport, core writer, and quantize subprocess checks.
OFF CTest passes 2/2: existing transport and unchanged unsupported quantize behavior.
Each transport run performs 39 process checks and six bytewise comparisons on synthetic tiny inputs, including info and a short bench.
The final quantize harness performs 26 process checks, including all three formats and a postpublication `/dev/full` report failure.
No real forecasting replay or model-quality evaluation runs for this CLI slice.
The ON/OFF inference shared libraries are byte-identical; only the enabled CLI links libcrypto.
The initial build archive SHA-256 is `4de473da75f389f52f9bb42dbe84fda57ff8c2977b046495e20736b5cf062874`.
The final tested archive, adding only the report-failure test, is `a5d8cfc2680152bdf8699b1cd61b8635c7d13f9f53cddc6650fb607d3bea1644`.

Forecast calls the flat C ABI. Repeated context files form a batch with identical
T/V. Every present covariate file flag repeats exactly B times, positionally,
with equal widths across the batch; no broadcasting. Input CSV has no headers,
one time step per row. Transpose into the ABI's row-major tensors.
Missing empty cells and case-insensitive nan become NaN. Reject textual infinity
for this CLI transport (the native API's broader numeric policy remains unchanged),
ragged rows, malformed numbers, and trailing garbage with file/row/column errors.

Use model-default averaging/clipping (-1), unless --no-symmetric/--no-clip sets
zero. Threads are positive. CSV output has zero-based series,variate,step indices
and q_<roundtrip-double-level> columns. F32 values use max_digits10, levels use
double max_digits10. JSON is {shape:[B,V,H,Q],quantiles:[...],values:[flatBVHQ...]}.
Only finite results reach stdout; diagnostics use stderr. Exit codes: 0 success,
2 usage/input, 1 load/inference/output. Info validates the model and emits version,
geometry, and quantile levels in readable text or --json, never tensor weights.

Verify strict parsing, missing values, batches, numeric/categorical covariates,
CSV/JSON float roundtrips, tiny and real uni_short/known_cov byte equivalence
against C ABI, and unchanged bench contract on a tiny bounded workload.
Builds/tests run serially under 12 GiB with -j2. Pin tested source/binaries
before/after; no full 32k performance or real aggregate parity replay.

## Commands

```sh
timestral-cli info --model model.gguf --json
timestral-cli forecast --model model.gguf --context target.csv --horizon 64 --json
timestral-cli forecast --model model.gguf --context a.csv --context b.csv \
  --known-cov a-known.csv --known-cov b-known.csv --horizon 64 --threads 2
```

Context rows are time steps; columns are target variates. Past numeric and
categorical files contain T rows; known files contain T+H rows. For each present
covariate flag, supply exactly one file per context in the same batch order.
Files within each group must have identical widths. Context T/V must agree.
Input is headerless, unquoted numeric CSV with optional surrounding spaces,
CRLF or LF, decimal/exponent notation, empty missing cells, and nan in any case.
Infinity, NaN payload syntax, hexadecimal numbers, overflowing numeric values,
and trailing garbage are rejected. This CLI policy is intentionally narrower
than native numeric-infinity preparation. Categorical values retain double
precision. Numeric values round once to F32 before calling the ABI.

CSV results use zero-based indices and quantile columns such as q_0.5.
Parsing each numeric result back to F32 reproduces the exact ABI output bytes.
JSON shape and values use B,V,H,Q ordering; quantiles contains ordered levels.
Info JSON includes version, name, patch_size, d_model, num_heads, layers,
context_length, block_size, and quantiles. It validates the model without
printing tensor weights. Neither command changes model defaults silently.

Exit status 2 indicates usage or input-file errors; 1 indicates load, inference,
or output-write failure; 0 indicates success. Output is fully rendered before
writing, and write errors are checked after flushing. A broken pipe can still
be subject to the platform's SIGPIPE policy. Quantize is explicitly unsupported.
The existing bench calculation and JSON code are unchanged; only dispatch to
the new commands precedes that branch.

## Verified preparatory transport

The shared Release CLI passes **34 process checks and 6 byte comparisons**
in the tiny harness: basic transport, two-series/two-variate missing inputs,
explicit flags, and two-series numeric plus categorical covariates.
Both JSON and CSV roundtrips match direct C ABI bytes. Invalid syntax, masks
represented by missing CSV cells, ragged rows, time/width mismatch, covariate
file-count/batch-width mismatch, unsupported commands, and load failures are
covered. Info readable/JSON and the unchanged tiny bench input/sample contract
are checked. A separate /dev/full probe returns output-failure status 1.

Actual real uni_short and known_cov fixtures pass **4 process checks and 4
byte comparisons**: JSON and CSV each match direct C ABI output for both cases.
Result byte sizes are 16128 and 5376 respectively. No real case skips occur.
The registered CTest passes 1/1 and executes the full tiny harness, including
inference; it is not a parser-only test. Real transport is a separate direct
run using the gguf-capable Python environment, not part of that CTest count.
Python is a test dependency only; the CLI requires no Python at inference.

Builds and tests execute serially under 12 GiB with -j2 compilation.
No kernel, Forecaster, C ABI, model, baseline, or numerical tolerance changes
accompany this transport. M6c performance/keypack retention and M7 quantization
acceptance remain pending; this is not full M8 acceptance.

## Reproduction and evidence

```sh
cmake -S . -B "$BUILD" -DTIMESTRAL_SHARED=ON -DTIMESTRAL_BUILD_TESTS=ON
cmake --build "$BUILD" --target timestral-cli -j2
ctest --test-dir "$BUILD" -R '^test_cli$' --output-on-failure
python3 tests/test_cli.py --cli "$BUILD/examples/cli/timestral-cli" \
  --library "$BUILD/libtimestral.so" --model "$TINY_MODEL" --scratch "$SCRATCH"
TIMESTRAL_TEST_REAL_BASELINES="$SUITE" "$REFERENCE_PYTHON" tests/test_cli.py \
  --real --cli "$BUILD/examples/cli/timestral-cli" --library "$BUILD/libtimestral.so" \
  --model "$MODEL" --scratch "$REAL_SCRATCH"
```

Evidence under `timestral-native/m8-cli`, relative to the work root:
`configure.log`, `build.log`, `flush-build.log`, `final-ctest.log`,
`final-tiny.log`, `final-real.log`, `output-failure.log`, generated CSV inputs,
`final-source.tar`, `final-before.json`, `final-after.json`, and
`source-commit.txt`. The archive/CLI/shared-library hashes match before/after
final direct suites. The CTest log precedes the final output-flush adjustment;
final direct suites rerun the same harness on the final binary.


## Numeric-sign parser correction

Review reproduced acceptance of malformed `+-1` and `+-1e2` by the original
transport: removing a leading plus exposed a minus accepted by from_chars.
The parser now permits only a single leading sign. Malformed double signs and
a lone plus return input status 2 with file/row/column diagnostics. Regression
cases include `+-1`, `+-1e2`, `++1`, `--1`, and `+`; valid `+1.0e0` remains
covered. This correction changes only CLI parsing, tests, and documentation.

The correction is tested from an isolated archive of `cf2bf33` with only these
three CLI/test/documentation changes and the pinned ggml source. It does not
include concurrent F16 runtime/converter edits in the shared worktree.
The old binary reproduces both malformed signs with success status zero;
the extended harness fails against it. The isolated corrected binary passes
**39 process checks / 6 bytewise comparisons**, including valid positive and
negative signed exponents, all malformed-sign cases, CSV/JSON, tiny inference,
and the existing bench contract. Final-binary CTest passes **1/1** and the
/dev/full output-error probe returns 1. No redundant real fixture replay runs
because runtime/C ABI source is exactly the reviewed baseline.

Evidence under `timestral-native/m8-cli-sign`: `base.tar`, `ggml.tar`,
`ggml-pin.txt`, `red-sign-harness.log`, `configure.log`, `build.log`,
`green-harness.log`, `green-ctest.log`, `output-failure.log`, `tested-source.tar`,
`before.json`, and `after.json`. Source archive, CLI, and shared-library hashes
match before/after testing. The tested archive precedes this results paragraph;
implementation and test content match the committed fix.

Archive-build metadata caveat: configure prints ggml commit `be955f52d` because
Git discovery escapes the extracted archive and finds the surrounding control
repository. This is misleading build metadata, not the tested vendor revision.
All **1962 vendor blob IDs**, archive files, extracted files, and the vendor
portion of the hash-pinned tested-source archive match actual submodule commit
`e705c5fed490514458bdd2eaddc43bd098fcce9b` exactly. The separate vendor archive
SHA-256 is `93ff424a3c5b1978df60bc3a4e0436ca4bfdc0f8223361c11df28cbeffff3fb0`.
`vendor-blob-manifest.json` and `vendor-verification.json` preserve this check.
No different ggml source is tested; no rebuild is needed for this metadata audit.
