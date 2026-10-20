# Evaluation driver: tsqbin-v1 and tsqbin-v2

## Complete recorded-matrix producer

`scripts/record_quant_matrix.py` is separate from the v1 F32 development producer.
It requires `--driver`, `--models`, `--build-sidecar`, and a new `--output` directory.
The models JSON object maps exactly `f32`, `f16`, `q8_0`, `q6_k`, `q5_k`, and `q4_k` to unique regular nonsymlink paths.
The build sidecar has exactly `version=quant-driver-build-v1`, `protocol_version=2`, `runtime_commit`, `source_sha256`, and `executable_sha256`.
Commit/source identity remains asserted build provenance; executable and model hashes are measured before and after execution.

The producer reads actual GGUF tensor metadata and compares logical shapes, configuration, licence/source fields, original weights identity, and QKV layout.
It requires exact requested storage on every B.4 weight, F32 exclusions, frozen options, and compatible rows.
K provenance must identify the mapped F32 GGUF as its actual input hash.
It never infers storage from filenames or manufactures six model identities.

For each frozen item, it constructs inputs/truth once, invokes six v2 requests, and captures anchors only in the F32 request.
That yields 474 invocations and 79 anchor captures for a complete matrix.
The scorer independently reconstructs fixtures during validation; this is separate from producer construction.
Each format retains request/result bytes and an identity audit; `build-audit.json` retains model metadata and sidecar provenance.
Each NPZ uses the scorer's exact array names, one frozen F32 anchor pair, and all six forecasts.
The producer invokes the actual recorded scorer before publishing `manifest.json` last.
A numerical FAIL report does not become a quality PASS merely because record production succeeds.

Any failed invocation, identity mismatch, or missing output writes `HOLD.json` and withholds the scoring manifest.
Native diagnostic equality never becomes an expected-reference-error assertion or a fabricated zero forecast.
All outputs use exclusive temporary files and no-overwrite publication inherited from the development publisher.
Successful partial item artifacts may remain for audit after HOLD; no success manifest labels them complete.

Model-free tests use a complete 79-item small-shape fixture mock and six distinct mock model files.
They verify all 474 requests, F32-only capture, actual scorer ingestion, lineage/storage negatives, response association, and missing-output HOLD.
No native matrix execution, trained-weight conversion, or quality acceptance occurs in these tests.

## Preparatory six-format extension

V1 encoding and F32 behavior remain unchanged.
V2 uses the same magic and fixed header sizes, with version fields set to 2.
Format values are F32=0, F16=1, Q8_0=2, Q6_K=3, Q5_K=4, and Q4_K=5.
Request offset 116 holds the format; offset 156 holds anchor-capture flags.
Only flag bit 0 exists: it must be set for F32 and clear for every reduced format.
Result offsets 240 and 244 hold format and anchor-present flags; bytes 248–255 remain zero.
Success requires F32 anchors exactly when format is F32; reduced successes contain levels and forecasts only.
Failure responses contain neither numerical arrays nor anchors, including F32 failures.
CLI format must equal request format before inference.
Reduced execution passes no capture trace, so it never recomputes metric anchors.

Production validation retains pinned 209M source/options/quantiles checks.
Every B.4 weight must have exactly the requested storage format, with compatible contracted rows.
Every excluded tensor remains F32; mixed formats or F32 fallbacks are invalid.
An internal storage-only validation seam permits synthetic tests without weakening production geometry checks.
Converter F16/Q8 policy/hash metadata must match the format; K metadata must match its type, policy, versions, input/source hashes, and ggml pin.
F32 rejects contradictory reduced-conversion metadata.
These fields remain provenance assertions, not proof of original model execution.

This slice changes only driver/codec transport and focused tests.
The F32 development producer remains v1; a six-format manifest producer is a separate step.
No real reduced-model conversion or quality matrix is authorized by this extension.

### Verified Step A checks

The isolated build uses `2528c6328331ae6ca58358e6ac8b1bdbc06564a2` plus only Step A sources/tests.
Final source archive SHA-256: `4ebc00be0ddecac6c9d9b35b3bdd43e762a3d25d3b3dad58540b97093f0e0bfe`.
The final driver SHA-256 is `1843d115dbc8e8ce2f5d934473a091f387f7843fe5ee500c3bba83000d91ebe6`.
The existing F32 model SHA-256 is `d1e614751f82bb074c6b7a139adf9670429bef27c9e9929b943ef235eb01ffb7`.
Executable/model hashes remain unchanged before and after both protocol regressions.
All 1962 ggml blobs match `e705c5fed490514458bdd2eaddc43bd098fcce9b`.

The native synthetic codec suite passes two cases and 172 checks, including all six result formats and anchor-presence negatives.
Storage tests use descriptor-only synthetic mutations, never inference through altered descriptors.
They check complete requested formats, excluded tensors, mixed storage, incompatible K rows, and per-format provenance conflicts.
These tests are not claims of native reduced-model forecast execution.
Python passes seven tests without skips, including actual v1 F32 driver processes for `uni_short` and `known_cov`.
Separate v2 F32 processes match accepted forecast, level, and anchor bytes on those same two cases.
An independent comparison with the prior encoder confirms v1 request byte compatibility.
No real reduced-model execution, conversion, or 474-run matrix occurs.
Builds use at most two compiler jobs; all execution uses the 12 GiB address-space cap and serial tests.

## Existing v1 protocol

Preparatory binary codec and internal F32 runner only. No record producer or
format matrix, no numerical acceptance. Truth is never an input. Public CLI,
C ABI, metrics, and inference arithmetic remain unchanged.

All integers and IEEE float bit patterns use little endian; native driver
rejects non-little-endian/non-IEEE hosts. Request/result files are at most 1 GiB,
checked before allocation. Diagnostics are at most 4096 bytes; case IDs are
1–256 ASCII alphanumeric, underscore, dash, or dot characters.

## Fixed request header (288 bytes)

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | ASCII TSQREQ01 |
| 8 | 4 | version=1 |
| 12 | 4 | header_bytes=288+case_bytes+6*32 |
| 16 | 8 | total file bytes |
| 24 | 64 | eight u64 B,V,T,H,P,K,CP,CK |
| 88 | 32 | u64 max_context,per_shot,patch_size; u32 flags=3, format=0(F32) |
| 120 | 32 | f64 cap_k,extreme_threshold,minimum_scale; u32 use_asinh=1,stabilize=0 |
| 152 | 4 | case ID byte count |
| 156 | 4 | reserved=0 |
| 160 | 32 | expected model SHA256 raw bytes |
| 192 | 32 | pinned HF revision ASCII40 is instead represented by SHA256 of its ASCII bytes |
| 224 | 32 | pinned reference SHA ASCII40 represented by SHA256 of ASCII bytes |
| 256 | 32 | canonical fixture input SHA256 raw bytes |

Case bytes follow; then six descriptors in context,mask,past,known,cat_past,
cat_known order. Each descriptor is u32 type,u32 reserved=0,u64 count,u64 bytes,
u64 offset. Types: F32=1,U8=2,F64=3. Counts/shapes derive strictly from header;
mask count zero means absent. Payloads must be contiguous in descriptor order,
with exact offsets and no overlap/gap/trailing bytes. Absent arrays have zero
count/bytes/offset and their declared type. Numeric covariates are F32,
categorical covariates F64. Frozen OPTIONS requires absent mask, averaging and
clipping true, max_context32768,shot32,patch32,cap10000,extreme1e30,minscale1e-6,
asinh true and stabilize false. Every dimension product is checked first.

## Result header and payload

Fixed 256-byte header: magic TSQRES01 (0), u32 version1(8), u32 status(12),
u64 total(16), u64 B,V,H,Q(24..55), model/executable/source SHA256 raw bytes
at 56/88/120, runtime commit ASCII40 at152, input-file SHA256 at192,
u32 case_bytes at224, u32 diagnostic_bytes at228, u64 payload_offset at232,
u64 reserved=0 at240, u64 reserved=0 at248.
Case then diagnostic bytes follow. Success status0 requires empty diagnostic,
then contiguous F64 levels[Q], F32 forecasts[B,V,H,Q], F32 loc[B,V],scale[B,V].
Failure status1 has zero dimensions and no payload; all failures are development
HOLD diagnostics, never expected-error numerical passes. Error text is bounded.

The optional driver depends on OpenSSL libcrypto for SHA256 only when enabled.
Build requires explicit runtime40hex and source-snapshot64hex definitions;
no automatic dirty-tree identity. Linux hashes /proc/self/exe. Model hash/stat
(dev,inode,size,mtime,ctime) is checked before/after execution. This detects
ordinary replacement, not adversarial race attacks or proof of source provenance.
F32 model tensors and pinned source identities/options must match before compute.

Output refuses existing paths/symlinks. Write an exclusive same-directory temp,
flush/close then publish without overwriting via hard-link+unlink. Reject output
aliases with input/model/executable. No stale success output is reused.

Code preparation and model-free Python checks may run now; native build/test
and CMake registration wait for root's slot after Q8 completes. Later checks
use isolated cd9097a+driver sources, hard12GiB and -j2; tiny and real two-case
output/anchor identity checks do not establish a complete evaluation matrix.

## Verified implementation slice

The internal driver is F32-only and built with explicit identity definitions.
The runtime commit identifies isolated base `cd9097ab74c2c0d9c3810d5aecbd77f2b134c8be`;
the independent source-snapshot SHA256 covers the actual base plus driver source
and tests before configuring. It is not a claim that the new driver existed
in that base commit. Build cache and binaries are outside that archive, avoiding
a self-referential source hash. These identities remain asserted build provenance,
not a cryptographic proof that a compiler used the claimed source.

The decoder validates descriptor counts and output size before allocating input
arrays. Frozen 209M geometry and quantile levels are validated on the loaded model.
Mask descriptor exists but must be absent under frozen OPTIONS. Python preserves
all inference-array payload bytes; native decoding uses memcpy on supported hosts.
The fixture input hash is caller-supplied provenance, distinct from the driver's
computed SHA256 of the actual request file. No truth array field is accepted.

No symlink input/model file or pre-existing output is accepted. Existing output
refusal also blocks ordinary hard-link aliases to input/model/executable.
Output publication uses an exclusive same-directory temporary, fsync/close,
then link/unlink without overwrite. Parent-directory/adversarial replacement
races and power-loss directory durability are not proven by this mechanism.
Model and executable before/after SHA256 plus model dev/inode/size/mtime/ctime
checks detect ordinary changes; they do not eliminate adversarial replacement
races during mmap execution. All failure results are bounded HOLD diagnostics,
never successful forecast payloads or expected-error numerical passes.

### Completed checks

- Python protocol suite: **5/5 pass, no skips**, including actual executable
  invocation for real uni_short/known_cov. It checks input-array payload layout,
  signed-zero/NaN bits, all nonempty covariate arrays, malformed identities/types,
  output header mutations/truncation/trailing data, scales, and result shape.
- Native tiny codec/direct transport: **2 cases / 50 checks**. Tiny geometry
  deliberately differs from frozen 209M OPTIONS, so the test overrides only its
  direct Forecaster max-context/shot controls; the actual executable rejects
  incompatible tiny models rather than falsely certifying frozen options.
- Native real codec/direct transport: **2 cases / 12 checks**. Export-mode adds
  two fixture-write checks, yielding 14 when exporting. Forecasts and captured
  anchors match direct Forecaster bytes.
- Separate process harness: **12 checks**. The actual executable runs both real
  cases; forecast, levels, location, and scale bytes match direct native exports.
  Actual model/executable/request hashes and compile identities are checked.
  Existing output remains unchanged on rerun. Format, wrong model hash,
  truncation, overlapping offsets, bad flags, symlink input, alias output, and
  a sparse request exceeding 1 GiB are rejected with no forecast payload.
- CTest: **2/2 pass**, including actual real artifacts, no skips. These native
  tests exercise codec/direct transport; executable-process evidence is reported
  separately, not implied by the CTest name.
- Final source snapshot, driver executable, and native test hashes match before/
  after all checks. Native execution rehashes input/model/executable around each
  successful forecast. No format matrix, model conversion, or performance run.

The optional Python executable test skips only when its explicit driver/export/
model paths are absent. The recorded final run provides all paths and has zero
skips. Python dependencies are test-side NumPy only, not native inference.
The protocol's result carries resolved shape and identity fields; options and
format are fixed by version1 and validated against the request/model, not an
additional mutable response option dictionary.

### Dependency resolution and build isolation

The host lacks OpenSSL development headers. A matching Ubuntu package is
extracted only under scratch: `libssl-dev 3.5.5-1ubuntu3.7`, matching installed
`libssl3t64 3.5.5-1ubuntu3.7`. Package SHA256 from the local APT index:
`d1cf6cafb141490b7020e392964b1f901835d60cb88c3a68ef0f95ef5e930de4`.
No system package installation or crypto implementation substitution occurs.
Headers: `timestral-native/m7-driver/dependencies/root/usr/include`, plus
`usr/include/x86_64-linux-gnu` for opensslconf.h. Link the matching system
`/usr/lib/x86_64-linux-gnu/libcrypto.so.3`. This OpenSSL dependency is scoped
to the default-OFF driver option; the library/public CLI do not acquire it.

Configure failure without headers and the first missing-multiarch compile are
preserved. Corrected configuration and final builds pass. Archive Git discovery
reports surrounding control-repository commit `be955f52d` as ggml metadata;
all **1962 vendor blobs** in the actual tested snapshot match pinned ggml
`e705c5fed490514458bdd2eaddc43bd098fcce9b`. This is metadata leakage, not a
change to the vendor source. Concurrent Q8/K-quant edits are excluded from
this isolated build. All builds/checks run serially under the hard 12 GiB limit,
with -j2 compilation; no sanitizer or uncapped run is claimed.

### Reproduction and artifacts

```sh
cmake -S . -B "$BUILD" -DTIMESTRAL_BUILD_TESTS=ON \
  -DTIMESTRAL_BUILD_EVAL_DRIVER=ON \
  -DTIMESTRAL_EVAL_RUNTIME_COMMIT="$RUNTIME_COMMIT" \
  -DTIMESTRAL_EVAL_SOURCE_SHA256="$SOURCE_SNAPSHOT_SHA256"
cmake --build "$BUILD" --target timestral-eval-driver test_eval_driver -j2
TIMESTRAL_TEST_TINY="$SUITE" TIMESTRAL_TEST_REAL_BASELINES="$SUITE" \
  TIMESTRAL_TEST_GGUF="$MODEL" ctest --test-dir "$BUILD" \
  -R '^test_eval_driver(_209m)?$' -j1 --output-on-failure
```

For scratch-only headers additionally set OPENSSL_INCLUDE_DIR to the include
root, OPENSSL_CRYPTO_LIBRARY to the system libcrypto.so.3, and add the extracted
multiarch include directory through isolated build CXX flags. No system setting
changes are needed. On supported systems with development packages, normal
find_package(OpenSSL) suffices. Linux is the supported platform for this slice.

Evidence under `timestral-native/m7-driver`: `configure.log`,
`build-missing-multiarch.log`, `configure2.log`, `configure3.log`,
`final-configure.log`, `final-build.log`, `final-native-tiny.log`,
`final-native-real.log`, `final-python.log`, `final-process.log`,
`final-ctest.log`, `final-source-snapshot.tar`, its SHA256 file,
`before.json`, `final-after.json`, `vendor-verification.json`, and dependency
package metadata/checksum logs. Final real binary requests/results and direct
native expected bytes are in `final-real-exports`.

The fixed codec does not produce quant-record-v1 manifests/NPZs or score all79.
A separately approved producer must verify fixture associations, actual hashes,
and successful F32 anchors before writing records. Reduced formats and expected-
error classification remain unimplemented. M6c, the 5M prerequisite, full M7
quality evaluation, and complete milestone acceptance remain pending.

## Approved F32-only development producer contract

`record_quant.py` produces **development-f32-record-v1**, explicitly non-scoring.
It never emits a quant-record-v1 manifest, six-format placeholders, or numerical
PASS. A run selects frozen case IDs, serializes inference arrays from unchanged
quant_fixtures, and sends no truth to the accepted native driver. A model-free
inventory validates all79 IDs and serialization without running inference.

Each selected item has its own NPZ, request, and strict JSON audit, plus a run
index. NPZ contains context, truth, quantile levels, final F32 values, anchor_loc,
anchor_scale and supplied covariate arrays. Load with allow_pickle=False.
Audit records canonical case/input/truth hashes, actual request/model/executable
hashes, asserted runtime/source identity, frozen options, shape and array hashes.
Response case/request/model/executable/source/commit identities must all match;
options are fixed by tsqbin-v1 and validated against freshly decoded fixture
shape/levels. Anchors must be finite F32 [B,V] with positive scales. The trusted
native capture remains the anchor source; Python cannot prove arbitrary plausible
anchor bytes came from execution merely by validating their shape and hashes.

Refuse existing outputs, symlinks, path aliases, unsafe item IDs, unknown response
fields, and nonfinite forecasts/truth/anchors. Validate all response data before
publishing NPZ/audit via exclusive same-directory temporaries with no overwrite.
No partial files masquerade as completed items: run index publishes last, and
errors leave no successful item entry. Existing outputs are never overwritten.
Model/executable hashes are independently checked before and after each process.
All driver errors remain development HOLD diagnostics, never expected passes.
Unit tests use small mocked fixtures/processes; verified real artifacts may be
consumed without launching inference. No heavy execution is authorized yet.

### Producer implementation and model-free verification

The producer implements the approved development-only contract. `--inventory`
constructs every frozen fixture and serializes all **79 identities**, totaling
**938712 request bytes**. It executes no model and stores no truth in requests.
Execution requires explicit case selection, actual driver/model files, runtime
commit, source snapshot hash, and a new output directory. Only F32 is invoked.
Run index publication occurs last; failure leaves no completed run index.
A previously completed item audit may survive a later item failure, but is not
a completed run or scoring manifest. Existing files are never replaced.

`tests/python/test_record_quant.py` passes **9/9** with mocked processes and
small fixtures. Coverage includes pickle-free NPZ roundtrip, exact input bytes,
identity mismatches, process failure, model/executable/request mutation,
nonfinite truth/forecasts/anchors, invalid anchor shape/scale, symlink/alias
rejection, overwrite refusal, and cleanup on injected publication failure.
Nine tests are grouped cases, not nine model executions.

Two already verified real native response artifacts (uni_short/known_cov) pass
producer response/hash/shape/anchor validation without launching another process.
Those old requests use the native test case ID, not canonical producer fixture
identities. Therefore this is response-consumption evidence only, **not** a claim
that the complete producer has executed real canonical cases. An actual two-case
producer run still requires a separately granted inference slot.

Finite but incorrect anchor values cannot be independently detected by Python
without a trusted native result oracle; the producer validates their type,
shape, finiteness, positive scale, request/model association, and response hashes.
The native driver's previously verified actual capture is the provenance basis.
No fabricated expected anchor values or quantized anchors are substituted.

Use the existing NumPy environment:

```sh
python scripts/record_quant.py --inventory
python scripts/record_quant.py --driver "$DRIVER" --model "$F32_MODEL" \
  --runtime-commit "$RUNTIME_COMMIT" --source-sha256 "$SOURCE_HASH" \
  --case uni_short --case known_cov --output "$NEW_OUTPUT_DIRECTORY"
```

The second command is documented, not executed in this slice. No native build,
inference, model conversion, quant-record-v1 manifest, or quality PASS runs.
All checks use the 12 GiB guard and single-threaded numerical Python settings.
Evidence under `timestral-native/m7-producer`: `verified-unit.log`,
`verified-inventory.json`, `archived-real.log`, and source/test pin records.
The native driver, protocol, metric/fixture modules, public CLI/C ABI, and
inference code remain unchanged. Performance and full quantization acceptance
remain pending. Publication uses same-directory hard-link/no-overwrite semantics;
adversarial filesystem races and power-loss directory durability are not proven.

### Canonical two-case F32 producer execution

A separately authorized end-to-end run executes canonical **uni_short and
known_cov** through the reviewed F32 driver, one process per case, serial and
single-threaded under the hard 12 GiB guard. Producer source matches `a733de2`;
the existing executable matches the reviewed driver pin. No rebuild occurs.
The resulting run index declares `development-f32-record-v1`, `scoring=false`,
and the two canonical identities. This supersedes the earlier producer execution
coverage gap for these two cases only, not the remaining 77 or reduced formats.

An independent artifact verifier reconstructs both fixtures and checks exact
request payload bytes, including NaN bit patterns and supplied covariates,
all fixed protocol options, canonical case/input/truth associations, and each
pickle-free NPZ array against fixture truth/inputs or the native binary response.
It completes **16 independent array-byte checks across 2 items**. Output shapes
are `[2,1,96,21]` and `[1,1,64,21]`. NPZ levels/forecasts/anchors match native
responses exactly; audit array hashes agree. The run index publishes after both
item audits, with no staging remnants. Truth is absent from request descriptors.

Actual model, executable, producer, protocol, fixture/metric/generator sources,
and case specifications retain their before/after hashes. Native responses
carry the expected runtime base `cd9097a`, actual source-snapshot identity,
model/executable hashes, request hash, and canonical item IDs. Native anchors
remain F32-owned and associated only with that model hash.

Evidence under `timestral-native/m7-producer/canonical-e2e`:
`producer-pinned.py`, `before.json`, `after.json`, `producer.log`,
`producer.stderr`, `verify.py`, `verification.log`, and `records/` containing
requests, binary results, NPZs, item audits, and final index.
No scoring manifest, six-format identities, quality PASS, complete 79-item
execution, performance benchmark, or model conversion is produced. This is
bounded canonical F32 producer transport evidence, not M7 acceptance.
