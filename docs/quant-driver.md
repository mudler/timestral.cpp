# Approved F32 evaluation driver: tsqbin-v1

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
