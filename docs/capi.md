# Flat C ABI

This document describes the eight flat C functions in `include/timestral_capi.h`.
The functions wrap the native forecaster without changing inference.
The `timestral-cli forecast` command calls this ABI.

Handles own Model then Forecaster and are non-reentrant; callers serialize all
operations on one handle. Independent handles may execute independently.
Saved CPU threads apply during each forecast with restoration of caller state.

`last_error(NULL)` returns thread-local load/null-handle diagnostics.
Successful load clears that storage; successful queries/forecast clear the
associated handle error. `last_error`, `free_buffer`, and `abi_version` do not
clear errors. Error text remains valid until the next modifying operation or
owner destruction. Error assignment failure uses a constant fallback string.
Null frees are no-ops. No C++ exception crosses the ABI.

Forecast requires out_q and sets it to zero first when nonnull. Validate signed
dimensions/counts, flags, mask bytes, pointer/count consistency, all shape and
byte products, and vector limits before reading caller buffers. Positive counts
require pointers; zero counts reject nonnull pointers. NaN missingness and
native defaults remain unchanged. Caller allocations must hold the implied
sizes because this ABI has no buffer-length fields.

Output is malloc-owned F32 [B,V,H,Q], checked finite and freed only with
`timestral_capi_free_buffer`. Invalid queries return zero or NaN with a message.
A private helper tests nonfinite output rejection without public ABI symbols.

Keep existing C++ exports. Shared builds statically embed PIC ggml. Require the
exact eight unmangled C API symbols, dlsym resolution, C99 header compilation
and layout agreement, tiny/error checks, and actual real uni_short/known_cov
bitwise comparisons against Forecaster. Shared/static builds use -j2 under
12 GiB. Missing real artifacts are skips, never acceptance passes.
No CLI, kernel, reference, tolerance, model, or licence change is included.

## Implemented contract and usage

The eight functions are `abi_version`, `load`, `free`, `last_error`,
`num_quantiles`, `quantile_level`, `forecast`, and `free_buffer`, each prefixed
with `timestral_capi_`. Version 1 preserves the PLAN request field order.

Initialize every request field. Zero-initialization disables symmetric averaging
and clipping; explicitly assign -1 to both fields to request model defaults.
A zero max_context retains the model default. The ABI does not expose the
C++ per-shot or quantile-cap overrides; those remain model defaults.
Invalid quantile count queries return zero; invalid level queries return NaN.
Both store a diagnostic. Successful queries and forecast clear handle errors.
Freeing a valid handle invalidates its error pointer; null frees are safe.
Input NaNs retain native missing-data behavior. Numeric/categorical preparation
otherwise follows Forecaster without numeric conversion at the ABI boundary.

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

Callers remain responsible for valid pointer lifetimes and implied buffer sizes.
No C ABI can validate an arbitrary dangling or undersized pointer. Dimension
and byte overflow checks reject invalid representable shapes before buffer reads.
Concurrent use or free of the same handle is unsupported; independent handles
have independent allocators/errors. Load and forecast never leave a changed
thread-count setting in the calling thread.

## Transport checks

Static and shared Release builds pass the following bounded checks, CPU-only,
serial under 12 GiB, using `-j2` compilation:

| Build | Tiny/error/layout checks | Tiny transport comparisons | Real checks | Real transport comparisons |
|---|---:|---:|---:|---:|
| Static | 140 total checks | 6 | 54 | 4 |
| Shared | 149 total checks | 6 | 54 | 4 |

Tiny transport covers uni_short, known_cov, and cat_uni. Each fixture runs with
model-default flags and then an explicit boolean observation mask with averaging
and clipping disabled. Real transport runs actual **uni_short and known_cov**
fixtures with both request variants. Every output float matches Forecaster
byte-for-byte; quantile levels match exactly. Real default result sizes are
16128 bytes for uni_short and 5376 bytes for known_cov.
Both builds pass CTest **2/2**, including the real test with no skips.

Error coverage includes bad/null file paths, invalid thread count, null handles,
requests and out_q, invalid dimensions/flags/masks/index, all covariate pointer/
count mismatches, overflow before reading undersized dummy buffers, error
clearing, TLS isolation, independent handles, output ownership, and thread-count
restoration on both success and a Forecaster categorical-infinity exception.
Private helper checks reject a nonfinite synthetic output and unsupported known
covariates using a synthetic config. The latter is not an on-disk unsupported
checkpoint test; no real model weights/config are modified to create one.
The allocation-safe constant-error fallback exists but allocation-failure
injection, sanitizer testing, and concurrent inference stress are not claimed.

Strict C99 compilation with `-std=c99 -pedantic-errors -Wall -Wextra` passes.
The C translation unit and C++ test agree on request size and all 17 offsets.
The shared library retains existing C++ exports and exports exactly the eight
unmangled C API names; every name resolves through dlsym. `ldd` contains no
external ggml library. Verification here is Linux/GCC; Windows export annotations
are present but Windows/macOS builds are unverified.

## Reproduction and pinned evidence

```sh
cmake -S . -B "$STATIC" -DTIMESTRAL_BUILD_TESTS=ON -DTIMESTRAL_SHARED=OFF
cmake --build "$STATIC" --target test_capi -j2
cmake -S . -B "$SHARED" -DTIMESTRAL_BUILD_TESTS=ON -DTIMESTRAL_SHARED=ON \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "$SHARED" --target test_capi -j2
export TIMESTRAL_TEST_TINY="$SUITE" TIMESTRAL_TEST_REAL_BASELINES="$SUITE"
export TIMESTRAL_TEST_GGUF="$MODEL"
ctest --test-dir "$STATIC" -R '^test_capi(_209m)?$' -j1 --output-on-failure
ctest --test-dir "$SHARED" -R '^test_capi(_209m)?$' -j1 --output-on-failure
```
