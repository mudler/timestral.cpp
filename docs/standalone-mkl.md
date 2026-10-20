# Optional standalone MKL CPU profile

The default build uses the existing native CPU implementation. It does not search for or link MKL.
Set `TIMESTRAL_STANDALONE_MKL=ON` only with a user-provided, qualified SDK.
This initial profile supports native Linux x86_64 builds with shared LP64, sequential, and core libraries.
It rejects cross-compilation, unsupported platforms, absent dependencies, and a wrong interface library name.
No download, vendor binary installation, or redistribution occurs.

## Build

Supply these CMake cache values explicitly:

- `TIMESTRAL_MKL_INCLUDE_DIR`: directory containing `mkl.h`.
- `TIMESTRAL_MKL_LP64_LIBRARY`: full path to `libmkl_intel_lp64.so.2`.
- `TIMESTRAL_MKL_SEQUENTIAL_LIBRARY`: full path to `libmkl_sequential.so.2`.
- `TIMESTRAL_MKL_CORE_LIBRARY`: full path to `libmkl_core.so.2`.

The compile/link probe requires four-byte `MKL_INT` and the SGEMM/version APIs.
Headers and the loaded runtime must report 2024.0 update2, build20240605.
The package label alone does not establish this identity.
The operating-system loader must resolve the supplied shared dependencies at deployment.
A missing DSO fails loading; an unsupported loaded numeric version fails initialization.
The project does not replace a failed ON backend with native arithmetic.
The SDK license and deployment obligations remain the user's responsibility.

## Arithmetic scope and threading

ON replaces SDPA and the three F32 `context_patch_projection` contractions only.
Patch contractions use bias prefill/beta1, or beta0 without bias, through documented SGEMM.
The implementation preserves whole GEMM geometry, SiLU, and separate residual addition.
Global affine arithmetic, binary64 heads, native test seams, and reduced-weight routes remain unchanged.
The sequential vendor library does not change process-global thread settings.
The custom nodes use worker0; other graph nodes retain the requested ggml thread count.
CLI provenance derives the backend identity from the validated loaded runtime.
The public C ABI remains version1 with the same eight exported functions.

## Boundary and failure behavior

Builders validate metadata and freeze tensor type, operation, dimensions, strides, source slots, and operation parameters.
Reachable callback metadata is checked before ggml graph expansion/allocation.
Storage preflight checks owning host buffers, nested view ancestors, offsets, alignment, byte extents, and output aliasing.
Unused callbacks pruned from the graph do not require storage.
Generated mask values are checked inside the callback after upstream computation.

Callbacks publish one bounded error without throwing across worker frames.
After execution failure, callbacks zero their retained, validated output view.
Downstream ordinary nodes can still run, but GraphRunner rejects the entire result after synchronization.
Invalid metadata or storage rejects before launch without writing an unvalidated output.
A fresh `begin()` recovers from actual allocation/upload/execution failures.
Pre-mutation validation rejection preserves Building/Allocated recovery when no readable result exists.
Every rejected modifying attempt on Computed invalidates the old readable result.

Fault setters exist only in test builds, in internal source headers, not the public C API.
Direct invocation of file-local callbacks with fabricated userdata is unsupported.
The shared test build can expose internal C++ symbols; do not treat internal headers as a security boundary.

## Initial verification and limits

The focused corpus covers37 boundary/route cases, seven runner-status cases, and14 lifecycle traces/120 state-event pairs.
It includes operation mutation before/after allocation, foreign intermediate host buffers, no-write canaries, scratch failure, downstream initialization, and same-runner recovery.
The unchanged RoPE test passes28 cases and384 rotations after validation/failure classification.

The frozen build matrix passes41/40/42/41 selected CTests for ON/static, OFF/static, ON/shared, and OFF/shared respectively.
Each profile passes28 tiny forecasts/684 comparisons and28 repeated/threaded determinism cases.
Shared profiles include CLI transport tests; the selections include C ABI and anchor tests.
A separate TESTS=OFF shared build lacks the injection setter.
ON/OFF/production builds retain the same eight C ABI exports without defined vendor API symbols.

This initial stage does not establish trained-model acceptance for the product integration.
It does not establish performance, quantization quality, support for other SDK versions, hostile-loader immunity, or sanitizer completeness.
Failure-chain zero testing uses a scalar payload; it is not a general proof for every allocation failure or arbitrary caller memory.
Further model gates require separate approval.

An external CMake consumer links the static ON target without manual vendor link flags and passes ABI/failure transport.
A separate OFF witness matches1188 forecast bytes from the accepted native baseline.
ON-to-OFF reconfiguration with cleared SDK paths does not execute the MKL probe.
Five negative configurations reject missing headers/libraries or the wrong interface identity.
The initial comparison harness mixed new internal headers with an old static archive and crashed before the OFF witness ran.
Matching headers to each archive corrects that harness defect; it does not change runtime arithmetic.
A separate wrong-source-root reconfigure failure is a harness error, not a backend failure.
Both failed logs remain in the private evidence.

Runtime-version rejection is implemented, but this stage does not substitute or modify vendor DSOs to test unsupported runtimes.
No proprietary interposition or binary modification occurs.

Closing hashes preserve all155 reference artifacts, both models,1962 vendor files, and199 SDK runtime/header files unchanged.
The immutable subset contains2319 entries. Final manifests also record each tested library/executable.
The frozen runtime/test source archive SHA-256 is recorded with the private acceptance artifacts.
No runtime/test source changes occur during the completed matrix.
