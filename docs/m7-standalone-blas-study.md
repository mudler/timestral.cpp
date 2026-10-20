# Private standalone SGEMM study: first value mismatch

## Scope

The diagnostic uses documented CBLAS SGEMM with a privately provisioned, hash-verified oneMKL 2024.2.0 LP64/sequential runtime.
Its version query reports major2024/minor0/update2/build20240605, matching the reference numeric release/build.
The executable links no Python or libtorch and does not run a model.
An offline exporter reads immutable tensors and uses pinned Torch only to create reference GEMM/flash observations.
No proprietary code inspection, interposition, kernel modification, runtime integration, package redistribution, or adoption occurs.

## Predeclared corpus and export

The corpus contains both signs of 5M multi layers0–3, time context/decoder and layer3 variate calls; both signs of the 209M layer5 witness; and twelve synthetic shape/stride/tail geometries.
It contains 552 per-head calls, 168,601 QK values, and 408,570 value-product outputs.
The reference exporter checks actual flash output against every stored model attention tensor byte-for-byte before recording per-head calls.
Synthetic dimensions include query counts1/2/3/4/7/8/9/17, key counts3/4/5/7/8/9/31/32/33/191/192/193, head dimensions7/8/17/33/64/65, and original padded/interleaved strides.
These are corpus definitions, not claims that all calls execute in the stopped standalone study.

Physical per-head files preserve token leading dimensions instead of silently packing token rows together.
QK uses ColMajor Trans/NoTrans, `(keys,queries,head_dim)`, alpha1/beta0.
Value multiplication uses ColMajor NoTrans/NoTrans, `(head_dim,queries,keys)`, alpha1/beta0.
Single-query leading dimensions follow the reference wrapper's normalization.
The standalone process uses the same stored F32 probability inputs for value comparison.
Known exp/reduction/reciprocal reconstruction remains frozen; differences from flash are separate from GEMM equivalence.

## Actual stop

The first eighteen standalone calls complete: 5M positive layer0 context, six rows by three heads.
Every QK and value product is byte-identical to the exported reference, and reconstructed flash output also matches.
The nineteenth call is 5M positive layer0 decoder, batch0/head0:
- QK `(34,2,64)`, leading dimensions192/192: exact, zero differing values.
- Values `(64,2,34)`, value leading dimension192 and probability leading dimension34: three differing outputs.
- Reconstructed flash output differs in three values, maximum `2.38419e-7`.

The executable exits1 at this genuine byte-equivalence failure as required.
No later calls, negative signs, layers1–3, variate calls, 209M calls, synthetic calls, or composed model metrics execute afterward.
The exported corpus remains available but is not relabeled as standalone execution coverage.
Matching version/build does not prove sequential standalone arithmetic equals the reference's embedded GNU-threaded build.
This diagnostic does not identify the cause as threading alone: dispatch, code packaging, alignment, or other execution details remain unresolved.
No changed settings, output selection, alternate library, or fallback retry follows.

## Loader observations and closure

Private absolute library paths and absolute RUNPATH are used; LD_PRELOAD/LD_LIBRARY_PATH are unset and execution starts in `/`.
ldd shows LP64 interface, sequential layer, core, and system C/C++/math/dl/pthread libraries.
Supported loader diagnostics additionally observe `libmkl_def.so.2` and `libmkl_vml_def.so.2` during compute.
No Python, libtorch, OpenMP, or TBB library appears in observed loading before failure.
Although the host supports AVX2, the observed loaded dispatch module is the default module; it is not assumed to be the reference's identical internal path.
Successful end-of-corpus loaded-object enumeration does not run because the failure occurs first; loader diagnostics provide the partial closure evidence.
The complete extracted runtime remains private, with all original licences/notices intact.

## Pins and limitations

Standalone executable SHA-256: `dd0acf44efed6efa5268c0b43414f62898e0a8ead5a3bcb231e6c612e8c85cdc`.
Standalone source SHA-256: `6f6fd77c9c4811e5f7c1a8dcf171f2909b2cf4b1e2baa5de0184c42c783b3a28`.
Reference manifest SHA-256: `23620339f7307987e36bd042c51e7ba72eb8b51290cdfa594accb5f08f9d6c5e`.
Default dispatch library SHA-256: `7443958030423ce859a41a6b0debbc7799e58b8a84a3304141545918d1f81432`.
VML default library SHA-256: `e86422e1f87b7128103732cfffdaa682b9da28f1de76ae78f0a0ef1f8bf2bdbb`.
All extracted runtime/header files match their original extraction inventory after execution.
All90 canonical/extended artifacts match retained pre-study pins; all65 5M artifacts and both model identities also match retained pins.
The initial exporter shape failure retained an extra singleton mask dimension; explicit shape inventory corrected that harness issue before the successful export.
No golden file is rewritten.

All execution is serial under12GiB; only small witnesses compile.
The outcome is partial feasibility with a failed exact-value-contraction equivalence check, not backend parity acceptance.
Further investigation needs a separately approved supported-API/dispatch-settings plan; no production dependency decision follows automatically.
