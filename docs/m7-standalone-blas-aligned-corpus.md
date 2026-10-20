# Standalone SGEMM: aligned diagnostic corpus

## Scope and generic allocation rule

This rerun corrects only diagnostic C-buffer allocation: every QK/value output base is 64-byte aligned.
Guard bytes surround the write region without offsetting its pointer.
Original leading dimensions remain unchanged, including awkward columns; no padded ldc is introduced.
A/B operands, flags, dimensions, threading layer, dynamic settings, arithmetic, and references do not change.
There is no outcome-selected alignment or production integration.
The original mismatch source/executable/log remains retained separately.

## Actual full-corpus result

All 552 predeclared per-head calls complete once:
- 492 model calls covering both signs of 5M multi layers0–3, including time context/decoder and layer3 variate calls;
- 48 model calls covering both signs of the 209M layer5 witness;
- 12 predeclared synthetic shape/stride/tail calls.

QK compares 168,601 outputs: zero byte differences to pinned reference GEMM.
Value multiplication compares 408,570 unnormalized outputs: zero byte differences.
The identical stored probability bytes feed reference and standalone value GEMM; this is not a comparison of separately reconstructed probability arrays.
All guards and operand-immutability checks pass.
Supported verbosity records 1104 GEMM calls; all observed C bases are mod64 zero, with original ldc.
A/B alignments remain recorded per call, not normalized or inferred from filenames.

## Flash reconstruction is a separate result

The frozen exp/reduction/reciprocal reconstruction differs from actual flash in 16 values across 11 calls, maximum `2.38419e-7`.
All 5M reconstructed flash calls are byte-exact.
Seven 209M decoder head calls differ by one output each; four synthetic calls contribute the other nine values.
Context calls remain exact in this corpus.
These residuals are report-only diagnostic output, not direct GEMM failure and not numerical-gate acceptance.
No composed model gate, full model inference, quality evaluation, or performance comparison runs.
Exact GEMM agreement does not establish a complete reference-compatible SDPA implementation.

## Native dependency observations

The standalone executable uses documented CBLAS calls, private LP64/sequential/core libraries, absolute RUNPATH, no preload/librarypath, and working directory `/`.
Supported loaded-object enumeration and loader logs show those libraries plus default dispatch and VML-default modules.
No Python, libtorch, OpenMP, or TBB loads in the standalone process.
Offline reference export remains separate and uses the pinned framework; it does not become an inference dependency.
No proprietary inspection, interposition, modification, or redistribution occurs.

## Pins and outcome

Source SHA-256: `3789e9c7172ccfa3a1716a5f18528ea5b8d78df8e3446b8ef41b42b394d1b98e`.
Executable SHA-256: `9bf2dc0c6073c96c9d1640168e46e17678d3c12fe56433cc5c2481c0fc15344b`.
Reference manifest SHA-256: `23620339f7307987e36bd042c51e7ba72eb8b51290cdfa594accb5f08f9d6c5e`.
Every input file hash matches before/after; all extracted runtime/header bytes match the retained verified-package inventory.
The selected runtime numeric version/build remains 2024/update2/build20240605; standalone dynamic state remains Dyn1.
Original licenses/notices remain private and intact.
All execution is serial under12GiB; only a small diagnostic executable compiles.
The result supports bounded standalone contraction feasibility, not runtime adoption, full SDPA parity, or M6c acceptance.
