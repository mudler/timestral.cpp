# Patch projection diagnostic: reference addmm bias placement

## Scope and exact reference

The witness uses exact stored 5M multi positive-block patch features, shape6x32x64, flattened to192 rows.
Fresh pinned Torch `ResidualBlock` operations reproduce stored hidden and final projection bytes exactly.
The public formula is `output(silu(hidden(x))) + residual(x)`; the relevant runtime path is `context_patch_projection`, not generic residual-block embedding.
No runtime implementation, precision policy, reference, floor, or tolerance changes.

## First internal discrepancy

Native hidden projection matches147456/147456 Torch values.
The reproduced native activation polynomial/operation boundaries match147456/147456 activation values, including activation on stored hidden inputs.
Native residual projection matches36864/36864 values.
The output projection on the exact stored activation first differs:16716/36864 values, maximum `1.52588e-5`.
The complete independently reconstructed native chain matches actual context_patch_projection exactly.
Adding the exact residual reduces differing final values to13066/36864 but does not remove the discrepancy.
This locates the first internal difference at output projection, not feature preparation, activation, or residual addition.

## Public execution semantics

Pinned Torch `Linear.cpp` at `5c4886908584029761b579af026dcfb627c84070` routes contiguous biased nD linear input through `_flatten_nd_linear` and `addmm(bias, input_flattened, weight.t())`.
`LinearAlgebra.cpp::addmm_impl_cpu_` copies the expanded self/bias into the output when beta is nonzero, then forwards alpha/beta to CPU BLAS.
The observed output call is SGEMM(T,N,192,192,768), with original leading dimensions768/768/192.
The native context projection instead computes192-wide F32 FMA panels, adds the panels, then separately adds F32 bias.
These are not guaranteed equivalent floating-point epilogues.
Hidden/residual K64 operations happen to match on this witness; that observation does not establish a different general bias contract or justify a model-specific branch.

Public source URLs:
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/Linear.cpp
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LinearAlgebra.cpp

## One-variable bias control

A documented standalone SGEMM call uses64-byte-aligned C, identical stored activation/weights, original flags/dimensions/leading dimensions, alpha1.
Beta0 followed by a separate bias addition reproduces the native discrepancy exactly:16716/36864 differences.
Prefilling C with bias and using beta1, with no separate bias addition, matches the reference output projection byte-for-byte.
Replacing only that output contraction in the observational chain, leaving native hidden/activation/residual/addition unchanged, makes the final projection byte-exact:0/36864 differences.
No gate-selected output or arithmetic-order sweep is performed.
This demonstrates the addmm bias-placement contract as a sufficient cause on this witness; no proprietary kernel ordering is inferred.

## Relation to prediction and limits

The earlier diagnostic establishes patch-projection drift is sufficient to trigger private integration prediction failure.
This diagnostic removes that projection drift in scratch but does not run a full prediction or model gate afterward.
It does not establish a generic dependency-free implementation of addmm semantics, nor authorize product backend expansion or bias policy changes.
A next separately approved correction needs explicit generic biased-linear semantics and regressions across K/panel/tail dimensions and models.
Head arithmetic remains a contributing downstream factor, not the sole explanation; no head switch follows.
Accepted and private candidate context-projection code is unchanged and shares this path; runtime/library/model/reference pins match retained before/after values.

Source witness SHA-256: `6502906fa6a34d342a8b97769bfcc66437be5e1ab75fbf410843c5baec0d7b7a`.
Executable SHA-256: `ab3636c5353e4c841b02f204b048aabbc8a0c8c60f73cbb8d6d6b61cfc059dcb`.
The initial witness compile failed from a missing cmath include; its log is retained separately and the corrected executable completes.
All execution is serial under12GiB, using only small diagnostic builds and public APIs.
All65 reference artifacts, model, accepted/private libraries, and forecast executable hashes remain unchanged.
No full replay, runtime fix, licence/publication claim, proprietary inspection, or push occurs.
