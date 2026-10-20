# Scratch-only binary64 head experiment

## Contract and isolation

This is an arithmetic-contract experiment, not a production fix or bitwise optimization.
Only the three F32 contractions owned by `Head::project` change in the isolated candidate: hidden, output, and residual.
Every dot sums binary64 products in ascending contracted-dimension order, then rounds once to F32.
Activation and residual-add boundaries remain unchanged.
There are no model, missing-row, execution-shape, or winning-case branches.
Global `linear_f32`, patch projection, QK/value arithmetic, scaler behavior, and production worktree sources remain unchanged.
Reduced-format weights retain the existing GGML dispatch; no reduced-format execution is claimed for this experiment.

The baseline source is `4e77007bc37034e60331edae7aed746e1afb8272`.
Source inspection verifies the only changed archived production file is `src/head.cpp`.
All 1962 ggml blobs match `e705c5fed490514458bdd2eaddc43bd098fcce9b`, regardless of misleading outer-repository configure metadata.
The source archive SHA-256 is `e322d4f9aad302dab9c20b0ff6b3d41d26747976e528d7649dcffc52833cc278`.

## Causal head pilot

A standalone witness first applies the uniform binary64 contractions to exact stored encoder outputs.
Source label 3 in this experiment means binary64-head raw output through native monotone/unscale/cap/average, not stored reference raw output.
Its combined final ratio is 0.12724489, versus baseline 1.9828851 on the same immutable `all_missing` artifacts.
Maximum raw difference from Torch falls from `7.62939e-5` to `3.8147e-5`; this is not bitwise reference parity.
The independent scalar binary64-head reconstruction matches the compiled candidate exactly over 1344 raw values for each sign.
That reconstruction does not call the candidate contraction callback.

## Unchanged strict gates

The native candidate completes `all_missing` M6b: two teacher blocks, two free blocks, one final, nine comparisons, maximum ratio 0.360359, zero failures.
The intended 5M profile explicitly selects all 13 non-slow names; PARTIAL/EXPLICIT SUBSET diagnostics remain visible.
M3 passes 30 selections, 240 local layers, and 510 comparisons, with no missing layers.
M6b passes 13 cases, 32 teacher blocks, 32 free blocks, 13 finals, and 141 comparisons, maximum ratio 0.947251, zero failures.
Nineteen repeated-amplification diagnostics remain visible; the maximum adjacent amplification is 20.8664.
The bounded 209M uni_short/known_cov/all_missing subset passes 27 comparisons, maximum ratio 0.547373, zero failures.
No floor, tolerance, case input, model weight, or reference artifact changes.

An initial M3 command used `selection=all`, which included absent canonical slow cases and stopped at `MISSING real/ctx_cap`.
That configuration failure is retained separately; the corrected command uses the exact approved 13-case profile.
No numerical failure is hidden by changing that selection.

## Limits and pins

The candidate is not retained in production. Production adoption needs a separate precision decision, TDD scope, component coverage, and broader aggregate replay.
The existing `test_head` executable is built but not run; no claim is made that old component-oracle expectations accept the new contract.
No full tiny/component, reduced-format, full 209M, performance, or M6c acceptance is claimed.
The successful scratch gates establish evidence for a proposal, not automatic completion of the production 5M prerequisite.

Candidate library SHA-256: `249abc41273ab45edefb9b257462bfd203fa0ac5bab25f2344ab800ce8f285a6`.
Candidate forecast executable SHA-256: `62e0ca35e64b1abc668216dd901bd03a500659bcca7836344ab849d2f3da6a3b`.
Candidate encoder executable SHA-256: `da2eed4ef711c65d18525afc02d8721de0a19576f725854d3fe6adf5a9f2710d`.
Before/after hashes cover the retained baseline library/forecast executable, 5M GGUF, and all 65 reference files; all match.
All execution is serial under a 12 GiB address-space limit; builds use at most two compiler jobs.
No model generation, backend dependency, proprietary kernel copying, production runtime edit, or push accompanies this experiment.
