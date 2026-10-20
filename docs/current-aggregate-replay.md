# Current aggregate replay: M6b pass, supplemental M3 HOLD

## Pinned runtime and scope

This replay uses the reviewed `e70af1dc545fc83d4b61391e062c39f00e35337c` runtime implementation.
The retained static build's tested source archive SHA-256 is `a3eb6d6d79c4cc1a028c4846c6c6988dbd410820c682612f5b5a09b28d30f456`.
The archive covers actual implementation/test sources before final result documentation; it is not represented as the complete commit archive.
The forecast executable SHA-256 is `9d660a672dd97b7953f1dc380f516d8fbeba110976327c3790b79912ebea91f3`.
The encoder executable SHA-256 is `a51bec0e7ae54b85702fe7c7b5852fa1d29e2bedcd39efcad46f4f8ad5e1bfea`.
The 209M model SHA-256 is `d1e614751f82bb074c6b7a139adf9670429bef27c9e9929b943ef235eb01ffb7`.
No reference inputs, floors, tolerances, or runtime sources change during this run.
All processes execute serially under 12 GiB, with OpenMP/OpenBLAS limited to one thread.

## Full unchanged 209M M6b: PASS

All 15 canonical cases execute, including both slow cases and `all_missing` as a numerical forecast.
The complete summary is 15 cases, 36 teacher blocks, 36 free blocks, 15 finals, and 159 comparisons.
The invariant model check passes; failed=0 and selection=all.
Maximum error/tolerance ratio is 1.0.
Expected deliberately corrupted negative controls remain rejected.
Wall time is 13:25.39; peak RSS is 3,461,584 KiB.
This is numerical replay evidence, not a performance comparison or M6c acceptance.

## Supplemental strict M3: genuine failure

The next command selects five canonical all-layer cases before the three existing all-layer extensions.
It stops on `uni_one_patch`, positive block 0, local teacher-forced layer 5 output.
The case identification follows canonical iteration order and the exact tensor size: B1 x S9 x D768 = 6912 values.
The preceding `uni_short` block selections have 29,184 values and complete both signs.
The failing comparison is the first comparison for layer 5, which uses `floor_tf`, not the subsequent free-running comparison.
Reported normalized error is `3.004e-6`, tolerance `2.867e-6`, floor `1.434e-6`, cap `2.165e-4`.
The displayed precision gives an approximate error/tolerance ratio 1.048; no higher-precision ratio is inferred.
Amplification diagnostics remain visible.

The command exits 1 after this genuine numerical failure.
No remaining M3 cases, extended union, or shared-build transport tests run afterward.
There is no supplemental 8-case/784 comparison pass and no current aggregate acceptance.
The head precision change does not itself establish the encoder-only M3 result; causality is not diagnosed by this stopped run.
Earlier successful forecast subsets are M6b evidence, not equivalent local encoder gates.

## Preserved evidence and remaining work

Before/after streaming hashes cover all 75 canonical real artifacts and all 15 existing extended artifacts.
All 90 identities match, along with the model and retained library/encoder/forecast/head-test executables.
No skip, selected-reference option, tolerance change, floor rewrite, or missing-input substitution occurs.
The fresh shared source tree/build is not created because the failure precedes that stage.
CLI, C ABI, anchor, exact-export/dependency, and current aggregate determinism checks remain planned, not executed here.
Prior bounded transport/determinism evidence remains separate and is not relabeled as this aggregate run.

The next step requires a separately authorized bounded local-layer diagnosis.
No numerical correction or retry follows automatically from this report.
