# Coherent binary64 SDPA candidate: HOLD, not adopted

The rejected-for-adoption candidate uses ascending binary64 QK products/sums, promoted reference F32 scale, binary64 mask/max/subtraction/exp/weight sum/value contraction, and direct division.
Each result rounds once to F32. Products/additions remain separate under `-ffp-contract=off`.
This is an arithmetic-contract change, not a bitwise optimization or universal backend match.
All model shapes use the same formula without reference-selected branches.

Internal policies remain distinct: frozen legacy F32, mixed binary64-QK/F32-rest SIMD, mixed scalar fallback, and candidate production-route coherent binary64.
Candidate legacy and mixed oracles retain their arithmetic expectations through explicit internal callback names.
The new coherent test first fails on the actual production graph route against the previous accepted library.
It then checks production graph routing, an exact scalar policy oracle, independently materialized long-double pairwise accuracy, analytic cases, guards, byte immutability, and actual concurrent partitions 1/2/3/6.

Supported graph inputs are canonical contiguous tensors with valid masks and at least one allowed key per query.
Encoder graph construction rejects fully masked queries; low-level direct callbacks do not gain a zero-output guarantee.
A direct fully masked call still produces NaN from negative-infinity subtraction.
Nonfinite direct inputs remain outside the tested accuracy domain; no broader exception policy is introduced.
Long-double evidence is bounded and not a universal correct-rounding proof.
Parity references, floors, and tolerances remain unchanged.

## Stopped implementation attempt

The actual production graph regression test fails against accepted e70af1d before the dispatch change: exit1 at the coherent-route assertion.
The isolated candidate then passes exact route, analytic, independent long-double accuracy, partition, and byte-guard checks.
The complete frozen legacy and mixed-policy tests pass separately.
A component selection passes 36/36, and the full tiny run passes 28 cases/684 comparisons.

The first broader 5M M3 command genuinely fails, so the implementation is not adopted.
Exact case is `multi`, positive block0, layer3, free-running `floor_fr` comparison.
Case identity follows manifest order: uni_short and uni_one_patch complete both signs, then multi begins; geometry is B2*V3*S34*D192=39168.
The layer3 local `floor_tf` comparison passes (error4.303e-6, tolerance1.147e-5).
The following free-running comparison fails (error7.171e-6, tolerance7.121e-6, floor3.560e-6; approximate displayed ratio1.007).
The raw partial log contains 75 passing comparisons and one failing comparison; no complete 13-case summary is invented.
Tiny 28-case determinism passes before the M3 failure; no later real determinism is claimed.
No 5M M6b, supplemental 209M M3, bounded 209M forecast, or later determinism command runs after failure.
The partial result is not a full-profile pass.

The owned runtime/test/CMake changes are restored to their accepted pre-attempt state.
The failed candidate source archive, patch, new test source, and binaries remain in scratch for review.
Only this public-safe result document is committed.
No numerical tuning, tolerance/floor/reference change, or automatic retry follows.
The next task requires separate causal diagnosis of accumulated versus local layer3 error, not another blanket precision guess.

The failed candidate source archive SHA-256 is `f75f61572f208f7be4579a8255a477bf7e71f0baa73cdb7d003366910994da57`.
Candidate library SHA-256 is `ab40d754a42d506842137b1571f17e4746d64c153b96cd26364ec6a0da595289`.
Candidate coherent-test SHA-256 is `22b4c4d5003c77c5a4b93889d3d1867a0e5d71d58c272c80e0bf3346ee0ca531`.
All90 canonical/extended reference hashes and the 5M model/all65 5M artifact pins match before/after.
Execution remains serial under12GiB; builds use at most two compiler jobs.

Review follow-ups remain unimplemented after the stop: separate wider-oracle availability from exact-policy coverage on platforms where long double is not wider, add S1/17/33 and split3/4 focused coverage.
No skip or compiler-platform claim is made from the current host's 64-bit long-double mantissa evidence.
