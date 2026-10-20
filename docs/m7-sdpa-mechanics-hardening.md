# Private SDPA mechanics hardening

## Scope

Only scratch `sdpa.hpp` and observational tests change. Product runtime remains unchanged.
The original callback/mechanics logs and earlier flash residuals remain retained separately.
No arithmetic order, ISA/dynamic setting, leading-dimension padding, or reference-output selection changes.

Preflight now validates ith/nth, source/destination pointers, positive dimensions, rows64 before narrowing to int, LP64 dimensions/stride, and split bounds before pointer arithmetic.
Checked full Q/K/V/output element and mask products, byte bounds, task counts, and ptrdiff limits establish a canonical-layout offset envelope.
Tile loop increments use the remaining valid count to avoid signed overflow at final tiles.
Output scratch indices promote products to size_t before indexing.
The adapter requires caller-provided arrays of the declared canonical extent; it does not discover allocation lengths from raw pointers.
Six malformed partition/dimension cases reject before allocation, including giant dimensions; no giant test array is allocated.

## Single-query leading dimension

Pinned public CPUBlas `normalize_last_dims` normalizes based on the current GEMM m/n/k, not the logical full attention call.
For QK T/N, current n is the query tile count; when n==1 it sets ldb=head_dim and ldc=keyBlockSize.
Only one column is read from B in that case, so its original token stride is not traversed.
For value N/N with n==1 it sets ldb=keyBlockSize and ldc=head_dim; the callback already uses those values.
No generic layout packing or ldc padding follows.
New S33, S193, and S769 multihead tests exercise one-query tile tails and compare actual pinned flash output.
Those three controls are byte-exact; S31 retains a small reference residual and is not tuned.

## Actual hardening checks

Twenty mechanics shapes pass 80 actual concurrent partition checks at ith/nth1/2/3/6 over265,968 checked output values.
The corpus retains all earlier query/key tiling boundaries and adds query31/32/33 and single-query tails.
Guards, bytewise Q/K/V/mask immutability, finite outputs, and serial/concurrent byte equality pass.
Synthetic actual-flash comparisons remain report-only: the added S33/193/769 and S32 controls are exact; S31 has86 differences/max `2.235174179e-8`.
No composed gate is inferred from those residuals.

A separate S1025 corpus spans three key chunks and exercises:
- masked first chunk, with later visible keys;
- masked middle chunk, with visible first/final keys;
- masked final chunk, with earlier visible keys;
- a late large maximum jump that underflows prior weights.
All16 concurrent partition checks pass, along with analytic visible-key means and late-maximum output checks.
Actual pinned Torch flash independently matches all four generated cases byte-for-byte: zero differences.
Overall visibility is retained; no fully masked-query policy is broadened.

The final scratch edit promotes output indexing to size_t without changing values.
The malformed/masked test recompiles and passes afterward; wider mechanics/reference checks precede only this index-type hardening.
The original60 checks and16 earlier reconstruction residuals remain separate evidence and are not rewritten.

## Integrity and limitations

Final callback SHA-256: `7e291781b7f3cc7efd2d4d9664bf1adcfc3a5bfc008245bca2f234bb6b8dcb19`.
Final malformed/masked executable SHA-256: `8e25c8ef0fe7f924f9fe3158e7daff0aaf20db8070480604154ffeb6556aad8c`.
Accepted library/encoder pins, all90 canonical/extended references, all65 5M references, and model identities match retained before/after values.
Public-source attribution and full PyTorch licence remain alongside the private scratch adaptation.
Execution is serial under12GiB; only small diagnostic executables compile.
No tiny/full-model replay, production backend integration, numerical tuning, performance claim, publication, or push occurs.
Broader integration remains a separate approval decision.
