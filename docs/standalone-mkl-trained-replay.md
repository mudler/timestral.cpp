# Trained replay of the optional standalone profile

## Frozen runtime and scope

This replay tests runtime commit `68a5a5394fda54bb852f3824e0a0465285edfe15` with the explicit standalone MKL profile enabled.
It uses the existing static and shared builds from the initial integration matrix.
No rebuild, arithmetic change, reference rewrite, or tolerance/floor adjustment occurs during replay.
The default native profile remains unchanged.

## Numerical results

All49 model-case invocations pass, with no missing-layer acceptance or skipped input.

| Gate | Cases | Comparisons | Additional coverage | Maximum error/tolerance |
| --- | --- | --- | --- | --- |
| Full 5M strict M3 | 13 | 510 | 30 selections, 240 local layers | 0.649274 |
| Full 5M M6b | 13 | 141 | 32 teacher blocks, 32 free blocks, 13 finals | 0.552672 |
| Supplemental 209M strict M3 | 8 | 784 | 16 selections, 384 local layers | 0.5 |
| Full 209M M6b | 15 | 159 | 36 teacher blocks, 36 free blocks, 15 finals | 0.764853 |

The supplemental canonical cases are `uni_short`, `uni_one_patch`, `nonneg`, `const`, and `all_missing`.
The extended cases are `multi`, `known_cov`, and `nans`.
This is not full15-case M3 acceptance.
The canonical `bench_32k` fixture runs as a parity case, not a performance measurement.

## Trained transport and determinism

Both models pass the following14 additional invocations:

- Static and shared C ABI: four invocations, each54 checks and four byte-exact transports.
- Static and shared anchor checks: four invocations, each two cases and68 checks.
- Static and shared real512/32 determinism: four invocations, byte-exact repeats at threads1,6,6,1.
- Shared CLI/C ABI: two invocations, each four process checks and four byte-exact comparisons.

Thus all63 test invocations return0.
Negative-input assertions inside transport harnesses do not count as numerical parity failures.
No standalone expected-FAIL numerical control enters these totals.

## Audit correction and immutable identities

The initial outer job returns1 after every test passes because its scratch audit expects two C ABI transports.
The unchanged C ABI harness increments once for default input and once for explicit mask/flags, for each of two cases.
Its correct summary is `PASS capi checks=54 transports=4`.
The original audit and failure log remain retained.
A source-checked scratch-only correction passes the final count, transport-summary, and identity audit without any inference rerun.
Anchor and CLI summary patterns also match their source counters and completed logs.
This is an audit-schema correction, not a numerical correction.

Before/after manifests match2388 entries, including all155 references, both models,1962 vendor files,199 SDK files, and tested static/shared binaries.
The exact committed source archive SHA-256 is:
`6c87921e5e1898a82d87dec5750926ea3fe2882d115669f03148795a8c8f28de`.
The vendor archive SHA-256 is:
`93ff424a3c5b1978df60bc3a4e0436ca4bfdc0f8223361c11df28cbeffff3fb0`.
The vendor commit is `e705c5fed490514458bdd2eaddc43bd098fcce9b`.

The run uses serial CPU execution under a hard12GiB virtual-memory ceiling.
The highest observed resident set across timed invocations is3462432KiB.
This resource observation is not a benchmark claim.

## Remaining decisions

This evidence closes the explicitly granted trained replay scope.
It does not establish performance, quantization quality, other vendor versions, full15-case M3 coverage, or release/publication readiness.
The earlier boundary and loader limitations remain documented in [the profile guide](standalone-mkl.md).
The owner retains the final acceptance and integration decision.
