# Replay of the coherent 209M F32 artifact

The new F32 artifact has SHA-256
`bf7605545245a2dd4d32f7ac2f7bbe5c2a12d8f47370814a254862150255b8e8`
and size838347872 bytes. It preserves all206 original tensor payloads exactly:
838331008 bytes, including RoPE tables and signed zeros. Typed metadata differs
only at the declared original-main versus pinned-revision licence locator.
The original F32 and historical reduced artifacts remain unchanged.

The fresh six-format cohort passes the unchanged strict metadata/storage/lineage
validator with a uniform pinned licence URI. This does not change validator
policy or establish reduced-model quality. The following replay requalifies the
new F32 artifact using existing runtime `68a5a53` ON/static and ON/shared builds.
No rebuild, runtime edit, reference generation, or tolerance change occurs.

## Completed trained checks

| Gate | Coverage | Maximum error/tolerance |
| --- | --- | --- |
| Supplemental strict M3 | 8 cases,784 comparisons | 0.5 |
| Full M6b | 15 cases,159 comparisons | 0.764853 |
| C ABI static/shared | Two runs, each54 checks/four transports | Byte-exact |
| Anchor static/shared | Two runs, each two cases/68 checks | Pass |
| Real512/32 static/shared | Two runs, threads1,6,6,1 | Byte-exact |
| Shared CLI/C ABI | Four process checks/four comparisons | Byte-exact |

All30 test invocations return0; no skip or failed comparison occurs.
The M3 union contains five canonical and three extended cases, not full15-case M3.
The unchanged5M artifact/runtime is not replayed in this artifact-specific grant.
Prior [trained runtime evidence](standalone-mkl-trained-replay.md) retains its
original artifact identities and scope; it is not relabeled as this new file.

## Closing identity and limits

Before/after manifests match2394 entries, including all six coherent models,
155 references,1962 vendor files, SDK files, source archives, and tested binaries.
The final execution, pin, and comparison statuses are all0.
Source archives identify exact runtime `68a5a5394fda54bb852f3824e0a0465285edfe15`
and vendor `e705c5fed490514458bdd2eaddc43bd098fcce9b`, independent of later docs.
The run uses serial CPU execution under a hard12GiB virtual-memory ceiling.
Highest observed RSS across timed tests is3471844KiB; this is not a speed result.

No reduced-model inference, pilot,474-call quality matrix, performance test,
weight upload, or redistribution occurs in this replay. Fresh same-profile F32
anchors are still required for a separately authorized quality run.
The owner retains final acceptance and subsequent execution decisions.
