# Coherent 209M six-format quality result: FAIL

The full recorded evaluation uses the unchanged ON runtime `68a5a53`, source
archive `ca9019d9d63ad7ab2b8f440df7e3d668905bf9ee368113975d3aebfc6acf592e`,
and the coherent pinned-revision cohort described in [the F32 replay](coherent-f32-replay.md).
No tolerance, anchor policy, format ordering, model, or reference changes occur.

## Coverage and outcome

The producer completes **79 items ×6 formats =474 driver calls**, with474
request/result files and79 fresh same-profile F32 anchor captures. Every format
has79 forecasts and zero coverage gaps. Anchor records identify F32 SHA-256
`bf7605545245a2dd4d32f7ac2f7bbe5c2a12d8f47370814a254862150255b8e8`.
An independent exact item/format inventory passes after scoring.

The overall quality result is **FAIL**. Q8_0 aggregate mean normalized error
exceeds its unchanged0.005 bound. F16 passes both bounds; strict ordering passes.

| Format | Aggregate mean normalized deviation | Signed aggregate WQL change, percent |
| --- | ---: | ---: |
| F32 | 0 | 0 |
| F16 | 0.00015529176950670028 | -0.00274317535567626% |
| Q8_0 | 0.005657737466196431 | -0.04116458747472307% |
| Q6_K | 0.013512040093425381 | -0.06180385023158404% |
| Q5_K | 0.02008887882387394 | +1.3580854256823647% |
| Q4_K | 0.033048053192290705 | +2.770376555053301% |

F16 mean0.000155292 is below0.001; absolute WQL change0.00274318% is below0.5%.
Q8_0 mean0.00565774 is above0.005; absolute WQL change0.0411646% is below1%.
The WQL values are already percentages, not fractional values to multiply again.
Q8 therefore fails the mean-deviation bound, **not** the WQL bound.
K formats have no separate loss bound in this contract; their measured means
satisfy the strict F32<F16<Q8_0<Q6_K<Q5_K<Q4_K ordering.
This does not turn the failed six-format evaluation into a pass.

Full aggregate MASE remains undefined because `all_missing` has no observed
adjacent pairs. The report retains that limitation rather than filtering it out.
No MASE acceptance bound is introduced.

## Execution and immutable closure

The producer returns0, the scorer returns1, and the independent inventory returns0.
The final outer status is `execution_exit=1 pin_exit=0`.
This is a genuine numerical quality failure, not missing data or an audit-schema error.
No retry, tuning, replacement anchors, reference rewrite, or gate relaxation follows.

Before/after checks preserve2361 source/checkpoint/SDK files, eight tools,
six coherent models, and13744 reference/environment/dependency files.
The envelope and admitted closing record also recheck the approved
source/helper/admission/status/mapping evidence and all15 representative mapped DSOs.
Those DSO identities come from the reviewed12-call pilot;20ms sampling is not
an exhaustive proof of every transient load in all474 processes.

The run is serial N1 under a hard virtual-memory limit with bytecode writes disabled.
The producer time wrapper records52:04.61 wall time and maximum RSS948292KiB.
These are execution resource observations, not formal performance acceptance.

Raw requests, results, per-call audits,79 NPZ records, manifest, scorer report,
independent inventory, failure status, and closing logs remain retained.
The scorer identifies itself as recorded-output-only: execution evidence and
actual tool/model identity checks remain separate from its asserted provenance.

## Limits and publication

This result applies to the209M coherent cohort and this exact ON profile.
It does not establish quality for5M,28M,123M, another backend, or another SDK.
F32 trained parity remains separately documented; it does not imply Q8 quality.
Any model card or publication must disclose this failed Q8 mean-error gate.
Do not advertise full quality acceptance or infer redistribution permission.
Further diagnosis or implementation requires a separate decision.
