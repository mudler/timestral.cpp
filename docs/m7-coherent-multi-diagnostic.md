# Coherent candidate: 5M multi layer-3 accumulated error

## Scope and preservation

This observational diagnostic uses the retained unaccepted coherent candidate archive/library without restoring it into production.
The earlier restoration explicitly rejected an unaccepted candidate; it did not discard accepted work.
The failed source archive, patch, test source, library, and executable remain intact in scratch.
No further reset, revert, runtime edit, reference rewrite, or candidate arithmetic sweep occurs.

The case is `multi`, positive block0, R6, S34, D192, three heads, head dimension64, context split32.
Rows are packed B2*V3. Layer3 is a composite time/variate layer.
Starting input is the exact stored `tf.in`; each free input thereafter is the candidate output of the previous layer.
Every local comparison separately uses that layer's stored F32 input.
Masks/positions/layout come from the same immutable fixture fields as the original gate.

## First divergence versus first failure

The first nonzero independent teacher boundary is layer0 time SDPA, not layer3.
All independently teacher-forced non-SDPA operations through layer3 are byte-exact on their stored inputs.
The coherent candidate has differences at time SDPA in layers0–3 and variate SDPA in layer3.
Those standalone teacher boundaries remain within their unchanged bounds.
An exact fully overridden layer output does not establish exact native composition.

| Layer | Local floor_tf ratio | Free floor_fr ratio | Free maximum signed error |
|---|---:|---:|---:|
| 0 | 0.263790 | 0.265268 | `-4.57763671875e-5` |
| 1 | 0.500000 | 0.292297 | `-6.103515625e-5` |
| 2 | 0.277223 | 0.413354 | `-9.1552734375e-5` |
| 3 | 0.375000 | 1.00705503406 FAIL | `-0.00030517578125` |

The signed errors in this table are each layer's own worst coordinate, not a claim that one coordinate monotonically propagates.
The first gate failure is layer3 free-running output: normalized error `7.1708615786e-6`, tolerance `7.12062532438e-6`.
Worst coordinate is row3 (batch1,variate0), token32 (first decoder token), feature5.
Free output is `373.220825195`, local output `373.220947266`, reference `373.221130371`.
At this coordinate, local error is `-0.00018310546875`; inherited/composed drift increases it by `-0.0001220703125`.
Thus local layer correctness within bounds does not prevent the accumulated free-running failure.

## Controlled layer-3 boundary interventions

Each intervention retains the candidate free input and replaces only the named downstream boundary with its stored reference value.
It is causal evidence, not a proposed production substitution or new gate.

| Reference boundary supplied | Free layer3 ratio |
|---|---:|
| None | 1.007055 FAIL |
| Time SDPA (`attn.0.sdpa`) | 0.402822 PASS |
| Time output projection (`attn.0.out`) | 0.402822 PASS |
| Variate SDPA (`attn.1.sdpa`) | 1.107761 FAIL |
| Variate output projection (`attn.1.out`) | 1.208466 FAIL |
| MLP norm or MLP output | 0.714380 PASS |

Replacing the time-attention boundary removes enough error to pass, but this does not uniquely attribute the error to layer3's own SDPA versus inherited input drift.
Replacing the later variate boundary worsens the result: error contributions can cancel as well as amplify.
These are non-monotone composition effects, not proof of a defective gate/MLP/variate implementation.

## Accepted mixed-policy control

The same harness linked to the accepted mixed-QK64 runtime uses identical inputs, masks, and references.
Layers0–2 have the same maximum normalized free errors, but differing coordinates/bit patterns.
Its local layer3 ratio is also 0.375000.
Its free layer3 ratio is 0.805644027252 PASS, error `5.73668926288e-6`, maximum absolute `0.000244140625`.
The worst coordinate remains row3/token32/feature5, value `373.22088623`.
This establishes policy-dependent accumulation, not universal superiority of either policy.
No model/layer/shape-specific policy selection follows.

## Outcome

The coherent policy remains unadopted and aggregate acceptance HOLD.
A next proposal requires further causal reasoning about accumulated versus locally introduced drift; no blanket precision adjustment is justified here.
All65 model reference artifacts, 5M model, candidate library/encoder, and accepted library hashes match before/after.
Witness sources/executables are separately pinned. All execution is serial under12GiB; only small observational programs compile.
No full replay, new reference generation, numerical fix, tolerance/floor change, or push occurs.
