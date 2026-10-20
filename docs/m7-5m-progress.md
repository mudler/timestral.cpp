# Bounded 5M prerequisite progress: multi and long_h

This records two new cases only, using accepted scalar binary64 QK policy
`7831255` and existing pinned native binaries from `timestral-m7-qk-policy`.
No SIMD experiment, runtime/source change, selected-reference bypass, tolerance
change, or prior artifact rewrite is part of this run.
Reference source remains a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff; 5M revision
bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f. Model SHA256:
`c320276d8db95fb72fe5f4e043147e29142a1bf197c1b45daf6736780f6736ed`.
The custom 13-case specification and inference inputs remain unchanged.

Five new modes are generated per case: f32, f32-math, f64, f64-rerun,
f32-math-rerun. Existing contract validation and floor computation run without
formula changes. Each strict M3/M6b test exits zero with explicit subset status;
these two cases do not establish the full 13-case prerequisite.

## Strict gate results

| Case | M3 selections | Local layers | M3 comparisons | Missing layers | Worst M3 error/tolerance | M6b teacher/free blocks | Finals | M6b comparisons | Worst M6b ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | 2 | 16 | 34 | 0 | 0.933687 | 2 / 2 | 1 | 9 | 0.788755 |
| long_h | 4 | 32 | 68 | 0 | 0.560730 | 6 / 6 | 1 | 25 | 0.846886 |

M3 selections follow the existing strict encoder selection contract; long_h's
four selections are not all six autoregressive sign/block combinations.
All eight layers exist in every stored sign/block for every mode: multi has
positive/negative block0; long_h has positive/negative blocks0,1,2.
The independent container audit confirms this 10-file layer coverage.
M6b checks all teacher and free-running blocks, both signs, and the final output.
No failed/skipped case or missing comparison is hidden by subset reporting.

Contract outputs report 408 records/20 counters for multi and 1178/20 for long_h.
Floor computation preserves warnings: **28 multi** and **199 long_h** tensors
have 2*floor above cap. These warnings are not omitted or reclassified as an
automatic gate failure when the actual established M3 comparisons pass.
No cap/floor is loosened and no warning is used to assert universal parity.
M3 repeated-amplification diagnostics also remain: multi1, long_h4.

## Measured resource cost

All generation/checks execute serially, CPU only, with OMP/MKL/OpenBLAS threads
one and the hard 12 GiB address-space guard. No competing Timestral build runs.

| Case/mode | Wall seconds | Peak RSS KiB | Artifact bytes |
|---|---:|---:|---:|
| multi f32 | 15.79 | 509556 | 54857888 |
| multi f32-math | 9.50 | 459572 | 54810752 |
| multi f64 | 12.79 | 640280 | 109427008 |
| multi f64-rerun | 14.59 | 890632 | 102121984 |
| multi f32-math-rerun | 9.69 | 507596 | 51076896 |
| long_h f32 | 21.39 | 704856 | 152301920 |
| long_h f32-math | 14.21 | 651260 | 152163168 |
| long_h f64 | 20.56 | 1020124 | 303700608 |
| long_h f64-rerun | 27.19 | 1691896 | 281842752 |
| long_h f32-math-rerun | 18.78 | 788872 | 140979168 |

Native M3/M6b wall/RSS respectively:
- multi: **0.97s /94980 KiB**, **2.05s /405464 KiB**.
- long_h: **2.07s /188112 KiB**, **4.98s /1051836 KiB**.

The five-mode generation totals are 62.36s and 102.13s, before contract/floor
and native checks. Suite disk grows from **16782720** to **1419931225 bytes**,
approximately 1.40 GB added. Peak observed generation RSS is approximately
1.61 GiB, comfortably within the hard guard.

For planning only, ten more cases at these two observed costs would require
roughly 10–17 minutes of generation and 3.7–10.3 GB of artifacts, plus validation.
This is a rough envelope, not a bound or a claim about unmeasured cases:
row counts, known-covariate expansion, and autoregressive windows differ.
No remaining case is authorized or executed automatically.

## Pins and reproduction

Before/after SHA256 checks preserve the actual 5M model, both native executables,
reference generator, floor/contract scripts, 13-case spec, accepted source archive,
and every existing uni_one_patch artifact. Existing pilot artifacts remain
byte-identical. Ten newly generated artifacts have individual size/SHA256 records.
The accepted native source archive hash remains
`17694ecf6c8d0f1bf415bb57a6e45476e0aec27722545692d1b8833f22c64ba1`.
No native build is needed for this bounded run.

Generation uses the exact offline Hub ID/revision and `--layers all`:

```sh
# Run each mode separately under the memory guard and thread-count environment.
python scripts/gen_reference_baseline.py --model mistralai/Timestral-5M \
  --revision bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f \
  --cases "$CASES_5M" --case "$CASE" --layers all --mode "$MODE" \
  --output-dir "$SUITE/real"
python tests/python/check_baseline.py --dir "$SUITE/real" --cases "$CASES_5M" --case "$CASE"
python scripts/compute_floors.py --dir "$SUITE/real" --case "$CASE"
TIMESTRAL_TEST_REAL_CASES="$CASE" "$BUILD/tests/test_encoder" --real
TIMESTRAL_TEST_REAL_CASES="$CASE" "$BUILD/tests/test_forecast" --real
```

Evidence under `timestral-m7-5m/progress-multi-long` relative to the work root:
`before.json`, `after.json`, `prior-pin-check.log`, `new-artifacts.json`,
`resource-summary.json`, `coverage-audit.log`, `completed.txt`, per-case/mode
resource logs, contract/floor logs (including all warnings), strict M3/M6b logs,
and disk measurements. No full-profile claim, quantized-quality gate, or licence
publication decision follows. Root selects the next bounded prerequisite stage.

## Remaining-case stage: STOP at known_cov failure

The separately authorized remaining-ten sequence runs accepted scalar policy
7831255, not the concurrent SIMD candidate, and stops on the first strict error.
It generates exactly 15 new artifacts: five modes each for uni_short, past_cov,
and known_cov. Each contract passes, floors use the unchanged formula, and M3
passes. No subsequent case, final65 audit, or aggregate13 gate runs.

| Case | M3 selections/layers/comparisons | Worst M3 ratio | M6b completed cases/comparisons | M6b result |
|---|---|---:|---|---|
| uni_short | 2 /16 /34 | 0.510238 | 1 /9 | PASS, worst0.481416 |
| past_cov | 2 /16 /34 | 0.741841 | 1 /9 | PASS, worst0.557982 |
| known_cov | 2 /16 /34 | 0.500000 | 0 /4 successful comparisons | FAIL at fifth attempted comparison |

Exact first failure: **G6.2 floor_tf predict neg.b0**, known_cov.
Normalized error **1.34354679e-5**, floor **6.52044475e-6**, unchanged tolerance
**1.30408895e-5**, ratio **1.03025702**. Four positive-pass comparisons complete;
one teacher block and one free block are counted, zero finals. The executable's
summary `max_ratio=0.304715` covers only successful completed comparisons and
must not hide the actual failing ratio. Deliberately corrupt negative controls
also print FAIL; they are not the reason for this case failure.
Known_cov M3 still passes all16 selected local layers with no missing references.
No inference is made yet about the causal operation behind the negative failure.

Known_cov floor generation reports **2 tensors with 2*floor above cap**;
all warnings and new artifacts remain intact. Existing fifteen pilot/multi/long_h
artifacts retain their original hashes. Atomic no-overwrite publication uses
same-filesystem hard links from per-case staging; no old file is regenerated.
Current inventory is **30 artifacts /6 cases**, not the required65/13.
Five cases now have passing strict M3/M6b evidence including earlier stages;
known_cov remains HOLD. Seven untouched cases are cat_uni,cat_multi,long_h_known,
nans,nonneg,const,all_missing. No case exclusion or gate reinterpretation occurs.

Generation wall seconds by modes f32/f32-math/f64/f64-rerun/f32-math-rerun:
- uni_short: **33.41 /12.59 /13.84 /16.31 /11.03**; peak509456 KiB.
- past_cov: **15.12 /16.27 /17.67 /24.10 /14.04**; peak795532 KiB.
- known_cov: **19.85 /18.55 /17.77 /23.27 /14.15**; peak796248 KiB.
All modes exit zero. Strict known_cov M6b exits one as required by the stop rule.
Hard12GiB and serial OMP/MKL/OpenBLAS1 apply; no native build runs.

After stopping, all **8 model/binary/source/spec/script pins** match the execution
before-manifest, and all **15 protected existing artifact pins** match separately.
The new15 sizes/hashes/resources are recorded without modifying their contents.
Scratch under `timestral-m7-5m/remaining-prepared`: `execution-before.json`,
`execution-after-stopped.json`, `existing15-pins.json`,
`existing15-after-stopped.json`, `stopped-resources-artifacts.json`,
per-case five-mode/contract/floor/M3/M6b logs, staging directories, and
`completed.txt` containing only uni_short/past_cov PASS.

Prepared aggregate expectations (30 M3 selections/510 comparisons and141 M6b
comparisons) remain unexecuted expectations, not results. No automatic diagnosis,
policy change, tolerance/floor/reference rewrite, or retry follows this report.
The 5M prerequisite remains **HOLD** pending root's next causal-analysis decision.

## Corrected-reduction resume: all_missing final forecast HOLD

Root authorizes the remaining seven cases after independent acceptance of
`6f2bab197ff19969c41bf80169e660ba4de09350` (generic known-row reduction).
The actual build is `timestral-m7-known-sum/build`, not the earlier QK-only
build. Every runtime/include source file matches that commit; accepted final
source/library/forecast/test pins pass before and after. The model, encoder,
case spec, reference scripts, and all **30 existing artifacts** also remain
byte-identical under separate before/after manifests. No build is required.

All35 missing artifacts are generated with unchanged five modes, source/model
pins, all-layers scratch spec, contracts, and floor formula. Generation runs
serially at OMP/MKL/OpenBLAS1 under the hard12GiB address-space guard.
The first six cases pass strict M3/M6b. The last case, all_missing, passes M3
and eight block comparisons but fails its final forecast. This is an actual
numerical gate failure, not a tooling-only failure. Stop-on-first-failure means
no aggregate13 encoder or forecast inference run follows.

| Case | M3 selections/local layers/comparisons | Worst M3 ratio | M6b completed teacher/free/finals/comparisons | Worst actual M6b result |
|---|---|---:|---|---|
| cat_uni | 2/16/34 | 0.900000 | 2/2/1/9 | PASS 0.414504 |
| cat_multi | 2/16/34 | 0.654731 | 2/2/1/9 | PASS 0.541468 |
| long_h_known | 4/32/68 | 0.614359 | 4/4/1/17 | PASS 0.389194 |
| nans | 2/16/34 | 0.764751 | 2/2/1/9 | PASS 0.471470 |
| nonneg | 2/16/34 | 0.520913 | 2/2/1/9 | PASS 0.524350 |
| const | 2/16/34 | 0.541153 | 2/2/1/9 | PASS 0 |
| all_missing | 2/16/34 | 0.501643 | 2/2/0/8 | FAIL final 1.9828851 |

Exact failure: **G6.3 final floor_fr original-positive-target-anchor**.
Normalized error **2.08006087e-5**, floor **5.24503633e-6**, tolerance
**1.04900727e-5**, ratio **1.9828851**. The final comparison is the ninth
attempt, not a completed comparison. Summary max_ratio0.537648 describes the
preceding eight successful block comparisons, not the failing final ratio.
Intentional negative-control FAIL lines are distinct and correctly rejected.
No causal operation is established by this final-output failure alone.

Floor warnings remain intact: long_h_known45, nans5, nonneg179 tensors with
2*floor above cap; cat_uni/cat_multi/const/all_missing have zero such warnings.
No cap, floor, tolerance, reference tensor, or case selection changes.

An artifact-only external completeness audit runs after stopping: **65 files,
13 exact case IDs, all five modes, all eight layers for all32 sign/block groups
per mode**. The 13-case spec deep-matches canonical inputs except approved
layers=all and exclusion of the two slow cases. This proves stored coverage,
not passing aggregate inference gates. The expected M3 30/240/510 and M6b
32/32/13/141 remain **unexecuted aggregate expectations**.

Five-mode generation wall totals/peak RSS KiB:
cat_uni90.60/713688; cat_multi39.98/713124;
long_h_known118.03/2198028; nans75.75/1279336;
nonneg34.19/474160; const37.55/460676; all_missing30.58/496796.
Total generation wall **426.68 seconds**; maximum RSS approximately2.10GiB.
The35 new artifacts total **2706293632 bytes**; suite disk is **4787261595
bytes**, including existing artifacts and auxiliary files. Staging hard links
do not duplicate artifact data blocks. No reference files are committed.

Evidence under `timestral-m7-5m/remaining-seven`:
`runtime-source-identity.json`, `final-pins-before.log`,
`final-pins-after-stopped.log`, `execution-before.json`,
`execution-after-stopped.json`, `existing30-before.json`,
`existing30-after-stopped.json`, `resource-artifact-summary.json`,
`artifact-coverage-stopped.log`, all per-case/mode resource logs,
contract/floor/M3/M6b logs, and `completed.txt` listing only the six successes.
The all65 hash inventory resides in `remaining-prepared/all65-artifacts.json`.
A foreground artifact-audit command was interrupted before completion; its
bounded background rerun completes and verifies pins, without model execution.
All generation/gate jobs are accounted: the stage exits1 on the strict failure;
the follow-up artifact/pin audit exits0. No retry or numerical correction is
performed. Root owns the next diagnosis. **5M prerequisite remains HOLD.**
