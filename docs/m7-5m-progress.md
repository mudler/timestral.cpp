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
