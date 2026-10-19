#!/usr/bin/env python3
"""Acceptance check of a baseline directory (docs/PLAN.md, M0c).

    check_baseline.py --dir DIR [--cases tests/cases.json] [--case NAME ...]

For every required case in DIR (15 real or 14 with --tiny):
  - the record names of C.3 are all present (derived from the case and the model config);
  - the statistics stat.<name>.rms / .numel exist for every float tensor, and kappa, rho,
    max_logit exist where the cap table needs them (f32 file);
  - every wrapped function was called when the case should call it, and not otherwise;
  - the case ran twice with bitwise equal records (repro.bitwise_equal);
  - the f32-math and f64 files hold the same record names as the f32 file;
  - the rerun files hold only names of the f32 file, cover the components, and the f64-rerun
    file passed the rerun self-check;
  - exception artifacts fail closed pending proven reference provenance.
It also validates cases.json against the schema of M0d.

Exit code 0 when everything holds, 1 otherwise.
"""
import argparse
import json
import math

import numpy as np
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rerun_contracts import expected_reruns, source_name
import baseline_io as bio  # noqa: E402

# The import alias keeps the historical checker API for callers and tests.
import baseline_contract as contract
from baseline_contract import *  # noqa: F403

def check_case(directory, case, tiny=False):
    contract.errors = errors
    return contract.check_case(directory, case, tiny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cases", default=str(Path(__file__).resolve().parents[2] / "tests" / "cases.json"))
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--tiny", action="store_true", help="validate scaled tiny suite (14 cases, no bench_32k)")
    a = ap.parse_args()
    errors.clear()
    spec = bio.load_cases(a.cases)
    validate_cases(spec)
    if errors:
        return 1
    eligible = [c for c in spec['cases'] if not a.tiny or c.get('tiny')]
    unknown = set(a.case) - {c['name'] for c in eligible}
    if unknown:
        bad(f'unknown/ineligible selected cases: {sorted(unknown)}')
    canonical = bio.load_cases(Path(__file__).resolve().parents[1] / 'cases.json')
    if not a.case and spec != canonical:
        bad('full milestone check requires the canonical case specification; use --case for partial checks')
    n = 0
    selected = [c for c in eligible if not a.case or c['name'] in a.case]
    for c in selected:
        found = check_case(a.dir, c, tiny=True) if a.tiny else check_case(a.dir, c)
        if not found:
            bad(f"{c['name']}: required case not accepted")
        n += int(found)
    if n == 0:
        bad(f"no baseline found in {a.dir}")
    label = "PARTIAL (not a milestone gate)" if a.case else "FULL"
    print(f"{label}: checked {n}/{len(selected)} cases: {'PASS' if not errors else 'FAIL (%d problems)' % len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
