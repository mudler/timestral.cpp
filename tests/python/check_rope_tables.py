#!/usr/bin/env python3
"""Gate G0.3: the stored rope.cos / rope.sin tables against every table the reference used.

    check_rope_tables.py --gguf MODEL.gguf --dir BASELINES [--mode f32 --mode f32-math] [--case NAME ...]

The baselines hold the cos and sin tables that `_apply_rotary` received at run time
(layer.<l>.attn.<j>.cos / .sin, shape (R, N, head_dim)) and the positions of the pass
(tf.positions). For every dumped table row, the script looks up the stored row of the same
position (position - table_pos_min) and compares the values.

Counted separately, per file and in total:
  value differences     the two floats are not equal as numbers (+0 == -0 counts as equal)
  signed-zero only      equal as numbers but different bit patterns (+0 against -0)
  out of range          a position outside the stored table (the runtime would refuse it)
Also reported: the largest ulp distance of the value differences, and the first positions.

EXACT on floats is IEEE value equality (plan D.1); the gate passes when there are no value
differences and no positions out of range. Signed-zero differences are reported, not failed.
Exit code 0 on pass, 1 on fail.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import baseline_io as bio  # noqa: E402


def ukey(a):
    b = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(b < 0, -(b & 0x7FFFFFFF), b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", action="append", default=[])
    ap.add_argument("--case", action="append", default=[])
    a = ap.parse_args()
    modes = a.mode or ["f32", "f32-math"]
    mt, mkv = bio.read_baseline(a.gguf)
    cos_t, sin_t = mt["rope.cos"], mt["rope.sin"]
    pos_min = int(mkv["timestral.rope.table_pos_min"])
    P = cos_t.shape[0]
    print(f"stored table: {P} positions from {pos_min}, head_dim {cos_t.shape[1]}")
    tot = dict(files=0, values=0, vdiff=0, szero=0, oor=0, maxulp=0, rows=0)
    for mode in modes:
        for path in sorted(Path(a.dir).glob(f"baseline-*-{mode}.gguf")):
            case = re.match(rf"baseline-(.*)-{mode}\.gguf$", path.name).group(1)
            if a.case and case not in a.case:
                continue
            t, kv = bio.read_baseline(path)
            c = dict(values=0, vdiff=0, szero=0, oor=0, maxulp=0, rows=0, tables=0)
            first = []
            for name in sorted(t):
                m = re.match(r"^((?:pos|neg)\.b\d+\.)layer\.(\d+)\.attn\.(\d+)\.cos$", name)
                if not m:
                    continue
                pre = m.group(1)
                cos_run, sin_run = t[name], t[name[:-3] + "sin"]
                pos = t[pre + "tf.positions"].astype(np.int64)
                R, N = pos.shape[0], cos_run.shape[1]
                # the transformer sequence may be [Enc;Dec] (2N positions); the table covers the
                # positions given to the rope call
                if pos.shape[1] != N:
                    c["oor"] += 1
                    first.append(f"{name}: positions {pos.shape} do not match table {cos_run.shape}")
                    continue
                idx = pos - pos_min
                bad = (idx < 0) | (idx >= P)
                c["oor"] += int(bad.sum())
                idx = np.clip(idx, 0, P - 1)
                for run, tab, nm in ((cos_run, cos_t, "cos"), (sin_run, sin_t, "sin")):
                    ref = tab[idx]                     # (R, N, head_dim)
                    ok = ~bad[..., None]
                    eq = (run == ref) | (np.isnan(run) & np.isnan(ref))
                    bitsame = run.view(np.uint32) == ref.view(np.uint32)
                    vd = (~eq) & ok
                    sz = eq & (~bitsame) & ok
                    c["values"] += int(ok.sum()) * run.shape[-1]
                    c["vdiff"] += int(vd.sum())
                    c["szero"] += int(sz.sum())
                    if vd.any():
                        c["maxulp"] = max(c["maxulp"], int(np.abs(ukey(run) - ukey(ref))[vd].max()))
                        if len(first) < 3:
                            r, n, d = np.argwhere(vd)[0]
                            first.append(f"{name[:-3]}{nm} row {r} pos {pos[r, n]} dim {d}: run={run[r, n, d]!r} stored={ref[r, n, d]!r}")
                c["tables"] += 1
                c["rows"] += R * N
            if c["tables"] == 0:
                continue
            print(f"{case:15s} {mode:9s} tables={c['tables']:4d} rows={c['rows']:8d} values={c['values']:10d} "
                  f"value-diff={c['vdiff']} signed-zero-only={c['szero']} out-of-range={c['oor']} max-ulp={c['maxulp']}")
            for f in first[:3]:
                print("   ", f)
            tot["files"] += 1
            for k in ("values", "vdiff", "szero", "oor", "rows"):
                tot[k] += c[k]
            tot["maxulp"] = max(tot["maxulp"], c["maxulp"])
    ok = tot["files"] > 0 and tot["vdiff"] == 0 and tot["oor"] == 0
    print(f"G0.3 {'PASS' if ok else 'FAIL'}: {tot['files']} files, {tot['rows']} table rows, {tot['values']} values, "
          f"value differences {tot['vdiff']} (max {tot['maxulp']} ulp), signed-zero-only {tot['szero']}, "
          f"out of range {tot['oor']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
