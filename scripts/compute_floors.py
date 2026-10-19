#!/usr/bin/env python3
"""Add the floors and caps of docs/PLAN.md (C.4, D.1, D.3) to the f32 baselines.

    compute_floors.py --dir DIR [--case NAME ...]

For every case with baseline-<case>-f32.gguf in DIR, and for every float tensor t in it:

  floor_fr.t = max(max|f32 - f64|, max|f32 - f32math|) / rms(f64)
               the free-running floor (needs the f64 file; the f32-math file when present)
  floor_tf.t = max(max|f32 - f64rerun|, max|f32 - f32mathrerun|) / rms(f64rerun)
               the teacher-forced floor, only for tensors that a rerun produces
  cap.t      the a-priori bound of the cap table (D.3), from the stat.* values of the f32 file

and writes them as KV to the f32 file (an earlier floor_*/cap.* is replaced). The script exits
with an error when a float tensor has no floor_fr or matches no cap pattern. A tensor whose cap
formula the plan does not define (inputs, host-computed values) matches an explicit `n/a` or
`copy` entry below; those entries are additions to the table of D.3 and are listed in
docs/parity.md.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_io as bio  # noqa: E402

U = 2.0 ** -24
DELTA = 1e-6


def lam(n):
    return math.sqrt(2.0 * math.log(2.0 * n / DELTA))


def gamma(k, n):
    return lam(n) * math.sqrt(k) * U


class Caps:
    """Evaluates the cap table of D.3 for the tensors of one f32 baseline."""

    def __init__(self, t, kv):
        self.t, self.kv = t, kv
        self.D = kv["model.d_model"]
        self.F = kv["model.mlp_hidden_dim"]
        self.ps = kv["model.patch_size"]
        self.hd = kv["model.head_dim"]
        self.n_layers = kv["model.n_layers"]
        self.n_blocks = kv.get("case.n_blocks", 1)
        self.memo = {}

    def stat(self, full, key):
        return self.kv.get(f"stat.{full}.{key}")

    def N(self, full):
        return self.stat(full, "numel")

    def cap(self, full):
        """Cap of the tensor `full`, or None for n/a (printed as n/a, no warning)."""
        if full in self.memo:
            return self.memo[full]
        v = self._cap(full)
        self.memo[full] = v
        return v

    def _cap(self, full):
        m = re.match(r"^((?:pos|neg)\.(?:b(\d+)\.)?)?(.*)$", full)
        pre, blk, rel = m.group(1) or "", m.group(2), m.group(3)
        P = lambda r: pre + r
        D, F, ps, hd = self.D, self.F, self.ps, self.hd
        N = self.N(full)

        def need(*xs):
            return all(x is not None for x in xs)

        for pat, kind in EXTRA_NOCAP:
            if re.search(pat, rel):
                return 0.0 if kind == "copy" else None
        if re.fullmatch(r"patch\.hidden_out", rel):
            k = self.stat(full, "kappa")
            return gamma(2 * ps, N) * k + 2 * U if need(k, N) else None
        if re.fullmatch(r"patch\.out", rel):
            k, kr = self.stat(full, "kappa"), self.stat(full, "kappa_res")
            return (gamma(2 * ps, N) + gamma(4 * D, N)) * k + gamma(2 * ps, N) * kr + 4 * U if need(k, kr, N) else None
        if re.fullmatch(r"layer\.\d+\.attn\.\d+\.(q|k|v)", rel):
            k = self.stat(full, "kappa")
            return gamma(D, N) * k if need(k, N) else None
        if re.fullmatch(r"layer\.\d+\.attn\.\d+\.qkv", rel):
            ks = [self.stat(full[:-3] + s, "kappa") for s in ("q", "k", "v")]
            return gamma(D, N) * max(ks) if need(*ks, N) else None
        if re.fullmatch(r"layer\.\d+\.attn\.\d+\.(q|k)_rope", rel):
            k = self.stat(full, "kappa")
            return gamma(D, N) * k + 4 * U if need(k, N) else None
        if re.fullmatch(r"layer\.\d+\.attn\.\d+\.sdpa", rel):
            ml = self.stat(full, "max_logit")
            S = self.t[full].shape[1]
            return gamma(hd, N) * (1 + 2 * ml) + gamma(S, N) + 8 * U if need(ml, N) else None
        if re.fullmatch(r"layer\.\d+\.attn\.\d+\.gate", rel):
            k = self.stat(full, "kappa")
            return gamma(D, N) * k if need(k, N) else None
        if re.fullmatch(r"layer\.\d+\.attn\.\d+\.out", rel):
            c = self.cap(full[:-3] + "sdpa")
            k = self.stat(full, "kappa")
            return c + gamma(D, N) * k + 4 * U if need(c, k, N) else None
        if re.fullmatch(r"layer\.\d+\.mlp", rel):
            k = self.stat(full, "kappa")
            return (gamma(D, N) + gamma(F, N)) * k + 4 * U if need(k, N) else None
        mo = re.fullmatch(r"layer\.(\d+)\.out", rel)
        if mo:
            l = int(mo.group(1))
            total = 0.0
            j = 0
            while P(f"layer.{l}.attn.{j}.out") in self.t:
                c, rho = self.cap(P(f"layer.{l}.attn.{j}.out")), self.stat(P(f"layer.{l}.attn.{j}.out"), "rho")
                if not need(c, rho):
                    return None
                total += rho * c
                j += 1
            c, rho = self.cap(P(f"layer.{l}.mlp")), self.stat(P(f"layer.{l}.mlp"), "rho")
            if not need(c, rho):
                return None
            return total + rho * c
        if rel == "out_norm":
            c = self.cap(P(f"layer.{self.n_layers - 1}.out"))
            return c + 4 * U if c is not None else None
        if rel == "head.raw":
            k, kr = self.stat(full, "kappa"), self.stat(full, "kappa_res")
            return (gamma(D, N) + gamma(4 * D, N)) * k + gamma(D, N) * kr + 4 * U if need(k, kr, N) else None
        if re.fullmatch(r"(predict|refined)", rel) or re.match(r"final\.", rel):
            single = (blk is not None and int(blk) == 0) or (blk is None and self.n_blocks <= 2)
            if not single:
                return None            # free-running through feedback: no cap
            a, b = self.cap(P("out_norm")), self.cap(P("head.raw"))
            return a + b if need(a, b) else None
        raise KeyError(f"no cap pattern matches tensor {full}")


# Additions to the cap table (D.3): tensors that are inputs, exact copies, or host-computed
# values for which the plan defines no a-priori bound. `copy` means the floor is expected to be 0.
EXTRA_NOCAP = [
    (r"^(input|args|assemble|scale\.in|refine\.context|cap\.rows)", "copy"),
    (r"^(tf\.in|layer\.\d+\.in|patch\.in|head\.in|denorm\.in|cap\.in|refine\.clamp\.in)$", "na"),
    (r"^(scaled|loc|scale|known\.|anchor\.|frames\.|refine\.|cap\.|target_encode|enc_cat|mask\.|first_obs|positions)", "na"),
    (r"^layer\.\d+\.attn\.\d+\.(norm|gated|cos|sin)$", "na"),
    (r"^layer\.\d+\.mlp_norm$", "na"),
    (r"^(head\.monotone|tf\.)", "na"),
]


def floors(t32, other_pairs, name):
    """max over the `other` files of max|f32 - other| / rms(first other), or None."""
    diffs, rr = [], None
    for o in other_pairs:
        if name not in o:
            continue
        if o[name].shape != t32[name].shape:
            raise SystemExit(f"{name}: shape {o[name].shape} vs {t32[name].shape}")
        diffs.append(bio.max_abs_diff(t32[name], o[name]))
        if rr is None:
            rr = bio.rms(o[name])
    if not diffs:
        return None
    d = max(diffs)
    if rr and rr > 0:
        return d / rr
    return 0.0 if d == 0 else float("inf")


def process(directory, case):
    f32p = bio.case_file(directory, case, "f32")
    if not f32p.exists():
        return None
    t, kv = bio.read_baseline(f32p)
    load = lambda m: bio.read_baseline(bio.case_file(directory, case, m))[0] if bio.case_file(directory, case, m).exists() else None
    t64, t32m, r64, r32m = load("f64"), load("f32-math"), load("f64-rerun"), load("f32-math-rerun")
    kv = {k: v for k, v in kv.items() if not (k.startswith("floor_fr.") or k.startswith("floor_tf.") or k.startswith("cap."))}
    caps = Caps(t, kv)
    out, rows, missing, warn = {}, [], [], []
    for name, a in t.items():
        if not bio.is_float(a):
            continue
        fr = floors(t, [x for x in (t64, t32m) if x is not None], name) if (t64 is not None or t32m is not None) else None
        tf = floors(t, [x for x in (r64, r32m) if x is not None], name)
        if fr is None:
            missing.append(name)
        else:
            out[f"floor_fr.{name}"] = fr
        if tf is not None:
            out[f"floor_tf.{name}"] = tf
        cap = caps.cap(name)
        if cap is not None:
            out[f"cap.{name}"] = cap
        for kind, fl in (("fr", fr), ("tf", tf)):
            if fl is not None and cap is not None and cap > 0 and 2 * fl > cap:
                warn.append((name, kind, fl, cap))
        rows.append((name, fr, tf, cap))
    if missing:
        raise SystemExit(f"{case}: {len(missing)} float tensors have no floor_fr (is the f64 file there?): {missing[:3]}")
    kv.update(out)
    bio.write_baseline(f32p, t, kv)
    return rows, warn


def kind_of(name):
    n = re.sub(r"^(pos|neg)\.b\d+\.", "", name)
    n = re.sub(r"layer\.\d+\.", "layer.L.", n)
    n = re.sub(r"attn\.\d+\.", "attn.J.", n)
    n = re.sub(r"(target_encode|enc_cat)\.\d+\.", r"\1.N.", n)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--case", action="append", default=[])
    a = ap.parse_args()
    cases = sorted({re.match(r"baseline-(.*)-f32\.gguf$", p.name).group(1) for p in Path(a.dir).glob("baseline-*-f32.gguf")})
    if a.case:
        cases = [c for c in cases if c in a.case]
    if not cases:
        raise SystemExit("no f32 baselines found")
    summary = {}
    for c in cases:
        res = process(a.dir, c)
        rows, warn = res
        n_tf = sum(1 for r in rows if r[2] is not None)
        n_cap = sum(1 for r in rows if r[3] is not None)
        print(f"{c}: {len(rows)} float tensors, floor_tf for {n_tf}, cap for {n_cap}, "
              f"{len(warn)} with 2*floor > cap")
        for name, kind, fl, cap in warn[:5]:
            print(f"   WARNING 2*floor_{kind}={2*fl:.3e} > cap={cap:.3e}: {name}")
        agg = summary.setdefault(c, {})
        for name, fr, tf, cap in rows:
            k = kind_of(name)
            e = agg.setdefault(k, {"floor_fr": 0.0, "floor_tf": None, "cap": None, "n": 0})
            e["n"] += 1
            e["floor_fr"] = max(e["floor_fr"], fr if fr is not None else 0.0)
            if tf is not None:
                e["floor_tf"] = tf if e["floor_tf"] is None else max(e["floor_tf"], tf)
            if cap is not None:
                e["cap"] = cap if e["cap"] is None else max(e["cap"], cap)
        agg["__warnings__"] = [{"tensor": n, "kind": k, "floor": f, "cap": cp} for n, k, f, cp in warn]
    (Path(a.dir) / "floors-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
