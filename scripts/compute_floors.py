#!/usr/bin/env python3
"""Add the floors and caps of docs/PLAN.md (C.4, D.1, D.3) to the f32 baselines.

    compute_floors.py --dir DIR [--case NAME ...]

For every case with baseline-<case>-f32.gguf in DIR, and for every float tensor t in it:

  floor_fr.t = max(max|f32 - f64|, max|f32 - f32math|) / rms(f64)
               the free-running floor (both companion modes are required)
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
from rerun_contracts import floor_inputs, expected_reruns
from baseline_contract import validate_artifacts, MODES, allows_input_nan
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
        value = self.kv.get(f"stat.{full}.{key}")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'missing/invalid stat.{full}.{key}')
        return value

    def N(self, full):
        value = self.stat(full, "numel")
        if value <= 0 or value != self.t[full].size:
            raise ValueError(f'{full}: invalid numel')
        return value

    def cap(self, full):
        """Cap of the tensor `full`, or None for n/a (printed as n/a, no warning)."""
        if full in self.memo:
            return self.memo[full]
        v = self._cap(full)
        if v is not None and (not math.isfinite(v) or v < 0):
            raise ValueError(f'{full}: invalid cap {v}')
        self.memo[full] = v
        return v

    def applicability(self, full):
        value = self.cap(full)
        if value is None:
            return 'na'
        rel = re.sub(r'^(pos|neg)\.(b\d+\.)?', '', full)
        return 'copy' if any(kind == 'copy' and re.search(pat, rel)
                             for pat, kind in EXTRA_NOCAP) else 'bounded'

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
            key = self.t[full[:-4] + 'k']
            if key.ndim != 4 or key.shape[1] <= 0:
                raise ValueError(f'{full}: expected [batch, keys, heads, head_dim] key tensor')
            S = key.shape[1]
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
        if rel.startswith('final.') and rel not in ('final.raw.pos', 'final.raw.neg', 'final.quantiles'):
            raise KeyError(f'no final cap pattern matches {full}')
        if re.fullmatch(r"(predict|refined)", rel) or re.match(r"final\.", rel):
            single = (blk is not None and int(blk) == 0) or (blk is None and self.n_blocks == 1)
            if not single:
                return None            # free-running through feedback: no cap
            if rel.startswith('final.'):
                passes = ('pos', 'neg') if rel == 'final.quantiles' else (rel.rsplit('.', 1)[-1],)
                if any(p not in ('pos', 'neg') for p in passes):
                    raise KeyError(f'no final cap pattern matches {full}')
                return max(self.cap(f'{p}.b0.out_norm') + self.cap(f'{p}.b0.head.raw') for p in passes)
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
    """C.4 numerator from both modes; denominator is the first (f64) mode."""
    if len(other_pairs) != 2 or any(name not in o for o in other_pairs):
        raise ValueError(f'{name}: both required floor sources must exist')
    arrays = [t32[name]] + [o[name] for o in other_pairs]
    if any(np.isinf(a).any() or (not allows_input_nan(name) and np.isnan(a).any()) for a in arrays):
        raise ValueError(f'{name}: nonfinite floor evidence')
    diffs = []
    for o in other_pairs:
        if o[name].shape != t32[name].shape:
            raise ValueError(f'{name}: floor shape mismatch')
        diffs.append(bio.max_abs_diff(t32[name], o[name]))
    d, rr = max(diffs), bio.rms(other_pairs[0][name])
    if not math.isfinite(d) or not math.isfinite(rr):
        raise ValueError(f'{name}: nonfinite floor evidence')
    if rr == 0:
        if d != 0:
            raise ValueError(f'{name}: nonzero error with zero reference RMS')
        return 0.0
    value = d / rr
    if not math.isfinite(value) or value < 0 or not math.isfinite(2 * value):
        raise ValueError(f'{name}: invalid floor/tolerance')
    return value


def prepare(directory, case):
    data = {}
    for mode in MODES:
        path = bio.case_file(directory, case, mode)
        if not path.exists():
            raise ValueError(f'{case}: required mode {mode} missing')
        data[mode] = bio.read_baseline(path)
    t, original = data['f32']
    spec = json.loads(original['case.json'])
    if spec['name'] != case:
        raise ValueError(f'{case}: case name mismatch')
    validate_artifacts(data, spec)
    kv = {k: v for k, v in original.items() if not k.startswith(('floor_fr.', 'floor_tf.', 'cap.'))}
    caps = Caps(t, kv)
    out, rows, warn = {}, [], []
    tf_sources = [floor_inputs(data[m][0]) for m in ('f64-rerun', 'f32-math-rerun')]
    tf_required = set(expected_reruns(t).values())
    for name, a in t.items():
        if not bio.is_float(a):
            continue
        fr = floors(t, [data[m][0] for m in ('f64', 'f32-math')], name)
        tf = floors(t, tf_sources, name) if name in tf_required else None
        out[f'floor_fr.{name}'] = fr
        if tf is not None:
            out[f'floor_tf.{name}'] = tf
        cap = caps.cap(name)
        if cap is not None:
            out[f'cap.{name}'] = cap
        for kind, fl in (('fr', fr), ('tf', tf)):
            if fl is not None and cap is not None and cap > 0 and 2 * fl > cap:
                warn.append((name, kind, fl, cap))
        rows.append((name, fr, tf, cap))
    kv.update(out)
    return bio.case_file(directory, case, 'f32'), t, kv, rows, warn


def process(directory, case):
    path, tensors, kv, rows, warn = prepare(directory, case)
    bio.write_baseline(path, tensors, kv)
    return rows, warn


def publish(directory, cases):
    # Discard each case after validation, avoiding all-suite tensor retention.
    for case in cases:
        prepare(directory, case)
    results = {}
    for case in cases:
        results[case] = process(directory, case)
    return results


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
        unknown = set(a.case) - set(cases)
        if unknown:
            raise ValueError(f'missing selected cases: {sorted(unknown)}')
        cases = [c for c in cases if c in a.case]
    if not cases:
        raise SystemExit("no f32 baselines found")
    label = 'PARTIAL (not a milestone gate)' if a.case else 'DIRECTORY (not a suite-completeness check)'
    print(f'{label}: {len(cases)} selected cases')
    summary = {}
    results = publish(a.dir, cases)
    for c in cases:
        res = results[c]
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
            e = agg.setdefault(k, {"floor_fr": 0.0, "floor_tf": None, "cap": None, "cap_applicability": {}, "n": 0})
            e["n"] += 1
            rel = re.sub(r'^(pos|neg)\.(b\d+\.)?', '', name)
            state = 'na' if cap is None else ('copy' if any(
                kind == 'copy' and re.search(pat, rel) for pat, kind in EXTRA_NOCAP) else 'bounded')
            e['cap_applicability'][state] = e['cap_applicability'].get(state, 0) + 1
            e["floor_fr"] = max(e["floor_fr"], fr if fr is not None else 0.0)
            if tf is not None:
                e["floor_tf"] = tf if e["floor_tf"] is None else max(e["floor_tf"], tf)
            if cap is not None:
                e["cap"] = cap if e["cap"] is None else max(e["cap"], cap)
        agg["__warnings__"] = [{"tensor": n, "kind": k, "floor": f, "cap": cp} for n, k, f, cp in warn]
    bio.write_json(Path(a.dir) / "floors-summary.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
