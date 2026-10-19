#!/usr/bin/env python3
"""Acceptance check of a baseline directory (docs/PLAN.md, M0c).

    check_baseline.py --dir DIR [--cases tests/cases.json] [--case NAME ...]

For every case with a baseline in DIR:
  - the record names of C.3 are all present (derived from the case and the model config);
  - the statistics stat.<name>.rms / .numel exist for every float tensor, and kappa, rho,
    max_logit exist where the cap table needs them (f32 file);
  - every wrapped function was called when the case should call it, and not otherwise;
  - the case ran twice with bitwise equal records (repro.bitwise_equal);
  - the f32-math and f64 files hold the same record names as the f32 file;
  - the rerun files hold only names of the f32 file, cover the components, and the f64-rerun
    file passed the rerun self-check;
  - const and all_missing have outputs or an expect.error record.
It also validates cases.json against the schema of M0d.

Exit code 0 when everything holds, 1 otherwise.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import baseline_io as bio  # noqa: E402

LAYER_KIND_AXES = {"time": (True,), "variate": (False,), "time_variate": (True, False), "variate_time": (False, True)}
errors = []


def bad(msg):
    errors.append(msg)
    print("FAIL:", msg)


# --- cases.json schema (M0d) ---
INT, FLT, STR, BOOL = int, (int, float), str, bool


def validate_cases(spec):
    if spec.get("version") != 1 or not isinstance(spec.get("cases"), list):
        bad("cases.json: need {'version': 1, 'cases': [...]}")
        return
    req = {"name": STR, "B": INT, "V": INT, "T": INT, "H": INT, "seed": INT}
    opt_obj = {
        "series": {"period": INT, "level": FLT, "trend": FLT, "noise_std": FLT, "positive": BOOL, "constant": FLT},
        "missing": {"leading": INT, "scattered_frac": FLT, "missing_patches": list, "all_missing_series": list},
        "past": {"numeric": INT, "nan_frac": FLT},
        "known": {"numeric": INT, "indicator": INT, "nan_frac": FLT},
        "cat_past": {"rows": INT, "n_codes": INT, "nan_frac": FLT, "unseen_in_horizon": BOOL},
        "cat_known": {"rows": INT, "n_codes": INT, "nan_frac": FLT, "unseen_in_horizon": BOOL},
    }
    names = set()
    for c in spec["cases"]:
        n = c.get("name", "?")
        if n in names:
            bad(f"cases.json: duplicate case {n}")
        names.add(n)
        for k, t in req.items():
            if not isinstance(c.get(k), t) or isinstance(c.get(k), bool):
                bad(f"cases.json[{n}]: field {k} missing or wrong type")
        for k, fields in opt_obj.items():
            if k in c:
                if not isinstance(c[k], dict):
                    bad(f"cases.json[{n}]: {k} must be an object")
                    continue
                for fk, fv in c[k].items():
                    if fk not in fields:
                        bad(f"cases.json[{n}]: unknown field {k}.{fk}")
                    elif fv is not None and not isinstance(fv, fields[fk]):
                        bad(f"cases.json[{n}]: {k}.{fk} has the wrong type")
        if "labels" in c and not (isinstance(c["labels"], list) and all(isinstance(x, str) for x in c["labels"])):
            bad(f"cases.json[{n}]: labels must be a list of strings")
        lay = c.get("layers", "all")
        if lay != "all" and not (isinstance(lay, list) and all(isinstance(x, int) for x in lay)):
            bad(f"cases.json[{n}]: layers must be 'all' or a list of ints")
        if "tiny" in c and not isinstance(c["tiny"], bool):
            bad(f"cases.json[{n}]: tiny must be a bool")
        extra = set(c) - set(req) - set(opt_obj) - {"labels", "layers", "tiny"}
        if extra:
            bad(f"cases.json[{n}]: unknown fields {sorted(extra)}")


# --- expected names ---
def expected(kv, tensors, case):
    cfg = json.loads(kv["model.config_json"])
    sched = cfg["block_schedule"]
    n_layers = len(sched)
    dumped = set(range(n_layers)) if kv["layers.dumped"] == "all" else {int(x) for x in kv["layers.dumped"].split(",")}
    gate = {True: cfg["time_attn_gate"] != "none", False: cfg["variate_attn_gate"] != "none"}
    has_cov = any(case.get(k, {}).get(f, 0) for k, f in (("past", "numeric"), ("known", "numeric"), ("known", "indicator"),
                                                          ("cat_past", "rows"), ("cat_known", "rows")))
    has_known = bool(case.get("known", {}).get("numeric", 0) or case.get("known", {}).get("indicator", 0) or
                     case.get("cat_known", {}).get("rows", 0))
    cov_path = has_cov and cfg["covariate_enabled"]
    R_rows = None
    names = set()
    calls = {}

    def add(*n):
        names.update(n)

    blocks = {}
    for p in ("pos", "neg"):
        k = 0
        while f"{p}.b{k}.args.inputs" in tensors:
            blocks.setdefault(p, []).append(k)
            k += 1
    any_multi = False
    n_blocks_total = 0
    for p, ks in blocks.items():
        for k in ks:
            n_blocks_total += 1
            pre = f"{p}.b{k}."
            R = tensors[pre + "args.inputs"].shape[0]
            h = int(tensors[pre + "args.horizon_patches"][0])
            multi = h > 1
            any_multi |= multi
            add(*[pre + n for n in (
                "args.inputs", "args.observed", "args.group_ids", "args.horizon_patches", "predict", "scale.in",
                "scale.observed", "scaled", "loc", "scale", "patch.in", "patch.hidden_out", "patch.out", "tf.in",
                "tf.positions", "out_norm", "head.in", "head.raw", "head.monotone", "denorm.in", "anchor.loc",
                "anchor.scale", "positions", "mask.key_valid", "mask.obs_per_patch", "mask.time", "first_obs.group",
                "group.inverse", "refined", "refine.context", "refine.mask")])
            if cfg["split_context_horizon_attn"]:
                add(pre + "tf.enc_len")
            if R > 1:
                add(pre + "mask.variate")
            if multi:
                add(*[pre + n for n in ("frames.in.sinh_median", "frames.loc", "frames.scale", "refine.loc",
                                         "refine.scale", "refine.clamp.in", "refine.clamp.out")])
            if (pre + "args.role_ids") in tensors:
                add(pre + "args.role_ids", pre + "refine.target_rows", pre + "mask.block_anchor",
                    pre + "mask.block_active")
                if (pre + "known.rows") in tensors:
                    add(pre + "known.rows", pre + "known.scaled")
                    if bool(tensors[pre + "known.rows"].any()):
                        add(pre + "known.loc", pre + "known.scale", pre + "known.is_binary")
                if has_known:
                    add(pre + "args.known_future", pre + "args.known_future_mask")
            for l in sorted(dumped):
                add(pre + f"layer.{l}.in", pre + f"layer.{l}.out", pre + f"layer.{l}.mlp_norm", pre + f"layer.{l}.mlp")
                for j, is_time in enumerate(LAYER_KIND_AXES[sched[l]]):
                    b = pre + f"layer.{l}.attn.{j}."
                    add(*[b + n for n in ("norm", "out", "qkv", "q", "k", "v", "gated", "sdpa")])
                    if gate[is_time]:
                        add(b + "gate")
                    if is_time:
                        add(*[b + n for n in ("q_rope", "k_rope", "cos", "sin")])
    for p in blocks:
        add(f"{p}.assemble.rows", f"{p}.assemble.mask", f"{p}.assemble.n_tgt", f"{p}.cap.in", f"{p}.cap.rows",
            f"{p}.cap.mask", f"{p}.cap.out", f"{p}.cap.scale", f"final.raw.{p}")
    add("input.context")
    return names, dict(has_cov=has_cov, has_known=has_known, cov_path=cov_path, any_multi=any_multi,
                       n_blocks=n_blocks_total, cat=bool(case.get("cat_past", {}).get("rows", 0) or
                                                         case.get("cat_known", {}).get("rows", 0)),
                       single_target=(case["V"] == 1), multi_blocks=n_blocks_total > 2)


def check_case(directory, case):
    name = case["name"]
    paths = {m: bio.case_file(directory, name, m) for m in ("f32", "f32-math", "f64", "f64-rerun", "f32-math-rerun")}
    if not paths["f32"].exists():
        return False
    print(f"== {name}")
    t, kv = bio.read_baseline(paths["f32"])
    case_run = json.loads(kv["case.json"])
    err = kv.get("expect.error.type")
    if err:
        print(f"   reference raised {err}: {kv.get('expect.error.message','')[:100]}")
    exp, info = expected(kv, t, case_run)
    if not err:
        exp.add("final.quantiles")
    missing = sorted(n for n in exp if n not in t)
    if missing:
        bad(f"{name}: {len(missing)} expected records missing, e.g. {missing[:4]}")
    if not kv.get("repro.bitwise_equal"):
        bad(f"{name}: no bitwise reproducibility record")
    # statistics
    for n, a in t.items():
        if bio.is_float(a):
            if f"stat.{n}.rms" not in kv or f"stat.{n}.numel" not in kv:
                bad(f"{name}: no rms/numel statistic for {n}")
                break
    cap_stats = [k for k in kv if k.endswith(".kappa")]
    if not cap_stats and not err:
        bad(f"{name}: no kappa statistics")
    if not any(k.endswith(".max_logit") for k in kv) and not err:
        bad(f"{name}: no max_logit statistics")
    if not any(k.endswith(".rho") for k in kv) and not err:
        bad(f"{name}: no rho statistics")
    # wrapped functions
    C = lambda f: kv.get(f"calls.{f}", 0)
    n_blocks = info["n_blocks"]
    must = {"forecaster._forecast_raw": lambda: C("forecaster._forecast_raw") == 2,
            "forecaster._assemble_rows": lambda: C("forecaster._assemble_rows") == 2,
            "forecaster._cap_quantiles": lambda: C("forecaster._cap_quantiles") == 2,
            "model.predict_block": lambda: C("model.predict_block") == n_blocks,
            "model.scale": lambda: C("model.scale") >= n_blocks,
            "model.denormalize_quantiles": lambda: C("model.denormalize_quantiles") == n_blocks,
            "model.compute_enc_key_validity": lambda: C("model.compute_enc_key_validity") == n_blocks,
            "model.group_first_observed_patch": lambda: C("model.group_first_observed_patch") == n_blocks,
            "forecaster.causal_patch_loc_scale": lambda: C("forecaster.causal_patch_loc_scale") == 2,
            "layers._sdpa": lambda: C("layers._sdpa") > 0, "layers._apply_rotary": lambda: C("layers._apply_rotary") > 0,
            "layers.Attention._qkv": lambda: C("layers.Attention._qkv") > 0}
    cov_calls = {"model.build_two_stream_time_mask": info["cov_path"], "model.build_two_stream_positions": info["cov_path"],
                 "masks.compute_block_layout": info["cov_path"],
                 "model.build_forecast_time_mask": not info["cov_path"], "model.build_forecast_positions": not info["cov_path"],
                 "forecaster.advancing_block_frames": info["any_multi"], "forecaster.clamp_nonfinite": info["any_multi"],
                 "model._known_cov_scaling": info["cov_path"],
                 "model.instance_norm_stats": info["has_known"] and info["cov_path"],
                 "forecaster._encode_categorical": info["cat"] and True,
                 "forecaster.target_encode": info["cat"] and info["single_target"]}
    if not err:
        for f, fn in must.items():
            if not fn():
                bad(f"{name}: wrapped function {f} called {C(f)} times (unexpected)")
        for f, should in cov_calls.items():
            c = C(f)
            if should and c == 0:
                bad(f"{name}: wrapped function {f} was never called")
            if not should and c != 0:
                bad(f"{name}: wrapped function {f} called {c} times but the case should not call it")
    # same names across f32 / f32-math / f64
    for m in ("f32-math", "f64"):
        if paths[m].exists():
            t2, kv2 = bio.read_baseline(paths[m])
            if set(t2) != set(t):
                d = sorted(set(t2) ^ set(t))
                bad(f"{name}: {m} record names differ from f32: {d[:4]}")
            if not kv2.get("repro.bitwise_equal"):
                bad(f"{name}: {m}: no reproducibility record")
            want = np_dtype = "float64"
            if m == "f64" and not kv2.get("truth.patches"):
                bad(f"{name}: f64 file has no truth.patches")
    # reruns
    for m in ("f64-rerun", "f32-math-rerun"):
        if not paths[m].exists():
            continue
        t3, kv3 = bio.read_baseline(paths[m])
        extra = sorted(set(t3) - set(t))
        if extra:
            bad(f"{name}: {m} has records not in the f32 file: {extra[:4]}")
        if not t3:
            bad(f"{name}: {m} is empty")
        if not kv3.get("repro.bitwise_equal"):
            bad(f"{name}: {m}: no reproducibility record")
        # coverage: each rerunnable group of the f32 file appears
        need_suffix = ["scaled", "patch.out", "out_norm", "head.raw", "head.monotone", "predict", "refined"]
        for p in ("pos", "neg"):
            for k in range(1000):
                if f"{p}.b{k}.args.inputs" not in t:
                    break
                for s in need_suffix:
                    if f"{p}.b{k}.{s}" not in t3:
                        bad(f"{name}: {m} lacks {p}.b{k}.{s}")
                        break
        if m == "f64-rerun":
            if not kv3.get("selfcheck.ok"):
                bad(f"{name}: f64-rerun self-check missing or failed: {kv3.get('selfcheck.max_ratio')}")
            else:
                print(f"   f64-rerun self-check: max ratio {kv3['selfcheck.max_ratio']:.2e} (limit 1e-12) at {kv3.get('selfcheck.where')}")
    if name in ("const", "all_missing") and not (("final.quantiles" in t) or err):
        bad(f"{name}: neither outputs nor expect.error")
    print(f"   {len(t)} records, {sum(1 for k in kv if k.startswith('calls.'))} call counters, "
          f"{'ok' if not [e for e in errors if e.startswith(name)] else 'problems'}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cases", default=str(Path(__file__).resolve().parents[2] / "tests" / "cases.json"))
    ap.add_argument("--case", action="append", default=[])
    a = ap.parse_args()
    spec = bio.load_cases(a.cases)
    validate_cases(spec)
    n = 0
    for c in spec["cases"]:
        if a.case and c["name"] not in a.case:
            continue
        n += int(check_case(a.dir, c))
    if n == 0:
        bad(f"no baseline found in {a.dir}")
    print(f"checked {n} cases: {'PASS' if not errors else 'FAIL (%d problems)' % len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
