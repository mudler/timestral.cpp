#!/usr/bin/env python3
"""Shared baseline acceptance contract for the checker and floor computation."""
from contextvars import ContextVar
import json
import math

import numpy as np
import re

from rerun_contracts import expected_reruns, source_name
import baseline_io as bio  # noqa: E402

LAYER_KIND_AXES = {"time": (True,), "variate": (False,), "time_variate": (True, False), "variate_time": (False, True)}
errors = []
_diagnostics = ContextVar("baseline_diagnostics", default=None)


def bad(msg):
    sink = _diagnostics.get()
    if sink is not None:
        sink.append(msg)
    else:
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
    names = set()

    def add(*n):
        names.update(n)

    ps, bs = cfg['patch_size'], cfg['block_size']
    patches = (case['H'] + ps - 1) // ps
    blocks = {p: range((patches + bs - 1) // bs) for p in ('pos', 'neg')}
    any_multi = False
    n_blocks_total = 0
    for p, ks in blocks.items():
        for k in ks:
            n_blocks_total += 1
            pre = f"{p}.b{k}."
            R = case['B'] * (case['V'] + sum(case.get(c, {}).get(f, 0) for c, f in
                (('past', 'numeric'), ('known', 'numeric'), ('known', 'indicator'),
                 ('cat_past', 'rows'), ('cat_known', 'rows'))))
            h = min(bs, patches - k * bs)
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
            if cov_path:
                add(pre + "args.role_ids", pre + "refine.target_rows", pre + "mask.block_anchor",
                    pre + "mask.block_active")
                if cov_path:
                    add(pre + "known.rows", pre + "known.scaled")
                    if has_known:
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
    for key, present in [('past', case.get('past', {}).get('numeric', 0)),
                         ('known', case.get('known', {}).get('numeric', 0) or case.get('known', {}).get('indicator', 0)),
                         ('cat_past', case.get('cat_past', {}).get('rows', 0)),
                         ('cat_known', case.get('cat_known', {}).get('rows', 0))]:
        if present:
            add('input.' + key)
    return names, dict(has_cov=has_cov, has_known=has_known, cov_path=cov_path, any_multi=any_multi,
                       n_blocks=n_blocks_total, cat=bool(case.get("cat_past", {}).get("rows", 0) or
                                                         case.get("cat_known", {}).get("rows", 0)),
                       single_target=(case["V"] == 1), multi_blocks=n_blocks_total > 2)


MODES = ('f32', 'f32-math', 'f64', 'f64-rerun', 'f32-math-rerun')


def scaled_case(case, tiny):
    result = json.loads(json.dumps(case))
    if tiny:
        for key in ('T', 'H'):
            result[key] = max(1, result[key] // 4)
        if result.get('missing', {}).get('leading'):
            result['missing']['leading'] = max(1, result['missing']['leading'] // 4)
    return result


def rerun_names(names):
    return set(expected_reruns(names))


def allows_input_nan(name):
    """Raw context copies retain the missing-value positions from the source."""
    return name.startswith('input.') or bool(re.search(
        r'\.(args\.(inputs|known_future)|scale\.in|refine\.context|assemble\.rows|cap\.rows)$', name))


def finite_nonnegative(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def input_shapes(case):
    shapes = {'input.context': (case['B'], case['V'], case['T'])}
    for key, rows, horizon in (
            ('past', case.get('past', {}).get('numeric', 0), False),
            ('known', sum(case.get('known', {}).get(k, 0) for k in ('numeric', 'indicator')), True),
            ('cat_past', case.get('cat_past', {}).get('rows', 0), False),
            ('cat_known', case.get('cat_known', {}).get('rows', 0), True)):
        if rows:
            shapes['input.' + key] = (case['B'], rows, case['T'] + (case['H'] if horizon else 0))
    return shapes


def has_exception(kv):
    return any(k.startswith('expect.error') for k in kv)


def validate_exception(t, kv, case, mode, cfg):
    name = case['name'] + ':' + mode
    # No pre-block refusal for the canonical const/all_missing inputs is proven
    # at the pinned reference. Matching wrapper metadata is not provenance.
    bad(f'{name}: no proven reference exception contract; acceptance blocked pending generator provenance')
    validate_records(t, kv, case, mode, cfg)
    if kv.get('case.n_blocks') != 0:
        bad(f'{name}: pre-block exception requires case.n_blocks = 0')
    # These are the only wrapped calls possible before predict_block. Unknown
    # counters fail closed too, rather than silently authorizing new block work.
    preblock = {'forecaster._forecast_raw', 'forecaster._assemble_rows',
                'forecaster._encode_categorical', 'forecaster.target_encode'}
    block_records = any(re.match(r'^(pos|neg)\.b\d+\.', n) for n in t)
    block_metadata = any(re.match(r'^(empty\.|stat\.)(pos|neg)\.b\d+\.', k) for k in kv)
    block_calls = any(k.startswith('calls.') and k[6:] not in preblock and v != 0
                      for k, v in kv.items())
    if block_records or block_metadata or block_calls:
        bad(f'{name}: block execution evidence; post-block errors need a step/coverage contract (follow-up)')
    for key in ('type', 'message', 'step'):
        value = kv.get('expect.error.' + key)
        if not isinstance(value, str) or not value:
            bad(f'{name}: reference error needs {key}')
    if any(n.startswith('final.') for n in t):
        bad(f'{name}: reference error and outputs mixed')


def validate_records(t, kv, case, mode, cfg):
    name = case['name'] + ':' + mode
    for n, a in t.items():
        if not a.size:
            bad(f'{name}: empty tensor {n}')
        if bio.is_float(a):
            # Input/category and host-double islands need not follow model precision.
            if not n.startswith('input.') and '.frames.' not in n:
                want = np.float64 if mode.startswith('f64') else np.float32
                if a.dtype != want:
                    bad(f'{name}: wrong dtype for {n}: {a.dtype}')
            if not mode.endswith('rerun'):
                rms = kv.get(f'stat.{n}.rms')
                if not finite_nonnegative(rms) or kv.get(f'stat.{n}.numel') != a.size:
                    bad(f'{name}: invalid rms/numel for {n}')
        if mode == 'f32':
            stats = []
            if re.search(r'\.(patch\.(hidden_out|out)|head\.raw|layer\.\d+\.(mlp|attn\.\d+\.(q|k|v|q_rope|k_rope|gate|out)))$', n):
                stats.append('kappa')
            if n.endswith(('.patch.out', '.head.raw')):
                stats.append('kappa_res')
            if n.endswith('.sdpa'):
                stats.append('max_logit')
            if re.search(r'\.layer\.\d+\.(mlp|attn\.\d+\.out)$', n):
                stats.append('rho')
            for stat in stats:
                if not finite_nonnegative(kv.get(f'stat.{n}.{stat}')):
                    bad(f'{name}: missing/invalid {stat} for {n}')
    shapes = input_shapes(case)
    if not mode.endswith('rerun'):
        for n in shapes:
            if n not in t:
                bad(f'{name}: missing stored input {n}')
            else:
                want = np.float64 if n.startswith('input.cat_') else np.float32
                if t[n].dtype != want:
                    bad(f'{name}: wrong dtype for {n}: {t[n].dtype}')
    q = len(cfg['quantile_knots'])
    for n in ('final.quantiles', 'final.raw.pos', 'final.raw.neg'):
        shapes[n] = (case['B'], case['V'], case['H'], q)
    for n, shape in shapes.items():
        if n in t and t[n].shape != shape:
            bad(f'{name}: wrong shape {n}: {t[n].shape}, expected {shape}')


def validate_metadata(kv, case, mode, tiny, reference=None):
    name = case['name'] + ':' + mode
    cfg = json.loads(kv['model.config_json'])
    # head_dim is a dataclass property, not a field in serialized config JSON.
    if type(cfg['num_heads']) is not int or cfg['num_heads'] <= 0:
        raise ValueError(f'{name}: invalid num_heads')
    cfg['head_dim'] = cfg['d_model'] // cfg['num_heads']
    for key in ('patch_size', 'block_size', 'context_length', 'd_model', 'num_heads', 'head_dim', 'mlp_hidden_dim'):
        if type(cfg[key]) is not int or cfg[key] <= 0 or kv.get('model.' + key) != cfg[key]:
            raise ValueError(f'{name}: invalid model.{key}')
    sched = cfg['block_schedule']
    if not sched or any(k not in LAYER_KIND_AXES for k in sched) or kv.get('model.n_layers') != len(sched):
        raise ValueError(f'{name}: invalid block schedule')
    if cfg['d_model'] != cfg['num_heads'] * cfg['head_dim']:
        raise ValueError(f'{name}: inconsistent head dimensions')
    if kv.get('mode') != mode or kv.get('case.tiny') is not tiny:
        bad(f'{name}: inconsistent mode/case.tiny')
    if json.loads(kv['case.json']) != case or kv.get('case.name') != case['name']:
        bad(f'{name}: case.json/name differs from requested case')
    for k in ('B', 'V', 'T', 'H'):
        if kv.get('case.' + k) != case[k]:
            bad(f'{name}: inconsistent case.{k}')
    blocks = (case['H'] + cfg['patch_size'] * cfg['block_size'] - 1) // (cfg['patch_size'] * cfg['block_size'])
    if not has_exception(kv) and kv.get('case.n_blocks') != blocks:
        bad(f'{name}: case.n_blocks must be {blocks}')
    layers = set(range(len(sched))) if tiny or case.get('layers', 'all') == 'all' else set(case['layers']) | {len(sched)-1}
    dumped = set(range(len(sched))) if kv['layers.dumped'] == 'all' else {int(v) for v in kv['layers.dumped'].split(',')}
    if dumped != layers or not layers <= set(range(len(sched))):
        bad(f'{name}: layers.dumped differs from required layers')
    if reference:
        for k in reference:
            if k.startswith(('model.', 'env.hf')) and kv.get(k) != reference[k]:
                bad(f'{name}: inconsistent {k}')
    if kv.get('repro.bitwise_equal') is not True:
        bad(f'{name}: bitwise reproducibility missing')
    if not mode.endswith('rerun') and kv.get('repro.runs') != 2:
        bad(f'{name}: expected two reproducibility runs')
    if mode.startswith('f64') and not kv.get('truth.patches'):
        bad(f'{name}: missing truth.patches')
    if mode.endswith('rerun') and kv.get('rerun.source') != 'f32':
        bad(f'{name}: rerun.source must be f32')
    if mode == 'f64-rerun':
        ratio = kv.get('selfcheck.max_ratio')
        if (kv.get('selfcheck.ok') is not True or not finite_nonnegative(ratio)
                or ratio > 1e-12 or kv.get('selfcheck.limit') != 1e-12):
            bad(f'{name}: invalid selfcheck (required ratio <= 1e-12)')
    return cfg


def check_case(directory, case, tiny=False):
    try:
        return _check_case(directory, scaled_case(case, tiny), tiny)
    except (KeyError, ValueError, TypeError, IndexError, OSError) as exc:
        bad(f"{case['name']}: malformed/unsupported baseline: {exc}")
        return False


def _check_case(directory, case, tiny):
    name = case["name"]
    paths = {m: bio.case_file(directory, name, m) for m in ("f32", "f32-math", "f64", "f64-rerun", "f32-math-rerun")}
    if not paths["f32"].exists():
        bad(f'{name}: required f32 baseline missing')
        return False
    print(f"== {name}")
    t, kv = bio.read_baseline(paths["f32"])
    cfg = validate_metadata(kv, case, 'f32', tiny)
    case_run = case
    err = has_exception(kv)
    if err:
        validate_exception(t, kv, case, 'f32', cfg)
        for mode in ('f32-math', 'f64'):
            if not paths[mode].exists():
                bad(f'{name}: missing error companion {mode}')
                continue
            et, ek = bio.read_baseline(paths[mode])
            ecfg = validate_metadata(ek, case, mode, tiny, kv)
            validate_exception(et, ek, case, mode, ecfg)
            if any(ek.get(k) != kv.get(k) for k in
                   ('expect.error.type', 'expect.error.message', 'expect.error.step')):
                bad(f'{name}: inconsistent reference errors in {mode}')
        for mode in ('f64-rerun', 'f32-math-rerun'):
            if paths[mode].exists():
                bad(f'{name}: unexpected rerun artifact for a pre-block reference error: {mode}')
        return True
    exp, info = expected(kv, t, case_run)
    if not err:
        exp.add("final.quantiles")
    missing = sorted(n for n in exp if n not in t)
    if missing:
        bad(f"{name}: {len(missing)} expected records missing, e.g. {missing[:4]}")
    if not kv.get("repro.bitwise_equal"):
        bad(f"{name}: no bitwise reproducibility record")
    validate_records(t, kv, case, 'f32', cfg)
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
        if not paths[m].exists():
            bad(f'{name}: required mode {m} missing')
            continue
        if paths[m].exists():
            t2, kv2 = bio.read_baseline(paths[m])
            validate_metadata(kv2, case, m, tiny, kv)
            validate_records(t2, kv2, case, m, cfg)
            if has_exception(kv2):
                bad(f'{name}: unexpected error in {m}')
            for n in set(t2) & set(t):
                if t2[n].shape != t[n].shape or (not bio.is_float(t[n]) and t2[n].dtype != t[n].dtype):
                    bad(f'{name}: {m} shape/dtype mismatch for {n}')
            if set(t2) != set(t):
                d = sorted(set(t2) ^ set(t))
                bad(f"{name}: {m} record names differ from f32: {d[:4]}")
            if not kv2.get("repro.bitwise_equal"):
                bad(f"{name}: {m}: no reproducibility record")
            if m == "f64" and not kv2.get("truth.patches"):
                bad(f"{name}: f64 file has no truth.patches")
    # reruns
    for m in ("f64-rerun", "f32-math-rerun"):
        if not paths[m].exists():
            bad(f'{name}: required mode {m} missing')
            continue
        t3, kv3 = bio.read_baseline(paths[m])
        validate_metadata(kv3, case, m, tiny, kv)
        validate_records(t3, kv3, case, m, cfg)
        if has_exception(kv3):
            bad(f'{name}: unexpected error in {m}')
        need = rerun_names(exp)
        if set(t3) != need:
            bad(f'{name}: {m} component coverage differs: missing {sorted(need-set(t3))[:4]}, extra {sorted(set(t3)-need)[:4]}')
        for n in set(t3) & need:
            if source_name(n) not in t:
                continue  # Missing gold coverage is already reported above.
            if t3[n].shape != t[source_name(n)].shape:
                bad(f'{name}: {m} shape mismatch for {n}')
        extra = sorted(n for n in t3 if source_name(n) not in t)
        if extra:
            bad(f"{name}: {m} has records not in the f32 file: {extra[:4]}")
        if not t3:
            bad(f"{name}: {m} is empty")
        if not kv3.get("repro.bitwise_equal"):
            bad(f"{name}: {m}: no reproducibility record")
    if name in ("const", "all_missing") and not (("final.quantiles" in t) or err):
        bad(f"{name}: neither outputs nor expect.error")
    print(f"   {len(t)} records, {sum(1 for k in kv if k.startswith('calls.'))} call counters, "
          f"{'ok' if not [e for e in errors if e.startswith(name)] else 'problems'}")
    return True



def validate_artifacts(data, case):
    """Validate in-memory artifacts without CLI state or filesystem mutation."""
    messages = []
    token = _diagnostics.set(messages)
    try:
        t, kv = data['f32']
        tiny = kv.get('case.tiny')
        if type(tiny) is not bool:
            bad('case.tiny must be a boolean')
        validate_cases({'version': 1, 'cases': [case]})
        cfg = validate_metadata(kv, case, 'f32', tiny)
        exp, _ = expected(kv, t, case)
        exp.add('final.quantiles')
        if not exp <= set(t):
            bad(f"missing required records: {sorted(exp-set(t))[:4]}")
        if has_exception(kv):
            bad('floor computation requires numerical artifacts, not exceptions')
        for mode in MODES:
            a, meta = data[mode]
            validate_metadata(meta, case, mode, tiny, kv)
            for key in ('env.timestral_sha', 'env.hf_revision'):
                if not isinstance(meta.get(key), str) or not meta[key].strip():
                    bad(f'{mode}: missing source provenance {key}')
            validate_records(a, meta, case, mode, cfg)
            want = set(expected_reruns(t)) if mode.endswith('rerun') else set(t)
            if set(a) != want:
                bad(f'{mode}: required record coverage differs')
            for n, arr in a.items():
                target = source_name(n) if mode.endswith('rerun') else n
                if target in t and (arr.shape != t[target].shape or
                        (bio.is_float(arr) != bio.is_float(t[target])) or
                        (not bio.is_float(t[target]) and arr.dtype != t[target].dtype)):
                    bad(f'{mode}: shape/dtype mismatch for {n}')
                if bio.is_float(arr):
                    # Raw inputs may retain NaNs, but infinities are never evidence.
                    if np.isinf(arr).any() or (not allows_input_nan(n) and np.isnan(arr).any()):
                        bad(f'{mode}: nonfinite evidence {n}')
            for k, v in meta.items():
                if k.startswith(('stat.', 'floor_fr.', 'floor_tf.', 'cap.')) and not finite_nonnegative(v):
                    bad(f'{mode}: invalid numerical metadata {k}')
            for k in kv:
                if k.startswith(('env.timestral', 'env.hf', 'model.')) and meta.get(k) != kv[k]:
                    bad(f'{mode}: inconsistent provenance {k}')
        if messages:
            raise ValueError('; '.join(messages))
    finally:
        _diagnostics.reset(token)
