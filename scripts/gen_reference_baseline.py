#!/usr/bin/env python3
"""Dump reference tensors of the Timestral Python package to baseline GGUF files.

    gen_reference_baseline.py --model <hub id or dir> [--revision SHA] --cases tests/cases.json
        [--case NAME ...] --mode f32|f32-math|f64|f64-rerun|f32-math-rerun
        --output-dir DIR [--layers all|0,1,3,11,23] [--tiny] [--include-slow] [--no-repeat]

docs/PLAN.md, section C, describes the modes, the instrumentation and the file format.
Short version:

  f32             the reference as shipped (the gold files)
  f32-math        the same inside the math attention kernel
  f64             "truth" run: model and inputs in float64; RMSNorm and the RoPE tables are
                  replaced by float64 copies of the same formula (listed in truth.patches)
  f64-rerun       each component rerun alone in float64 on the dumped f32 input
                  (teacher-forced floors), plus the rerun self-check on the f64 file
  f32-math-rerun  the same component reruns in float32 with the math kernel

Each case runs twice and the two runs must be bitwise equal. The file for a case and mode is
baseline-<case>-<mode>.gguf in the output directory.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# --- determinism, before the model loads (plan C.3) ---
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float32)

import timestral
import timestral.covariate_encoding as _cov
import timestral.forecaster as _fc
import timestral.layers as _layers
import timestral.masks as _masks
import timestral.model as _model
import timestral.scaler as _scaler
from timestral.config import LAYER_KIND_AXES
from timestral.forecaster import TimestralForecaster
from timestral.layers import RMSNorm, SinCosRoPE, TwoStreamMask
from timestral.loading import load_model
from timestral.model import TimestralModel, monotone_quantiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_io as bio  # noqa: E402

MODES = ["f32", "f32-math", "f64", "f64-rerun", "f32-math-rerun"]
CODE_SHA = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"

# Names of recorded tensors that hold booleans / integers (stored as I8 / I32), used when a
# rerun reads them back.
BOOL_NAMES = {"args.observed", "args.known_future_mask", "scale.observed", "mask.time", "mask.variate",
              "mask.key_valid", "mask.obs_per_patch", "mask.block_active", "known.is_binary", "known.rows",
              "refine.mask", "refine.target_rows", "cap.mask", "assemble.mask", "assemble.known_future_mask"}
LONG_NAMES = {"args.group_ids", "args.role_ids", "positions", "tf.positions", "first_obs.group",
              "group.inverse", "mask.block_anchor", "assemble.role_ids"}


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Ctx:
    pass_name = None      # 'pos' | 'neg' | None
    block = None          # block index inside the pass, or None
    attn = None           # (layer, attn index) while inside Attention.forward
    in_refine = False
    rerun = False
    n_raw = 0             # _forecast_raw calls inside the current forecast
    enc_n = 0
    target_n = 0
    sdpa_buf: list = []
    logit_max = 0.0
    last_call = ""


CTX = Ctx()


def prefix() -> str:
    if CTX.pass_name is None:
        return ""
    if CTX.block is None:
        return f"{CTX.pass_name}."
    return f"{CTX.pass_name}.b{CTX.block}."


def to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, (bool, int, np.integer)):
        x = np.array([int(x)], dtype=np.int32)
    elif isinstance(x, (float, np.floating)):
        x = np.array([float(x)], dtype=np.float64)
    x = np.array(x, copy=True)
    if x.dtype == np.bool_:
        x = x.astype(np.int8)
    elif x.dtype == np.int64:
        x = x.astype(np.int32)
    if x.ndim == 0:
        x = x.reshape(1)
    return x


class Rec:
    def __init__(self, stats: bool = True):
        self.t: dict[str, np.ndarray] = {}
        self.stat: dict[str, float] = {}
        self.kv: dict = {}
        self.calls = Counter()
        self.allow = None      # callable(rel_name) -> bool, or None
        self.stats = stats
        self.layers = None     # set of layer indices to record, or None for all

    def put(self, rel: str, value, absolute: bool = False):
        if value is None:
            return
        if self.allow is not None and not self.allow(rel):
            return
        name = rel if absolute else prefix() + rel
        a = to_np(value)
        if a.size == 0:
            self.kv[f"empty.{name}"] = "x".join(map(str, a.shape))
            return
        if name in self.t:
            raise RuntimeError(f"duplicate record {name}")
        self.t[name] = a

    def put_stat(self, rel: str, key: str, value: float):
        if self.stats and (self.allow is None or self.allow(rel)):
            self.stat[f"stat.{prefix()}{rel}.{key}"] = float(value)

    def want_layer(self, l: int) -> bool:
        return self.layers is None or l in self.layers


REC = Rec()


@contextlib.contextmanager
def only(names=None, pred=None):
    """Record only the given relative names (or those accepted by `pred`)."""
    old = REC.allow
    if pred is not None:
        REC.allow = pred
    else:
        s = set(names)
        REC.allow = lambda n: n in s
    try:
        yield
    finally:
        REC.allow = old


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

def _wrap(owner, name: str, tag: str, before=None, after=None):
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def w(*a, **k):
        REC.calls[tag] += 1
        CTX.last_call = tag
        if before is not None:
            before(*a, **k)
        out = orig(*a, **k)
        if after is not None:
            after(out, *a, **k)
        return out

    w._orig = orig
    setattr(owner, name, w)
    return orig


_INSTALLED = False


def kappa_of(lin: torch.nn.Linear, x: torch.Tensor, y: torch.Tensor, rows: slice | None = None) -> float:
    """max_i (|x| @ |W|^T + |b|)_i / rms(y), over the output rows `rows`."""
    with torch.no_grad():
        w = lin.weight.abs()
        b = lin.bias.abs() if lin.bias is not None else None
        yy = y
        if rows is not None:
            w = w[rows]
            b = b[rows] if b is not None else None
            yy = y[..., rows]
        t = torch.nn.functional.linear(x.abs(), w, b)
        r = yy.double().pow(2).mean().sqrt()
        return float(t.amax().double() / r) if r > 0 else 0.0


def install_instrumentation():
    """Wrap functions in the namespace that calls them, and add the forward-hook helpers.

    Wrapping only records; it does not change any computation (plan C.3).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    P = lambda rel, v: REC.put(rel, v)

    # --- timestral.layers ---
    def after_rotary(out, q, k, cos, sin):
        if CTX.attn is not None and REC.want_layer(CTX.attn[0]):
            l, j = CTX.attn
            base = f"layer.{l}.attn.{j}."
            P(base + "q_rope", out[0])
            P(base + "k_rope", out[1])
            P(base + "cos", cos)
            P(base + "sin", sin)
    _wrap(_layers, "_apply_rotary", "layers._apply_rotary", after=after_rotary)

    def after_sdpa(out, query, key, value, mask, is_causal=False):
        if CTX.attn is None or not REC.want_layer(CTX.attn[0]):
            return
        CTX.sdpa_buf.append(out.detach().clone())
        if REC.stats and REC.allow is None:
            with torch.no_grad():
                hd = query.shape[-1]
                lg = torch.einsum("bqhd,bkhd->bhqk", query.double(), key.double()) * (hd ** -0.5)
                if mask is not None:
                    lg = lg.masked_fill(~mask, 0.0)
                elif is_causal:
                    sq, sk = lg.shape[-2:]
                    lg = lg.masked_fill(~torch.tril(torch.ones(sq, sk, dtype=torch.bool)), 0.0)
                CTX.logit_max = max(CTX.logit_max, float(lg.abs().amax()))
    _wrap(_layers, "_sdpa", "layers._sdpa", after=after_sdpa)

    def after_qkv(out, self, x):
        if CTX.attn is not None and REC.want_layer(CTX.attn[0]):
            base = f"layer.{CTX.attn[0]}.attn.{CTX.attn[1]}."
            P(base + "q", out[0])
            P(base + "k", out[1])
            P(base + "v", out[2])
    _wrap(_layers.Attention, "_qkv", "layers.Attention._qkv", after=after_qkv)

    # --- timestral.model ---
    _wrap(_model, "build_forecast_time_mask", "model.build_forecast_time_mask",
          after=lambda out, *a, **k: P("mask.time", out))
    _wrap(_model, "build_forecast_positions", "model.build_forecast_positions",
          after=lambda out, *a, **k: P("positions", out))
    _wrap(_model, "build_two_stream_time_mask", "model.build_two_stream_time_mask",
          after=lambda out, *a, **k: P("mask.time", out))
    _wrap(_model, "build_two_stream_variate_mask", "model.build_two_stream_variate_mask",
          after=lambda out, *a, **k: P("mask.variate", out))
    _wrap(_model, "build_two_stream_positions", "model.build_two_stream_positions",
          after=lambda out, *a, **k: P("positions", out))

    def after_kv(out, obs_per_patch, attend_missing):
        P("mask.obs_per_patch", obs_per_patch)
        P("mask.key_valid", out)
    _wrap(_model, "compute_enc_key_validity", "model.compute_enc_key_validity", after=after_kv)

    def after_first(out, obs_per_patch, inverse, num_groups):
        P("first_obs.group", out)
        P("group.inverse", inverse)
    _wrap(_model, "group_first_observed_patch", "model.group_first_observed_patch", after=after_first)
    _wrap(_model, "causal_patch_loc_scale", "model.causal_patch_loc_scale")
    _wrap(_model, "apply_affine_asinh_masked", "model.apply_affine_asinh_masked")

    def after_denorm(out, preds, loc, scale, use_asinh):
        P("denorm.in", preds)
        P("anchor.loc", loc)
        P("anchor.scale", scale)
    _wrap(_model, "denormalize_quantiles", "model.denormalize_quantiles", after=after_denorm)

    def after_inst(out, values, mask, minimum_scale):
        P("known.loc", out[0])
        P("known.scale", out[1])
        P("known.is_binary", out[2])
    _wrap(_model, "instance_norm_stats", "model.instance_norm_stats", after=after_inst)

    # --- timestral.masks ---
    def after_layout(out, min_context, n_patches, block_size):
        P("mask.block_anchor", out[0])
        P("mask.block_active", out[1])
    _wrap(_masks, "compute_block_layout", "masks.compute_block_layout", after=after_layout)

    # --- timestral.forecaster ---
    def after_frames(out, context, mask, sinh_median, minimum_scale):
        P("frames.in.sinh_median", sinh_median)
        P("frames.loc", out[0])
        P("frames.scale", out[1])
    _wrap(_fc, "advancing_block_frames", "forecaster.advancing_block_frames", after=after_frames)

    def after_cpls(out, data, mask, ps, ms, sg):
        P("cap.scale", out[1])
    _wrap(_fc, "causal_patch_loc_scale", "forecaster.causal_patch_loc_scale", after=after_cpls)

    def after_clamp(out, vals):
        P("refine.clamp.in", vals)
        P("refine.clamp.out", out)
    _wrap(_fc, "clamp_nonfinite", "forecaster.clamp_nonfinite", after=after_clamp)

    def after_te(out, target_context, past_codes, future_codes=None, smooth=1.0):
        n = CTX.target_n
        CTX.target_n += 1
        P(f"target_encode.{n}.ctx", target_context)
        P(f"target_encode.{n}.past", past_codes)
        if future_codes is not None:
            P(f"target_encode.{n}.future", future_codes)
        P(f"target_encode.{n}.out_past", out[0])
        if out[1] is not None:
            P(f"target_encode.{n}.out_future", out[1])
    _wrap(_fc, "target_encode", "forecaster.target_encode", after=after_te)

    # --- methods ---
    def before_predict(self, inputs, observed, group_ids, horizon_patches, role_ids=None, known_future=None,
                       known_future_mask=None):
        if not CTX.rerun:
            CTX.block = 0 if CTX.block is None else CTX.block + 1
        P("args.inputs", inputs)
        P("args.observed", observed)
        P("args.group_ids", group_ids)
        P("args.horizon_patches", horizon_patches)
        P("args.role_ids", role_ids)
        P("args.known_future", known_future)
        P("args.known_future_mask", known_future_mask)
    _wrap(TimestralModel, "predict_block", "model.predict_block", before=before_predict,
          after=lambda out, *a, **k: P("predict", out))

    def after_scale(out, self, inputs, observed):
        if CTX.in_refine:
            P("refine.loc", out[1])
            P("refine.scale", out[2])
        else:
            P("scale.in", inputs)
            P("scale.observed", observed)
            P("scaled", out[0])
            P("loc", out[1])
            P("scale", out[2])
    _wrap(TimestralModel, "scale", "model.scale", after=after_scale)

    def after_known(out, self, inputs, observed, scaled, loc, scale, known_rows):
        P("known.rows", known_rows)
        P("known.scaled", out[0])
    _wrap(TimestralModel, "_known_cov_scaling", "model._known_cov_scaling", after=after_known)

    def after_assemble(out, self, inputs, mask, plen, past, known):
        rows, m, role_ids, kf, kfm, n_tgt = out
        P("assemble.rows", rows)
        P("assemble.mask", m)
        P("assemble.role_ids", role_ids)
        P("assemble.known_future", kf)
        P("assemble.known_future_mask", kfm)
        P("assemble.n_tgt", n_tgt)
    _wrap(TimestralForecaster, "_assemble_rows", "forecaster._assemble_rows", after=after_assemble)

    def before_refine(self, quantiles, context, mask, target_rows):
        CTX.in_refine = True
        P("refine.context", context)
        P("refine.mask", mask)
        P("refine.target_rows", target_rows)

    def after_refine(out, self, quantiles, context, mask, target_rows):
        CTX.in_refine = False
        P("refined", out)
    _wrap(TimestralForecaster, "_refine_block", "forecaster._refine_block", before=before_refine, after=after_refine)

    def before_raw(self, *a, **k):
        if not CTX.rerun:
            CTX.pass_name = "pos" if CTX.n_raw == 0 else "neg"
            CTX.n_raw += 1
            CTX.block = None

    # final.raw is recorded with an absolute name, so it must not get the pass prefix
    def after_raw_abs(out, self, *a, **k):
        nm = CTX.pass_name
        REC.put(f"final.raw.{nm}", out, absolute=True)
        CTX.pass_name = None
        CTX.block = None
    _wrap(TimestralForecaster, "_forecast_raw", "forecaster._forecast_raw", before=before_raw, after=after_raw_abs)

    def before_cap(self, quantiles, inputs, mask):
        CTX.block = None
        P("cap.in", quantiles)
        P("cap.rows", inputs)
        P("cap.mask", mask)
    _wrap(TimestralForecaster, "_cap_quantiles", "forecaster._cap_quantiles", before=before_cap,
          after=lambda out, *a, **k: P("cap.out", out))

    def after_enc(out, self, codes, target):
        n = CTX.enc_n
        CTX.enc_n += 1
        P(f"enc_cat.{n}.codes", codes)
        P(f"enc_cat.{n}.target", target)
        P(f"enc_cat.{n}.out", out)
    _wrap(TimestralForecaster, "_encode_categorical", "forecaster._encode_categorical", after=after_enc)


def register_hooks(model: TimestralModel):
    """Forward hooks of plan C.3 on `model`. Returns the list of handles."""
    hs = []
    cfg = model.config
    P = lambda rel, v: REC.put(rel, v)
    ps = cfg.patch_size

    # patch embedding
    pp = model.representation.patch_proj
    kap = {}

    def lin_hook(key):
        def h(m, inp, out):
            if REC.stats and REC.allow is None:
                kap[key] = kappa_of(m, inp[0], out)
        return h
    hs.append(pp.hidden_layer.register_forward_hook(lambda m, i, o: (P("patch.hidden_out", o), lin_hook("h")(m, i, o))[0]))
    hs.append(pp.output_layer.register_forward_hook(lin_hook("o")))
    hs.append(pp.residual_layer.register_forward_hook(lin_hook("r")))

    def patch_post(m, inp, out):
        P("patch.in", inp[0])
        P("patch.out", out)
        if "h" in kap:
            REC.put_stat("patch.hidden_out", "kappa", kap["h"])
            REC.put_stat("patch.out", "kappa", kap["o"])
            REC.put_stat("patch.out", "kappa_res", kap["r"])
        kap.clear()
    hs.append(pp.register_forward_hook(patch_post))

    # transformer
    def tf_pre(m, args):
        x, tmask, vmask, positions = args[:4]
        P("tf.in", x)
        P("tf.positions", positions)
        if isinstance(tmask, TwoStreamMask):
            P("tf.enc_len", tmask.enc_len)
    hs.append(model.transformer.register_forward_pre_hook(tf_pre))
    hs.append(model.transformer.out_norm.register_forward_hook(lambda m, i, o: P("out_norm", o)))

    for li, layer in enumerate(model.transformer.layers):
        def layer_pre(m, args, li=li):
            if REC.want_layer(li):
                P(f"layer.{li}.in", args[0])
        hs.append(layer.register_forward_pre_hook(layer_pre))

        def layer_post(m, args, out, li=li):
            if not REC.want_layer(li):
                return
            P(f"layer.{li}.out", out)
            if REC.stats and REC.allow is None:
                _layer_rho(li, m)
        hs.append(layer.register_forward_hook(layer_post))

        for j, (norm, attn) in enumerate(zip(layer.attn_norms, layer.attns)):
            base = f"layer.{li}.attn.{j}."

            def norm_hook(m, i, o, li=li, base=base):
                if REC.want_layer(li):
                    P(base + "norm", o)
            hs.append(norm.register_forward_hook(norm_hook))

            def attn_pre(m, args, li=li, j=j):
                CTX.attn = (li, j)
                CTX.sdpa_buf = []
                CTX.logit_max = 0.0
            hs.append(attn.register_forward_pre_hook(attn_pre))

            def attn_post(m, args, out, li=li, j=j, base=base):
                if REC.want_layer(li):
                    P(base + "out", out)
                    if CTX.sdpa_buf:
                        P(base + "sdpa", torch.cat(CTX.sdpa_buf, dim=1))
                        if REC.stats and REC.allow is None:
                            REC.put_stat(base + "sdpa", "max_logit", CTX.logit_max)
                CTX.attn = None
                CTX.sdpa_buf = []
            hs.append(attn.register_forward_hook(attn_post))

            D = cfg.d_model

            def qkv_hook(m, i, o, li=li, base=base, D=D):
                if not REC.want_layer(li):
                    return
                P(base + "qkv", o)
                if REC.stats and REC.allow is None:
                    for s, nm in enumerate("qkv"):
                        REC.put_stat(base + nm, "kappa", kappa_of(m, i[0], o, slice(s * D, (s + 1) * D)))
                        if nm != "v":
                            REC.put_stat(base + nm + "_rope", "kappa", REC.stat.get(f"stat.{prefix()}{base}{nm}.kappa", 0.0))
            hs.append(attn.wQKV.register_forward_hook(qkv_hook))

            if attn._gate is not None:
                def gate_hook(m, i, o, li=li, base=base):
                    if REC.want_layer(li):
                        P(base + "gate", o)
                        if REC.stats and REC.allow is None:
                            REC.put_stat(base + "gate", "kappa", kappa_of(m, i[0], o))
                hs.append(attn._gate.register_forward_hook(gate_hook))

            def wo_pre(m, args, li=li, base=base):
                if REC.want_layer(li):
                    P(base + "gated", args[0])
            hs.append(attn.wO.register_forward_pre_hook(wo_pre))

            def wo_post(m, i, o, li=li, base=base):
                if REC.want_layer(li) and REC.stats and REC.allow is None:
                    REC.put_stat(base + "out", "kappa", kappa_of(m, i[0], o))
            hs.append(attn.wO.register_forward_hook(wo_post))

        def mnorm_hook(m, i, o, li=li):
            if REC.want_layer(li):
                P(f"layer.{li}.mlp_norm", o)
        hs.append(layer.mlp_norm.register_forward_hook(mnorm_hook))

        mlp_kap = {}

        def w_hook(key, li=li, mlp_kap=mlp_kap):
            def h(m, i, o):
                if REC.want_layer(li) and REC.stats and REC.allow is None:
                    mlp_kap[key] = kappa_of(m, i[0], o)
            return h
        hs.append(layer.mlp.w12.register_forward_hook(w_hook("w12")))
        hs.append(layer.mlp.w3.register_forward_hook(w_hook("w3")))

        def mlp_post(m, i, o, li=li, mlp_kap=mlp_kap):
            if REC.want_layer(li):
                P(f"layer.{li}.mlp", o)
                if REC.stats and REC.allow is None and mlp_kap:
                    REC.put_stat(f"layer.{li}.mlp", "kappa", max(mlp_kap.values()))
            mlp_kap.clear()
        hs.append(layer.mlp.register_forward_hook(mlp_post))

    # head
    hk = {}
    hs.append(model.head.projection.output_layer.register_forward_hook(
        lambda m, i, o: hk.__setitem__("o", kappa_of(m, i[0], o)) if (REC.stats and REC.allow is None) else None))
    hs.append(model.head.projection.residual_layer.register_forward_hook(
        lambda m, i, o: hk.__setitem__("r", kappa_of(m, i[0], o)) if (REC.stats and REC.allow is None) else None))

    def proj_post(m, i, o):
        P("head.raw", o)
        if "o" in hk:
            REC.put_stat("head.raw", "kappa", hk["o"])
            REC.put_stat("head.raw", "kappa_res", hk["r"])
        hk.clear()
    hs.append(model.head.projection.register_forward_hook(proj_post))
    hs.append(model.head.register_forward_pre_hook(lambda m, a: P("head.in", a[0])))
    hs.append(model.head.register_forward_hook(lambda m, i, o: P("head.monotone", o)))
    return hs


def _rms_t(a) -> float:
    return float(a.double().pow(2).mean().sqrt())


def _layer_rho(li: int, layer):
    """rho = rms(sublayer out) / rms(residual stream) for each sublayer of layer `li`."""
    pre = prefix()
    get = lambda n: REC.t.get(pre + n)
    x = get(f"layer.{li}.in")
    if x is None:
        return
    x = torch.from_numpy(x)
    for j in range(len(layer.attns)):
        o = get(f"layer.{li}.attn.{j}.out")
        if o is None:
            return
        o = torch.from_numpy(o)
        r = _rms_t(x)
        REC.put_stat(f"layer.{li}.attn.{j}.out", "rho", _rms_t(o) / r if r > 0 else 0.0)
        x = x + o
    m = get(f"layer.{li}.mlp")
    if m is not None:
        m = torch.from_numpy(m)
        r = _rms_t(x)
        REC.put_stat(f"layer.{li}.mlp", "rho", _rms_t(m) / r if r > 0 else 0.0)


# ---------------------------------------------------------------------------
# Model setup per mode
# ---------------------------------------------------------------------------

def _truth_rmsnorm_forward(self, x):
    xf = x.double()
    out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
    return out * self.scale


def _truth_inv_freq(rope: SinCosRoPE) -> torch.Tensor:
    inv = torch.zeros(rope.dim // 2, dtype=torch.float64)
    idx = torch.arange(0, rope.rotary_dim, 2, dtype=torch.float64)
    inv[: rope.rotary_dim // 2] = 1.0 / (rope.base ** (idx / rope.rotary_dim))
    return inv


def _truth_rope_forward(self, x, position_ids):
    inv = self.inv_freq64[None, :, None].expand(position_ids.shape[0], -1, 1)
    pos = position_ids[:, None, :].double()
    freqs = (inv @ pos).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


TRUTH_PATCHES = ["timestral.layers.RMSNorm.forward: x.double() copy of the same formula",
                 "timestral.layers.SinCosRoPE.forward: float64 inv_freq and float64 cos/sin tables",
                 "timestral.forecaster.TimestralForecaster._as_context: float32-prepared inputs upcast to float64",
                 "timestral.forecaster.TimestralForecaster._as_covariates: float32 numeric covariates upcast to float64",
                 "timestral.forecaster.TimestralForecaster._encode_categorical: float32 encodings upcast to float64"]


def apply_truth_patches(model: TimestralModel) -> None:
    """f64 mode only: replace the two float32 islands by float64 copies (listed in TRUTH_PATCHES)."""
    RMSNorm.forward = _truth_rmsnorm_forward
    SinCosRoPE.forward = _truth_rope_forward
    model.to(torch.float64)
    rope = model.transformer.rope
    rope.inv_freq64 = _truth_inv_freq(rope)

    o1 = TimestralForecaster._as_context
    o2 = TimestralForecaster._as_covariates
    o3 = TimestralForecaster._encode_categorical

    def ctx(self, c, om):
        x, m = o1(self, c, om)
        return x.double(), m

    def cov(self, values, batch, length, name, dtype=torch.float32):
        v = o2(self, values, batch, length, name, dtype)
        return v.double() if (v is not None and dtype == torch.float32) else v

    def enc(self, codes, target):
        v = o3(self, codes, target)
        return v.double() if (v is not None and v.dtype == torch.float32) else v
    TimestralForecaster._as_context = ctx
    TimestralForecaster._as_covariates = cov
    TimestralForecaster._encode_categorical = enc


def build_model(model_arg: str, revision: str | None, truth: bool) -> TimestralModel:
    kw = {"revision": revision} if (revision and not Path(model_arg).is_dir()) else {}
    model = load_model(model_arg, device="cpu", dtype=torch.float32, **kw)
    model.eval()
    if truth:
        apply_truth_patches(model)
    return model


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def scaled_case(case: dict, tiny: bool) -> dict:
    """The tiny variant of a case: T / 4, H / 4 (plan C.2)."""
    c = json.loads(json.dumps(case))
    if tiny:
        c["T"] = max(1, case["T"] // 4)
        c["H"] = max(1, case["H"] // 4)
        if "missing" in c and c["missing"].get("leading"):
            c["missing"]["leading"] = max(1, c["missing"]["leading"] // 4)
    return c


def make_inputs(case: dict, patch_size: int) -> dict:
    """Deterministic inputs of a case (plan C.2). numpy default_rng(seed)."""
    rng = np.random.default_rng(case["seed"])
    B, V, T, H = case["B"], case["V"], case["T"], case["H"]
    s = case.get("series", {})
    period = s.get("period", 24)
    t = np.arange(T, dtype=np.float64)
    x = np.empty((B, V, T))
    for b in range(B):
        for v in range(V):
            walk = np.cumsum(rng.normal(0.0, s.get("noise_std", 1.0), T))
            wave = 10.0 * np.sin(2.0 * math.pi * t / (period * (1.0 + 0.25 * v)) + 0.3 * b)
            x[b, v] = s.get("level", 0.0) + s.get("trend", 0.0) * t + wave + walk
    if s.get("positive"):
        x = np.abs(x)
    if s.get("constant") is not None:
        x[:] = s["constant"]
    x = x.astype(np.float32)
    m = case.get("missing", {})
    flat = x.reshape(B * V, T)
    if m.get("leading"):
        flat[0, : m["leading"]] = np.nan
    for p in m.get("missing_patches", []):
        flat[0, p * patch_size:(p + 1) * patch_size] = np.nan
    if m.get("scattered_frac"):
        flat[rng.random(flat.shape) < m["scattered_frac"]] = np.nan
    for i in m.get("all_missing_series", []):
        flat[i, :] = np.nan
    out = {"context": x}

    def smooth_rows(n_rows, length, nan_frac):
        tt = np.arange(length, dtype=np.float64)
        r = np.empty((B, n_rows, length))
        for b in range(B):
            for c in range(n_rows):
                r[b, c] = 5.0 * np.sin(2 * math.pi * tt / (period + 7 + 5 * c) + 0.5 * b) + 0.5 * rng.normal(size=length)
        r = r.astype(np.float32)
        if nan_frac:
            r[rng.random(r.shape) < nan_frac] = np.nan
        return r

    pc = case.get("past", {})
    if pc.get("numeric"):
        out["past"] = smooth_rows(pc["numeric"], T, pc.get("nan_frac", 0.0))
    kc = case.get("known", {})
    if kc.get("numeric") or kc.get("indicator"):
        rows = []
        if kc.get("numeric"):
            rows.append(smooth_rows(kc["numeric"], T + H, kc.get("nan_frac", 0.0)))
        if kc.get("indicator"):
            ind = np.empty((B, kc["indicator"], T + H), dtype=np.float32)
            tt = np.arange(T + H)
            for c in range(kc["indicator"]):
                ind[:, c, :] = ((tt // (5 + 2 * c)) % 2 == 0).astype(np.float32)
            rows.append(ind)
        out["known"] = np.concatenate(rows, axis=1)
    cp = case.get("cat_past", {})
    if cp.get("rows"):
        codes = rng.integers(0, cp["n_codes"], size=(B, cp["rows"], T)).astype(np.float64)
        if cp.get("nan_frac"):
            codes[rng.random(codes.shape) < cp["nan_frac"]] = np.nan
        out["cat_past"] = codes
    ck = case.get("cat_known", {})
    if ck.get("rows"):
        codes = rng.integers(0, ck["n_codes"], size=(B, ck["rows"], T + H)).astype(np.float64)
        if ck.get("unseen_in_horizon"):
            codes[:, :, T + H // 2] = ck["n_codes"] + 7     # a code that never occurs in the context
        if ck.get("nan_frac"):
            codes[rng.random(codes.shape) < ck["nan_frac"]] = np.nan
        out["cat_known"] = codes
    return out


def case_layers(case: dict, n_layers: int, cli: str | None, tiny: bool):
    spec = cli if cli is not None else (None if tiny else case.get("layers", "all"))
    if spec in (None, "all"):
        return None
    if isinstance(spec, str):
        spec = [int(x) for x in spec.split(",") if x != ""]
    s = set(int(i) for i in spec) | {n_layers - 1}
    return s


def run_forecast(fc: TimestralForecaster, case: dict, inputs: dict, layers, stats: bool = True) -> Rec:
    """One instrumented forecast of a case. Returns the recorder (records, stats, KV, call counts)."""
    global REC
    REC = Rec(stats=stats)
    REC.layers = layers
    CTX.pass_name = None
    CTX.block = None
    CTX.attn = None
    CTX.in_refine = False
    CTX.rerun = False
    CTX.n_raw = 0
    CTX.enc_n = 0
    CTX.target_n = 0
    for k, nm in (("context", "input.context"), ("past", "input.past"), ("known", "input.known"),
                  ("cat_past", "input.cat_past"), ("cat_known", "input.cat_known")):
        if k in inputs:
            REC.put(nm, inputs[k], absolute=True)
    try:
        res = fc.forecast(inputs["context"], case["H"],
                          past_covariates=inputs.get("past"), known_covariates=inputs.get("known"),
                          categorical_past_covariates=inputs.get("cat_past"),
                          categorical_known_covariates=inputs.get("cat_known"))
        REC.put("final.quantiles", res.quantiles, absolute=True)
    except Exception as e:   # the reference may refuse a case; the baseline records how
        REC.kv["expect.error.type"] = type(e).__name__
        REC.kv["expect.error.message"] = str(e)
        REC.kv["expect.error.step"] = CTX.last_call
        CTX.pass_name = None
    return REC


def same_records(a: Rec, b: Rec) -> str | None:
    if set(a.t) != set(b.t):
        return f"record sets differ: {sorted(set(a.t) ^ set(b.t))[:5]}"
    for k in a.t:
        x, y = a.t[k], b.t[k]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return f"record {k} differs between the two runs"
    if a.stat != b.stat and any(not (math.isclose(a.stat[k], b.stat.get(k, math.nan), rel_tol=0, abs_tol=0)) for k in a.stat):
        return "statistics differ between the two runs"
    if a.kv != b.kv:
        return "KV differ between the two runs"
    return None


# ---------------------------------------------------------------------------
# Environment KV
# ---------------------------------------------------------------------------

def sdpa_kernel_seen(model: TimestralModel) -> str:
    """Name the attention kernel torch uses for one forward (torch.profiler trace)."""
    from torch.profiler import ProfilerActivity, profile
    cfg = model.config
    ps = cfg.patch_size
    r = 2
    inputs = torch.randn(r, ps * 4, dtype=next(model.parameters()).dtype)
    observed = torch.ones(r, ps * 4, dtype=torch.bool)
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
        model.predict_block(inputs, observed, torch.arange(r), 2)
    names = sorted({e.key for e in prof.key_averages()
                    if "scaled_dot_product" in e.key or "sdpa" in e.key.lower() or "flash_attention" in e.key})
    return ",".join(names) if names else "unknown"


def env_kv(model: TimestralModel, args, mode: str, cfg_json: str) -> dict:
    lib, ver = platform.libc_ver()
    cfg = model.config
    kv = {
        "mode": mode,
        "env.torch": torch.__version__,
        "env.numpy": np.__version__,
        "env.timestral_sha": CODE_SHA,
        "env.timestral_version": timestral.__version__,
        "env.hf_revision": args.revision or "local",
        "env.cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "env.threads": torch.get_num_threads(),
        "env.python": platform.python_version(),
        "env.glibc": f"{lib} {ver}".strip(),
        "model.id": Path(args.model).name if Path(args.model).is_dir() else args.model,
        "model.patch_size": cfg.patch_size,
        "model.d_model": cfg.d_model,
        "model.num_heads": cfg.num_heads,
        "model.head_dim": cfg.head_dim,
        "model.mlp_hidden_dim": cfg.mlp_hidden_dim,
        "model.n_layers": len(cfg.block_schedule),
        "model.block_size": cfg.block_size,
        "model.context_length": cfg.context_length,
        "model.config_json": cfg_json,
    }
    return kv


# ---------------------------------------------------------------------------
# Reruns (f64-rerun, f32-math-rerun)
# ---------------------------------------------------------------------------

def _pass_blocks(src: dict):
    """[(pass, k)] present in a baseline, in order."""
    out = []
    for p in ("pos", "neg"):
        k = 0
        while f"{p}.b{k}.args.inputs" in src:
            out.append((p, k))
            k += 1
    return out


class Harness:
    """Reruns each component alone on tensors from a baseline (plan C.3, `f64-rerun`)."""

    def __init__(self, model: TimestralModel, fc: TimestralForecaster, dtype: torch.dtype, layers_dumped):
        self.model, self.fc, self.dtype = model, fc, dtype
        self.layers = layers_dumped

    def T(self, src, p, name):
        key = p + name
        if key not in src:
            return None
        a = src[key]
        if name in BOOL_NAMES:
            return torch.from_numpy(a.astype(bool))
        if name in LONG_NAMES:
            return torch.from_numpy(a.astype(np.int64))
        if a.dtype in (np.float32, np.float64):
            return torch.from_numpy(a).to(self.dtype)
        return torch.from_numpy(a)

    def run(self, src: dict) -> dict:
        global REC
        REC = Rec(stats=False)
        CTX.rerun = True
        CTX.in_refine = False
        cfg = self.model.config
        m = self.model
        sched = cfg.block_schedule
        with torch.inference_mode():
            for (p, k) in _pass_blocks(src):
                CTX.pass_name, CTX.block = p, k
                pre = f"{p}.b{k}."
                T = lambda n: self.T(src, pre, n)
                # scaler
                if T("scale.in") is not None:
                    with only(["scaled", "loc", "scale"]):
                        m.scale(T("scale.in"), T("scale.observed"))
                    if (pre + "known.rows") in src:
                        if bool(T("known.rows").any()):   # the model calls it only then
                            with only(["known.loc", "known.scale", "known.is_binary"]):
                                _model.instance_norm_stats(T("scale.in"), T("scale.observed"), cfg.minimum_scale)
                        with only(["known.scaled"]):
                            m._known_cov_scaling(T("scale.in"), T("scale.observed"), T("scaled"), T("loc"), T("scale"),
                                                 T("known.rows"))
                # patch embedding
                if T("patch.in") is not None:
                    with only(["patch.hidden_out", "patch.out"]):
                        m.representation.patch_proj(T("patch.in"))
                # transformer pieces
                positions = T("tf.positions")
                tmask = T("mask.time")
                vmask = T("mask.variate")
                if "tf.enc_len" in {n[len(pre):] for n in src if n.startswith(pre + "tf.enc_len")}:
                    tmask_obj = TwoStreamMask(mask=tmask, enc_len=int(src[pre + "tf.enc_len"][0]))
                else:
                    tmask_obj = tmask
                for l in sorted(self.layers):
                    if (pre + f"layer.{l}.in") not in src:
                        continue
                    layer = m.transformer.layers[l]
                    x = T(f"layer.{l}.in")
                    cos, sin = m.transformer.rope(x, positions)
                    with only([f"layer.{l}.out"]):
                        layer(x, tmask_obj, vmask, cos, sin)
                    axes = LAYER_KIND_AXES[sched[l]]
                    # sublayers, each on its dumped normed input
                    resid = src[pre + f"layer.{l}.in"]            # the stream in the source dtype
                    for j, is_time in enumerate(axes):
                        base = f"layer.{l}.attn.{j}."
                        nrm_in = torch.from_numpy(resid).to(self.dtype)
                        with only([base + "norm"]):
                            layer.attn_norms[j](nrm_in)
                        if (pre + base + "norm") in src:
                            xin = T(base + "norm")
                            with only(pred=lambda n, b=base: n.startswith(b) and n != b + "norm"):
                                layer.attns[j](xin, tmask_obj if is_time else vmask, cos, sin)
                        if (pre + base + "out") in src:
                            resid = resid + src[pre + base + "out"]
                    with only([f"layer.{l}.mlp_norm"]):
                        layer.mlp_norm(torch.from_numpy(resid).to(self.dtype))
                    if (pre + f"layer.{l}.mlp_norm") in src:
                        with only([f"layer.{l}.mlp"]):
                            layer.mlp(T(f"layer.{l}.mlp_norm"))
                last = len(sched) - 1
                if (pre + f"layer.{last}.out") in src:
                    with only(["out_norm"]):
                        m.transformer.out_norm(T(f"layer.{last}.out"))
                if T("head.in") is not None:
                    with only(["head.raw"]):
                        m.head.projection(T("head.in"))
                    raw = T("head.raw")
                    with only(["head.monotone"]):
                        REC.put("head.monotone", self._monotone(raw))
                # a whole block, from the dumped block arguments
                if T("args.inputs") is not None:
                    with only(["predict"]):
                        pred = m.predict_block(T("args.inputs"), T("args.observed"), T("args.group_ids"),
                                               int(src[pre + "args.horizon_patches"][0]), T("args.role_ids"),
                                               T("args.known_future"), T("args.known_future_mask"))
                    q_in = T("predict")
                    with only(["refined", "frames.loc", "frames.scale", "frames.in.sinh_median"]):
                        self.fc._refine_block(q_in, T("refine.context"), T("refine.mask"), T("refine.target_rows"))
            # pass-level: the cap
            for p in ("pos", "neg"):
                if f"{p}.cap.in" in src:
                    CTX.pass_name, CTX.block = p, None
                    pre = f"{p}."
                    with only(["cap.out", "cap.scale"]):
                        self.fc._cap_quantiles(self.T(src, pre, "cap.in"), self.T(src, pre, "cap.rows"),
                                               self.T(src, pre, "cap.mask"))
        CTX.rerun = False
        CTX.pass_name, CTX.block = None, None
        return REC.t

    def _monotone(self, raw):
        cfg = self.model.config
        preds = raw.reshape(raw.shape[0], raw.shape[1], cfg.patch_size, cfg.num_quantiles)
        preds = preds.to(torch.promote_types(preds.dtype, torch.float32))
        preds = monotone_quantiles(preds, cfg.quantile_knots, cfg.quantile_width_factor)
        return preds.reshape(raw.shape[0], raw.shape[1] * cfg.patch_size, cfg.num_quantiles)


def selfcheck(out: dict, ref: dict) -> tuple[float, str]:
    """Largest |rerun - original| / rms over the components. The rerun is the same computation
    on the same inputs, so the ratio must be at most 1e-12."""
    worst, where = 0.0, ""
    for k, a in out.items():
        if k not in ref:
            return float("inf"), f"{k}: not in the free-running file"
        b = ref[k]
        if not bio.is_float(a):
            continue
        d = bio.max_abs_diff(a, b)
        r = bio.rms(b)
        ratio = d / r if r > 0 else (0.0 if d == 0 else float("inf"))
        if ratio > worst:
            worst, where = ratio, k
    return worst, where


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def collect_kv(rec: Rec) -> dict:
    kv = dict(rec.kv)
    kv.update(rec.stat)
    for name, a in rec.t.items():
        if bio.is_float(a):
            kv[f"stat.{name}.rms"] = bio.rms(a)
            kv[f"stat.{name}.numel"] = int(a.size)
    for k, v in rec.calls.items():
        kv[f"calls.{k}"] = int(v)
    return kv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--tiny", action="store_true", help="use the tiny variant of each case (T/4, H/4)")
    ap.add_argument("--include-slow", action="store_true", help="also run cases labelled slow")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(args.cases).read_text())
    cases = spec["cases"]
    if args.case:
        cases = [c for c in cases if c["name"] in args.case]
        if len(cases) != len(set(args.case)):
            raise SystemExit("unknown case name")
    if args.tiny:
        cases = [c for c in cases if c.get("tiny")]
    if not args.include_slow and not args.case:
        cases = [c for c in cases if "slow" not in c.get("labels", [])]

    truth = args.mode in ("f64", "f64-rerun")
    model = build_model(args.model, args.revision, truth)
    cfg = model.config
    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True)
    install_instrumentation()
    hooks = register_hooks(model)
    fc = TimestralForecaster(model)
    base_kv = env_kv(model, args, args.mode, cfg_json)
    base_kv["env.sdpa_kernel"] = ("MATH (forced)" if "math" in args.mode else sdpa_kernel_seen(model))
    n_layers = len(cfg.block_schedule)

    math_ctx = (lambda: torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)) \
        if "math" in args.mode else contextlib.nullcontext

    for case0 in cases:
        t0 = time.time()
        case = scaled_case(case0, args.tiny)
        if args.tiny and case["name"] == "bench_32k":
            continue
        layers = case_layers(case, n_layers, args.layers, args.tiny)
        inputs = make_inputs(case, cfg.patch_size)
        path = bio.case_file(outdir, case["name"], args.mode)

        if args.mode in ("f32", "f32-math", "f64"):
            with math_ctx():
                r1 = run_forecast(fc, case, inputs, layers)
                r2 = run_forecast(fc, case, inputs, layers)
            diff = same_records(r1, r2)
            if diff:
                print(f"[{case['name']}] NOT REPRODUCIBLE: {diff}")
                return 1
            kv = dict(base_kv)
            kv.update(collect_kv(r1))
            kv["case.json"] = json.dumps(case, sort_keys=True)
            kv["case.name"] = case["name"]
            kv["case.B"], kv["case.V"], kv["case.T"], kv["case.H"] = case["B"], case["V"], case["T"], case["H"]
            kv["case.tiny"] = bool(args.tiny)
            kv["case.n_blocks"] = max([int(mm.group(1)) for mm in (re.match(r"pos\.b(\d+)\.", k) for k in r1.t) if mm] + [-1]) + 1
            kv["layers.dumped"] = ",".join(map(str, sorted(layers))) if layers else "all"
            kv["repro.runs"] = 2
            kv["repro.bitwise_equal"] = True
            if truth:
                kv["truth.patches"] = TRUTH_PATCHES
            bio.write_baseline(path, r1.t, kv)
        else:
            src_mode = "f32"
            src_path = bio.case_file(outdir, case["name"], src_mode)
            if not src_path.exists():
                print(f"[{case['name']}] missing {src_path.name}; run --mode f32 first")
                return 1
            src, src_kv = bio.read_baseline(src_path)
            if "expect.error.type" in src_kv and not any(k.endswith("args.inputs") for k in src):
                print(f"[{case['name']}] the reference raised; nothing to rerun")
                continue
            dumped = ({int(x) for x in src_kv["layers.dumped"].split(",")} if src_kv["layers.dumped"] != "all"
                      else set(range(n_layers)))
            h = Harness(model, fc, torch.float64 if truth else torch.float32, dumped)
            with math_ctx():
                o1 = h.run(src)
                o2 = h.run(src)
            for k in o1:
                if o1[k].tobytes() != o2[k].tobytes():
                    print(f"[{case['name']}] rerun NOT REPRODUCIBLE at {k}")
                    return 1
            kv = dict(base_kv)
            kv.update({k: v for k, v in src_kv.items() if k.startswith(("case.", "layers.", "model.", "env.hf"))})
            kv["rerun.source"] = src_mode
            kv["repro.bitwise_equal"] = True
            if truth:
                kv["truth.patches"] = TRUTH_PATCHES
                f64_path = bio.case_file(outdir, case["name"], "f64")
                if not f64_path.exists():
                    print(f"[{case['name']}] self-check needs {f64_path.name}; run --mode f64 first")
                    return 1
                s64, s64_kv = bio.read_baseline(f64_path)
                if "expect.error.type" in s64_kv and not any(k.endswith("args.inputs") for k in s64):
                    pass
                else:
                    own = Harness(model, fc, torch.float64, dumped).run(s64)
                    ratio, where = selfcheck(own, s64)
                    kv["selfcheck.max_ratio"] = ratio
                    kv["selfcheck.where"] = where
                    kv["selfcheck.limit"] = 1e-12
                    kv["selfcheck.ok"] = bool(ratio <= 1e-12)
                    if ratio > 1e-12:
                        print(f"[{case['name']}] RERUN SELF-CHECK FAILED: max ratio {ratio:.3e} at {where}")
                        return 1
            if not o1:
                kv["rerun.empty"] = True
                print(f"[{case['name']}] warning: no component was rerun")
                continue
            bio.write_baseline(path, o1, kv)
        sz = path.stat().st_size if path.exists() else 0
        print(f"[{case['name']}] {args.mode}: {path.name} {sz/1e6:.2f} MB in {time.time()-t0:.1f}s", flush=True)
    for h in hooks:
        h.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
