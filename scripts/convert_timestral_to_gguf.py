#!/usr/bin/env python3
"""Convert a Timestral checkpoint (config.json + model.safetensors) to one GGUF file.

Writes the schema of docs/PLAN.md, sections B.1 to B.3. The GGUF is self-contained: every
setting is a KV, and the rope tables come from the reference class. Values are never
changed at f32, except for the exact row permutation of the fused QKV projection (B.3).
The f16 and q8_0 outputs reduce only the projection weights in the B.4 allowlist.

    convert_timestral_to_gguf.py --model <hub id or dir> [--revision SHA] \
        --outtype f32|f16|q8_0 --output OUT.gguf [--license-file FILE]

Next to OUT.gguf the script also writes OUT.config.json, a copy of the config.json that
was converted. Only tests read it (gate G0.2); the GGUF does not need it.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf
from gguf import GGUFValueType as VT

from timestral.config import LAYER_KIND_AXES, TimestralConfig
from timestral.layers import SinCosRoPE

CONVERTER_VERSION = 1
CODE_SHA = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
HF_REPO_PREFIX = "https://huggingface.co/"
DEFAULT_FORECASTER = dict(per_shot_horizon_patches=32, quantile_real_cap_k=1e4,
                          extreme_value_threshold=1e30, symmetric_averaging=True,
                          clip_nonnegative=True)
TARGET_ENCODING_SMOOTH = 1.0


def resolve(model: str, revision: str | None):
    """Return (directory, hub_id or None, resolved revision or None)."""
    p = Path(model).expanduser()
    if p.is_dir():
        return p, None, None
    from huggingface_hub import snapshot_download
    d = snapshot_download(model, revision=revision,
                          allow_patterns=["config.json", "model.safetensors", "LICENSE"])
    return Path(d), model, Path(d).name  # the snapshot directory is named by the commit


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def check_no_shared_storage(header: dict) -> None:
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k)
                   for k, v in header.items() if k != "__metadata__")
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 < a1:
            raise SystemExit(f"tensors {an!r} and {bn!r} share storage")
    for s, e, n in spans:
        if e <= s:
            raise SystemExit(f"tensor {n!r} has an empty or negative byte range")


def expected_tensors(c: TimestralConfig):
    """Return [(safetensors name, gguf name, torch shape)] for the config, in write order."""
    D, F, ps, Q = c.d_model, c.mlp_hidden_dim, c.patch_size, c.num_quantiles
    out = []
    p = "representation.patch_proj."
    out += [(p + "hidden_layer.weight", "patch.hidden.weight", (4 * D, 2 * ps)),
            (p + "hidden_layer.bias", "patch.hidden.bias", (4 * D,)),
            (p + "output_layer.weight", "patch.out.weight", (D, 4 * D)),
            (p + "output_layer.bias", "patch.out.bias", (D,)),
            (p + "residual_layer.weight", "patch.res.weight", (D, 2 * ps)),
            (p + "residual_layer.bias", "patch.res.bias", (D,)),
            ("representation.forecast_query", "patch.forecast_query", (D,))]
    for i, kind in enumerate(c.block_schedule):
        for j, is_time in enumerate(LAYER_KIND_AXES[kind]):
            s = f"transformer.layers.{i}.attns.{j}."
            g = f"blk.{i}.attn.{j}."
            out.append((f"transformer.layers.{i}.attn_norms.{j}.scale", f"blk.{i}.attn_norm.{j}.weight", (D,)))
            out.append((s + "wQKV.weight", g + "qkv.weight", (3 * D, D)))
            if c.attn_bias:
                out.append((s + "wQKV.bias", g + "qkv.bias", (3 * D,)))
            gate = c.time_attn_gate if is_time else c.variate_attn_gate
            if gate == "sdpa_sigmoid":
                out.append((s + "_gate.weight", g + "gate.weight", (D, D)))
                if c.attn_bias:
                    out.append((s + "_gate.bias", g + "gate.bias", (D,)))
            out.append((s + "wO.weight", g + "out.weight", (D, D)))
            if c.attn_bias:
                out.append((s + "wO.bias", g + "out.bias", (D,)))
        m = f"transformer.layers.{i}."
        out.append((m + "mlp_norm.scale", f"blk.{i}.ffn_norm.weight", (D,)))
        out.append((m + "mlp.w12.weight", f"blk.{i}.ffn_w12.weight", (2 * F, D)))
        if c.mlp_bias:
            out.append((m + "mlp.w12.bias", f"blk.{i}.ffn_w12.bias", (2 * F,)))
        out.append((m + "mlp.w3.weight", f"blk.{i}.ffn_w3.weight", (D, F)))
        if c.mlp_bias:
            out.append((m + "mlp.w3.bias", f"blk.{i}.ffn_w3.bias", (D,)))
    out += [("transformer.out_norm.scale", "out_norm.weight", (D,)),
            ("head.projection.hidden_layer.weight", "head.hidden.weight", (4 * D, D)),
            ("head.projection.output_layer.weight", "head.out.weight", (ps * Q, 4 * D)),
            ("head.projection.residual_layer.weight", "head.res.weight", (ps * Q, D))]
    return out


def reduced_weight_name(name: str) -> bool:
    return name in {'patch.out.weight', 'head.hidden.weight', 'head.out.weight', 'head.res.weight'} or bool(
        re.fullmatch(r'blk\.[0-9]+\.(?:attn\.[0-9]+\.(?:qkv|gate|out)|ffn_w12|ffn_w3)\.weight', name))


def storage_tensor(name, a, c, outtype):
    if outtype not in ('f32', 'f16', 'q8_0'):
        raise ValueError('unsupported output type')
    if name.endswith(('qkv.weight', 'qkv.bias')):
        a = permute_qkv(a, c)
    if outtype == 'f16' and reduced_weight_name(name):
        return np.ascontiguousarray(a, dtype=np.float16), gguf.GGMLQuantizationType.F16
    if outtype == 'q8_0' and reduced_weight_name(name):
        if a.dtype != np.float32 or a.ndim != 2 or a.shape[-1] <= 0 or a.shape[-1] % 32:
            raise ValueError('Q8_0 requires F32 matrix rows divisible by 32')
        if not np.all(np.isfinite(a)):
            raise ValueError('Q8_0 requires finite source weights')
        from gguf.quants import quantize
        return np.ascontiguousarray(quantize(np.ascontiguousarray(a), gguf.GGMLQuantizationType.Q8_0)), gguf.GGMLQuantizationType.Q8_0
    return np.ascontiguousarray(a), gguf.GGMLQuantizationType.F32


def permute_qkv(a: np.ndarray, c: TimestralConfig) -> np.ndarray:
    """B.3: torch row o = s*D + d*H + h  ->  GGUF row s*D + h*head_dim + d."""
    hd, H = c.head_dim, c.num_heads
    if a.ndim == 2:
        return np.ascontiguousarray(a.reshape(3, hd, H, a.shape[1]).transpose(0, 2, 1, 3).reshape(a.shape))
    return np.ascontiguousarray(a.reshape(3, hd, H).transpose(0, 2, 1).reshape(a.shape))


def rope_tables(c: TimestralConfig):
    """inv_freq, cos, sin from the reference class (plan P9, P10)."""
    torch.set_num_threads(1)
    if c.context_length % c.patch_size:
        raise SystemExit("context_length must be a multiple of patch_size")
    pos_min = -(c.context_length // c.patch_size + c.block_size)
    pos_max = -pos_min
    rope = SinCosRoPE(dim=c.head_dim, base=c.rope_theta, rotary_dim=c.rope_dim)
    rope.rebuild_inv_freq()  # as loading.load_model does
    positions = torch.arange(pos_min, pos_max + 1, dtype=torch.int64)[None, :]
    x = torch.zeros(1, positions.shape[1], c.head_dim, dtype=torch.float32)
    cos, sin = rope(x, positions)
    assert cos.dtype == torch.float32 and cos.shape == (1, positions.shape[1], c.head_dim)
    return (pos_min, rope.inv_freq.detach().numpy().copy(),
            cos[0].numpy().copy(), sin[0].numpy().copy())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hub id or local directory")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--outtype", default="f32", choices=["f32", "f16", "q8_0"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--license-file", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    d, hub_id, snap = resolve(args.model, args.revision)
    cfg_path, w_path = d / "config.json", d / "model.safetensors"
    raw_cfg = json.loads(cfg_path.read_text())
    c = TimestralConfig.from_dict(raw_cfg)  # rejects unknown keys, runs __post_init__
    if c.stabilize_with_global:
        raise SystemExit("stabilize_with_global=true is not supported")

    lic_path = Path(args.license_file) if args.license_file else d / "LICENSE"
    if not lic_path.is_file():
        raise SystemExit("no LICENSE in the model directory; pass --license-file")
    license_text = lic_path.read_text()

    header = safetensors_header(w_path)
    check_no_shared_storage(header)
    spec = expected_tensors(c)
    st_names = {k for k in header if k != "__metadata__"}
    want = [s for s, _, _ in spec]
    if len(set(want)) != len(want):
        raise SystemExit("internal error: duplicate expected tensor")
    if st_names != set(want):
        raise SystemExit(f"safetensors and config disagree: missing {sorted(set(want) - st_names)[:5]}, "
                         f"unused {sorted(st_names - set(want))[:5]}")

    out = Path(args.output)
    w = gguf.GGUFWriter(str(out), "timestral")
    S = lambda k, v: w.add_key_value(k, v, VT.STRING)
    U = lambda k, v: w.add_key_value(k, int(v), VT.UINT32)
    F64 = lambda k, v: w.add_key_value(k, float(v), VT.FLOAT64)
    B = lambda k, v: w.add_key_value(k, bool(v), VT.BOOL)

    # A fixed name for a local directory, so the output does not depend on the directory name.
    name = args.name or (hub_id.split("/")[-1] if hub_id else "Timestral")
    S("general.name", name)
    S("general.license", license_text)
    S("general.license.name", "Modified MIT")
    lic_rev = args.revision or snap
    S("general.license.link", f"{HF_REPO_PREFIX}{hub_id}/blob/{lic_rev}/LICENSE" if hub_id else "")
    S("general.source.url", f"{HF_REPO_PREFIX}{hub_id}" if hub_id else "")
    S("timestral.source.hf_revision", args.revision or snap or "")
    S("timestral.source.code_sha", CODE_SHA)
    S("timestral.source.weights_sha256", sha256_of(w_path))
    U("timestral.converter_version", CONVERTER_VERSION)
    if args.outtype != "f32":
        S("timestral.conversion.outtype", args.outtype)
        S("timestral.conversion.type_policy", "b4-q8_0-v1" if args.outtype == "q8_0" else "b4-f16-v1")
        S("timestral.conversion.converter_sha256", sha256_of(Path(__file__)))
    U("timestral.patch_size", c.patch_size)
    U("timestral.embedding_length", c.d_model)
    U("timestral.attention.head_count", c.num_heads)
    U("timestral.feed_forward_length", c.mlp_hidden_dim)
    U("timestral.block_count", len(c.block_schedule))
    w.add_key_value("timestral.block_schedule", list(c.block_schedule), VT.ARRAY, VT.STRING)
    w.add_key_value("timestral.quantile_knots", [float(q) for q in c.quantile_knots], VT.ARRAY, VT.FLOAT64)
    U("timestral.block_size", c.block_size)
    U("timestral.context_length", c.context_length)
    F64("timestral.rope.freq_base", c.rope_theta)
    F64("timestral.rope.partial_factor", c.rope_partial_factor)
    U("timestral.rope.dimension_count", c.rope_dim)
    pos_min, inv_freq, cos, sin = rope_tables(c)
    w.add_key_value("timestral.rope.table_pos_min", int(pos_min), VT.INT32)
    w.add_key_value("timestral.attention.layer_norm_rms_epsilon", float(np.float32(c.norm_eps)), VT.FLOAT32)
    B("timestral.attention.bias", c.attn_bias)
    B("timestral.mlp.bias", c.mlp_bias)
    S("timestral.attention.time_gate", c.time_attn_gate)
    S("timestral.attention.variate_gate", c.variate_attn_gate)
    B("timestral.attention.split_context_horizon", c.split_context_horizon_attn)
    B("timestral.attention.qkv_head_major", True)
    F64("timestral.scaler.minimum_scale", c.minimum_scale)
    B("timestral.scaler.use_asinh", c.use_asinh)
    B("timestral.scaler.stabilize_with_global", c.stabilize_with_global)
    B("timestral.covariate_enabled", c.covariate_enabled)
    B("timestral.attend_missing_patches", c.attend_missing_patches)
    F64("timestral.quantile_width_factor", c.quantile_width_factor)
    f = DEFAULT_FORECASTER
    U("timestral.forecaster.per_shot_horizon_patches", f["per_shot_horizon_patches"])
    F64("timestral.forecaster.quantile_real_cap_k", f["quantile_real_cap_k"])
    F64("timestral.forecaster.extreme_value_threshold", f["extreme_value_threshold"])
    B("timestral.forecaster.symmetric_averaging", f["symmetric_averaging"])
    B("timestral.forecaster.clip_nonnegative", f["clip_nonnegative"])
    F64("timestral.target_encoding.smooth", TARGET_ENCODING_SMOOTH)

    n_written = 0
    with safe_open(str(w_path), framework="np") as sf:
        for st_name, gg_name, shape in spec:
            a = sf.get_tensor(st_name)
            if a.dtype != np.float32:
                raise SystemExit(f"{st_name}: dtype {a.dtype}, expected float32")
            if tuple(a.shape) != shape:
                raise SystemExit(f"{st_name}: shape {tuple(a.shape)}, expected {shape} from the config")
            a, storage_type = storage_tensor(gg_name, a, c, args.outtype)
            w.add_tensor(gg_name, a, raw_dtype=storage_type)
            n_written += 1
    w.add_tensor("rope.inv_freq", np.ascontiguousarray(inv_freq), raw_dtype=gguf.GGMLQuantizationType.F32)
    w.add_tensor("rope.cos", np.ascontiguousarray(cos), raw_dtype=gguf.GGMLQuantizationType.F32)
    w.add_tensor("rope.sin", np.ascontiguousarray(sin), raw_dtype=gguf.GGMLQuantizationType.F32)
    assert n_written == len(st_names)  # every safetensors tensor used exactly once

    out.parent.mkdir(parents=True, exist_ok=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    shutil.copyfile(cfg_path, out.with_suffix(".config.json"))
    print(f"wrote {out} ({n_written + 3} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
