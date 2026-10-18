#!/usr/bin/env python3
"""Gate G0.1: every tensor in the GGUF equals the safetensors tensor byte for byte.

    check_convert.py --gguf FILE.gguf --model <hub id or dir> [--revision SHA]

The check is independent of the converter: it maps names with its own rules, undoes the
QKV row permutation (plan B.3) on the GGUF side, and compares raw bytes. It also checks
that the GGUF holds exactly the mapped tensors plus the three rope tensors, that the rope
tensors equal a fresh run of the reference class, and that the KV of the config equal
config.json. The G0.2 check on the loader's own view of the config is in
tests/test_model_loader.cpp.

Exit code 0 when every check passes, 1 otherwise.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf
from timestral.config import LAYER_KIND_AXES, TimestralConfig
from timestral.layers import SinCosRoPE

failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print(f"FAIL: {what}")
    return cond


def map_name(st_name: str):
    rules = [
        (r"representation\.patch_proj\.hidden_layer\.(weight|bias)", r"patch.hidden.\1"),
        (r"representation\.patch_proj\.output_layer\.(weight|bias)", r"patch.out.\1"),
        (r"representation\.patch_proj\.residual_layer\.(weight|bias)", r"patch.res.\1"),
        (r"representation\.forecast_query", "patch.forecast_query"),
        (r"transformer\.layers\.(\d+)\.attn_norms\.(\d+)\.scale", r"blk.\1.attn_norm.\2.weight"),
        (r"transformer\.layers\.(\d+)\.attns\.(\d+)\.wQKV\.(weight|bias)", r"blk.\1.attn.\2.qkv.\3"),
        (r"transformer\.layers\.(\d+)\.attns\.(\d+)\._gate\.(weight|bias)", r"blk.\1.attn.\2.gate.\3"),
        (r"transformer\.layers\.(\d+)\.attns\.(\d+)\.wO\.(weight|bias)", r"blk.\1.attn.\2.out.\3"),
        (r"transformer\.layers\.(\d+)\.mlp_norm\.scale", r"blk.\1.ffn_norm.weight"),
        (r"transformer\.layers\.(\d+)\.mlp\.w12\.(weight|bias)", r"blk.\1.ffn_w12.\2"),
        (r"transformer\.layers\.(\d+)\.mlp\.w3\.(weight|bias)", r"blk.\1.ffn_w3.\2"),
        (r"transformer\.out_norm\.scale", "out_norm.weight"),
        (r"head\.projection\.hidden_layer\.weight", "head.hidden.weight"),
        (r"head\.projection\.output_layer\.weight", "head.out.weight"),
        (r"head\.projection\.residual_layer\.weight", "head.res.weight"),
    ]
    for pat, rep in rules:
        if re.fullmatch(pat, st_name):
            return re.sub(pat, rep, st_name)
    return None


def undo_qkv_permutation(a: np.ndarray, hd: int, H: int) -> np.ndarray:
    """GGUF row s*D + h*hd + d  ->  torch row s*D + d*H + h."""
    if a.ndim == 2:
        return np.ascontiguousarray(a.reshape(3, H, hd, a.shape[1]).transpose(0, 2, 1, 3).reshape(a.shape))
    return np.ascontiguousarray(a.reshape(3, H, hd).transpose(0, 2, 1).reshape(a.shape))


def kv(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    return f.contents()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    p = Path(args.model).expanduser()
    if p.is_dir():
        d = p
    else:
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(args.model, revision=args.revision,
                                   allow_patterns=["config.json", "model.safetensors", "LICENSE"]))
    cfg = TimestralConfig.load(d / "config.json")
    hd, H = cfg.head_dim, cfg.num_heads

    r = gguf.GGUFReader(args.gguf)
    tensors = {t.name: t for t in r.tensors}

    # --- G0.1: bytes ---
    seen = set()
    n_cmp = 0
    with safe_open(str(d / "model.safetensors"), framework="np") as sf:
        for st_name in sf.keys():
            gg_name = map_name(st_name)
            if not check(gg_name is not None, f"no mapping for safetensors tensor {st_name}"):
                continue
            if not check(gg_name in tensors, f"{gg_name} (from {st_name}) missing in the GGUF"):
                continue
            check(gg_name not in seen, f"{gg_name} mapped twice")
            seen.add(gg_name)
            src = sf.get_tensor(st_name)
            t = tensors[gg_name]
            check(t.tensor_type == gguf.GGMLQuantizationType.F32, f"{gg_name}: not F32")
            check(src.dtype == np.float32, f"{st_name}: not float32")
            a = np.asarray(t.data).reshape(tuple(t.shape[::-1]))  # gguf-py: t.shape is ne
            if gg_name.endswith(("qkv.weight", "qkv.bias")):
                a = undo_qkv_permutation(a, hd, H)
            ok = a.shape == src.shape and a.tobytes() == np.ascontiguousarray(src).tobytes()
            check(ok, f"{gg_name}: bytes differ from {st_name}")
            n_cmp += 1
    extra = set(tensors) - seen - {"rope.inv_freq", "rope.cos", "rope.sin"}
    check(not extra, f"GGUF has tensors that are not in the safetensors file: {sorted(extra)[:5]}")
    check({"rope.inv_freq", "rope.cos", "rope.sin"} <= set(tensors), "rope tensors missing")
    print(f"G0.1: {n_cmp} tensors compared byte for byte against safetensors")

    # --- rope tables vs a fresh run of the reference class ---
    torch.set_num_threads(1)
    pos_min = kv(r, "timestral.rope.table_pos_min")
    pos_min = int(np.asarray(pos_min).reshape(-1)[0])
    check(pos_min == -(cfg.context_length // cfg.patch_size + cfg.block_size), "table_pos_min")
    rope = SinCosRoPE(dim=hd, base=cfg.rope_theta, rotary_dim=cfg.rope_dim)
    rope.rebuild_inv_freq()
    inv = rope.inv_freq.numpy()
    check(np.asarray(tensors["rope.inv_freq"].data).tobytes() == inv.tobytes(), "rope.inv_freq differs from the reference")
    P = -2 * pos_min + 1
    pos = torch.arange(pos_min, -pos_min + 1, dtype=torch.int64)
    x = torch.zeros(1, P, hd)
    cos, sin = rope(x, pos[None, :])
    sc = np.asarray(tensors["rope.cos"].data).reshape(P, hd)
    ss = np.asarray(tensors["rope.sin"].data).reshape(P, hd)
    check(sc.tobytes() == cos[0].numpy().tobytes(), "rope.cos differs from a fresh reference run")
    check(ss.tobytes() == sin[0].numpy().tobytes(), "rope.sin differs from a fresh reference run")
    # Preview of G0.3 (the full gate, against the tables of the dumps, is in M0d): run the
    # reference at run-time-like layouts (R rows, N patches, positions p - first_obs) and
    # compare each value with the stored table row of its position.
    def ukey(a):
        b = np.ascontiguousarray(a).view(np.int32).astype(np.int64)
        return np.where(b < 0, -(b & 0x7FFFFFFF), b)
    rng = np.random.default_rng(0)
    tot = nbad = maxulp = nbits = 0
    max_n = min(cfg.context_length // cfg.patch_size + cfg.block_size, 1056)
    for R in (1, 2, 3, 6, 17):
        for N in sorted({1, 2, 3, 5, 8, 33, 100, max_n}):
            for _ in range(3):
                first = rng.integers(0, max(1, cfg.context_length // cfg.patch_size - 1), size=(R, 1))
                pp = torch.tensor(np.arange(N)[None, :] - first)
                pp = pp.clamp(min=pos_min, max=-pos_min)
                c2, s2 = rope(torch.zeros(R, N, hd), pp)
                idx = (pp - pos_min).numpy()
                for got, tab in ((c2.numpy(), sc), (s2.numpy(), ss)):
                    dd = np.abs(ukey(got) - ukey(tab[idx]))
                    tot += dd.size
                    nbad += int((dd > 0).sum())
                    nbits += int((np.ascontiguousarray(got).view(np.uint32) != tab[idx].view(np.uint32)).sum())
                    maxulp = max(maxulp, int(dd.max()))
    print(f"G0.3 preview: run-time-like layouts differ from the stored table in {nbad} of {tot} "
          f"values (max {maxulp} ulp); {nbits - nbad} more differ only in the sign of a zero" + ("" if nbad == 0 else
          "  (a mismatch decides G2.2: ULP(1) at these positions, plan A.5.4 P10)"))
    print(f"rope tables: {P} positions from {pos_min}")

    # --- KV vs config.json (python side; G0.2 proper is in test_model_loader) ---
    def eq(key, val):
        got = kv(r, key)
        got = got.tolist() if hasattr(got, "tolist") else got
        if isinstance(got, list) and len(got) == 1 and not isinstance(val, (list, tuple)):
            got = got[0]
        check(got == val, f"KV {key}: {got!r} != {val!r}")
    eq("timestral.patch_size", cfg.patch_size)
    eq("timestral.embedding_length", cfg.d_model)
    eq("timestral.attention.head_count", cfg.num_heads)
    eq("timestral.feed_forward_length", cfg.mlp_hidden_dim)
    eq("timestral.block_count", len(cfg.block_schedule))
    eq("timestral.quantile_knots", list(cfg.quantile_knots))
    eq("timestral.block_size", cfg.block_size)
    eq("timestral.context_length", cfg.context_length)
    eq("timestral.rope.freq_base", cfg.rope_theta)
    eq("timestral.rope.partial_factor", cfg.rope_partial_factor)
    eq("timestral.rope.dimension_count", cfg.rope_dim)
    eq("timestral.attention.bias", cfg.attn_bias)
    eq("timestral.mlp.bias", cfg.mlp_bias)
    eq("timestral.scaler.minimum_scale", cfg.minimum_scale)
    eq("timestral.quantile_width_factor", cfg.quantile_width_factor)
    sched = kv(r, "timestral.block_schedule")
    check(list(sched) == list(cfg.block_schedule), "KV block_schedule")

    if failures:
        print(f"G0.1: FAIL ({len(failures)} problems)")
        return 1
    print("G0.1: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
