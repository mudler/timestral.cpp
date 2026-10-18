#!/usr/bin/env python3
"""Write a tiny random Timestral checkpoint for tests (docs/PLAN.md, M0b).

    make_tiny_checkpoint.py --output-dir DIR [--bias]

Writes config.json, model.safetensors and LICENSE. The weights are random numbers
(seed 0, normal with std 0.05) that carry no trained information. The schedule has one
layer of each kind, so every attention path is covered. With --bias, attn_bias and
mlp_bias are true, so the bias tensors exist.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from timestral.config import TimestralConfig
from timestral.model import TimestralModel

LICENSE_TEXT = """The weights in this directory are random numbers made by
scripts/make_tiny_checkpoint.py of the timestral.cpp project, for tests. They are
the project's own work and carry no trained information. They are not the weights
of any Timestral model.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bias", action="store_true")
    args = ap.parse_args()

    cfg = TimestralConfig(
        patch_size=8, d_model=32, num_heads=4, mlp_hidden_dim=64,
        block_schedule=("time", "variate", "time_variate", "variate_time"),
        quantile_knots=(0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95),
        block_size=4, context_length=2048, quantile_width_factor=3.0,
        attn_bias=args.bias, mlp_bias=args.bias,
    )
    torch.manual_seed(0)
    model = TimestralModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.05)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    sd = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    (out / "LICENSE").write_text(LICENSE_TEXT)
    print(f"wrote {out} ({len(sd)} tensors, bias={args.bias})")


if __name__ == "__main__":
    main()
