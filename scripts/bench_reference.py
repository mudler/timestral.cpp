#!/usr/bin/env python3
"""CPU-only, uninstrumented shipped-reference forecast benchmark (M6c)."""
import argparse
import json
import platform
import resource
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    for name, default in (("context-len", 32768), ("horizon", 1024),
                          ("variates", 1), ("threads", 1), ("repeat", 5), ("warmup", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    for name in ("context_len", "horizon", "variates", "threads", "repeat", "warmup"):
        if getattr(args, name) < (0 if name == "warmup" else 1):
            parser.error(f"invalid {name}")

    import torch
    from timestral.forecaster import TimestralForecaster
    from timestral.loading import load_model

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    model = load_model(args.model, revision=args.revision, device="cpu", dtype=torch.float32)
    model.eval()
    fc = TimestralForecaster(model)
    i = torch.arange(args.context_len, dtype=torch.int64)
    v = torch.arange(args.variates, dtype=torch.int64)[:, None]
    context = (((i * 17 + v * 13) % 257 - 128).float() / 64).unsqueeze(0)
    samples = []
    checksum = None
    with torch.inference_mode():
        for iteration in range(args.warmup + args.repeat):
            start = time.perf_counter()
            result = fc.forecast(context, args.horizon)
            elapsed = time.perf_counter() - start
            values = torch.as_tensor(result.quantiles)
            if not torch.isfinite(values).all():
                raise RuntimeError("nonfinite output")
            checksum = values.double().sum().item()
            if iteration >= args.warmup:
                samples.append(elapsed)
            print(f"{'warmup' if iteration < args.warmup else 'sample'} {iteration} seconds={elapsed:.9f}",
                  file=sys.stderr, flush=True)
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as stream:
            cpu = next(line.split(":", 1)[1].strip() for line in stream if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(backend="reference-cpu", model=args.model, revision=args.revision,
                          cpu=cpu, torch_version=torch.__version__, context_len=args.context_len,
                          horizon=args.horizon, variates=args.variates, threads=args.threads,
                          repeat=args.repeat, warmup=args.warmup, input="dyadic-v1",
                          input_sum=context.double().sum().item(), output_sum=checksum,
                          median_seconds=statistics.median(samples),
                          peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
                          samples_seconds=samples)))


if __name__ == "__main__":
    main()
