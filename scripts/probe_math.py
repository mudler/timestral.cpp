#!/usr/bin/env python3
"""Sample torch CPU float32 math against real libm; never change gate tolerances."""
import argparse
import ctypes
import ctypes.util
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_io import write_json

SEED = 20261020
MIN_N, MAX_N = 8, 10_000_000
RANGES = {'asinh': (-1e7, 1e7), 'sinh': (-80, 80),
          'sin': (-1100, 1100), 'cos': (-1100, 1100), 'softplus': (-100, 30)}


def validate_n(n):
    if type(n) is not int or not MIN_N <= n <= MAX_N:
        raise ValueError(f'--n must be an integer in [{MIN_N}, {MAX_N}]')
    return n


def samples(name, n):
    """Generate once as float32; see sampling metadata and docs/parity.md."""
    validate_n(n)
    rng = np.random.Generator(np.random.PCG64(SEED))
    if name == 'asinh':
        k = n // 2
        negative = k // 2
        positive = k - negative
        logs = np.concatenate((-np.geomspace(1e-30, 1e7, negative),
                               np.geomspace(1e-30, 1e7, positive)))
        uniform = rng.uniform(-10, 10, n - k)
        uniform[:3] = [-10, 0, 10]
        return np.concatenate((logs, uniform)).astype(np.float32)
    lo, hi = RANGES[name]
    x = rng.uniform(lo, hi, n).astype(np.float32)
    x[:3] = [lo, 0, hi]
    return x


class Libm:
    def __init__(self):
        name = ctypes.util.find_library('m')
        if not name:
            raise RuntimeError('libm lookup failed: find_library("m") returned no library')
        try:
            self.library = ctypes.CDLL(name)
            for symbol in ('asinhf', 'sinhf', 'sinf', 'cosf', 'expf', 'log1pf'):
                fn = getattr(self.library, symbol)
                fn.argtypes = [ctypes.c_float]
                fn.restype = ctypes.c_float
        except (OSError, AttributeError) as exc:
            raise RuntimeError('libm load failed or required float symbol is missing') from exc

    def evaluate(self, name, x):
        if name == 'softplus':
            expf, log1pf = self.library.expf, self.library.log1pf
            return np.fromiter((log1pf(expf(float(v))) for v in x), np.float32, count=x.size)
        fn = getattr(self.library, name + 'f')
        return np.fromiter((fn(float(v)) for v in x), np.float32, count=x.size)


def finite_ulp(a, b):
    """Finite float32 distance with both zeros at the same ordered position."""
    def ordered(x):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('finite_ulp requires finite float32 arrays')
        bits = x.view(np.uint32).astype(np.int64)
        magnitude = bits & 0x7fffffff
        return np.where(bits & 0x80000000, -magnitude, magnitude)
    return np.abs(ordered(a) - ordered(b))


def compare(inputs, actual, reference):
    inputs, actual, reference = (np.asarray(x) for x in (inputs, actual, reference))
    if any(x.dtype != np.float32 or x.ndim != 1 for x in (inputs, actual, reference)):
        raise ValueError('comparison requires one-dimensional float32 arrays')
    if not (inputs.shape == actual.shape == reference.shape):
        raise ValueError('comparison shapes differ')
    finite = np.isfinite(actual) & np.isfinite(reference)
    both_nan = np.isnan(actual) & np.isnan(reference)
    same_inf = np.isinf(actual) & (actual == reference)
    special_mismatch = ~(finite | both_nan | same_inf)
    indices = np.flatnonzero(finite)
    distances = finite_ulp(actual[finite], reference[finite])
    values, counts = np.unique(distances, return_counts=True)
    mismatch = special_mismatch.copy()
    mismatch[indices] = distances != 0

    def example(i):
        result = {'index': int(i)}
        for label, arr in [('input', inputs), ('torch', actual), ('libm', reference)]:
            v = arr[i]
            result[label] = float(v) if np.isfinite(v) else str(v)
            result[label + '_bits'] = f'0x{int(v.view(np.uint32)):08x}'
        result['finite_ulp'] = (int(finite_ulp(actual[i:i+1], reference[i:i+1])[0])
                                if finite[i] else None)
        return result

    return {
        'total_count': int(inputs.size),
        'finite_count': int(finite.sum()),
        'finite_histogram': {str(int(v)): int(c) for v, c in zip(values, counts)},
        'special_counts': {'both_nan': int(both_nan.sum()),
                           'same_infinity': int(same_inf.sum()),
                           'nonfinite_mismatch': int(special_mismatch.sum())},
        'mismatch_count': int(mismatch.sum()),
        'max_finite_ulp': int(distances.max()) if distances.size else None,
        'max_example': example(indices[int(distances.argmax())]) if distances.size else None,
        'mismatch_examples': [example(i) for i in np.flatnonzero(mismatch)[:8]],
    }


def run_probe(n):
    validate_n(n)
    torch.set_num_threads(1)
    libm = Libm()
    results = {}
    with torch.inference_mode():
        for name in RANGES:
            x = samples(name, n)
            tensor = torch.from_numpy(x)
            fn = F.softplus if name == 'softplus' else getattr(torch, name)
            actual = fn(tensor).numpy()
            results[name] = compare(x, actual, libm.evaluate(name, x))
    libc, version = platform.libc_ver()
    return {
        'version': 1, 'n_per_function': n,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
                        'cpu_architecture': platform.machine(),
                        'torch_cpu_capability': torch.backends.cpu.get_cpu_capability(),
                        'libc': libc, 'libc_version': version},
        'sampling': {'seed': SEED, 'rng': 'numpy.PCG64', 'dtype': 'float32',
                     'ranges': {k: list(v) for k, v in RANGES.items()},
                     'asinh': {'log_count': n // 2, 'negative_log_count': n // 4,
                               'positive_log_count': n // 2 - n // 4,
                               'log_magnitude_endpoints': [1e-30, 1e7],
                               'uniform_count': n - n // 2, 'uniform_range': [-10, 10],
                               'uniform_first_three': [-10, 0, 10]},
                     'others': 'uniform; replace first three with lower bound, zero, upper bound'},
        'comparison': {'zeros': '+0 equals -0', 'nan': 'both NaN counted as matching',
                       'histogram_contract': 'finite_histogram counts + special_counts sum to total_count',
                       'examples': 'first eight mismatches; first maximum finite distance',
                       'torch_softplus': 'default beta=1 threshold=20; x when x>20',
                       'libm_softplus': 'log1pf(expf(x)); float32 argument and result at each stage',
                       'scope': 'sampled evidence only; no universal ULP(0) claim or tolerance changes'},
        'functions': results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=1_000_000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate_n(args.n)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        write_json(args.out, run_probe(args.n))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'math probe failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
