"""Model-free M7 inputs and independent c2-truth-v1 future observations."""
import ast
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'c2-truth-v1'
# Accepted source at 65933ae; update only through explicit source/fixture review.
# No runtime override or hash-the-current-file acceptance path.
GENERATOR_SHA256 = '596a45ed8dd14faae0b04718d79308c5fc37af41fb2599256fdf5bd0c185425f'
REFERENCE_SHA = 'a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'
HF_REVISION = '9874d252ab0aae16f7f7c999f98f98d6ad7d8365'
OPTIONS = dict(symmetric_averaging=True, clip_nonnegative=True, max_context=32768,
               per_shot_horizon_patches=32, quantile_real_cap_k=10000.,
               extreme_value_threshold=1e30, patch_size=32, minimum_scale=1e-6,
               use_asinh=True, stabilize_with_global=False, observation_mask=None,
               refinement='native-per-block-bypass-le-one-patch')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def input_generator():
    """Compile ONLY the unchanged pure make_inputs definition, never module imports.

    The reference module imports torch and changes process-wide determinism settings.
    AST isolation executes no module top-level code and records the source hash.
    """
    source = (ROOT / 'scripts/gen_reference_baseline.py').read_text()
    if digest(source.encode()) != GENERATOR_SHA256:
        raise ValueError('unreviewed reference generator source')
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == 'make_inputs']
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise ValueError('expected one undecorated make_inputs function')
    namespace = dict(np=np, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<make_inputs>', 'exec'), namespace)
    return namespace['make_inputs']


def load_cases():
    """Return all 15 canonical and 64 frozen synthetic mappings, without generating data."""
    canonical = json.loads((ROOT / 'tests/cases.json').read_text())['cases']
    synthetic = json.loads((ROOT / 'docs/synthetic64-v1.cases.json').read_text())['cases']
    if len(canonical) != 15 or len(synthetic) != 64:
        raise ValueError('M7 requires 15 canonical and 64 synthetic items')
    cases = canonical + synthetic
    if len({c['name'] for c in cases}) != 79:
        raise ValueError('duplicate case names')
    return cases


def future_truth(case):
    """Replay only original target draws; keep future innovations in another namespace."""
    B, V, T, H = (case[k] for k in ('B', 'V', 'T', 'H'))
    if min(B, V, T, H) <= 0:
        raise ValueError('positive dimensions required')
    s = case.get('series', {})
    rng = np.random.default_rng(case['seed'])
    out = np.empty((B, V, H), np.float32)
    t = np.arange(T, T + H, dtype=np.float64)
    for b in range(B):
        for v in range(V):
            context = rng.normal(0., s.get('noise_std', 1.), T)
            future_rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([case['seed'], 7001, b, v])))
            future = future_rng.normal(0., s.get('noise_std', 1.), H)
            walk = np.cumsum(np.concatenate((context, future)))[T:]
            wave = 10. * np.sin(2. * math.pi * t /
                                (s.get('period', 24) * (1. + .25*v)) + .3*b)
            x = s.get('level', 0.) + s.get('trend', 0.)*t + wave + walk
            if s.get('positive'):
                x = np.abs(x)
            if s.get('constant') is not None:
                x[:] = s['constant']
            out[b, v] = x.astype(np.float32)
    return out


def array_hash(a):
    a = np.ascontiguousarray(a)
    return digest(str((a.dtype.str, a.shape)).encode() + a.tobytes())


def make_fixture(case):
    inputs = input_generator()(case, OPTIONS['patch_size'])
    before = {k: array_hash(v) for k, v in inputs.items()}
    truth = future_truth(case)
    if before != {k: array_hash(v) for k, v in inputs.items()}:
        raise ValueError('truth generation mutated inference inputs')
    files = ['scripts/gen_reference_baseline.py', 'docs/synthetic64-v1.cases.json',
             'tests/cases.json', 'scripts/quant_fixtures.py', 'scripts/quant_metrics.py']
    provenance = dict(version=VERSION, reference_sha=REFERENCE_SHA,
                      hf_revision=HF_REVISION, python=platform.python_version(),
                      numpy=np.__version__, rng='PCG64',
                      options=dict(OPTIONS), case_sha256=digest(json.dumps(
                          case, sort_keys=True, allow_nan=False).encode()),
                      source_sha256={p: digest((ROOT/p).read_bytes()) for p in files},
                      input_sha256=before, truth_sha256=array_hash(truth))
    return inputs, truth, provenance
