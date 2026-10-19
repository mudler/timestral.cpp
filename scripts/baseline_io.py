"""Read and write baseline GGUF files (docs/PLAN.md, section C.4).

Shared by gen_reference_baseline.py, compute_floors.py and the checks in tests/python.
Tensors are numpy arrays in torch (row-major) shape: GGUF stores the shape reversed, and
gguf-py does the reversal. KV values are python values.
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf import GGUFValueType as VT

ARCH = "timestral-baseline"
MAX_TENSOR_NAME = 63   # GGML_MAX_NAME is 64 including the terminator


def case_file(directory, case: str, mode: str) -> Path:
    return Path(directory) / f"baseline-{case}-{mode}.gguf"


def _field_value(field):
    """Python value of a reader field (scalar, str, bool or list)."""
    main = field.types[0]
    if main == VT.STRING:
        return str(field.parts[field.data[0]].tobytes(), encoding="utf-8")
    if main == VT.ARRAY:
        sub = field.types[-1]
        if sub == VT.STRING:
            return [str(field.parts[i].tobytes(), encoding="utf-8") for i in field.data]
        return [x.item() for x in (field.parts[i][0] for i in field.data)]
    v = field.parts[field.data[0]][0]
    return v.item() if hasattr(v, "item") else v


def read_baseline(path) -> tuple[dict, dict]:
    """Return (tensors, kv). Tensors are copied into memory."""
    r = gguf.GGUFReader(str(path))
    kv = {k: _field_value(f) for k, f in r.fields.items()
          if k not in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count")}
    tensors = {}
    for t in r.tensors:
        shape = tuple(int(x) for x in t.shape[::-1])
        tensors[t.name] = np.array(t.data, copy=True).reshape(shape)
    return tensors, kv


def _kv_add(w: gguf.GGUFWriter, key: str, v) -> None:
    if isinstance(v, bool):
        w.add_key_value(key, v, VT.BOOL)
    elif isinstance(v, str):
        w.add_key_value(key, v, VT.STRING)
    elif isinstance(v, (int, np.integer)):
        v = int(v)
        if 0 <= v < 2**32:
            w.add_key_value(key, v, VT.UINT32)
        elif -2**63 <= v < 2**63:
            w.add_key_value(key, v, VT.INT64)
        else:
            w.add_key_value(key, v, VT.UINT64)
    elif isinstance(v, (float, np.floating)):
        w.add_key_value(key, float(v), VT.FLOAT64)
    elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
        w.add_key_value(key, list(v), VT.ARRAY, VT.STRING)
    elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
        w.add_key_value(key, [float(x) for x in v], VT.ARRAY, VT.FLOAT64)
    else:
        raise TypeError(f"unsupported KV {key!r}: {type(v)}")


def storage_array(a: np.ndarray) -> tuple[np.ndarray, Q]:
    """numpy array -> (array, ggml type). bool -> I8, int64 -> I32 (range checked)."""
    a = np.ascontiguousarray(a)
    if a.dtype == np.bool_:
        return a.astype(np.int8), Q.I8
    if a.dtype == np.int8:
        return a, Q.I8
    if a.dtype in (np.int32,):
        return a, Q.I32
    if a.dtype == np.int64:
        if a.size and (a.min() < -2**31 or a.max() >= 2**31):
            raise ValueError("int64 value does not fit in I32")
        return a.astype(np.int32), Q.I32
    if a.dtype == np.float32:
        return a, Q.F32
    if a.dtype == np.float64:
        return a, Q.F64
    raise TypeError(f"unsupported dtype {a.dtype}")


@contextmanager
def atomic_path(path):
    """Publish one complete sibling file; no directory-wide transaction is implied."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.pending', dir=path.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_path(path) as temporary:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())


def write_baseline(path, tensors: dict, kv: dict) -> None:
    with atomic_path(path) as temporary:
        _write_baseline(temporary, tensors, kv)


def _write_baseline(path, tensors, kv):
    w = gguf.GGUFWriter(str(path), ARCH)
    for k, v in kv.items():
        if k != "general.architecture":      # the writer sets it
            _kv_add(w, k, v)
    for name, a in tensors.items():
        if len(name.encode()) > MAX_TENSOR_NAME:
            raise ValueError(f"tensor name too long: {name}")
        if a.size == 0:
            raise ValueError(f"empty tensor {name}")
        arr, qt = storage_array(a)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim > 4:
            raise ValueError(f"{name}: more than 4 dimensions {arr.shape}")
        w.add_tensor(name, arr, raw_dtype=qt)
    try:
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
    finally:
        w.close()


def is_float(a: np.ndarray) -> bool:
    return a.dtype in (np.float32, np.float64)


def rms(a: np.ndarray) -> float:
    """Root mean square over the finite elements, in float64."""
    x = a.astype(np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| over positions where both are finite; inf if the non-finite parts disagree."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return float("inf")
    nn = ~fa
    if nn.any() and not np.array_equal(a[nn], b[nn], equal_nan=True):
        return float("inf")
    m = fa
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def load_cases(path) -> dict:
    return json.loads(Path(path).read_text())
