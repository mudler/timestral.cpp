"""Strict tsqbin-v1 codec only; no execution, NPZ writer, pickle, or truth input."""
import hashlib
import re
import struct
import numpy as np

LIMIT = 1 << 30
HF = '9874d252ab0aae16f7f7c999f98f98d6ad7d8365'
REFERENCE = 'a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'
NAMES = ('context', 'mask', 'past', 'known', 'cat_past', 'cat_known')
DTYPES = ('<f4', '|u1', '<f4', '<f4', '<f8', '<f8')
TYPES = (1, 2, 1, 1, 3, 3)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(s):
    require(isinstance(s, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,256}', s), 'invalid case ID')
    return s.encode('ascii')


def sha(s):
    require(isinstance(s, str) and re.fullmatch('[0-9a-f]{64}', s), 'invalid SHA256')
    return bytes.fromhex(s)


def encode_request(case_id, shape, arrays, model_sha256, input_sha256):
    """shape=(B,V,T,H,P,K,CP,CK); arrays contain inference inputs only."""
    name = identity(case_id)
    require(len(shape) == 8 and all(type(x) is int and 0 <= x < 2**64 for x in shape), 'invalid shape')
    B, V, T, H, P, K, CP, CK = shape
    require(min(B, V, T, H) > 0, 'positive dimensions required')
    counts = (B*V*T, 0, B*P*T, B*K*(T+H), B*CP*T, B*CK*(T+H))
    require(set(arrays) == {n for n, c in zip(NAMES, counts) if c}, 'unexpected or missing arrays')
    payloads = []
    for n, count, dt in zip(NAMES, counts, DTYPES):
        require(count <= LIMIT // np.dtype(dt).itemsize, 'array exceeds limit')
        a = np.asarray(arrays[n]) if count else np.empty(0, dtype=dt)
        require(a.dtype == np.dtype(dt) and a.size == count and a.flags.c_contiguous, 'array dtype/count/layout')
        payloads.append(a.tobytes())
    offset = 288 + len(name) + 192
    total = offset + sum(map(len, payloads))
    require(total <= LIMIT, 'request exceeds limit')
    header = bytearray(288)
    struct.pack_into('<8sIIQ8Q3QII3d4I', header, 0, b'TSQREQ01', 1, offset, total,
                     *shape, 32768, 32, 32, 3, 0, 10000., 1e30, 1e-6, 1, 0, len(name), 0)
    header[160:192] = sha(model_sha256)
    header[192:224] = hashlib.sha256(HF.encode()).digest()
    header[224:256] = hashlib.sha256(REFERENCE.encode()).digest()
    header[256:288] = sha(input_sha256)
    desc = bytearray()
    for typ, count, payload in zip(TYPES, counts, payloads):
        desc += struct.pack('<IIQQQ', typ, 0, count, len(payload), offset if count else 0)
        offset += len(payload)
    return bytes(header) + name + desc + b''.join(payloads)


def decode_result(data):
    require(256 <= len(data) <= LIMIT, 'result size')
    magic, version, status, total = struct.unpack_from('<8sIIQ', data)
    require(magic == b'TSQRES01' and version == 1 and status in (0, 1) and total == len(data), 'result header')
    B, V, H, Q = struct.unpack_from('<4Q', data, 24)
    nc, nd, offset, reserved1, reserved2 = struct.unpack_from('<IIQQQ', data, 224)
    require(nc <= 256 and nd <= 4096 and offset == 256+nc+nd and offset <= len(data) and not reserved1 and not reserved2, 'result offsets')
    case = data[256:256+nc].decode('ascii')
    if case:
        identity(case)
    diagnostic = data[256+nc:offset].decode('utf-8')
    commit = data[152:192].decode('ascii')
    require(re.fullmatch('[0-9a-f]{40}', commit), 'runtime commit')
    result = dict(status=status, case_id=case, diagnostic=diagnostic, model_sha256=data[56:88].hex(),
                  executable_sha256=data[88:120].hex(), source_sha256=data[120:152].hex(),
                  runtime_commit=commit, request_sha256=data[192:224].hex())
    if status:
        require(not any((B, V, H, Q)) and offset == len(data) and nd > 0, 'failure payload')
        return result
    require(nc > 0 and nd == 0 and min(B, V, H, Q) > 0, 'success shape')
    counts = (Q, B*V*H*Q, B*V, B*V)
    require(sum(c*s for c, s in zip(counts, (8, 4, 4, 4))) == len(data)-offset, 'result payload count')
    result['shape'] = (B, V, H, Q)
    for key, count, dt, shape in zip(('quantiles', 'values', 'anchor_loc', 'anchor_scale'), counts,
                                    ('<f8', '<f4', '<f4', '<f4'), ((Q,), (B,V,H,Q), (B,V), (B,V))):
        a = np.frombuffer(data, dtype=dt, count=count, offset=offset).copy().reshape(shape)
        require(np.isfinite(a).all(), 'nonfinite result')
        if key == 'anchor_scale': require((a > 0).all(), 'invalid scale')
        result[key] = a
        offset += a.nbytes
    return result
