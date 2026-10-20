"""Independent Q8 byte/dequantization oracles; optional built C++ cross-check."""
import ast
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT/'scripts/convert_timestral_to_gguf.py').read_text()
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
         and n.name in ('storage_tensor', 'reduced_weight_name', 'permute_qkv')]
ns = dict(np=np, re=re, gguf=gguf, TimestralConfig=object)
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<q8-storage>', 'exec'), ns)


def encode_scalar(a):
    out = bytearray()
    for row in a:
        for offset in range(0, len(row), 32):
            block = row[offset:offset+32]
            d = np.float32(max(abs(float(v)) for v in block))/np.float32(127)
            inverse = np.float32(1)/d if d else np.float32(0)
            out.extend(np.float16(d).tobytes())
            for v in block:
                z = float(np.float32(v*inverse))
                rounded = math.copysign(math.floor(abs(z)+.5), z)
                out.extend(int(rounded).to_bytes(1, 'little', signed=True))
    return bytes(out)


def sample():
    a = np.empty((3,64), np.float32)
    for row in range(3):
        for k in range(64):
            a[row,k] = 0 if row==0 else (127 if k in (31,63) else k%31-15+.5) if row==1 else (k-32)*.125
    return a


class Q8(unittest.TestCase):
    def test_blocks_independent_bytes_and_decode(self):
        a = sample()
        out, ty = ns['storage_tensor']('head.out.weight', a, None, 'q8_0')
        self.assertEqual(ty, gguf.GGMLQuantizationType.Q8_0)
        self.assertEqual(out.shape, (3,68))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.tobytes(), encode_scalar(a))
        decoded = np.empty_like(a)
        for row in range(3):
            for b in range(2):
                raw = out[row,b*34:(b+1)*34]
                scale = raw[:2].copy().view(np.float16).astype(np.float32)[0]
                decoded[row,b*32:(b+1)*32] = raw[2:].view(np.int8).astype(np.float32)*scale
        from gguf.quants import dequantize
        np.testing.assert_array_equal(decoded, dequantize(out, ty))
        self.assertEqual(out[1,2:34].view(np.int8)[0], -15)  # -14.5 rounds away.
        self.assertEqual(out[1,2:34].view(np.int8)[15], 1)   # +0.5 rounds away.

    def test_widths_exclusions_and_qkv(self):
        for width in [0,1,31,33,63]:
            with self.assertRaises(ValueError):
                ns['storage_tensor']('head.out.weight', np.zeros((2,width),np.float32),None,'q8_0')
        for width in [32,64,96]:
            a = np.arange(3*width,dtype=np.float32).reshape(3,width)/13
            out,_ = ns['storage_tensor']('head.out.weight',a,None,'q8_0')
            self.assertEqual(out.tobytes(),encode_scalar(a))
        for name in ['patch.hidden.weight','patch.res.weight','out_norm.weight','rope.cos',
                     'patch.forecast_query','head.out.bias','blk.0.ffn_norm.weight']:
            a = np.array([[0.,-0.,1.0003]],np.float32)
            out,ty = ns['storage_tensor'](name,a,None,'q8_0')
            self.assertEqual(out.tobytes(),a.tobytes())
            self.assertEqual(ty,gguf.GGMLQuantizationType.F32)
        c = SimpleNamespace(head_dim=4,num_heads=3)
        a = np.arange(36*32,dtype=np.float32).reshape(36,32)/17
        permuted = np.empty_like(a)
        for s in range(3):
            for h in range(3):
                for d in range(4):
                    permuted[s*12+h*4+d] = a[s*12+d*3+h]
        out,_ = ns['storage_tensor']('blk.0.attn.0.qkv.weight',a,c,'q8_0')
        self.assertEqual(out.tobytes(),encode_scalar(permuted))
        for a in [np.ones((1,32),np.float16),np.full((1,32),np.nan,np.float32),np.ones(32,np.float32)]:
            with self.assertRaises(ValueError):
                ns['storage_tensor']('head.out.weight',a,None,'q8_0')
        with self.assertRaises(ValueError):
            ns['storage_tensor']('head.out.weight',sample(),None,'q4_k')

    def test_cpp_reference_bytes(self):
        binary = os.environ.get('TIMESTRAL_TEST_Q8_BINARY')
        if not binary:
            self.skipTest('built Q8 oracle unavailable; required for slice gate')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'blocks.bin'
            subprocess.run([binary,'--encode-blocks',str(path)],check=True)
            self.assertEqual(path.read_bytes(),encode_scalar(sample()))
            out,_ = ns['storage_tensor']('head.out.weight',sample(),None,'q8_0')
            self.assertEqual(path.read_bytes(),out.tobytes())


if __name__ == '__main__':
    unittest.main()
