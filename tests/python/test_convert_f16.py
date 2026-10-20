"""Independent synthetic storage oracles; no reference imports or checkpoints."""
import ast
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT/'scripts/convert_timestral_to_gguf.py').read_text()
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
         and n.name in ('reduced_weight_name', 'storage_tensor', 'permute_qkv')]
ns = dict(np=np, re=re, TimestralConfig=object,
          gguf=SimpleNamespace(GGMLQuantizationType=SimpleNamespace(F32=0, F16=1)))
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<storage>', 'exec'), ns)


class Storage(unittest.TestCase):
    def test_policy_and_exclusions(self):
        yes = ['patch.out.weight', 'head.hidden.weight', 'head.out.weight', 'head.res.weight',
               'blk.0.attn.0.qkv.weight', 'blk.23.attn.1.gate.weight',
               'blk.2.attn.0.out.weight', 'blk.0.ffn_w12.weight', 'blk.9.ffn_w3.weight']
        no = ['patch.hidden.weight', 'patch.res.weight', 'patch.forecast_query', 'rope.cos',
              'rope.sin', 'rope.inv_freq', 'out_norm.weight', 'blk.0.ffn_norm.weight',
              'blk.0.attn_norm.0.weight', 'blk.x.ffn_w3.weight', 'head.out.bias',
              'xblk.0.ffn_w3.weight', 'blk.0.ffn_w3.weight.extra']
        a = np.array([[0., -0., 1.0003, -.003]], np.float32)
        for name in yes + no:
            self.assertEqual(ns['reduced_weight_name'](name), name in yes)
            # Separate policy from QKV shape requirements.
            c = SimpleNamespace(head_dim=1, num_heads=1)
            x = np.tile(a, (3,1)) if name.endswith('qkv.weight') else a
            out, ty = ns['storage_tensor'](name, x, c, 'f16')
            expected = x.astype(np.float16) if name in yes else x
            self.assertEqual(out.tobytes(), expected.tobytes())
            self.assertEqual(ty, 1 if name in yes else 0)
        with self.assertRaises(ValueError):
            ns['storage_tensor']('head.out.weight', a, None, 'q4_k')

    def test_independent_qkv_row_mapping_f32_and_f16(self):
        c = SimpleNamespace(head_dim=4, num_heads=3)
        a = (np.arange(36*7, dtype=np.float32).reshape(36,7)/13)
        expected = np.empty_like(a)
        for s in range(3):
            for h in range(3):
                for d in range(4):
                    expected[s*12+h*4+d] = a[s*12+d*3+h]
        for kind in ['f32', 'f16']:
            out, _ = ns['storage_tensor']('blk.0.attn.0.qkv.weight', a, c, kind)
            self.assertEqual(out.tobytes(), expected.astype(np.float16 if kind=='f16' else np.float32).tobytes())
        bias = a[:,0].copy()
        out, ty = ns['storage_tensor']('blk.0.attn.0.qkv.bias', bias, c, 'f16')
        self.assertEqual(out.tobytes(), expected[:,0].copy().tobytes())
        self.assertEqual(ty, 0)


if __name__ == '__main__':
    unittest.main()
