"""Bounded CPU hook regressions; no checkpoints, forecasts or baseline files."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from timestral.config import TimestralConfig
from timestral.model import TimestralModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import gen_reference_baseline as gen


class GeneratorHookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        cfg = TimestralConfig(
            patch_size=2, d_model=4, num_heads=1, mlp_hidden_dim=8,
            block_schedule=('time', 'variate', 'time_variate', 'variate_time'),
            quantile_knots=(0.5,), block_size=2, context_length=8)
        self.model = TimestralModel(cfg).eval()
        self.rec_patch = patch.object(gen, 'REC', gen.Rec())
        self.rec_patch.start()
        self.addCleanup(self.rec_patch.stop)
        self.ctx_patch = patch.object(gen, 'CTX', gen.Ctx())
        self.ctx_patch.start()
        self.addCleanup(self.ctx_patch.stop)
        self.x = torch.randn(2, 3, 4)
        self.positions = torch.arange(3).expand(2, -1)
        self.handles = []
        self.addCleanup(self.remove_hooks)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def install_hooks(self):
        self.handles = gen.register_hooks(self.model)

    def forward(self, x):
        with torch.inference_mode():
            return self.model.transformer(x, None, None, self.positions)

    def assert_rhos(self, rec, pre):
        expected = set()
        for li, layer in enumerate(self.model.transformer.layers):
            residual = rec.t[f'{pre}layer.{li}.in'].copy()
            names = [f'layer.{li}.attn.{j}.out' for j in range(len(layer.attns))]
            names += [f'layer.{li}.mlp']
            for name in names:
                key = f'stat.{pre}{name}.rho'
                expected.add(key)
                self.assertIn(key, rec.stat)
                out = rec.t[pre + name]
                rms = lambda a: float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
                self.assertAlmostEqual(rec.stat[key], rms(out) / rms(residual), places=14)
                residual = residual + out
        actual = {k for k in rec.stat if k.startswith('stat.' + pre) and k.endswith('.rho')}
        self.assertEqual(actual, expected)

    def test_mixed_schedule_rho_for_both_passes_and_blocks(self):
        self.install_hooks()
        for pass_name in ('pos', 'neg'):
            for block in (0, 1):
                gen.CTX.pass_name, gen.CTX.block = pass_name, block
                x = self.x * (1 if pass_name == 'pos' else -1) + block * 0.25
                self.forward(x)
                self.assert_rhos(gen.REC, f'{pass_name}.b{block}.')

    def test_hooks_leave_outputs_bitwise_unchanged(self):
        before = self.forward(self.x)
        self.install_hooks()
        after = self.forward(self.x)
        self.assertEqual(before.numpy().tobytes(), after.numpy().tobytes())
        self.remove_hooks()
        again = self.forward(self.x)
        self.assertEqual(before.numpy().tobytes(), again.numpy().tobytes())

    def test_new_recorder_and_stats_disabled_are_independent(self):
        self.install_hooks()
        self.forward(self.x)
        first = gen.REC
        gen.REC = gen.Rec(stats=False)
        self.forward(-self.x)
        self.assertEqual(gen.REC.stat, {})
        gen.REC = gen.Rec()
        self.forward(-self.x)
        self.assert_rhos(gen.REC, '')
        self.assertIsNot(gen.REC.stat, first.stat)
        self.assertFalse(np.array_equal(first.t['layer.0.in'], gen.REC.t['layer.0.in']))

    def test_filtered_mlp_rerun_does_not_publish_stale_stats(self):
        self.install_hooks()
        mlp = self.model.transformer.layers[0].mlp
        # An isolated linear probe leaves a pending condition statistic. A
        # filtered component rerun must neither publish it nor carry it onward.
        with torch.inference_mode():
            mlp.w12(self.x)
        gen.REC = gen.Rec()
        with torch.inference_mode(), gen.only(['layer.0.mlp']):
            mlp(-self.x)
        self.assertEqual(set(gen.REC.t), {'layer.0.mlp'})
        self.assertEqual(gen.REC.stat, {})
        gen.REC = gen.Rec()
        with torch.inference_mode(), patch.object(gen, 'kappa_of', return_value=7.0):
            mlp(self.x)
        self.assertEqual(gen.REC.stat, {'stat.layer.0.mlp.kappa': 7.0})

    def test_filtered_layer_rerun_records_no_rho_or_other_stats(self):
        self.install_hooks()
        self.forward(self.x)
        gen.REC = gen.Rec(stats=False)
        with gen.only(['layer.0.out']):
            self.forward(-self.x)
        self.assertEqual(set(gen.REC.t), {'layer.0.out'})
        self.assertEqual(gen.REC.stat, {})
        self.assertIsNone(gen.CTX.attn)
        self.assertEqual(gen.CTX.sdpa_buf, [])


if __name__ == '__main__':
    unittest.main()
