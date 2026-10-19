"""Dtype acceptance must follow record meaning, not the submitted array kind."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import baseline_io as bio
import test_check_baseline as acceptance
from test_check_baseline import MODES
import test_compute_floors as floor_tests
from test_compute_floors import floor_fixture


class SemanticDtypeTests(unittest.TestCase):
    def assert_rejected(self, data, name):
        errors = acceptance.AcceptanceTests().run_fixture(data)
        self.assertTrue(any(name in e and 'dtype' in e for e in errors), errors)
        self.assert_floor_rejected(data, name)

    def assert_floor_rejected(self, data, diagnostic):
        def mode(path):
            return next(m for m in sorted(MODES, key=len, reverse=True)
                        if str(path).endswith('-' + m + '.gguf'))
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(bio, 'read_baseline', lambda p: data[mode(p)]), \
             patch.object(bio, 'write_baseline') as write:
            with self.assertRaisesRegex(ValueError, diagnostic):
                floor_tests.cf.process('synthetic', acceptance.CASE['name'])
            write.assert_not_called()

    def test_all_modes_integer_numerical_components(self):
        # Mutate all five together: matching kinds between modes is not proof.
        for name in ('pos.b0.patch.out', 'pos.b0.layer.0.out',
                     'pos.b0.layer.0.mlp', 'pos.b0.layer.0.mlp_norm',
                     'pos.b0.layer.0.attn.0.q', 'pos.b0.layer.0.attn.0.cos',
                     'pos.b0.layer.0.attn.0.sdpa', 'pos.b0.head.raw',
                     'pos.b0.refined', 'pos.b0.anchor.loc', 'pos.b0.frames.loc',
                     'pos.b0.scale', 'pos.cap.out', 'final.raw.pos', 'final.quantiles'):
            with self.subTest(name=name):
                data = floor_fixture()
                for t, _ in data.values():
                    if name in t:
                        t[name] = t[name].astype(np.int32)
                self.assert_rejected(data, name)

    def test_all_modes_combined_refine_alias(self):
        data = floor_fixture()
        for mode, (t, _) in data.items():
            name = 'pos.b0.' + ('block_refined' if mode.endswith('rerun') else 'refined')
            t[name] = t[name].astype(np.int32)
        self.assert_rejected(data, 'refined')

    def test_each_mode_and_combined_alias(self):
        for mode in MODES:
            for name in ('pos.b0.patch.out', 'pos.b0.block_refined'):
                if name.endswith('block_refined') and not mode.endswith('rerun'):
                    continue
                with self.subTest(mode=mode, name=name):
                    data = floor_fixture()
                    data[mode][0][name] = data[mode][0][name].astype(np.int32)
                    self.assert_rejected(data, name)

    def test_unknown_integer_outputs_fail_closed(self):
        for name in ('pos.b0.layer.0.attn.0.surprise', 'pos.b0.surprise', 'final.surprise'):
            with self.subTest(name=name):
                data = floor_fixture()
                for t, _ in data.values():
                    t[name] = np.zeros(1, np.int32)
                errors = acceptance.AcceptanceTests().run_fixture(data)
                self.assertTrue(any('unknown record' in e and name in e for e in errors), errors)
                self.assert_floor_rejected(data, 'unknown record')

    def test_legitimate_integer_boolean_and_input_records(self):
        data = floor_fixture()
        for t, _ in data.values():
            for n in t:
                if '.frames.' in n:
                    t[n] = t[n].astype(np.float64)
        integers = ('positions', 'tf.positions', 'first_obs.group', 'group.inverse',
                    'mask.block_anchor', 'args.group_ids', 'args.role_ids',
                    'args.horizon_patches', 'tf.enc_len')
        booleans = ('mask.block_active', 'mask.key_valid', 'args.observed',
                    'refine.mask', 'refine.target_rows', 'known.rows')
        for mode, (t, kv) in data.items():
            if mode.endswith('rerun'):
                continue
            for rel in integers + booleans:
                t['pos.b0.' + rel] = np.zeros(1, np.int32 if rel in integers else np.int8)
            t['pos.assemble.role_ids'] = np.zeros(1, np.int32)
            t['pos.assemble.known_future_mask'] = np.zeros(1, np.int8)
        self.assertEqual(acceptance.AcceptanceTests().run_fixture(data), [])
        result, write = floor_tests.FloorTests().run_data(data)
        self.assertTrue(result[0])
        write.assert_called_once()
