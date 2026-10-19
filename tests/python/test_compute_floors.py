"""Synthetic floor and publication regressions; never touch reference artifacts."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import compute_floors as cf
import baseline_io as bio
from test_check_baseline import fixture, CASE, MODES

SCRATCH = Path(os.environ.get('TIMESTRAL_WORK', './work'))
SCRATCH.mkdir(parents=True, exist_ok=True)


def floor_fixture():
    data = fixture()
    for _, kv in data.values():
        kv.update({'env.timestral_sha': 'synthetic-source', 'env.hf_revision': 'synthetic-revision'})
    # The acceptance fixture deliberately uses placeholder component shapes.
    for mode, (t, kv) in data.items():
        for n in t:
            if n.endswith(('.group.inverse', '.positions', '.first_obs.group', '.tf.enc_len',
                           '.assemble.n_tgt')):
                t[n] = t[n].astype(np.int32)
            elif any(part in n for part in ('.mask', '.observed')):
                t[n] = t[n].astype(np.int8)
            if n.endswith(('.q', '.k', '.v', '.q_rope', '.k_rope', '.sdpa')):
                t[n] = np.zeros((1, 2, 1, 4), t[n].dtype)
                kv[f'stat.{n}.numel'] = t[n].size
    return data


class FloorTests(unittest.TestCase):
    def run_data(self, data):
        def mode(p):
            return next(m for m in sorted(MODES, key=len, reverse=True) if str(p).endswith('-'+m+'.gguf'))
        with patch.object(Path, 'exists', lambda p: mode(p) in data), \
             patch.object(bio, 'read_baseline', lambda p: data[mode(p)]), \
             patch.object(bio, 'write_baseline') as write:
            result = cf.process('synthetic', CASE['name'])
        return result, write

    def test_valid(self):
        result, write = self.run_data(floor_fixture())
        self.assertTrue(result[0]); write.assert_called_once()

    def test_missing_each_mode(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                data = floor_fixture(); del data[mode]
                with self.assertRaises((ValueError, SystemExit)):
                    self.run_data(data)

    def test_invalid_evidence(self):
        mutations = [
            lambda d: d['f64'][0].pop('pos.b0.patch.out'),
            lambda d: d['f64-rerun'][0].pop('pos.b0.block_refined'),
            lambda d: d['f32'][1].pop('stat.pos.b0.patch.out.kappa'),
            lambda d: d['f64'][1].update({'model.id': 'other'}),
            lambda d: d['f64'][1].update({'case.H': 99}),
            lambda d: d['f64-rerun'][1].update({'selfcheck.ok': False}),
            lambda d: d['f32-math-rerun'][1].update({'rerun.source': 'f64'}),
            lambda d: d['f64'][0].update({'pos.b0.patch.out': np.zeros((2,), np.float64)}),
            lambda d: d['f64'][0].update({'pos.b0.patch.out': np.zeros((1,), np.float32)}),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                data = floor_fixture(); mutate(data)
                with self.assertRaises((ValueError, SystemExit, KeyError)):
                    self.run_data(data)

    def test_nonfinite_negative_stats_and_evidence(self):
        for value in (float('nan'), float('inf'), -1.):
            data = floor_fixture(); data['f32'][1]['stat.pos.b0.patch.out.kappa'] = value
            with self.subTest(stat=value), self.assertRaises((ValueError, SystemExit)):
                self.run_data(data)
        for value in (float('nan'), float('inf')):
            data = floor_fixture(); data['f64'][0]['pos.b0.patch.out'][0] = value
            with self.subTest(evidence=value), self.assertRaises((ValueError, SystemExit)):
                self.run_data(data)

    def test_zero_rms_and_exact_denominator(self):
        t = {'x': np.array([1.], np.float32)}
        with self.assertRaises(ValueError):
            cf.floors(t, [{'x': np.array([0.])}, {'x': np.array([1.])}], 'x')
        self.assertEqual(cf.floors(t, [{'x': np.array([2.])}, {'x': np.array([4.])}], 'x'), 1.5)
        with self.assertRaises(ValueError):
            cf.floors(t, [{}, {'x': np.array([4.])}], 'x')

    def test_raw_nan_positions(self):
        data = floor_fixture()
        for t, _ in data.values():
            if 'input.context' in t:
                t['input.context'].flat[0] = np.nan
                t['pos.b0.args.inputs'].flat[0] = np.nan
                t['pos.b0.scale.in'].flat[0] = np.nan
        self.run_data(data)
        data['f64'][0]['input.context'].flat[0] = 0.
        with self.assertRaises((ValueError, SystemExit)): self.run_data(data)

    def test_final_one_and_two_blocks(self):
        data = floor_fixture(); t, kv = data['f32']; kv['case.n_blocks'] = 1
        caps = cf.Caps(t, kv)
        want = caps.cap('pos.b0.out_norm') + caps.cap('pos.b0.head.raw')
        self.assertEqual(caps.cap('final.raw.pos'), want)
        self.assertEqual(caps.cap('final.quantiles'), want)
        kv['case.n_blocks'] = 2
        self.assertIsNone(cf.Caps(t, kv).cap('final.quantiles'))

    def test_missing_cap_stat_is_not_na(self):
        t, kv = floor_fixture()['f32']; del kv['stat.pos.b0.patch.out.kappa']
        with self.assertRaises(ValueError): cf.Caps(t, kv).cap('pos.b0.patch.out')

    def test_required_provenance_and_float_kind(self):
        for mutate in (
            lambda d: d['f64'][1].pop('env.timestral_sha'),
            lambda d: d['f32'][1].pop('env.hf_revision'),
            lambda d: d['f64'][0].update({'pos.b0.patch.out': np.zeros((1,), np.int32)}),
        ):
            data = floor_fixture()
            for _, kv in data.values():
                kv.update({'env.timestral_sha': 'synthetic-source', 'env.hf_revision': 'synthetic-revision'})
            mutate(data)
            with self.assertRaises(ValueError): self.run_data(data)

    def test_cap_applicability_and_unknown_names(self):
        t, kv = floor_fixture()['f32']; caps = cf.Caps(t, kv)
        self.assertEqual(caps.applicability('pos.b0.layer.0.attn.0.q'), 'bounded')
        self.assertEqual(caps.applicability('input.context'), 'copy')
        self.assertEqual(caps.applicability('pos.b1.refined'), 'na')
        for n in ('pos.b0.surprise', 'final.surprise'):
            t[n] = np.zeros(1, np.float32); kv[f'stat.{n}.numel'] = 1
            with self.assertRaises(KeyError): caps.cap(n)

    def test_nonfinite_floor_helpers(self):
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                cf.floors({'x': np.array([value])}, [{'x': np.array([value])}] * 2, 'x')

    def test_sdpa_key_length_not_query_length(self):
        t, kv = floor_fixture()['f32']; n = 'pos.b0.layer.0.attn.0.sdpa'
        t[n] = np.zeros((1, 2, 1, 4), np.float32)
        t[n[:-4]+'k'] = np.zeros((1, 7, 1, 4), np.float32)
        want = cf.gamma(4, 8) + cf.gamma(7, 8) + 8 * cf.U
        self.assertEqual(cf.Caps(t, kv).cap(n), want)

    def test_json_failure_preserves_bytes(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            path = Path(root)/'summary.json'; path.write_bytes(b'original')
            for method in ('fsync', 'replace'):
                with patch.object(bio.os, method, side_effect=OSError('injected')):
                    with self.assertRaises(OSError): bio.write_json(path, {'ok': 1})
                self.assertEqual(path.read_bytes(), b'original')
                self.assertEqual(list(Path(root).iterdir()), [path])
            bio.write_json(path, {'ok': 1})
            self.assertIn('"ok": 1', path.read_text())

    def test_real_metadata_update_preserves_arrays(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            data = floor_fixture()
            for mode, (t, kv) in data.items(): bio.write_baseline(bio.case_file(root, CASE['name'], mode), t, kv)
            path = bio.case_file(root, CASE['name'], 'f32')
            before, _ = bio.read_baseline(path)
            cf.process(root, CASE['name'])
            after, kv = bio.read_baseline(path)
            self.assertTrue(any(k.startswith('floor_tf.') for k in kv))
            for n in before:
                self.assertEqual(before[n].dtype, after[n].dtype)
                self.assertEqual(before[n].tobytes(), after[n].tobytes())

    def test_atomic_failure_and_success(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            path = Path(root)/'test.gguf'; arrays = {'x': np.arange(8, dtype=np.float32)}
            bio.write_baseline(path, arrays, {'test': 1}); original = path.read_bytes()
            with patch.object(bio.gguf.GGUFWriter, 'write_tensors_to_file', side_effect=OSError('injected')):
                with self.assertRaises(OSError): bio.write_baseline(path, arrays, {'test': 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(root).iterdir()), [path])
            bio.write_baseline(path, arrays, {'test': 3})
            t, kv = bio.read_baseline(path)
            np.testing.assert_array_equal(t['x'], arrays['x']); self.assertEqual(kv['test'], 3)

    def test_cli_summary_and_actual_late_invalid_case(self):
        import contextlib
        import io
        import json
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            data = floor_fixture()
            for mode, (t, kv) in data.items():
                bio.write_baseline(bio.case_file(root, CASE['name'], mode), t, kv)
            first = bio.case_file(root, CASE['name'], 'f32'); original = first.read_bytes()
            bio.write_baseline(bio.case_file(root, 'zzz-invalid', 'f32'), *data['f32'])
            with patch.object(cf.sys, 'argv', ['floors', '--dir', root]), self.assertRaises(ValueError):
                cf.main()
            self.assertEqual(first.read_bytes(), original)
            self.assertFalse((Path(root)/'floors-summary.json').exists())
            with patch.object(cf.sys, 'argv', ['floors', '--dir', root, '--case', CASE['name']]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cf.main(), 0)
            summary = json.loads((Path(root)/'floors-summary.json').read_text())[CASE['name']]
            self.assertEqual(summary['input.context']['cap_applicability'], {'copy': 1})
            self.assertEqual(summary['layer.L.attn.J.q']['cap_applicability'], {'bounded': 6})
            self.assertEqual(summary['refined']['cap_applicability']['na'], 4)
            with patch.object(cf.sys, 'argv', ['floors', '--dir', root, '--case', 'missing']), \
                 self.assertRaises(ValueError):
                cf.main()

    def test_all_cap_classes_and_combined_floor_selection(self):
        data = floor_fixture(); t, kv = data['f32']; caps = cf.Caps(t, kv)
        names = ['patch.hidden_out', 'patch.out', 'layer.0.attn.0.q', 'layer.0.attn.0.k',
                 'layer.0.attn.0.v', 'layer.0.attn.0.qkv', 'layer.0.attn.0.q_rope',
                 'layer.0.attn.0.k_rope', 'layer.0.attn.0.sdpa', 'layer.0.attn.0.gate',
                 'layer.0.attn.0.out', 'layer.0.mlp', 'layer.0.out', 'out_norm',
                 'head.raw', 'predict', 'refined']
        for rel in names:
            self.assertIsNotNone(caps.cap('pos.b0.' + rel), rel)
        data['f32'][0]['pos.b0.refined'][:] = 1
        for mode in ('f64-rerun', 'f32-math-rerun'):
            data[mode][0]['pos.b0.refined'][:] = 100
            data[mode][0]['pos.b0.block_refined'][:] = 2
        data['f64'][0]['pos.b0.refined'][:] = 1
        result, _ = self.run_data(data)
        row = next(r for r in result[0] if r[0] == 'pos.b0.refined')
        self.assertEqual(row[2], 0.5)

    def test_late_invalid_preflight_no_writes(self):
        with patch.object(cf, 'prepare', side_effect=[('one', {}, {}, [], []), ValueError('late')]), \
             patch.object(bio, 'write_baseline') as write:
            with self.assertRaises(ValueError): cf.publish('synthetic', ['one', 'two'])
            write.assert_not_called()
