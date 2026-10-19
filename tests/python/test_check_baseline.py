"""Fast synthetic acceptance regressions; no reference inference or model weights."""
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import check_baseline as cb


CASE = dict(name='uni_short', B=1, V=1, T=8, H=12, seed=1, layers='all', tiny=True)
CFG = dict(patch_size=2, block_size=2, context_length=16, d_model=4,
           num_heads=1, head_dim=4, mlp_hidden_dim=8, block_schedule=['time'],
           time_attn_gate='sdpa_sigmoid', variate_attn_gate='none',
           covariate_enabled=True, split_context_horizon_attn=True, quantile_knots=[0.5])
MODES = ('f32', 'f32-math', 'f64', 'f64-rerun', 'f32-math-rerun')


def fixture():
    kv = {'model.config_json': json.dumps(CFG), 'model.id': 'synthetic',
          'case.json': json.dumps(CASE), 'case.name': CASE['name'], 'case.tiny': False,
          'case.n_blocks': 3, 'layers.dumped': '0', 'repro.bitwise_equal': True,
          'repro.runs': 2, 'truth.patches': ['RMSNorm', 'SinCosRoPE']}
    for k in ('B', 'V', 'T', 'H'):
        kv['case.' + k] = CASE[k]
    for k in ('patch_size', 'block_size', 'context_length', 'd_model', 'num_heads', 'head_dim', 'mlp_hidden_dim'):
        kv['model.' + k] = CFG[k]
    kv['model.n_layers'] = 1
    t = {}
    for p in ('pos', 'neg'):
        for k in range(3):
            pre = f'{p}.b{k}.'
            t[pre+'args.inputs'] = np.zeros((1, 8), np.float32)
            t[pre+'args.horizon_patches'] = np.array([2], np.int32)
    names, _ = cb.expected(kv, t, CASE)
    for n in names | {'final.quantiles'}:
        t.setdefault(n, np.zeros((1,), np.float32))
    t['input.context'] = np.zeros((1, 1, 8), np.float32)
    for n in ('final.quantiles', 'final.raw.pos', 'final.raw.neg'):
        t[n] = np.zeros((1, 1, 12, 1), np.float32)
    calls = {'forecaster._forecast_raw': 2, 'forecaster._assemble_rows': 2,
             'forecaster._cap_quantiles': 2, 'forecaster.causal_patch_loc_scale': 2,
             'model.predict_block': 6, 'model.scale': 6, 'model.denormalize_quantiles': 6,
             'model.compute_enc_key_validity': 6, 'model.group_first_observed_patch': 6,
             'layers._sdpa': 6, 'layers._apply_rotary': 6, 'layers.Attention._qkv': 6,
             'model.build_forecast_time_mask': 6, 'model.build_forecast_positions': 6,
             'forecaster.advancing_block_frames': 6, 'forecaster.clamp_nonfinite': 6}
    kv.update({'calls.'+k: v for k, v in calls.items()})
    for n, a in t.items():
        if cb.bio.is_float(a):
            for s, v in dict(rms=0., numel=a.size, kappa=0., kappa_res=0., rho=0., max_logit=0.).items():
                kv[f'stat.{n}.{s}'] = v
    out = {}
    for m in MODES:
        tt, kk = copy.deepcopy((t, kv))
        kk['mode'] = m
        if m.startswith('f64'):
            tt = {n: a.astype(np.float64) if cb.bio.is_float(a) and not n.startswith('input.') else a for n, a in tt.items()}
        if m.endswith('rerun'):
            tt = {n: tt[target].copy() for n, target in cb.expected_reruns(t).items()}
            kk.update({'rerun.source': 'f32', 'selfcheck.ok': True,
                       'selfcheck.max_ratio': 0., 'selfcheck.limit': 1e-12, 'selfcheck.where': ''})
        out[m] = tt, kk
    return out


class AcceptanceTests(unittest.TestCase):
    def run_fixture(self, data, case=CASE, tiny=False):
        cb.errors.clear()
        def mode(path):
            return next(m for m in sorted(MODES, key=len, reverse=True) if str(path).endswith('-'+m+'.gguf'))
        with patch.object(Path, 'exists', lambda p: mode(p) in data), \
             patch.object(cb.bio, 'read_baseline', lambda p: data[mode(p)]), \
             contextlib.redirect_stdout(io.StringIO()):
            cb.check_case('synthetic', case, tiny=tiny)
        return list(cb.errors)

    def test_valid(self):
        self.assertEqual(self.run_fixture(fixture()), [])

    def test_valid_tiny(self):
        original = dict(CASE, T=32, H=48)
        data = fixture()
        for _, kv in data.values():
            kv['case.tiny'] = True
        self.assertEqual(self.run_fixture(data, original, tiny=True), [])
        self.assertTrue(self.run_fixture(data, original, tiny=False))

    def test_subset_and_suite_selection(self):
        for args, count, label in [([], 15, 'FULL'), (['--tiny'], 14, 'FULL'),
                                    (['--case', 'uni_one_patch'], 1, 'PARTIAL'),
                                    (['--tiny', '--case', 'uni_one_patch'], 1, 'PARTIAL')]:
            with self.subTest(args=args):
                output = io.StringIO()
                with patch.object(cb, 'check_case', return_value=True) as check, \
                     patch.object(cb.sys, 'argv', ['check', '--dir', 'synthetic'] + args), \
                     contextlib.redirect_stdout(output):
                    self.assertEqual(cb.main(), 0)
                    self.assertEqual(check.call_count, count)
                    self.assertIn(label, output.getvalue())
                    if '--tiny' in args:
                        self.assertNotIn('bench_32k', [c.args[1]['name'] for c in check.call_args_list])
        with patch.object(cb.sys, 'argv', ['check', '--dir', 'synthetic', '--tiny', '--case', 'bench_32k']), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cb.main(), 1)

    def error_fixture(self, name='const'):
        case = dict(CASE, name=name)
        data = fixture()
        for mode in list(data):
            if mode.endswith('rerun'):
                del data[mode]
                continue
            t, kv = data[mode]
            data[mode] = {'input.context': t['input.context']}, kv
            kv.update({'case.name': name, 'case.json': json.dumps(case), 'case.n_blocks': 0,
                       'expect.error.type': 'ValueError', 'expect.error.message': 'reference rejects input',
                       'expect.error.step': 'forecaster._assemble_rows'})
        return case, data

    def test_unproven_preblock_errors_fail_closed(self):
        for name in ('const', 'all_missing'):
            case, data = self.error_fixture(name)
            self.assertTrue(any('no proven reference exception contract' in e
                                for e in self.run_fixture(data, case)))

    def test_exception_input_validation(self):
        for mode in ('f32', 'f32-math', 'f64'):
            for mutation, diagnostic in [('missing', 'missing stored input'),
                                         ('shape', 'wrong shape input.context'),
                                         ('dtype', 'wrong dtype for input.context'),
                                         ('stats', 'invalid rms/numel for input.context')]:
                with self.subTest(mode=mode, mutation=mutation):
                    case, data = self.error_fixture()
                    t, kv = data[mode]
                    if mutation == 'missing':
                        t.clear()
                    elif mutation == 'shape':
                        t['input.context'] = np.zeros((8,), np.float32)
                    elif mutation == 'dtype':
                        t['input.context'] = t['input.context'].astype(np.int32)
                    else:
                        del kv['stat.input.context.rms']
                    self.assertTrue(any(diagnostic in e for e in self.run_fixture(data, case)))

    def test_exception_any_block_evidence_rejected(self):
        for mode in ('f32', 'f32-math', 'f64'):
            for evidence in ('record', 'counter'):
                with self.subTest(mode=mode, evidence=evidence):
                    case, data = self.error_fixture()
                    t, kv = data[mode]
                    if evidence == 'record':
                        t['pos.b0.scaled'] = np.zeros((1,), np.float32)
                    else:
                        kv['calls.model.predict_block'] = 1
                    self.assertTrue(any('block execution evidence' in e
                                        for e in self.run_fixture(data, case)))

    def test_exception_step_and_harness_errors(self):
        for name in ('const', 'all_missing'):
            for step in (None, '', 'not.a.reference.step', 'forecaster._assemble_rows'):
                with self.subTest(name=name, step=step):
                    case, data = self.error_fixture(name)
                    for _, kv in data.values():
                        kv.update({'expect.error.type': 'AssertionError',
                                   'expect.error.message': 'instrumentation harness failed'})
                        if step is None:
                            del kv['expect.error.step']
                        else:
                            kv['expect.error.step'] = step
                    self.assertTrue(self.run_fixture(data, case))

    def test_error_contract_rejections(self):
        case, data = self.error_fixture('uni_short')
        self.assertTrue(self.run_fixture(data, case))
        for mutation in ('missing-mode', 'mixed-error', 'partial', 'rerun'):
            with self.subTest(mutation=mutation):
                case, data = self.error_fixture()
                if mutation == 'missing-mode':
                    del data['f64']
                elif mutation == 'mixed-error':
                    data['f64'][1]['expect.error.message'] = 'different exception'
                elif mutation == 'partial':
                    data['f32'][0]['pos.b0.args.inputs'] = np.zeros((1, 8), np.float32)
                else:
                    data['f64-rerun'] = fixture()['f64-rerun']
                self.assertTrue(self.run_fixture(data, case))

    def test_missing_companion_modes(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                data = fixture(); del data[mode]
                self.assertTrue(self.run_fixture(data))

    def test_missing_case(self):
        cb.errors.clear()
        spec = {'version': 1, 'cases': [CASE, dict(CASE, name='missing')]}
        with patch.object(cb.bio, 'load_cases', return_value=spec), \
             patch.object(cb, 'check_case', side_effect=[True, False]), \
             patch.object(cb.sys, 'argv', ['check', '--dir', 'synthetic']), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cb.main(), 1)

    def test_deleted_records(self):
        for prefix in ('neg.', 'pos.b2.', 'pos.b1.layer.0.attn.0.q', 'input.context'):
            with self.subTest(prefix=prefix):
                data = fixture()
                for t, _ in data.values():
                    for n in list(t):
                        if n.startswith(prefix): del t[n]
                self.assertTrue(self.run_fixture(data))

    def test_component_statistic(self):
        data = fixture(); del data['f32'][1]['stat.pos.b1.layer.0.attn.0.q.kappa']
        self.assertTrue(self.run_fixture(data))

    def test_combined_refine_contract_required_and_shape_checked(self):
        data = fixture()
        del data['f64-rerun'][0]['pos.b0.block_refined']
        self.assertTrue(any('component coverage' in e for e in self.run_fixture(data)))
        data = fixture()
        data['f64-rerun'][0]['pos.b0.block_refined'] = np.ones((2,), np.float64)
        self.assertTrue(any('shape mismatch' in e for e in self.run_fixture(data)))

    def test_rerun_component(self):
        data = fixture(); del data['f64-rerun'][0]['pos.b1.layer.0.attn.0.q']
        self.assertTrue(self.run_fixture(data))

    def test_bad_selfcheck(self):
        for key, value in [('selfcheck.max_ratio', 1e-6), ('selfcheck.max_ratio', float('nan')),
                           ('selfcheck.max_ratio', -1.), ('selfcheck.limit', 1e-6), ('selfcheck.ok', 'yes')]:
            with self.subTest(key=key, value=value):
                data = fixture(); data['f64-rerun'][1][key] = value
                self.assertTrue(self.run_fixture(data))

    def test_metadata(self):
        for key, value in [('mode', 'f64'), ('case.H', 99), ('case.tiny', True),
                           ('model.d_model', 99), ('case.n_blocks', 1), ('layers.dumped', '')]:
            with self.subTest(key=key):
                data = fixture(); data['f32'][1][key] = value
                self.assertTrue(self.run_fixture(data))

    def test_shape_dtype(self):
        for n, arr in [('input.context', np.zeros((8,), np.float32)),
                       ('pos.b0.args.inputs', np.zeros((1, 8), np.float64))]:
            data = fixture(); data['f32'][0][n] = arr
            self.assertTrue(self.run_fixture(data))


if __name__ == '__main__':
    unittest.main()
