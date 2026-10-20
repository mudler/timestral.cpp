"""Recorded-transport mocks only: no models, canonical-size arrays, or golden files."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import eval_quant as eq
import quant_fixtures as qf


class Recorded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cases = [dict(c, B=1, V=1, T=3, H=2) for c in qf.load_cases()]
        cls.inputs = {'context': np.array([[[1, 2, 3]]], dtype='<f4')}
        cls.truth = np.full((1,1,2), 3., dtype='<f4')
        cls.provenance = dict(source_sha256={'mock-source': 'a'*64},
                              case_sha256='b'*64, input_sha256={'context': qf.array_hash(cls.inputs['context'])},
                              truth_sha256=qf.array_hash(cls.truth))
        models = {f: dict(sha256=str(i+1)*64, hf_revision=qf.HF_REVISION,
                          reference_sha=qf.REFERENCE_SHA) for i,f in enumerate(eq.FORMATS)}
        cls.base = dict(version='quant-record-v1', fixture_version=qf.VERSION,
                        options=qf.OPTIONS, source_sha256=cls.provenance['source_sha256'],
                        driver=dict(source_sha256='d'*64, runtime_commit='e'*40), models=models, items=[])
        cls.arrays = dict(context=cls.inputs['context'], truth=cls.truth, levels=eq.LEVELS,
                          anchor_loc=np.zeros((1,1), '<f4'), anchor_scale=np.ones((1,1), '<f4'))
        for i,f in enumerate(eq.FORMATS):
            cls.arrays['forecast_'+f] = np.full((1,1,2,21), 2.+i*.0001, '<f4')
        np.savez(cls.root/'mock.npz', **cls.arrays)
        for c in cls.cases:
            cls.base['items'].append(dict(id=c['name'], **{k: cls.provenance[k] for k in
                ('case_sha256','input_sha256','truth_sha256')}, npz='mock.npz',
                arrays=cls.descriptors(cls.arrays), anchor_model_sha256=models['f32']['sha256'],
                outputs={f: {'status':'forecast'} for f in eq.FORMATS}))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def descriptors(arrays):
        return {k: dict(dtype=a.dtype.str, shape=list(a.shape), sha256=qf.array_hash(a))
                for k,a in arrays.items()}

    def evaluate(self, manifest):
        path = self.root/'manifest.json'
        path.write_text(json.dumps(manifest))
        with patch.object(qf, 'load_cases', return_value=self.cases), patch.object(
                qf, 'make_fixture', return_value=(self.inputs,self.truth,self.provenance)):
            return eq.evaluate(path)

    def test_mock_pass_rows_suites_gates_and_atomic_json(self):
        out = self.evaluate(self.base)
        self.assertEqual(out['result'], 'PASS')
        self.assertEqual(out['mode'], 'recorded-output-only')
        self.assertTrue(out['gates']['strict_monotonic_mean'])
        for f in eq.FORMATS:
            self.assertEqual(out['coverage'][f]['forecasts'], 79)
            self.assertEqual(out['summaries'][f]['c2']['mean']['item_count'], 15)
            self.assertEqual(out['summaries'][f]['synthetic']['mean']['item_count'], 64)
            self.assertEqual(len(out['rows'][f]), 79)
        path = self.root/'report.json'
        eq.write_report(path, out)
        self.assertEqual(eq.read_json(path)['result'], 'PASS')
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            eq.write_report(path, {'bad': float('nan')})
        self.assertEqual(path.read_bytes(), before)

    def test_identity_provenance_and_anchor_rejection(self):
        mutations = [lambda m: m['items'].pop(),
                     lambda m: m['items'][0].update(id='unknown'),
                     lambda m: m['items'][0].update(id=m['items'][1]['id']),
                     lambda m: m.update(options={}),
                     lambda m: m.update(source_sha256={}),
                     lambda m: m['models']['f32'].update(hf_revision='wrong'),
                     lambda m: m['items'][0].update(anchor_model_sha256=m['models']['q8_0']['sha256']),
                     lambda m: m['items'][0]['arrays']['truth'].update(sha256='0'*64),
                     lambda m: m['items'][0].update(npz='../escape.npz'),
                     lambda m: m['items'][0]['arrays'].update(anchor_q8={})]
        for mutate in mutations:
            m = copy.deepcopy(self.base)
            mutate(m)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.evaluate(m)

    def test_invalid_npz_values_dtype_shape_and_pickle(self):
        for value in [np.full((1,1,2,21), np.nan, '<f4'),
                      np.ones((1,1,2,21), '<f8'), np.ones((1,1,2,20), '<f4'),
                      np.array([object()], dtype=object),
                      np.broadcast_to(np.arange(21, 0, -1, dtype='<f4'), (1,1,2,21))]:
            arrays = dict(self.arrays)
            arrays['forecast_q8_0'] = value
            np.savez(self.root/'invalid.npz', **arrays)
            m = copy.deepcopy(self.base)
            m['items'][0].update(npz='invalid.npz', arrays=self.descriptors(arrays))
            with self.subTest(dtype=str(value.dtype), shape=value.shape), self.assertRaises(ValueError):
                self.evaluate(m)

    def test_expected_errors_hold_named_coverage(self):
        m = copy.deepcopy(self.base)
        error = dict(status='expected-error', type='ValueError', message='degenerate', stage='prepare')
        m['items'][0]['outputs'] = {f: error for f in eq.FORMATS}
        arrays = {k:v for k,v in self.arrays.items() if k in ('context','truth','levels')}
        np.savez(self.root/'error.npz', **arrays)
        m['items'][0].update(npz='error.npz', arrays=self.descriptors(arrays), anchor_model_sha256=None)
        out = self.evaluate(m)
        self.assertEqual(out['result'], 'HOLD')
        name = m['items'][0]['id']
        self.assertEqual(out['coverage']['f32']['gaps'][name], 'expected-error-match')
        self.assertEqual(out['rows']['q8_0'][name], [])
        self.assertEqual(out['summaries']['f16']['overall']['mean']['aggregate']['kind'], 'undefined')

    def test_large_error_and_monotonic_failure(self):
        m = copy.deepcopy(self.base)
        arrays = dict(self.arrays)
        arrays['forecast_f16'] = np.full((1,1,2,21), 100., '<f4')
        np.savez(self.root/'bad.npz', **arrays)
        m['items'][0].update(npz='bad.npz', arrays=self.descriptors(arrays))
        out = self.evaluate(m)
        self.assertEqual(out['result'], 'FAIL')
        self.assertFalse(out['gates']['bounds']['f16'])
        self.assertFalse(out['gates']['strict_monotonic_mean'])

    def command(self, manifest_path, report_path):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as result:
                eq.main(['--manifest', str(manifest_path), '--report', str(report_path)])
        self.assertEqual(result.exception.code, 2)
        self.assertTrue(stderr.getvalue().startswith('error: '))
        self.assertNotIn('Traceback', stderr.getvalue())

    def test_malformed_container_command_exit(self):
        bad = [[], None, 7, {'items': None}, {'items': [7]}, {'items': [None]}]
        for items in [None, {}, [7]*79, [None]*79, ['bad']*79]:
            m = copy.deepcopy(self.base)
            m['items'] = items
            bad.append(m)
        for value in bad:
            with self.subTest(value_type=type(value).__name__):
                path = self.root/'malformed.json'
                report = self.root/'never-created.json'
                path.write_text(json.dumps(value))
                self.command(path, report)
                self.assertFalse(report.exists())

    def test_symlink_and_overwrite_protection(self):
        m = copy.deepcopy(self.base)
        path = self.root/'protected-manifest.json'
        path.write_text(json.dumps(m))
        original = path.read_bytes()
        self.command(path, path)
        self.assertEqual(path.read_bytes(), original)
        arrays = self.root/'mock.npz'
        original_arrays = arrays.read_bytes()
        self.command(path, arrays)
        self.assertEqual(arrays.read_bytes(), original_arrays)
        alias = self.root/'report-alias.npz'
        alias.symlink_to(arrays.name)
        self.command(path, alias)
        self.assertEqual(arrays.read_bytes(), original_arrays)
        self.assertTrue(alias.is_symlink())
        m['items'][0]['npz'] = alias.name
        with self.assertRaises(ValueError):
            self.evaluate(m)
        self.assertEqual(arrays.read_bytes(), original_arrays)

    def test_duplicate_json_and_command_errors(self):
        path = self.root/'duplicate.json'
        path.write_text('{"version":1,"version":2}')
        with self.assertRaises(ValueError):
            eq.read_json(path)
        with self.assertRaises(SystemExit) as result:
            eq.main(['--help'])
        self.assertEqual(result.exception.code, 0)
        with self.assertRaises(SystemExit) as result:
            eq.main(['--manifest', str(self.root/'absent'), '--report', str(self.root/'out')])
        self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
