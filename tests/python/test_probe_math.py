"""Synthetic math-probe checks; no reference artifacts or full probe required."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import probe_math as pm


class ProbeTests(unittest.TestCase):
    def test_known_ulp_pairs(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        a = np.array([0., -tiny, -tiny, -1., 1.], np.float32)
        b = np.array([-0., tiny, 0., np.nextafter(np.float32(-1), np.float32(-2)),
                      np.nextafter(np.float32(1), np.float32(2))], np.float32)
        self.assertEqual(pm.finite_ulp(a, b).tolist(), [0, 2, 1, 1, 1])

    def test_nonfinite_accounting(self):
        a = np.array([0, np.nan, np.inf, -np.inf, np.nan, np.inf, 1], np.float32)
        b = np.array([-0., np.nan, np.inf, -np.inf, 1, -np.inf, np.inf], np.float32)
        r = pm.compare(a, a, b)
        self.assertEqual(r['finite_histogram'], {'0': 1})
        self.assertEqual(r['special_counts'], {'both_nan': 1, 'same_infinity': 2,
                                               'nonfinite_mismatch': 3})
        self.assertEqual(r['total_count'], 7)
        self.assertEqual(r['mismatch_count'], 3)
        self.assertEqual(sum(r['finite_histogram'].values()) + sum(r['special_counts'].values()), 7)
        json.dumps(r, allow_nan=False)

    def test_all_special_has_no_finite_maximum(self):
        x = np.array([np.nan, np.inf], np.float32)
        r = pm.compare(x, x, x)
        self.assertEqual(r['finite_histogram'], {})
        self.assertIsNone(r['max_finite_ulp'])
        self.assertIsNone(r['max_example'])
        self.assertEqual(r['mismatch_count'], 0)
        with self.assertRaises(ValueError):
            pm.finite_ulp(x, x)

    def test_injected_mismatch(self):
        x = np.array([-1, 0, 1], np.float32)
        y = x.copy(); y[2] = np.nextafter(y[2], np.float32(2))
        r = pm.compare(x, x, y)
        self.assertEqual(r['finite_histogram'], {'0': 2, '1': 1})
        self.assertEqual(r['max_finite_ulp'], 1)
        self.assertEqual(r['mismatch_count'], 1)
        self.assertEqual(r['max_example']['input_bits'], '0x3f800000')
        self.assertEqual(r['mismatch_examples'][0]['libm_bits'], '0x3f800001')

    def test_samples(self):
        for n in (8, 9, 1000):
            for name, bounds in pm.RANGES.items():
                a = pm.samples(name, n)
                self.assertEqual(a.dtype, np.float32)
                self.assertEqual(a.size, n)
                np.testing.assert_array_equal(a, pm.samples(name, n))
                self.assertGreaterEqual(float(a.min()), bounds[0])
                self.assertLessEqual(float(a.max()), bounds[1])
                self.assertIn(np.float32(0), a)
                self.assertIn(np.float32(bounds[0]), a)
                self.assertIn(np.float32(bounds[1]), a)
        a = pm.samples('asinh', 1000)
        self.assertTrue(np.any((np.abs(a[:500]) > 0) & (np.abs(a[:500]) < 1e-6)))
        self.assertTrue(np.all(np.abs(a[500:]) <= 10))

    def test_bad_counts(self):
        for n in (0, -1, 7, 10000001, True, 8.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                pm.samples('sin', n)

    def test_ctypes_bindings(self):
        lib = pm.Libm()
        for name in ('asinhf', 'sinhf', 'sinf', 'cosf', 'expf', 'log1pf'):
            fn = getattr(lib.library, name)
            self.assertEqual(fn.argtypes, [ctypes.c_float])
            self.assertIs(fn.restype, ctypes.c_float)
        for name, expected in [('asinh', 0), ('sinh', 0), ('sin', 0), ('cos', 1)]:
            self.assertEqual(lib.evaluate(name, np.array([0], np.float32))[0], expected)
        x = np.float32(.5)
        expected = lib.library.log1pf(lib.library.expf(x))
        self.assertEqual(lib.evaluate('softplus', np.array([x], np.float32))[0], expected)

    def test_missing_libm(self):
        with patch.object(pm.ctypes.util, 'find_library', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'libm'):
                pm.Libm()

    def test_missing_symbol(self):
        with patch.object(pm.ctypes, 'CDLL', return_value=object()):
            with self.assertRaisesRegex(RuntimeError, 'libm'):
                pm.Libm()

    def test_run_deterministic(self):
        a = pm.run_probe(16)
        self.assertEqual(a, pm.run_probe(16))
        self.assertEqual(a['environment']['torch_threads'], 1)
        for r in a['functions'].values():
            self.assertEqual(r['total_count'], 16)
            self.assertEqual(sum(r['finite_histogram'].values()) + sum(r['special_counts'].values()), 16)

    def test_cli_atomic_failure(self):
        root = Path(os.environ.get('TIMESTRAL_WORK', './work'))
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as d:
            out = Path(d) / 'probe.json'; out.write_text('old')
            with patch.object(pm, 'run_probe', return_value={'bad': float('nan')}):
                self.assertEqual(pm.main(['--n', '8', '--out', str(out)]), 1)
            self.assertEqual(out.read_text(), 'old')
            self.assertEqual(list(Path(d).iterdir()), [out])

    def test_cli_invalid_count_preserves_old(self):
        root = Path(os.environ.get('TIMESTRAL_WORK', './work'))
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as d:
            out = Path(d) / 'probe.json'; out.write_text('old')
            for n in ('0', '-1', '10000001', '1.5'):
                p = subprocess.run([sys.executable, pm.__file__, '--n', n, '--out', str(out)],
                                   capture_output=True, text=True)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn('--n', p.stderr)
                self.assertEqual(out.read_text(), 'old')


if __name__ == '__main__':
    unittest.main()
