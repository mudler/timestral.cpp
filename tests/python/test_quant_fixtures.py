"""Model-free input preservation and independent future checks."""
import ast
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import quant_fixtures as qf


class Fixtures(unittest.TestCase):
    def test_coverage(self):
        cases = qf.load_cases()
        self.assertEqual(len(cases), 79)
        self.assertEqual(sum('slow' in c['labels'] for c in cases), 2)
        self.assertEqual([c['seed'] for c in cases[15:]], list(range(20000, 20064)))
        self.assertEqual(len({c['name'] for c in cases}), 79)

    def test_exact_original_body_without_imports(self):
        # Independent extraction oracle: source text body, not helper AST selection.
        source = (qf.ROOT/'scripts/gen_reference_baseline.py').read_text()
        start = source.index('def make_inputs(')
        end = source.index('\n\ndef case_layers(', start)
        ns = dict(np=np, math=math)
        exec(source[start:end], ns)
        generator = qf.input_generator()
        for original in qf.load_cases():
            case = copy.deepcopy(original)
            case.update(T=64, H=9)  # Small shapes only; retain every covariate/missingness path.
            a = ns['make_inputs'](case, 32)
            inputs, truth, provenance = qf.make_fixture(case)
            self.assertEqual(set(a), set(inputs))
            for key in a:
                self.assertEqual(a[key].dtype, inputs[key].dtype)
                self.assertEqual(a[key].tobytes(), inputs[key].tobytes())
                self.assertEqual(generator(case, 32)[key].tobytes(), inputs[key].tobytes())
            self.assertEqual(truth.shape, (case['B'], case['V'], 9))
            self.assertTrue(np.all(np.isfinite(truth)))
            self.assertEqual(provenance['truth_sha256'], qf.array_hash(truth))
            self.assertEqual(provenance['version'], 'c2-truth-v1')

    def test_future_independence_and_prefix(self):
        case = dict(name='unit', B=2, V=2, T=32, H=8, seed=104, series={})
        a = qf.future_truth(case)
        self.assertEqual(a.tobytes(), qf.future_truth(case).tobytes())
        changed = copy.deepcopy(case)
        changed.update(past=dict(numeric=2, nan_frac=.2),
                       known=dict(numeric=1, indicator=1), missing=dict(scattered_frac=.5))
        self.assertEqual(a.tobytes(), qf.future_truth(changed).tobytes())
        changed['H'] = 16
        np.testing.assert_array_equal(a, qf.future_truth(changed)[..., :8])
        changed['seed'] = 105
        self.assertFalse(np.array_equal(a, qf.future_truth(changed)[..., :8]))

    def test_formula_and_constant_positive(self):
        case = dict(B=1, V=1, T=7, H=3, seed=5,
                    series=dict(level=-100, trend=.1, period=12, noise_std=0, positive=True))
        t = np.arange(7, 10, dtype=np.float64)
        expected = np.abs(-100+.1*t+10*np.sin(2*math.pi*t/12)).astype(np.float32)
        np.testing.assert_array_equal(qf.future_truth(case)[0, 0], expected)
        case['series']['constant'] = 5
        self.assertTrue(np.all(qf.future_truth(case) == 5))

    def test_stochastic_truth_oracle(self):
        case = dict(B=1, V=2, T=5, H=3, seed=77, series={})
        rng = np.random.default_rng(77)
        expected = []
        for v in range(2):
            increments = rng.normal(size=5).tolist()
            future = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([77, 7001, 0, v]))).normal(size=3)
            walk = 0.
            for value in increments:
                walk += value
            row = []
            for h, value in enumerate(future):
                walk += value
                row.append(10*math.sin(2*math.pi*(5+h)/(24*(1+.25*v)))+walk)
            expected.append(row)
        np.testing.assert_array_equal(qf.future_truth(case)[0], np.asarray(expected, np.float32))

    def test_no_heavy_imports(self):
        tree = ast.parse((qf.ROOT/'scripts/quant_fixtures.py').read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) or 'timestral' in ast.unparse(n)
                             for n in imports))


if __name__ == '__main__':
    unittest.main()
