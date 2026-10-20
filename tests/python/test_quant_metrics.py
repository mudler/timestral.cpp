"""Known-good and known-bad model-free metric contracts."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import quant_metrics as qm


class Metrics(unittest.TestCase):
    def test_normalized_frozen_anchor(self):
        f = np.ones((2, 3))
        a = qm.normalized(f, f+.2, 0., 1.)
        expected = .2/(math.sqrt(2)*math.asinh(1))
        self.assertAlmostEqual(a['mean'].value, expected)
        self.assertAlmostEqual(a['maximum'].value, expected)
        self.assertEqual(qm.normalized(f, f, 0., 1.)['mean'].value, 0)
        self.assertNotEqual(qm.normalized(f, f+.2, -1., 2.)['mean'].value, a['mean'].value)
        self.assertEqual(qm.normalized(f, f, 1., 1.)['mean'].status, 'zero-rms-identical')
        self.assertEqual(qm.normalized(f, f+.1, 1., 1.)['mean'].value, math.inf)
        for args in [(f, f, 0, 0), (f, f, math.nan, 1), (f, f[:, :1], 0, 1),
                     (f, f*np.nan, 0, 1)]:
            with self.assertRaises(ValueError):
                qm.normalized(*args)

    def test_wql(self):
        self.assertEqual(qm.wql([2], [[1]], [.5]).value, .5)
        self.assertEqual(qm.wql([2], [[2]], [.5]).value, 0)
        self.assertEqual(qm.wql([0], [[0]], [.5]).status, 'zero-truth-exact')
        self.assertEqual(qm.wql([0], [[1]], [.5]).value, math.inf)
        self.assertGreater(qm.wql([2], [[-100]], [.5]).value, 1)
        with self.assertRaises(ValueError):
            qm.wql([2], [[1, 2]], [.8, .2])

    def test_mase(self):
        self.assertEqual(qm.mase([5], [3], [1, 2, 4]).value, 2/1.5)
        self.assertEqual(qm.mase([5], [3], [100, 2, 4], max_context=2).value, 1)
        self.assertEqual(qm.mase([5], [5], [5, 5]).status, 'zero-scale-exact')
        self.assertEqual(qm.mase([5], [4], [5, 5]).value, math.inf)
        self.assertIsNone(qm.mase([5], [5], [1, np.nan, 2]).value)
        self.assertEqual(qm.mase([5], [3], [1, 2, np.nan, 4, 5]).value, 2)

    def test_aggregation_no_hidden_exclusions(self):
        s = qm.summarize({'a': [qm.Score(0), qm.Score(2)], 'b': [qm.Score(5)]})
        self.assertEqual(s['aggregate']['value'], 3)
        self.assertEqual(s['item_count'], 2)
        self.assertEqual(s['row_count'], 3)
        s = qm.summarize({'a': [qm.Score(math.inf)], 'b': [qm.Score(None, 'missing')]})
        self.assertEqual(s['aggregate']['kind'], 'undefined')
        self.assertEqual(s['row_status_counts']['missing'], 1)
        self.assertEqual(qm.average([qm.Score(math.inf), qm.Score(1)]).value, math.inf)
        json.dumps(s, allow_nan=False)
        json.dumps(qm.Score(math.inf).json(), allow_nan=False)
        with self.assertRaises(ValueError):
            qm.average([])

    def test_relative_changes(self):
        a = qm.relative_change(qm.Score(10), qm.Score(9))
        self.assertEqual(a['signed'].value, -10)
        self.assertEqual(a['absolute'].value, 10)
        self.assertEqual(qm.relative_change(qm.Score(0), qm.Score(0))['signed'].value, 0)
        self.assertEqual(qm.relative_change(qm.Score(0), qm.Score(1))['signed'].value, math.inf)
        self.assertIsNone(qm.relative_change(qm.Score(math.inf), qm.Score(math.inf))['signed'].value)
        self.assertIsNone(qm.relative_change(qm.Score(None), qm.Score(1))['signed'].value)

    def test_coverage_and_order(self):
        c = qm.coverage(['a', 'b'], {'a': 'forecast', 'b': 'expected-error-match'})
        self.assertEqual(c['numeric_acceptance'], 'HOLD')
        self.assertEqual(c['forecasts'], 1)
        self.assertEqual(c['gaps'], {'b': 'expected-error-match'})
        self.assertEqual(qm.coverage(['a'], {'a': 'forecast'})['numeric_acceptance'], 'READY')
        with self.assertRaises(ValueError):
            qm.coverage(['a', 'b'], {'a': 'forecast'})
        self.assertTrue(qm.monotonic([qm.Score(i) for i in range(6)]))
        self.assertFalse(qm.monotonic([qm.Score(i) for i in [0, 1, 1, 3, 4, 5]]))
        self.assertFalse(qm.monotonic([qm.Score(i) for i in [0, 1, 3, 2, 4, 5]]))
        self.assertFalse(qm.monotonic([qm.Score(i) for i in [0, 1, 2, 3, 4, math.inf]]))


if __name__ == '__main__':
    unittest.main()
