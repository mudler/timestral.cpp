"""Exercise categorical contracts through the pinned, instrumented CPU producer."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
from timestral.config import TimestralConfig
from timestral.model import TimestralModel

import gen_reference_baseline as gen
import baseline_contract as contract
import baseline_io as bio
import test_compute_floors as floor_tests
import test_check_baseline as acceptance


def produce(mode, variant='known'):
    # Restore every global wrapper/truth patch: no model loading or forecast.
    with ExitStack() as stack:
        stack.enter_context(patch.object(gen, 'REC', gen.Rec()))
        stack.enter_context(patch.object(gen, 'CTX', gen.Ctx()))
        stack.enter_context(patch.object(gen, '_INSTALLED', False))
        for owner, names in ((gen.RMSNorm, ('forward',)), (gen.SinCosRoPE, ('forward',)),
                             (gen.TimestralForecaster, ('_as_context', '_as_covariates', '_encode_categorical'))):
            for name in names:
                stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        original_wrap = gen._wrap

        def wrap(owner, name, *args, **kwargs):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
            return original_wrap(owner, name, *args, **kwargs)

        stack.enter_context(patch.object(gen, '_wrap', side_effect=wrap))
        model = TimestralModel(TimestralConfig(
            patch_size=2, d_model=4, num_heads=1, mlp_hidden_dim=8,
            block_schedule=('time',), quantile_knots=(0.5,), block_size=2, context_length=8))
        if mode == 'f64':
            gen.apply_truth_patches(model)
        gen.install_instrumentation()
        fc = gen.TimestralForecaster(model)
        gen.CTX.pass_name = 'pos'
        context = [[[1., 2., 3., 4.]]]
        if variant == 'missing_context':
            context[0][0][0] = np.nan
        if variant in ('raw', 'raw_missing'):
            context[0].append([2., 3., 4., 5.])
        x, mask = fc._as_context(context, None)
        target = fc._encoding_target(x, mask)
        codes = [1., np.nan, 1., 2.]
        if variant != 'past':
            codes += [np.nan, 3.]
        if variant == 'raw':
            codes = np.nan_to_num(codes).tolist()
        codes = fc._as_categorical([[codes]], 1, len(codes), 'codes')
        fc._encode_categorical(codes, target)
        return {n: a.copy() for n, a in gen.REC.t.items()}


def evidence(variant='known'):
    data = floor_tests.floor_fixture()
    for mode in ('f32', 'f32-math', 'f64'):
        t, kv = data[mode]
        records = produce('f64' if mode == 'f64' else 'f32', variant)
        t.update(records)
        for n, a in records.items():
            kv[f'stat.{n}.numel'] = a.size
            kv[f'stat.{n}.rms'] = bio.rms(a)
    return data


class CategoricalContractTests(unittest.TestCase):
    def test_actual_producer_precision_and_missing_inputs(self):
        for variant in ('known', 'past', 'missing_context', 'raw'):
            with self.subTest(variant=variant):
                data = evidence(variant)
                for mode in ('f32', 'f64'):
                    records = data[mode][0]
                    model_type = np.float64 if mode == 'f64' else np.float32
                    self.assertEqual(records['pos.enc_cat.0.codes'].dtype, np.float64)
                    self.assertEqual(records['pos.enc_cat.0.out'].dtype, model_type)
                    if variant != 'raw':
                        self.assertEqual(records['pos.enc_cat.0.target'].dtype, model_type)
                        for suffix in ('ctx', 'past') + (() if variant == 'past' else ('future',)):
                            self.assertEqual(records['pos.target_encode.0.' + suffix].dtype, np.float64)
                        for suffix in ('out_past',) + (() if variant == 'past' else ('out_future',)):
                            self.assertEqual(records['pos.target_encode.0.' + suffix].dtype, np.float32)
                self.assertEqual(acceptance.AcceptanceTests().run_fixture(data), [])
                result, write = floor_tests.FloorTests().run_data(data)
                self.assertTrue(result[0])
                write.assert_called_once()

    def test_each_record_wrong_precision_and_integer_rejected(self):
        for mode in ('f32', 'f32-math', 'f64'):
            records = produce('f64' if mode == 'f64' else 'f32')
            for n, arr in records.items():
                for dtype in (np.int32, np.float32 if arr.dtype == np.float64 else np.float64):
                    with self.subTest(mode=mode, name=n, dtype=dtype):
                        data = evidence()
                        data[mode][0][n] = np.nan_to_num(arr).astype(dtype)
                        errors = acceptance.AcceptanceTests().run_fixture(data)
                        self.assertTrue(any(n in e and 'dtype' in e for e in errors), errors)
                        with self.assertRaisesRegex(ValueError, 'dtype'):
                            floor_tests.FloorTests().run_data(data)

    def test_nan_mismatch_and_nonfinite_outputs_rejected(self):
        for n in produce('f64'):
            with self.subTest(name=n):
                data = evidence()
                if contract.allows_input_nan(n):
                    # Ensure a mismatch even for context with no original NaN.
                    arr = data['f64'][0][n]
                    arr.flat[0] = 0. if np.isnan(arr.flat[0]) else np.nan
                else:
                    for t, _ in data.values():
                        if n in t:
                            t[n].flat[0] = np.nan
                with self.assertRaises(ValueError):
                    floor_tests.FloorTests().run_data(data)
                data = evidence()
                data['f64'][0][n].flat[0] = np.inf
                with self.assertRaises(ValueError):
                    floor_tests.FloorTests().run_data(data)

    def test_raw_missing_codes_are_nonfinite_output_not_input(self):
        # The multivariate bypass casts raw codes, retaining their NaNs in out.
        # Do not let input permission authorize this output as floor evidence.
        data = evidence('raw_missing')
        self.assertTrue(np.isnan(data['f32'][0]['pos.enc_cat.0.out']).any())
        with self.assertRaisesRegex(ValueError, 'nonfinite evidence pos.enc_cat.0.out'):
            floor_tests.FloorTests().run_data(data)

    def test_unknown_category_record_fails_closed(self):
        self.assertIsNone(contract.semantic_type('pos.enc_cat.0.surprise'))
        self.assertFalse(contract.allows_input_nan('pos.enc_cat.0.out'))
        self.assertFalse(contract.allows_input_nan('pos.target_encode.0.out_past'))
        self.assertFalse(contract.allows_input_nan('input.surprise'))
