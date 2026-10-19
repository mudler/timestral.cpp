"""Tiny CPU teacher-forced boundary regressions (no artifact regeneration)."""
import test_generator_hooks as hooks
import unittest
import gen_reference_baseline as gen
import numpy as np
import torch
from timestral.forecaster import TimestralForecaster
from unittest.mock import patch


class RerunContractTests(unittest.TestCase):
    setUp = hooks.GeneratorHookTests.setUp
    remove_hooks = hooks.GeneratorHookTests.remove_hooks
    install_hooks = hooks.GeneratorHookTests.install_hooks
    forward = hooks.GeneratorHookTests.forward
    def source(self, split=False):
        gen.install_instrumentation()
        self.install_hooks()
        gen.REC = gen.Rec()
        gen.CTX.pass_name, gen.CTX.block = 'pos', 0
        mask = None
        if split:
            mask = gen.TwoStreamMask(torch.ones(2, 1, 3, 3, dtype=torch.bool), 2)
        with torch.inference_mode():
            self.model.transformer(self.x, mask, None, self.positions)
        src = dict(gen.REC.t)
        src['pos.b0.args.inputs'] = self.x.numpy()
        src['pos.b0.tf.positions'] = self.positions.numpy()
        if split:
            src['pos.b0.mask.time'] = mask.mask.numpy()
            src['pos.b0.tf.enc_len'] = np.array([2])
        return src

    def rerun(self, src):
        h = gen.Harness(self.model, TimestralForecaster(self.model), next(self.model.parameters()).dtype, set(range(4)))
        # The fixture isolates transformer components, not predict/refine.
        with patch.object(h, 'T', side_effect=lambda s, p, n: None if n == 'args.inputs' else gen.Harness.T(h, s, p, n)):
            return h.run(src)

    def test_dumped_boundaries_ignore_upstream_perturbations(self):
        src = self.source()
        baseline = self.rerun(src)
        # A local output must not inherit rounding/errors before its dumped input.
        for li, layer in enumerate(self.model.transformer.layers):
            for j, attn in enumerate(layer.attns):
                b = f'pos.b0.layer.{li}.attn.{j}.'
                for input_name, outputs in [('norm', ['q_rope', 'sdpa', 'gated', 'out']),
                                            ('q', (['sdpa'] if attn.is_time else []) + ['gated', 'out']),
                                            ('sdpa', ['out']), ('gate', ['out'])]:
                    if b + input_name not in src:
                        continue
                    changed = dict(src)
                    changed[b + input_name] = src[b + input_name] + 0.75
                    out = self.rerun(changed)
                    for output in outputs:
                        if b + output in baseline:
                            with self.subTest(layer=li, axis=j, input=input_name, output=output):
                                np.testing.assert_array_equal(out[b + output], baseline[b + output])

    def test_missing_expected_output_fails_selfcheck(self):
        src = self.source()
        out = self.rerun(src)
        del out['pos.b0.layer.0.attn.0.q']
        ratio, where = gen.selfcheck(out, src)
        self.assertTrue(np.isinf(ratio), where)

    def test_missing_each_contract_fails_selfcheck(self):
        src = self.source()
        out = self.rerun(src)
        for name in out:
            incomplete = dict(out)
            del incomplete[name]
            ratio, where = gen.selfcheck(incomplete, src)
            self.assertTrue(np.isinf(ratio), (name, where))

    def test_harness_exception_resets_context(self):
        src = self.source()
        h = gen.Harness(self.model, None, torch.float32, set(range(4)))
        with patch.object(h, 'attention', side_effect=RuntimeError('rerun failed')):
            with self.assertRaisesRegex(RuntimeError, 'rerun failed'):
                h.run(src)
        self.assertFalse(gen.CTX.rerun)
        self.assertIsNone(gen.CTX.attn)
        self.assertIsNone(gen.CTX.pass_name)

    def test_math_selfcheck_split_and_plain(self):
        for split in (False, True):
            src = self.source(split)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                out = self.rerun(src)
            self.assertEqual(set(out), gen.expected_reruns(src).keys())
            ratio, where = gen.selfcheck(out, src)
            self.assertLess(ratio, 2e-6, where)
            self.remove_hooks()

    def test_f64_truth_all_components_and_contexts(self):
        self.model.double()
        self.x = self.x.double()
        self.model.transformer.rope.inv_freq64 = gen._truth_inv_freq(self.model.transformer.rope)
        with patch.object(gen.RMSNorm, 'forward', gen._truth_rmsnorm_forward), \
             patch.object(gen.SinCosRoPE, 'forward', gen._truth_rope_forward), \
             torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for split in (False, True):
                src0 = self.source(split)
                self.remove_hooks()
                self.install_hooks()
                src = {}
                for p in ('pos', 'neg'):
                    for k in (0, 1):
                        src.update({n.replace('pos.b0.', f'{p}.b{k}.'): a for n, a in src0.items()})
                out = self.rerun(src)
                self.assertEqual(set(out), set(gen.expected_reruns(src)))
                ratio, where = gen.selfcheck(out, src)
                self.assertLessEqual(ratio, 1e-12, where)
                self.remove_hooks()

    def test_no_gate_layout_and_gold_invariance(self):
        for layer in self.model.transformer.layers:
            for attn in layer.attns:
                attn._gate = None
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            before = self.forward(self.x)
            src = self.source()
            out = self.rerun(src)
            ratio, where = gen.selfcheck(out, src)
            self.assertLess(ratio, 2e-6, where)
            self.assertFalse(any(n.endswith('.gate') for n in out))
            self.remove_hooks()
            after = self.forward(self.x)
            np.testing.assert_array_equal(before.numpy(), after.numpy())

    def test_combined_refine_uses_computed_prediction(self):
        from types import SimpleNamespace
        computed = torch.full((1, 2, 1), 7.)
        dumped = np.full((1, 2, 1), 3., dtype=np.float32)
        model = SimpleNamespace(config=SimpleNamespace(block_schedule=()),
                                predict_block=lambda *args: computed)
        calls = []
        def refine(q, *args):
            calls.append(q.clone())
            result = q + 1
            gen.REC.put('refined', result)
            return result
        fc = SimpleNamespace(_refine_block=refine)
        h = gen.Harness(model, fc, torch.float32, set())
        src = {'pos.b0.args.inputs': np.ones((1, 2), np.float32),
               'pos.b0.args.horizon_patches': np.array([1]), 'pos.b0.predict': dumped}
        out = h.run(src)
        torch.testing.assert_close(calls[0], computed)
        np.testing.assert_array_equal(calls[1].numpy(), dumped)
        np.testing.assert_array_equal(out['pos.b0.block_refined'], computed.numpy() + 1)
        np.testing.assert_array_equal(out['pos.b0.refined'], dumped + 1)

    def test_failures_propagate_and_nonfinite_selfcheck_fails(self):
        from types import SimpleNamespace
        fc = SimpleNamespace(forecast=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('hook failed')))
        with self.assertRaisesRegex(RuntimeError, 'hook failed'):
            gen.run_forecast(fc, {'H': 1}, {'context': np.ones((1, 1, 2))}, set())
        ref = {'pos.b0.predict': np.ones(1)}
        ratio, _ = gen.selfcheck({'pos.b0.predict': np.array([np.nan])}, ref)
        self.assertTrue(np.isinf(ratio))

    def test_floor_selects_combined_not_isolated_refine(self):
        from rerun_contracts import floor_inputs
        data = {'pos.b0.refined': np.array([3.]), 'pos.b0.block_refined': np.array([8.])}
        self.assertEqual(floor_inputs(data)['pos.b0.refined'][0], 8.)
        del data['pos.b0.block_refined']
        with self.assertRaisesRegex(ValueError, 'combined block contract missing'):
            floor_inputs(data)

    def test_explicit_attention_inputs_with_spies(self):
        src = self.source(True)
        h = gen.Harness(self.model, None, torch.float32, set(range(4)))
        for li, layer in enumerate(self.model.transformer.layers):
            for j, attn in enumerate(layer.attns):
                base = f'layer.{li}.attn.{j}.'
                T = lambda n: h.T(src, 'pos.b0.', n)
                x = T(base + 'norm')
                axis = x if attn.is_time else x.permute(1, 0, 2)
                mask = gen.TwoStreamMask(T('mask.time').bool(), 2) if attn.is_time else None
                gen.REC = gen.Rec(stats=False)
                with patch.object(attn.wQKV, 'forward', wraps=attn.wQKV.forward) as projection, \
                     patch.object(attn.wO, 'forward', wraps=attn.wO.forward) as wo, \
                     patch.object(attn._gate, 'forward', wraps=attn._gate.forward) as gate, \
                     patch.object(gen._layers, '_sdpa', wraps=gen._layers._sdpa) as sdpa, \
                     patch.object(gen._layers, '_apply_rotary', wraps=gen._layers._apply_rotary) as rotary, \
                     torch.inference_mode():
                    h.attention(attn, T, base, mask)
                    torch.testing.assert_close(projection.call_args.args[0], axis, rtol=0, atol=0)
                    torch.testing.assert_close(gate.call_args.args[0], axis, rtol=0, atol=0)
                    torch.testing.assert_close(wo.call_args.args[0], T(base + 'gated'), rtol=0, atol=0)
                    q = T(base + ('q_rope' if attn.is_time else 'q'))
                    k = T(base + ('k_rope' if attn.is_time else 'k'))
                    v = T(base + 'v')
                    expected = [(q[:, :2], k[:, :2], v[:, :2], mask.mask[:, :, :2, :2]),
                                (q[:, 2:], k, v, mask.mask[:, :, 2:, :])] if attn.is_time else [(q, k, v, None)]
                    self.assertEqual(len(sdpa.call_args_list), len(expected))
                    for call, args in zip(sdpa.call_args_list, expected):
                        for actual, want in zip(call.args, args):
                            if want is None:
                                self.assertIsNone(actual)
                            else:
                                torch.testing.assert_close(actual, want, rtol=0, atol=0)
                    if attn.is_time:
                        for actual, name in zip(rotary.call_args.args, ('q', 'k', 'cos', 'sin')):
                            torch.testing.assert_close(actual, T(base + name), rtol=0, atol=0)
                    gated = T(base + 'sdpa') * T(base + 'gate').sigmoid()
                    if not attn.is_time:
                        gated = gated.permute(1, 0, 2)
                    np.testing.assert_array_equal(gen.REC.t['pos.b0.' + base + 'gated'], gated.numpy())
