"""Negative CLI fixtures; originals remain read-only. No Python packages required."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class Coverage(unittest.TestCase):
    def run_gate(self, root, real=False):
        env = dict(os.environ, TIMESTRAL_TEST_TINY=str(root),
                   TIMESTRAL_TEST_REAL_BASELINES=str(root),
                   TIMESTRAL_TEST_GGUF=str(SOURCE / "tests/fixtures/tiny/timestral-tiny-f32.gguf"))
        return subprocess.run([BINARY] + (['--real'] if real else []), env=env,
                              capture_output=True, text=True)

    def test_missing_bench(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            (root / 'real').mkdir()
            # Presence preflight must reject the absent manifest case before opening files.
            import json
            for case in json.loads((SOURCE / 'tests/cases.json').read_text())['cases']:
                if case['name'] != 'bench_32k':
                    (root / 'real' / ('baseline-' + case['name'] + '-f32.gguf')).touch()
            result = self.run_gate(root, True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('MISSING real/bench_32k', result.stderr)

    def test_missing_required_tensors(self):
        for case, names in [
            ('long_h', ['pos.b19.scale.in', 'neg.b19.scale.in']),
            ('known_cov', ['pos.b0.known.rows']),
            ('known_cov', ['neg.b0.known.scale']),
            ('uni_short', ['pos.b0.refine.clamp.in']),
            ('uni_short', ['neg.b0.refine.clamp.out']),
        ]:
            with self.subTest(names=names), tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                root = Path(tmp)
                for variant in ('tiny', 'tiny-bias'):
                    (root / variant).mkdir()
                    for original in (SUITE / variant).glob('baseline-*-f32.gguf'):
                        (root / variant / original.name).symlink_to(original)
                target = root / 'tiny' / ('baseline-' + case + '-f32.gguf')
                data = target.read_bytes()
                target.unlink()
                for name in names:
                    old = name.encode()
                    self.assertTrue(old in data, name)
                    data = data.replace(old, b'X' + old[1:])
                target.write_bytes(data)
                result = self.run_gate(root)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('missing ' + names[0], result.stderr)


if __name__ == '__main__':
    BINARY = sys.argv.pop(1)
    SUITE = Path(sys.argv.pop(1)).resolve()
    SCRATCH = sys.argv.pop(1)
    SOURCE = Path(__file__).resolve().parents[1]
    unittest.main()
