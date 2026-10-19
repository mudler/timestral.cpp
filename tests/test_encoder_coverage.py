"""Replay CLI must fail closed; temporary fixture copies never change originals."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY, SOURCE, SUITE, SCRATCH = sys.argv[1:5]
sys.argv = sys.argv[:1]


class EncoderCoverage(unittest.TestCase):
    def invoke(self, root, selection=None):
        env = dict(os.environ, TIMESTRAL_TEST_TINY=str(root),
                   TIMESTRAL_TEST_REAL_BASELINES=str(root))
        if selection is not None:
            env['TIMESTRAL_TEST_REAL_CASES'] = selection
        return subprocess.run([BINARY] + (['--real'] if selection else []),
                              env=env, capture_output=True, text=True)

    def test_invalid_selection(self):
        for selection, message in [('not_a_case', 'unknown real case'),
                                   ('multi,multi', 'duplicate real case')]:
            result = self.invoke(SUITE, selection)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(message, result.stderr)

    def test_missing_case(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            result = self.invoke(tmp)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('MISSING', result.stderr)

    def test_missing_families(self):
        # Same-length GGUF name edits preserve offsets and produce valid files.
        for name in ['pos.b0.layer.0.out',
                     'floor_tf.pos.b0.layer.0.out',
                     'floor_fr.pos.b0.layer.0.out',
                     'stat.pos.b0.layer.0.out.numel',
                     'stat.pos.b0.layer.0.out.rms',
                     'cap.pos.b0.layer.0.out',
                     'floor_fr.pos.b0.out_norm',
                     'stat.pos.b0.out_norm.rms',
                     'cap.pos.b0.out_norm']:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                root = Path(tmp)
                for variant in ['tiny', 'tiny-bias']:
                    (root / variant).mkdir()
                    for path in (Path(SUITE) / variant).glob('*-f32.gguf'):
                        (root / variant / path.name).symlink_to(path)
                path = root / 'tiny' / 'baseline-uni_short-f32.gguf'
                data = path.read_bytes()
                # Include string length to avoid replacing a prefix of qkv.
                key = len(name).to_bytes(8, 'little') + name.encode()
                self.assertIn(key, data)
                replacement = len(name).to_bytes(8, 'little') + ('X' + name[1:]).encode()
                path.unlink()
                path.write_bytes(data.replace(key, replacement))
                result = self.invoke(root)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('missing', result.stderr.lower())


unittest.main()
