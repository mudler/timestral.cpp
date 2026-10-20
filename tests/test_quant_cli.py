#!/usr/bin/env python3
"""Synthetic CLI subprocess checks. No models, NumPy, or external corpus required."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--fixture-tool')
    parser.add_argument('--disabled', action='store_true')
    args = parser.parse_args()
    checks = 0

    def invoke(argv, code, stdout_empty=False):
        nonlocal checks
        out = subprocess.run([args.cli]+argv, text=True, capture_output=True)
        assert out.returncode == code, (argv, out.returncode, out.stdout, out.stderr)
        if stdout_empty:
            assert out.stdout == '', out.stdout
        checks += 1
        return out

    with tempfile.TemporaryDirectory(prefix='timestral-quant-cli-') as directory:
        root = Path(directory)
        source, destination = root/'source.gguf', root/'output.gguf'
        if args.disabled:
            result = invoke(['quantize', str(source), str(destination), 'q4_k'], 2, True)
            assert 'unsupported' in result.stderr.lower() or 'unknown' in result.stderr.lower()
            assert not destination.exists()
            print('PASS disabled quantize unchanged unsupported contract')
            return
        assert args.fixture_tool, 'fixture tool required for enabled tests'
        subprocess.run([args.fixture_tool, '--fixture', str(source), 'f32'], check=True)
        original = source.read_bytes()
        fields = {'input_sha256','output_sha256','input_bytes','output_bytes',
                  'converted_tensors','copied_tensors','type'}
        for kind in ['q6_k','q5_k','q4_k']:
            destination = root/(kind+'.gguf')
            result = invoke(['quantize', str(source), str(destination), kind], 0)
            record = json.loads(result.stdout)
            assert set(record) == fields
            assert record['type'] == kind
            assert record['input_sha256'] == digest(source)
            assert record['output_sha256'] == digest(destination)
            assert record['input_bytes'] == source.stat().st_size
            assert record['output_bytes'] == destination.stat().st_size
            assert record['converted_tensors'] > 0 and record['copied_tensors'] > 0
            subprocess.run([args.fixture_tool, '--validate', str(destination)], check=True)
            saved = destination.read_bytes()
            invoke(['quantize', str(source), str(destination), kind], 1, True)
            assert destination.read_bytes() == saved
        for arguments in [[], [str(source)], [str(source), str(root/'missing')],
                          [str(source), str(root/'missing'), 'q4_k', 'extra']]:
            invoke(['quantize']+arguments, 2, True)
        for kind in ['f32','f16','q8_0','q4_K','Q4_K','q2_k','unknown']:
            invoke(['quantize',str(source),str(root/'missing'),kind],2,True)
        assert not (root/'missing').exists()
        invoke(['quantize',str(source),str(source),'q4_k'],1,True)
        alias = root/'alias.gguf'
        alias.hardlink_to(source)
        invoke(['quantize',str(source),str(alias),'q4_k'],1,True)
        alias.unlink()
        alias.symlink_to(source.name)
        invoke(['quantize',str(source),str(alias),'q4_k'],1,True)
        invoke(['quantize',str(alias),str(root/'missing'),'q4_k'],1,True)
        for mode in ['f16','q8_0']:
            bad = root/(mode+'.gguf')
            subprocess.run([args.fixture_tool,'--fixture',str(bad),mode],check=True)
            invoke(['quantize',str(bad),str(root/'missing'),'q4_k'],1,True)
        for content in [b'not GGUF', original[:-100]]:
            bad = root/'truncated.gguf'
            bad.write_bytes(content)
            invoke(['quantize',str(bad),str(root/'missing'),'q4_k'],1,True)
            assert not (root/'missing').exists()
        if Path('/dev/full').exists():
            published = root/'report-failure.gguf'
            with open('/dev/full', 'w') as sink:
                failed = subprocess.run([args.cli,'quantize',str(source),str(published),'q4_k'],
                                        text=True,stdout=sink,stderr=subprocess.PIPE)
            assert failed.returncode == 1 and published.exists()
            assert 'output file exists' in failed.stderr
            checks += 1
        assert source.read_bytes() == original
        assert not list(root.glob('.timestral-quant-*'))
    print(f'PASS quantize CLI checks={checks}; all three formats, hashes, schema, errors, no clobber')


if __name__ == '__main__':
    main()
