"""Score recorded quant-record-v1 artifacts; never execute models."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

import quant_fixtures as fixtures
import quant_metrics as metrics

FORMATS = ('f32', 'f16', 'q8_0', 'q6_k', 'q5_k', 'q4_k')
LEVELS = np.array([.01, .05] + [i/20 for i in range(2, 19)] + [.95, .99], dtype='<f8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(obj, expected):
    require(isinstance(obj, dict) and set(obj) == set(expected), 'unexpected fields')


def sha(value, length=64):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value),
            'invalid identity hash')


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _record(record):
    require(isinstance(record, dict), 'invalid output record')
    if record.get('status') == 'forecast':
        keys(record, ['status'])
    else:
        keys(record, ['status', 'type', 'message', 'stage'])
        require(record['status'] in ('expected-error', 'error'), 'invalid output status')
        require(all(isinstance(record[k], str) and record[k] for k in ('type', 'stage'))
                and isinstance(record['message'], str), 'invalid error detail')


def evaluate(manifest_path):
    path = Path(manifest_path)
    manifest = read_json(path)
    keys(manifest, ['version', 'fixture_version', 'options', 'source_sha256',
                    'driver', 'models', 'items'])
    require(manifest['version'] == 'quant-record-v1' and
            manifest['fixture_version'] == fixtures.VERSION, 'unsupported protocol')
    require(manifest['options'] == fixtures.OPTIONS, 'forecast options differ')
    keys(manifest['driver'], ['source_sha256', 'runtime_commit'])
    sha(manifest['driver']['source_sha256'])
    sha(manifest['driver']['runtime_commit'], 40)
    keys(manifest['models'], FORMATS)
    for model in manifest['models'].values():
        keys(model, ['sha256', 'hf_revision', 'reference_sha'])
        sha(model['sha256'])
        require(model['hf_revision'] == fixtures.HF_REVISION and
                model['reference_sha'] == fixtures.REFERENCE_SHA, 'wrong source model')
    cases = fixtures.load_cases()
    expected = {c['name']: c for c in cases}
    records = manifest['items']
    require(isinstance(records, list) and len(records) == 79, 'exactly 79 items required')
    ids = [r.get('id') for r in records]
    require(len(set(ids)) == 79 and set(ids) == set(expected), 'case identities differ')
    rows = {fmt: {} for fmt in FORMATS}
    coverage = {fmt: {} for fmt in FORMATS}
    for record in records:
        keys(record, ['id', 'case_sha256', 'input_sha256', 'truth_sha256', 'npz',
                      'arrays', 'anchor_model_sha256', 'outputs'])
        name = record['id']
        case = expected[name]
        inputs, truth, provenance = fixtures.make_fixture(case)
        require(manifest['source_sha256'] == provenance['source_sha256'], 'source hash mismatch')
        for field in ('case_sha256', 'input_sha256', 'truth_sha256'):
            require(record[field] == provenance[field], field + ' mismatch')
        basename = record['npz']
        require(isinstance(basename, str) and Path(basename).name == basename
                and basename.endswith('.npz'), 'NPZ must be a sibling basename')
        archive = path.parent / basename
        require(not archive.is_symlink(), 'NPZ symlinks not permitted')
        keys(record['outputs'], FORMATS)
        for output in record['outputs'].values():
            _record(output)
        f32_present = record['outputs']['f32']['status'] == 'forecast'
        wanted = {'context', 'truth', 'levels'}
        wanted.update('forecast_'+f for f in FORMATS if record['outputs'][f]['status'] == 'forecast')
        if f32_present:
            wanted.update(('anchor_loc', 'anchor_scale'))
            require(record['anchor_model_sha256'] == manifest['models']['f32']['sha256'],
                    'anchors must belong to F32')
        else:
            require(record['anchor_model_sha256'] is None, 'anchors without F32')
        keys(record['arrays'], wanted)
        with np.load(archive, allow_pickle=False) as stored:
            require(len(stored.files) == len(wanted) and set(stored.files) == wanted,
                    'NPZ keys differ')
            arrays = {}
            B, V, T, H = (case[k] for k in ('B', 'V', 'T', 'H'))
            for key in wanted:
                a = stored[key]
                descriptor = record['arrays'][key]
                keys(descriptor, ['dtype', 'shape', 'sha256'])
                shape = ((len(LEVELS),) if key == 'levels' else
                         (B, V, T) if key == 'context' else
                         (B, V, H) if key == 'truth' else
                         (B, V) if key.startswith('anchor_') else (B, V, H, len(LEVELS)))
                dtype = '<f8' if key == 'levels' else '<f4'
                require(a.dtype.str == dtype and a.shape == shape, 'array dtype/shape differs')
                require(descriptor == dict(dtype=dtype, shape=list(shape), sha256=fixtures.array_hash(a)),
                        'array hash/descriptor differs')
                arrays[key] = a
        require(arrays['context'].tobytes() == inputs['context'].tobytes() and
                arrays['truth'].tobytes() == truth.tobytes() and
                np.array_equal(arrays['levels'], LEVELS), 'fixture array differs')
        for fmt in FORMATS:
            output = record['outputs'][fmt]
            if output['status'] == 'forecast':
                p = arrays['forecast_'+fmt]
                require(np.all(np.isfinite(p)) and np.all(np.diff(p, axis=-1) >= 0),
                        'invalid forecast values/order')
            if not f32_present or output['status'] != 'forecast':
                base = record['outputs']['f32']
                match = base['status'] == 'expected-error' and output == base
                coverage[fmt][name] = 'expected-error-match' if match else 'error-mismatch'
                rows[fmt][name] = []
                continue
            coverage[fmt][name] = 'forecast'
            f = arrays['forecast_f32']
            item = []
            for b in range(B):
                for v in range(V):
                    n = metrics.normalized(f[b,v], p[b,v], float(arrays['anchor_loc'][b,v]),
                                           float(arrays['anchor_scale'][b,v]))
                    item.append(dict(batch=b, variate=v, mean=n['mean'], maximum=n['maximum'],
                                     wql=metrics.wql(truth[b,v], p[b,v], LEVELS),
                                     mase=metrics.mase(truth[b,v], p[b,v,:,10], inputs['context'][b,v])))
            rows[fmt][name] = item
    summaries = {}
    scores = {}
    groups = dict(overall=set(expected), c2={c['name'] for c in cases[:15]},
                  synthetic={c['name'] for c in cases[15:]})
    for fmt in FORMATS:
        summaries[fmt], scores[fmt] = {}, {}
        for group, names in groups.items():
            summaries[fmt][group], scores[fmt][group] = {}, {}
            for metric in ('mean', 'maximum', 'wql', 'mase'):
                items = {name: [r[metric] for r in rows[fmt][name]] or
                         [metrics.Score(None, 'undefined-member')] *
                         (expected[name]['B'] * expected[name]['V']) for name in sorted(names)}
                summary = metrics.summarize(items)
                if metric == 'maximum':
                    vals = [s.value for values in items.values() for s in values]
                    score = metrics.Score(None, 'undefined-member') if None in vals else metrics.Score(max(vals))
                    summary['global_maximum'] = score.json()
                score = metrics.average([metrics.average(v) for v in items.values()])
                summaries[fmt][group][metric] = summary
                scores[fmt][group][metric] = score
    cover = {f: metrics.coverage(list(expected), coverage[f]) for f in FORMATS}
    changes = {}
    for fmt in FORMATS:
        changes[fmt] = {g: {m: {k: v.json() for k, v in metrics.relative_change(
            scores['f32'][g][m], scores[fmt][g][m]).items()} for m in ('wql', 'mase')} for g in groups}
    ordered = metrics.monotonic([scores[f]['overall']['mean'] for f in FORMATS])
    gates = {}
    for fmt, bound, lossbound in [('f16', .001, .5), ('q8_0', .005, 1.)]:
        mean = scores[fmt]['overall']['mean'].value
        loss = changes[fmt]['overall']['wql']['absolute']
        gates[fmt] = bool(mean is not None and math.isfinite(mean) and mean <= bound and
                          loss['kind'] == 'finite' and loss['value'] <= lossbound)
    hold = any(c['numeric_acceptance'] == 'HOLD' for c in cover.values())
    return dict(version='quant-report-v1', mode='recorded-output-only',
                provenance=dict(manifest_sha256=fixtures.digest(path.read_bytes()),
                                source_sha256=manifest['source_sha256'], options=manifest['options'],
                                checked_fixture_and_array_hashes=True,
                                driver_and_model_identities='asserted-not-execution-verified',
                                manifest=manifest['driver'], models=manifest['models']),
                coverage=cover, summaries=summaries, relative_changes=changes,
                rows={f: {n: [{k: v.json() if isinstance(v, metrics.Score) else v
                               for k,v in r.items()} for r in rs] for n,rs in items.items()}
                      for f,items in rows.items()},
                gates=dict(bounds=gates, strict_monotonic_mean=ordered),
                result='HOLD' if hold else 'PASS' if ordered and all(gates.values()) else 'FAIL')


def write_report(path, report):
    path = Path(path)
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2)+'\n'
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as out:
            name = out.name
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    try:
        manifest_path = Path(args.manifest)
        report_path = Path(args.report).resolve()
        require(manifest_path.resolve() != report_path, 'report overwrites manifest')
        raw = read_json(manifest_path)
        for record in raw.get('items', []):
            if isinstance(record, dict) and isinstance(record.get('npz'), str):
                require((manifest_path.parent / record['npz']).resolve() != report_path,
                        'report overwrites recorded arrays')
        report = evaluate(args.manifest)
        write_report(args.report, report)
        return 0 if report['result'] == 'PASS' else 1
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(2, 'error: '+str(error)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
