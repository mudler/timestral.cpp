#!/usr/bin/env python3
"""Produce complete recorded six-format inputs; never convert models or claim quality."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

import eval_quant
import quant_fixtures as fixtures
import quant_protocol as protocol
import record_quant as development

FORMATS = eval_quant.FORMATS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metadata(path):
    """Measure GGUF metadata/storage, not filename claims. Read-only mmap."""
    from gguf import GGUFReader, GGMLQuantizationType as Type
    import re
    reader = GGUFReader(path, mode='r')
    def field(key):
        value = reader.get_field(key)
        require(value is not None, 'missing model metadata: '+key)
        # ReaderField.contents preserves scalar/string/array semantics.
        return value.contents()
    fields = {key: field(key) for key in reader.fields
              if key.startswith(('timestral.', 'general.license', 'general.source'))}
    kv_types = {key:[int(t) for t in reader.get_field(key).types] for key in fields}
    pattern = re.compile(r'blk\.[0-9]+\.(?:attn\.[0-9]+\.(?:qkv|gate|out)|ffn_w12|ffn_w3)\.weight')
    extras = {'patch.out.weight','head.hidden.weight','head.out.weight','head.res.weight'}
    tensors = {}
    for t in reader.tensors:
        require(t.name not in tensors, 'duplicate model tensor')
        tensors[t.name] = dict(shape=[int(x) for x in t.shape], type=int(t.tensor_type),
                               allow=bool(pattern.fullmatch(t.name) or t.name in extras))
    return dict(fields=fields, kv_types=kv_types, tensors=tensors)


def validate_models(paths, inspector=metadata):
    require(isinstance(paths, dict) and set(paths) == set(FORMATS), 'exact six model mappings required')
    resolved = [Path(paths[f]).resolve() for f in FORMATS]
    require(all(isinstance(paths[f], (str,Path)) for f in FORMATS), 'model paths required')
    for path in paths.values():development.nonsymlink_chain(path)
    require(len({(Path(p).stat().st_dev,Path(p).stat().st_ino) for p in paths.values()}) == 6, 'model inode aliases')
    require(len(set(resolved)) == 6, 'model path aliases')
    hashes, audits = {}, {}
    for fmt in FORMATS:
        hashes[fmt] = development.file_hash(paths[fmt])
        audits[fmt] = inspector(paths[fmt])
    require(len(set(hashes.values())) == 6, 'duplicate model file identities')
    expected_types = dict(f32=0, f16=1, q8_0=8, q6_k=14, q5_k=13, q4_k=12)
    base = audits['f32']
    def lineage(fields):
        return {k:v for k,v in fields.items() if not k.startswith(
            ('timestral.conversion.', 'timestral.quantization.'))}
    require(base['fields']['timestral.source.hf_revision'] == fixtures.HF_REVISION and
            base['fields']['timestral.source.code_sha'] == fixtures.REFERENCE_SHA,
            'source pins differ')
    require(base['fields'].get('timestral.attention.qkv_head_major') is True, 'QKV layout differs')
    frozen = {'timestral.patch_size':32,'timestral.context_length':32768,
              'timestral.forecaster.per_shot_horizon_patches':32,
              'timestral.forecaster.quantile_real_cap_k':10000.,
              'timestral.forecaster.extreme_value_threshold':1e30,
              'timestral.scaler.minimum_scale':1e-6,'timestral.scaler.use_asinh':True,
              'timestral.scaler.stabilize_with_global':False,
              'timestral.forecaster.symmetric_averaging':True,
              'timestral.forecaster.clip_nonnegative':True}
    require(all(base['fields'].get(k) == v for k,v in frozen.items()), 'frozen model options differ')
    for key in ('general.license','general.license.name','general.license.link','general.source.url'):
        require(isinstance(base['fields'].get(key),str), 'missing licence/source field')
    require(base['fields'].get('timestral.quantile_knots') == eval_quant.LEVELS.tolist(), 'model levels differ')
    weights = base['fields'].get('timestral.source.weights_sha256')
    protocol.sha(weights)
    for fmt in FORMATS:
        audit = audits[fmt]
        require(lineage(audit['fields']) == lineage(base['fields']), 'model logical metadata/licence lineage differs')
        require(lineage(audit['kv_types']) == lineage(base['kv_types']), 'model metadata KV types differ')
        require(set(audit['tensors']) == set(base['tensors']), 'model schema names differ')
        for name,t in audit['tensors'].items():
            require(t['shape'] == base['tensors'][name]['shape'] and t['allow'] == base['tensors'][name]['allow'], 'model shape differs')
            require(t['type'] == (expected_types[fmt] if t['allow'] else 0), 'model storage differs')
            block = 256 if fmt in ('q6_k','q5_k','q4_k') else 32 if fmt == 'q8_0' else 1
            require(t['shape'] and t['shape'][0] > 0 and (not t['allow'] or t['shape'][0] % block == 0), 'incompatible model row')
        fields = audit['fields']
        if fmt in ('f16','q8_0'):
            require(fields.get('timestral.conversion.outtype') == fmt and fields.get('timestral.conversion.type_policy') == 'b4-'+fmt+'-v1', 'conversion provenance differs')
            protocol.sha(fields.get('timestral.conversion.converter_sha256'))
            require(not any(k.startswith('timestral.quantization.') for k in fields), 'conflicting quantization history')
        elif fmt != 'f32':
            require(fields.get('timestral.quantization.type') == fmt and fields.get('timestral.quantization.policy') == 'b4-k-v1', 'K provenance differs')
            require(fields.get('timestral.quantization.input_gguf_sha256') == hashes['f32'], 'K input is not mapped F32')
            require(fields.get('timestral.quantization.version') == '1' and fields.get('timestral.quantization.tool_version') == '1', 'K provenance version')
            require(fields.get('timestral.quantization.ggml_commit') == 'e705c5fed490514458bdd2eaddc43bd098fcce9b', 'K vendor lineage')
            protocol.sha(fields.get('timestral.quantization.source_sha256'))
            require(isinstance(fields.get('timestral.quantization.runtime_commit'), str) and len(fields['timestral.quantization.runtime_commit']) == 40 and set(fields['timestral.quantization.runtime_commit']) <= set('0123456789abcdef'), 'K build commit')
            require(not any(k.startswith('timestral.conversion.') for k in fields), 'conflicting conversion history')
        else:
            require(not any(k.startswith(('timestral.conversion.','timestral.quantization.')) for k in fields), 'F32 reduced provenance')
    return hashes, audits


def produce(driver, models, output, sidecar, *, runner=subprocess.run, maker=fixtures.make_fixture,
            inspector=metadata, cases=None, scorer=eval_quant.evaluate):
    cases = fixtures.load_cases() if cases is None else cases
    canonical = fixtures.load_cases()
    require(cases == canonical and len(cases) == 79 and len({c['name'] for c in cases}) == 79,
            'exact frozen 79 mappings required')
    require(isinstance(sidecar, dict) and set(sidecar) == {'version','runtime_commit','source_sha256','executable_sha256','protocol_version'}, 'build sidecar fields')
    require(sidecar['version'] == 'quant-driver-build-v1' and sidecar['protocol_version'] == 2, 'build sidecar version')
    protocol.sha(sidecar['source_sha256']);protocol.sha(sidecar['executable_sha256'])
    require(isinstance(sidecar['runtime_commit'],str) and len(sidecar['runtime_commit']) == 40 and set(sidecar['runtime_commit']) <= set('0123456789abcdef'), 'build commit')
    executable_hash = development.file_hash(driver)
    require(executable_hash == sidecar['executable_sha256'], 'driver sidecar executable differs')
    model_hashes, model_audits = validate_models(models, inspector)
    require(all(development.file_hash(models[f]) == model_hashes[f] for f in FORMATS), 'model changed during metadata audit')
    require(Path(driver).resolve() not in [Path(p).resolve() for p in models.values()], 'driver/model alias')
    output = Path(output);development.nonsymlink_chain(output)
    require(not output.exists() and not output.is_symlink(), 'output directory already exists')
    output.mkdir()
    manifest = dict(version='quant-record-v1', fixture_version=fixtures.VERSION, options=fixtures.OPTIONS,
                    source_sha256=None, driver={k:sidecar[k] for k in ('source_sha256','runtime_commit')},
                    models={f:dict(sha256=model_hashes[f],hf_revision=fixtures.HF_REVISION,reference_sha=fixtures.REFERENCE_SHA) for f in FORMATS},items=[])
    try:
        development.publish(output/'build-audit.json', development.json_bytes(dict(sidecar=sidecar, model_hashes=model_hashes, metadata=model_audits,
                            provenance='actual file hashes; build identity asserted, not independently compiled-proof')))
        for case in cases:
            prepared = development.prepare(case, maker)
            inputs, truth, provenance, shape = prepared
            if manifest['source_sha256'] is None:manifest['source_sha256'] = provenance['source_sha256']
            require(manifest['source_sha256'] == provenance['source_sha256'], 'fixture source drift')
            arrays = dict(context=inputs['context'],truth=truth,levels=eval_quant.LEVELS)
            for fmt in FORMATS:
                stem = case['name']+'.'+fmt
                request = protocol.encode_request(case['name'],shape,inputs,model_hashes[fmt],
                    hashlib.sha256(development.json_bytes(provenance['input_sha256'])).hexdigest(),version=2,format=fmt)
                req, res = output/(stem+'.request'), output/(stem+'.result')
                development.publish(req, request)
                completed = runner([str(driver),'--model',str(models[fmt]),'--format',fmt,'--threads','1','--input',str(req),'--output',str(res)],capture_output=True,timeout=900)
                require(development.file_hash(driver) == executable_hash and development.file_hash(models[fmt]) == model_hashes[fmt], 'model/driver changed')
                require(req.read_bytes() == request, 'request mutation')
                require(res.is_file() and not res.is_symlink() and res.stat().st_size <= protocol.LIMIT, 'missing/invalid driver result')
                data = res.read_bytes();result = protocol.decode_result(data)
                expected = dict(case_id=case['name'],model_sha256=model_hashes[fmt],executable_sha256=executable_hash,
                                source_sha256=sidecar['source_sha256'],runtime_commit=sidecar['runtime_commit'],
                                request_sha256=hashlib.sha256(request).hexdigest(),version=2,format=fmt)
                for key,value in expected.items():require(result.get(key) == value, 'response association: '+key)
                audit = dict(id=case['name'],format=fmt,request_sha256=expected['request_sha256'],result_sha256=hashlib.sha256(data).hexdigest(),model_sha256=model_hashes[fmt],executable_sha256=executable_hash,status=result['status'],diagnostic=result['diagnostic'])
                development.publish(output/(stem+'.audit.json'),development.json_bytes(audit))
                require(completed.returncode == 0 and result['status'] == 0, 'driver failure: coverage HOLD')
                require(result['shape'] == (case['B'],case['V'],case['H'],21) and result['quantiles'].tobytes() == eval_quant.LEVELS.tobytes(), 'response shape/levels')
                require(result['anchors_present'] == (fmt == 'f32'), 'anchor association')
                if fmt == 'f32':
                    arrays.update(anchor_loc=result['anchor_loc'],anchor_scale=result['anchor_scale'])
                else:require('anchor_loc' not in result and 'anchor_scale' not in result, 'reduced anchors forbidden')
                arrays['forecast_'+fmt] = result['values']
            buf=io.BytesIO();np.savez(buf,**arrays)
            npz = case['name']+'.npz';development.publish(output/npz,buf.getvalue())
            manifest['items'].append(dict(id=case['name'],case_sha256=provenance['case_sha256'],input_sha256=provenance['input_sha256'],truth_sha256=provenance['truth_sha256'],npz=npz,
                arrays={k:dict(dtype=a.dtype.str,shape=list(a.shape),sha256=fixtures.array_hash(a)) for k,a in arrays.items()},
                anchor_model_sha256=model_hashes['f32'],outputs={f:dict(status='forecast') for f in FORMATS}))
        require(development.file_hash(driver) == executable_hash and all(development.file_hash(models[f]) == model_hashes[f] for f in FORMATS), 'final file identity drift')
        # Validate with the actual scorer before publishing the final manifest.
        candidate=output/'candidate.json';development.publish(candidate,development.json_bytes(manifest))
        try: report=scorer(candidate)
        finally:candidate.unlink()
        development.publish(output/'recorded-checks.json',development.json_bytes(report))
        development.publish(output/'manifest.json',development.json_bytes(manifest))
        return manifest
    except Exception as error:
        development.publish(output/'HOLD.json',development.json_bytes(dict(status='HOLD',scoring_manifest_published=False,diagnostic=str(error),completed_items=[i['id'] for i in manifest['items']])))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver',required=True);parser.add_argument('--models',required=True)
    parser.add_argument('--build-sidecar',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        produce(args.driver,eval_quant.read_json(args.models),args.output,eval_quant.read_json(args.build_sidecar))
        return 0
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        print('matrix HOLD: '+str(error),file=__import__('sys').stderr);return 1


if __name__=='__main__':raise SystemExit(main())
