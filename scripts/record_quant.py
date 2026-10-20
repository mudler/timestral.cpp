#!/usr/bin/env python3
"""Produce F32-only development records, never a scoring manifest or quality PASS."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import quant_fixtures as fixtures
import quant_protocol as protocol

VERSION = 'development-f32-record-v1'
LEVELS = np.array([.01, .05] + [i/20 for i in range(2, 19)] + [.95, .99], dtype='<f8')


def require(ok, message):
    if not ok: raise ValueError(message)


def nonsymlink_chain(path):
    path=Path(path).absolute()
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'symlink path component')


def file_hash(path):
    path = Path(path)
    nonsymlink_chain(path)
    require(not path.is_symlink() and path.is_file(), 'regular nonsymlink file required: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def publish(path, data):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'refuse existing output: '+str(path))
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.tmp.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(tmp, path)  # atomic, refuses overwrite, including dangling symlinks
    finally: os.unlink(tmp)


def prepare(case, maker=fixtures.make_fixture):
    protocol.identity(case['name'])
    inputs, truth, provenance = maker(case)
    B,V,T,H = (case[k] for k in ('B','V','T','H'))
    require(provenance['options'] == fixtures.OPTIONS, 'fixture options differ')
    require(provenance['case_sha256']==hashlib.sha256(json.dumps(case,sort_keys=True,allow_nan=False).encode()).hexdigest(), 'case association')
    require(set(inputs).issubset({'context','past','known','cat_past','cat_known'}), 'unexpected inference array')
    counts=[]
    for name, width, dt in [('context',T,'<f4'),('past',T,'<f4'),('known',T+H,'<f4'),('cat_past',T,'<f8'),('cat_known',T+H,'<f8')]:
        if name not in inputs:
            require(name != 'context', 'missing context'); counts.append(0); continue
        a=inputs[name]
        require(isinstance(a,np.ndarray) and a.dtype==np.dtype(dt) and a.ndim==3 and a.shape[0]==B and a.shape[2]==width and a.flags.c_contiguous, 'fixture input shape/type')
        require(a.shape[1] > 0 and (name!='context' or a.shape[1]==V), 'fixture input width')
        counts.append(a.shape[1])
        require(fixtures.array_hash(a)==provenance['input_sha256'][name], 'fixture input association')
    require(set(provenance['input_sha256'])==set(inputs), 'input hash keys')
    require(truth.dtype==np.dtype('<f4') and truth.shape==(B,V,H) and np.isfinite(truth).all(), 'invalid truth')
    require(fixtures.array_hash(truth)==provenance['truth_sha256'], 'truth association')
    shape=(B,V,T,H,*counts[1:])
    return inputs,truth,provenance,shape


def request_bytes(case, prepared, model_hash):
    inputs,_,provenance,shape=prepared
    input_hash=hashlib.sha256(json_bytes(provenance['input_sha256'])).hexdigest()
    return protocol.encode_request(case['name'],shape,inputs,model_hash,input_hash)


def validate_result(data,case,request,model_hash,exe_hash,runtime_commit,source_hash):
    result=protocol.decode_result(data)
    require(result['status']==0, 'driver HOLD: '+result['diagnostic'])
    require(set(result)=={'status','case_id','diagnostic','model_sha256','executable_sha256','source_sha256','runtime_commit','request_sha256','shape','quantiles','values','anchor_loc','anchor_scale'}, 'unexpected response fields')
    for key,expected in [('case_id',case['name']),('model_sha256',model_hash),('executable_sha256',exe_hash),('source_sha256',source_hash),('runtime_commit',runtime_commit),('request_sha256',hashlib.sha256(request).hexdigest())]: require(result[key]==expected, 'response identity mismatch: '+key)
    require(result['shape']==(case['B'],case['V'],case['H'],len(LEVELS)), 'response fixture shape')
    require(result['quantiles'].dtype==LEVELS.dtype and result['quantiles'].tobytes()==LEVELS.tobytes(), 'response levels/options')
    for key in ('anchor_loc','anchor_scale'):
        require(result[key].dtype==np.dtype('<f4') and result[key].shape==(case['B'],case['V']), 'anchor type/shape')
    # Frozen options/format are fixed and validated by tsqbin-v1, not mutable
    # fields accepted from an arbitrary driver JSON object.
    return result


def produce_item(case,driver,model,output,runtime_commit,source_hash,runner=subprocess.run,maker=fixtures.make_fixture):
    prepared=prepare(case,maker);inputs,truth,provenance,_=prepared
    model_hash=file_hash(model);exe_hash=file_hash(driver);request=request_bytes(case,prepared,model_hash)
    output=Path(output);nonsymlink_chain(output);require(output.is_dir() and not output.is_symlink(), 'output directory required')
    stem=case['name'];paths={suffix:output/(stem+suffix) for suffix in ('.request','.result','.npz','.audit.json')}
    for p in paths.values(): require(not p.exists() and not p.is_symlink(), 'refuse existing item output')
    # Existing paths/aliases cannot be overwritten; only safe item names are used.
    require(Path(model).resolve()!=Path(driver).resolve(), 'model/executable alias')
    with tempfile.TemporaryDirectory(prefix='item-',dir=output) as staging:
        req=Path(staging)/'input.bin';res=Path(staging)/'output.bin';req.write_bytes(request)
        completed=runner([str(driver),'--model',str(model),'--format','f32','--threads','1','--input',str(req),'--output',str(res)],capture_output=True,timeout=900)
        require(file_hash(model)==model_hash and file_hash(driver)==exe_hash, 'model/executable changed')
        require(req.read_bytes()==request, 'request mutated by driver')
        require(completed.returncode==0, 'driver invocation HOLD')
        require(res.is_file() and not res.is_symlink() and res.stat().st_size<=protocol.LIMIT, 'invalid result file')
        data=res.read_bytes();result=validate_result(data,case,request,model_hash,exe_hash,runtime_commit,source_hash)
    arrays=dict(inputs,truth=truth,quantiles=result['quantiles'],f32=result['values'],anchor_loc=result['anchor_loc'],anchor_scale=result['anchor_scale'])
    require(all(a.dtype.kind!='O' for a in arrays.values()), 'object arrays forbidden')
    buf=io.BytesIO();np.savez(buf,**arrays)
    audit=dict(version=VERSION,scoring=False,status='development-record',id=stem,options=fixtures.OPTIONS,
               fixture=provenance,model_sha256=model_hash,executable_sha256=exe_hash,runtime_commit=runtime_commit,
               source_sha256=source_hash,request_sha256=hashlib.sha256(request).hexdigest(),result_sha256=hashlib.sha256(data).hexdigest(),
               anchor_model_sha256=model_hash,shape=list(result['shape']),arrays={k:fixtures.array_hash(v) for k,v in arrays.items()})
    created=[]
    try:
        for suffix,payload in [('.request',request),('.result',data),('.npz',buf.getvalue()),('.audit.json',json_bytes(audit))]: publish(paths[suffix],payload);created.append(paths[suffix])
    except Exception:
        for p in created:p.unlink()
        raise
    return audit


def inventory(cases=None):
    cases=fixtures.load_cases() if cases is None else cases
    require(len(cases)==79 and len({c['name'] for c in cases})==79, 'exactly 79 fixture IDs required')
    total=0
    for case in cases:
        prepared=prepare(case);request=request_bytes(case,prepared,'0'*64)
        total+=len(request)
    return dict(version=VERSION,scoring=False,status='inventory-only',items=79,serialized_bytes=total)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',action='store_true');parser.add_argument('--driver');parser.add_argument('--model');parser.add_argument('--output');parser.add_argument('--case',action='append');parser.add_argument('--runtime-commit');parser.add_argument('--source-sha256')
    args=parser.parse_args(argv)
    try:
        if args.inventory:
            require(not any((args.driver,args.model,args.output,args.case,args.runtime_commit,args.source_sha256)), 'inventory takes no execution options')
            print(json_bytes(inventory()).decode(),end='');return 0
        require(all((args.driver,args.model,args.output,args.case,args.runtime_commit,args.source_sha256)), 'execution options required')
        require(re.fullmatch('[0-9a-f]{40}',args.runtime_commit), 'invalid runtime commit');protocol.sha(args.source_sha256)
        cases={c['name']:c for c in fixtures.load_cases()};require(len(set(args.case))==len(args.case) and all(n in cases for n in args.case), 'unknown/duplicate case')
        out=Path(args.output);nonsymlink_chain(out);require(not out.exists() and not out.is_symlink(), 'output directory must not exist');require(not out.parent.is_symlink(), 'symlink output parent')
        out.mkdir();audits=[]
        for name in args.case:audits.append(produce_item(cases[name],args.driver,args.model,out,args.runtime_commit,args.source_sha256))
        publish(out/'index.json',json_bytes(dict(version=VERSION,scoring=False,status='development-only',items=[{'id':a['id'],'audit':a['id']+'.audit.json'} for a in audits])))
        return 0
    except (ValueError,OSError,subprocess.SubprocessError) as e:
        print('development HOLD: '+str(e),file=__import__('sys').stderr);return 1


if __name__=='__main__':raise SystemExit(main())
