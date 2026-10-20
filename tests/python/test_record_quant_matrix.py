"""Small mocked 79x6 matrix; no native model execution or trained weights."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import record_quant_matrix as matrix
import quant_fixtures as fixtures
import eval_quant


class Matrix(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.driver=self.root/'driver';self.driver.write_bytes(b'mock executable')
        self.models={f:self.root/(f+'.gguf') for f in matrix.FORMATS}
        for f,p in self.models.items():p.write_bytes(('mock '+f).encode())
        self.hashes={f:hashlib.sha256(p.read_bytes()).hexdigest() for f,p in self.models.items()}
        self.side=dict(version='quant-driver-build-v1',runtime_commit='a'*40,source_sha256='b'*64,executable_sha256=hashlib.sha256(self.driver.read_bytes()).hexdigest(),protocol_version=2)
        self.cases=[dict(c,B=1,V=1,T=3,H=2) for c in fixtures.load_cases()]
        self.made=0;self.called=[]

    def inspect(self,path):
        f=next(f for f,p in self.models.items() if p==Path(path))
        fields={'timestral.source.hf_revision':fixtures.HF_REVISION,'timestral.source.code_sha':fixtures.REFERENCE_SHA,
                'timestral.source.weights_sha256':'c'*64,'timestral.attention.qkv_head_major':True,'general.license':'synthetic'}
        fields.update({'general.license.name':'test','general.license.link':'','general.source.url':'',
                       'timestral.patch_size':32,'timestral.context_length':32768,
                       'timestral.forecaster.per_shot_horizon_patches':32,
                       'timestral.forecaster.quantile_real_cap_k':10000.,'timestral.forecaster.extreme_value_threshold':1e30,
                       'timestral.scaler.minimum_scale':1e-6,'timestral.scaler.use_asinh':True,
                       'timestral.scaler.stabilize_with_global':False,'timestral.forecaster.symmetric_averaging':True,
                       'timestral.forecaster.clip_nonnegative':True,'timestral.quantile_knots':eval_quant.LEVELS.tolist()})
        if f in ('f16','q8_0'):fields.update({'timestral.conversion.outtype':f,'timestral.conversion.type_policy':'b4-'+f+'-v1','timestral.conversion.converter_sha256':'d'*64})
        elif f!='f32':fields.update({'timestral.quantization.type':f,'timestral.quantization.policy':'b4-k-v1','timestral.quantization.input_gguf_sha256':self.hashes['f32'],'timestral.quantization.version':'1','timestral.quantization.tool_version':'1','timestral.quantization.source_sha256':'e'*64,'timestral.quantization.runtime_commit':'f'*40,'timestral.quantization.ggml_commit':'e705c5fed490514458bdd2eaddc43bd098fcce9b'})
        return dict(fields=fields,kv_types={k:[8] for k in fields},tensors={'head.out.weight':dict(shape=[256,21],type=dict(f32=0,f16=1,q8_0=8,q6_k=14,q5_k=13,q4_k=12)[f],allow=True),'out_norm.weight':dict(shape=[256],type=0,allow=False)})

    def maker(self,case):
        self.made+=1
        x=np.array([[[1,2,3]]],dtype='<f4');truth=np.full((1,1,2),3,dtype='<f4')
        provenance=dict(options=fixtures.OPTIONS,case_sha256=hashlib.sha256(json.dumps(case,sort_keys=True,allow_nan=False).encode()).hexdigest(),input_sha256={'context':fixtures.array_hash(x)},truth_sha256=fixtures.array_hash(truth),source_sha256={'mock':'a'*64})
        return {'context':x},truth,provenance

    def runner(self,argv,**kwargs):
        options=dict(zip(argv[1::2],argv[2::2]));fmt=options['--format'];request=Path(options['--input']).read_bytes()
        self.called.append((fmt,struct.unpack_from('<I',request,156)[0]))
        nc=struct.unpack_from('<I',request,152)[0];name=request[288:288+nc]
        index=matrix.FORMATS.index(fmt);payload=eval_quant.LEVELS.tobytes()+np.full((1,1,2,21),2+index*.0001,dtype='<f4').tobytes()
        if fmt=='f32':payload+=np.zeros((1,1),'<f4').tobytes()+np.ones((1,1),'<f4').tobytes()
        data=bytearray(256+len(name))+payload
        struct.pack_into('<8sIIQ4Q',data,0,b'TSQRES01',2,0,len(data),1,1,2,21)
        data[56:88]=bytes.fromhex(self.hashes[fmt]);data[88:120]=bytes.fromhex(self.side['executable_sha256']);data[120:152]=bytes.fromhex(self.side['source_sha256']);data[152:192]=self.side['runtime_commit'].encode();data[192:224]=hashlib.sha256(request).digest()
        struct.pack_into('<IIQIIQ',data,224,len(name),0,256+len(name),index,int(fmt=='f32'),0);data[256:256+len(name)]=name
        Path(options['--output']).write_bytes(data)
        return subprocess.CompletedProcess(argv,0,b'',b'')

    def produce(self,**kw):
        with patch.object(fixtures,'load_cases',return_value=self.cases),patch.object(fixtures,'make_fixture',side_effect=self.maker):
            return matrix.produce(self.driver,self.models,self.root/'output',self.side,runner=kw.pop('runner',self.runner),maker=self.maker,inspector=kw.pop('inspector',self.inspect),cases=self.cases,**kw)

    def test_complete_matrix_actual_scorer(self):
        manifest=self.produce()
        self.assertEqual(len(manifest['items']),79)
        self.assertEqual(len(self.called),474)
        self.assertEqual(sum(flag for _,flag in self.called),79)
        self.assertTrue(all(flag==(fmt=='f32') for fmt,flag in self.called))
        # Once per producer item plus once per scorer item, never once per format.
        self.assertEqual(self.made,158)
        report=json.loads((self.root/'output/recorded-checks.json').read_text())
        self.assertEqual(report['result'],'PASS')
        for item in manifest['items']:
            self.assertEqual(item['anchor_model_sha256'],self.hashes['f32'])
            self.assertEqual(set(item['outputs']),set(matrix.FORMATS))
        self.assertEqual(len(list((self.root/'output').glob('*.request'))),474)
        self.assertEqual(len(list((self.root/'output').glob('*.result'))),474)
        self.assertTrue((self.root/'output/manifest.json').is_file())

    def test_failure_hold_no_manifest(self):
        def bad(argv,**kw):
            result=self.runner(argv,**kw)
            path=Path(dict(zip(argv[1::2],argv[2::2]))['--output'])
            data=bytearray(path.read_bytes());data[56]^=1;path.write_bytes(data)
            return result
        with self.assertRaises(ValueError):self.produce(runner=bad)
        self.assertTrue((self.root/'output/HOLD.json').is_file())
        self.assertFalse((self.root/'output/manifest.json').exists())

    def test_response_associations_and_missing_output(self):
        mutations=[('format',240),('case',256),('request',192),('executable',88),('source',120)]
        for index,(name,offset) in enumerate(mutations):
            with self.subTest(name=name):
                def bad(argv,**kw):
                    result=self.runner(argv,**kw);path=Path(dict(zip(argv[1::2],argv[2::2]))['--output'])
                    data=bytearray(path.read_bytes());data[offset]^=1;path.write_bytes(data);return result
                with patch.object(fixtures,'load_cases',return_value=self.cases):
                    with self.assertRaises(ValueError):
                        matrix.produce(self.driver,self.models,self.root/('bad'+str(index)),self.side,runner=bad,maker=self.maker,inspector=self.inspect)
                self.assertFalse((self.root/('bad'+str(index))/'manifest.json').exists())
        def missing(argv,**kw):return subprocess.CompletedProcess(argv,1,b'',b'failure')
        with self.assertRaises(ValueError):self.produce(runner=missing)
        self.assertTrue((self.root/'output/HOLD.json').exists())

    def test_mapping_and_lineage_negatives(self):
        for mode in ['alias','missing','mixed','licence','input-hash']:
            with self.subTest(mode=mode):
                models=dict(self.models)
                if mode=='alias':models['f16']=models['f32']
                if mode=='missing':models.pop('q4_k')
                def inspect(path):
                    a=self.inspect(path)
                    if Path(path)==self.models['q4_k']:
                        if mode=='mixed':a['tensors']['head.out.weight']['type']=0
                        if mode=='licence':a['fields']['general.license']='changed'
                        if mode=='input-hash':a['fields']['timestral.quantization.input_gguf_sha256']='0'*64
                    return a
                with self.assertRaises(ValueError):matrix.validate_models(models,inspect)


if __name__=='__main__':unittest.main()
