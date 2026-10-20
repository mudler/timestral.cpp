import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import record_quant as r


class Producer(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('TIMESTRAL_TEST_TMP_DIR',str(Path(__file__).resolve().parents[2]/'work')))
        self.root=Path(self.tmp.name);self.model=self.root/'model';self.driver=self.root/'driver'
        self.model.write_bytes(b'model');self.driver.write_bytes(b'driver');self.out=self.root/'out';self.out.mkdir()
        self.case=dict(name='small',B=1,V=1,T=3,H=2)
        self.inputs={'context':np.array([[[1.,np.nan,-0.]]],dtype='<f4')};self.truth=np.ones((1,1,2),dtype='<f4')
        self.prov=dict(options=r.fixtures.OPTIONS,input_sha256={'context':r.fixtures.array_hash(self.inputs['context'])},truth_sha256=r.fixtures.array_hash(self.truth),case_sha256=hashlib.sha256(json.dumps(self.case,sort_keys=True,allow_nan=False).encode()).hexdigest(),source_sha256={})
    def tearDown(self):self.tmp.cleanup()
    def maker(self,case):return self.inputs,self.truth,self.prov
    def runner(self,args,**kw):
        req=Path(args[args.index('--input')+1]).read_bytes();case=struct.unpack_from('<I',req,152)[0];name=req[288:288+case]
        levels=r.LEVELS.tobytes();values=np.ones((1,1,2,21),dtype='<f4').tobytes();loc=np.zeros((1,1),dtype='<f4').tobytes();scale=np.ones((1,1),dtype='<f4').tobytes()
        b=bytearray(256+len(name));struct.pack_into('<8sIIQ4Q',b,0,b'TSQRES01',1,0,len(b)+len(levels+values+loc+scale),1,1,2,21)
        b[56:88]=bytes.fromhex(r.file_hash(self.model));b[88:120]=bytes.fromhex(r.file_hash(self.driver));b[120:152]=bytes.fromhex('b'*64);b[152:192]=b'a'*40;b[192:224]=hashlib.sha256(req).digest();struct.pack_into('<IIQQQ',b,224,len(name),0,len(b),0,0);b[256:]=name
        Path(args[args.index('--output')+1]).write_bytes(b+levels+values+loc+scale)
        return types.SimpleNamespace(returncode=0)
    def produce(self,runner=None):return r.produce_item(self.case,self.driver,self.model,self.out,'a'*40,'b'*64,runner=runner or self.runner,maker=self.maker)
    def test_roundtrip_non_scoring(self):
        audit=self.produce();self.assertFalse(audit['scoring']);self.assertEqual(audit['version'],'development-f32-record-v1')
        with np.load(self.out/'small.npz',allow_pickle=False) as z:
            self.assertEqual(z['context'].tobytes(),self.inputs['context'].tobytes());self.assertEqual(z['truth'].tobytes(),self.truth.tobytes());self.assertEqual(z['anchor_scale'].item(),1)
        with self.assertRaises(ValueError):self.produce()
    def test_mutated_id_and_anchor_nonfinite(self):
        for mode in ['id','anchor','levels']:
            def runner(args,**kw):
                result=self.runner(args);p=Path(args[args.index('--output')+1]);b=bytearray(p.read_bytes())
                if mode=='id':b[256]=ord('X')
                elif mode=='anchor':struct.pack_into('<f',b,len(b)-4,float('nan'))
                else:struct.pack_into('<d',b,261,.2)
                p.write_bytes(b);return result
            with self.assertRaises(ValueError):self.produce(runner)
            self.assertFalse((self.out/'small.audit.json').exists())
    def test_process_and_hash_failure(self):
        with self.assertRaises(ValueError):self.produce(lambda *a,**k:types.SimpleNamespace(returncode=1))
        def runner(args,**kw):
            result=self.runner(args);self.model.write_bytes(b'changed');return result
        with self.assertRaises(ValueError):self.produce(runner)
    def test_paths_and_truth(self):
        (self.out/'small.npz').symlink_to(self.model)
        with self.assertRaises(ValueError):self.produce()
        (self.out/'small.npz').unlink();self.truth[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.produce()
    def test_wrong_identity_fields(self):
        for offset in [56,88,120,152,192]:
            def runner(args,**kw):
                result=self.runner(args);p=Path(args[args.index('--output')+1]);b=bytearray(p.read_bytes());b[offset]=ord('c') if offset==152 else b[offset]^1;p.write_bytes(b);return result
            with self.assertRaises(ValueError):self.produce(runner)
    def test_request_mutation_and_executable_mutation(self):
        for target in ('request','executable'):
            def runner(args,**kw):
                result=self.runner(args)
                path=Path(args[args.index('--input')+1]) if target=='request' else self.driver
                path.write_bytes(path.read_bytes()+b'changed')
                return result
            with self.assertRaises(ValueError):self.produce(runner)
            self.assertEqual(list(self.out.iterdir()),[])
    def test_invalid_anchor_shape_and_scale(self):
        for mode in ('shape','scale'):
            def runner(args,**kw):
                result=self.runner(args);path=Path(args[args.index('--output')+1]);b=bytearray(path.read_bytes())
                if mode=='shape':struct.pack_into('<Q',b,24,2)
                else:struct.pack_into('<f',b,len(b)-4,-1.)
                path.write_bytes(b);return result
            with self.assertRaises(ValueError):self.produce(runner)
    def test_symlink_parent_and_alias(self):
        alias=self.root/'alias';alias.symlink_to(self.out,target_is_directory=True)
        with self.assertRaises(ValueError):r.produce_item(self.case,self.driver,self.model,alias,'a'*40,'b'*64,runner=self.runner,maker=self.maker)
        with self.assertRaises(ValueError):r.produce_item(self.case,self.model,self.model,self.out,'a'*40,'b'*64,runner=self.runner,maker=self.maker)

    def test_publication_cleanup(self):
        original=r.publish
        def fail(path,data):
            if str(path).endswith('.audit.json'):raise OSError('injected')
            original(path,data)
        with patch.object(r,'publish',fail):
            with self.assertRaises(OSError):self.produce()
        self.assertEqual(list(self.out.iterdir()),[])


if __name__=='__main__':unittest.main()
