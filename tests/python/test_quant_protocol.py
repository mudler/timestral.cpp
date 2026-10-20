import importlib.util
from pathlib import Path
import struct
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('quant_protocol', Path(__file__).resolve().parents[2]/'scripts/quant_protocol.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


class Protocol(unittest.TestCase):
    def test_request_layout_bits(self):
        x = np.array([0., -0., np.nan, 1e-38], dtype='<f4')
        b = p.encode_request('test-1', (1,1,4,2,0,0,0,0), {'context': x}, 'a'*64, 'b'*64)
        self.assertEqual(b[:8], b'TSQREQ01')
        self.assertEqual(struct.unpack_from('<Q', b, 16)[0], len(b))
        self.assertEqual(b[-16:], x.tobytes())
        self.assertEqual(struct.unpack_from('<IIQQQ', b, 294), (1,0,4,16,486))

    def test_all_arrays(self):
        shape=(2,2,3,2,1,1,1,1)
        counts=(12,0,6,10,6,10)
        arrays={n:np.arange(c,dtype=dt) for n,c,dt in zip(p.NAMES,counts,p.DTYPES) if c}
        data=p.encode_request('arrays',shape,arrays,'a'*64,'b'*64)
        offset=288+6+192
        for name in p.NAMES:
            if name in arrays:
                value=arrays[name].tobytes()
                self.assertEqual(data[offset:offset+len(value)],value)
                offset+=len(value)
        self.assertEqual(offset,len(data))

    def test_rejects_input(self):
        for name in ('', '../case', 'x'*257, 'é'):
            with self.assertRaises(ValueError): p.encode_request(name, (1,1,1,1,0,0,0,0), {'context':np.ones(1,np.float32)}, 'a'*64,'b'*64)
        with self.assertRaises(ValueError): p.encode_request('x',(1,1,1,1,0,0,0,0), {'context':np.ones(1,np.float64)},'a'*64,'b'*64)
        with self.assertRaises(ValueError): p.encode_request('x',(1,1,1,1,0,0,0,0), {'context':np.ones(1,np.float32),'truth':np.ones(1,np.float32)},'a'*64,'b'*64)
        with self.assertRaises(ValueError): p.encode_request('x',(2**63,2,2,1,0,0,0,0), {'context':np.ones(1,np.float32)},'a'*64,'b'*64)

    def test_result_and_mutations(self):
        b=bytearray(257+8+4+4+4)
        struct.pack_into('<8sIIQ4Q',b,0,b'TSQRES01',1,0,len(b),1,1,1,1)
        b[152:192]=b'a'*40
        struct.pack_into('<IIQQQ',b,224,1,0,257,0,0);b[256]=ord('x')
        struct.pack_into('<dfff',b,257,.5,2.,0.,1.)
        r=p.decode_result(b);self.assertEqual(r['values'].item(),2.)
        for offset in (0,8,12,16,224,228,232,240,248):
            bad=bytearray(b);bad[offset]^=255
            with self.assertRaises((ValueError,UnicodeError)):p.decode_result(bad)
        for bad in (b[:-1],b+b'x'):
            with self.assertRaises(ValueError):p.decode_result(bad)
        struct.pack_into('<f',b,273,0.)
        with self.assertRaises(ValueError):p.decode_result(b)



# Optional executable integration uses native test exports, not regenerated truth.
class NativeProcess(unittest.TestCase):
    def test_real_exports(self):
        import os
        import subprocess
        import tempfile
        import hashlib
        driver = os.environ.get('TIMESTRAL_EVAL_DRIVER')
        exports = os.environ.get('TIMESTRAL_EVAL_EXPORTS')
        model = os.environ.get('TIMESTRAL_TEST_GGUF')
        if not all((driver, exports, model)):
            self.skipTest('native driver/export/model paths unset')
        with tempfile.TemporaryDirectory(dir=os.environ['TIMESTRAL_EVAL_SCRATCH']) as tmp:
            for name in ('uni_short', 'known_cov'):
                req = Path(exports) / (name+'.request')
                out = Path(tmp)/(name+'.result')
                args = [driver, '--model', model, '--input', str(req), '--output', str(out), '--format', 'f32', '--threads', '1']
                run = subprocess.run(args, capture_output=True, text=True, timeout=240)
                self.assertEqual(run.returncode, 0, run.stderr)
                result = p.decode_result(out.read_bytes())
                expected = p.decode_result((Path(exports)/(name+'.expected')).read_bytes())
                for key in ('values', 'quantiles', 'anchor_loc', 'anchor_scale'):
                    self.assertEqual(result[key].tobytes(), expected[key].tobytes())
                self.assertEqual(result['executable_sha256'], hashlib.sha256(Path(driver).read_bytes()).hexdigest())
                self.assertEqual(result['model_sha256'], hashlib.sha256(Path(model).read_bytes()).hexdigest())
                self.assertEqual(result['request_sha256'], hashlib.sha256(req.read_bytes()).hexdigest())
                original = out.read_bytes()
                self.assertEqual(subprocess.run(args, capture_output=True).returncode, 1)
                self.assertEqual(out.read_bytes(), original)

if __name__ == '__main__':unittest.main()
