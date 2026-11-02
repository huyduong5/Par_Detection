import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from PIL import Image
import pickle
from experiments.par_resolution.core import atomic_json,sha
from experiments.par_resolution.generate import generate
from experiments.par_resolution.evaluate import evaluate_entry
from experiments.par_resolution.analyze import analyze,paired_bootstrap
from experiments.par_resolution.worker import idle
from experiments.par_resolution.gate import seal,validate

class TinyModel(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.,-1.]),requires_grad=False);self.eval()
    def forward(self,image,tokens,attrs):
        return image.mean((1,2,3)).unsqueeze(1)*self.weight,None

def transform(image):return torch.from_numpy(np.array(image.resize((8,8)),copy=True)).permute(2,0,1).float()/255

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name);images=self.root/'images';images.mkdir()
        names=[]
        for i in range(6):
            name=f'part{i}_a.png';Image.fromarray(np.random.default_rng(i).integers(0,256,(13,11,3),dtype=np.uint8)).save(images/name);names.append(name)
        pkl=self.root/'labels.pkl';pkl.write_bytes(pickle.dumps(SimpleNamespace(attributes=['hat','shoe'],image_name=names,label=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]]),partition={'val':[np.array([0,1,2])],'test':[np.array([3,4,5])]})))
        self.bench=self.root/'benchmark';generate(pkl,images,self.bench);self.manifest=self.bench/'manifests/index.json'
        self.entry=dict(id='fixture',seed=42,family='aq',checkpoint_sha256='fixture',attribute_binding='synthetic test only')
        self.registry=self.root/'registry.json';atomic_json(self.registry,dict(source_fingerprint='fixed',models=[self.entry]))
        self.report=self.root/'cpu.json';atomic_json(self.report,dict(state='PASS',source_fingerprint='fixed'))
        config=SimpleNamespace(use_domain_tokens=False,embed_dim=8,use_amp=False,img_height=8,img_width=8)
        self.patch=patch('experiments.par_resolution.adapters.load',side_effect=lambda *args:(TinyModel(),config,transform));self.patch.start()
        self.cuda=patch('torch.cuda.is_available',return_value=False);self.cuda.start()
    def tearDown(self):self.patch.stop();self.cuda.stop();self.tmp.cleanup()
    def test_production_evaluator_roundtrip_resume_and_analysis(self):
        dest=self.root/'results/fixture'
        first=evaluate_entry(self.entry,self.manifest,'test',dest,batch_size=2,workers=0)
        again=evaluate_entry(self.entry,self.manifest,'test',dest,batch_size=1,workers=0)
        self.assertEqual(first,again)
        self.assertEqual(first['1']['clean_reference_parity']['max_input_tensor_error'],0)
        rows=analyze(self.root/'results',self.root/'analysis',draws=10)
        self.assertEqual(len(rows),1);self.assertTrue((self.root/'analysis/resolution_curve.png').exists())
        p=dest/'test_x8.npz';p.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):evaluate_entry(self.entry,self.manifest,'test',dest,workers=0)
    def test_gate_requires_complete_matching_smoke(self):
        smoke=self.root/'smoke';gate=self.root/'gate.json'
        subset=self.bench/'manifests/val_smoke128_ids.json'
        evaluate_entry(self.entry,self.manifest,'val',smoke,subset,2,0)
        with patch('experiments.par_resolution.registry.source_fingerprint',return_value=('fixed',{})),patch('experiments.par_resolution.gate.source_fingerprint',return_value=('fixed',{})):
            seal(gate,self.registry,self.manifest,smoke,self.report);self.assertEqual(validate(gate,self.registry,self.manifest)['state'],'PASS')
            atomic_json(self.report,dict(state='FAIL'))
            with self.assertRaises(ValueError):validate(gate,self.registry,self.manifest)
    def test_empty_or_bad_runtime_arguments(self):
        for batch,workers in ((0,1),(1,-1)):
            with self.assertRaises(ValueError):evaluate_entry(self.entry,self.manifest,'val',self.root/'bad',batch_size=batch,workers=workers)
    def test_bootstrap_identical_and_direction(self):
        labels=np.array([[1,0],[0,1],[1,1],[0,0]])
        good=labels*.8+.1;bad=1-good
        self.assertEqual(paired_bootstrap(good,good,labels,50)['ci95_pp'],[0.,0.])
        self.assertEqual(paired_bootstrap(good,bad,labels,50)['delta_mA_pp'],-100.)
    def test_idle_denies_busy_processes_errors_and_checks_timeout(self):
        for responses,expected in ((['24000,0',''],True),(['24000,0','123'],False),(['1000,0',''],False),(['24000,99',''],False)):
            with patch('subprocess.check_output',side_effect=responses) as mock:
                self.assertEqual(idle(0),expected);self.assertEqual(mock.call_args.kwargs['timeout'],10)
        with patch('subprocess.check_output',side_effect=OSError('missing')):self.assertFalse(idle(0))

if __name__=='__main__':unittest.main()
