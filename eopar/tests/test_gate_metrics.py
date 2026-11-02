import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader

from eopar.eval.metrics import par_metrics,risk_coverage
from eopar.train.run import evaluate
from eopar.verification.toy_process import Data,Model
from eopar.verification.gate import validate_gate,REQUIRED_GPU
from eopar.scripts.queue import idle
from eopar.train.runtime import ROOT,digest


class TestGateAndMetrics(unittest.TestCase):
    def test_gpu_busy_process_and_query_failure(self):
        for outputs,expected in [(['24000, 0',''],True),(['24000, 0','1234'],False),
                                 (['17000, 0',''],False),(['24000, 99',''],False)]:
            with patch('eopar.scripts.queue.subprocess.check_output',side_effect=outputs):
                self.assertEqual(idle(0,18000),expected)
        with patch('eopar.scripts.queue.subprocess.check_output',side_effect=OSError('no driver')):
            with self.assertRaises(OSError): idle(0,18000)

    def test_pilot_cannot_launch_without_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-m','eopar.scripts.queue','--gpu','0','--variant','B1',
                                   '--run-pilot','--output',str(Path(tmp)/'out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertFalse((Path(tmp)/'out').exists())

    def test_gate_missing_gpu_check_or_stale_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json'; source={'runtime.py':'hash'}; env={'python':'test'}
            config=ROOT/'eopar/configs/B1.json'
            report=dict(state='PASS',authorization='pilot_only',release_root=str(ROOT.resolve()),
                source_hashes=source,environment=env,config_hashes={'B1':digest(config)},
                cpu='PASS',data='PASS',gpu_checks={key:'PASS' for key in REQUIRED_GPU},input_hashes={str(config):digest(config)})
            with patch('eopar.verification.gate.source_hashes',return_value=source),patch('eopar.verification.gate.environment_identity',return_value=env):
                path.write_text(json.dumps(report)); validate_gate(path,'B1',config)
                for field,value in [('source_hashes',{}),('environment',{}),('state','FAIL'),('gpu_checks',{}),('config_hashes',{})]:
                    path.write_text(json.dumps(dict(report,**{field:value})))
                    with self.assertRaises(ValueError): validate_gate(path,'B1',config)

    def test_metrics_batch_partition_invariance_roundtrip(self):
        torch.manual_seed(1); model=Model()
        results=[]
        with tempfile.TemporaryDirectory() as tmp:
            for size in (2,3,7):
                torch.manual_seed(14)
                path=Path(tmp)/f'{size}.npz'
                result=evaluate(model,DataLoader(Data('val'),batch_size=size,num_workers=0),torch.device('cpu'),['a','b'],prediction_path=path)
                a=np.load(path)
                self.assertEqual(result['clean']['par'],par_metrics(a['clean'],a['labels']))
                results.append(result['clean']['par'])
            self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])

    def test_aurc_manual_and_permutation(self):
        # Errors [0,1,0], distinct descending confidence: risks [0,1/2,1/3].
        p=np.array([[.9,.9,.9]]);y=np.array([[1,0,1]]);s=np.array([[3.,2.,1.]])
        result=risk_coverage(p,y,s)
        self.assertAlmostEqual(result['aurc'],(0+.5+1/3)/3)
        self.assertAlmostEqual(result['at_coverage']['0.95']['risk'],1/3)
        perm=[2,0,1]
        self.assertEqual(result,risk_coverage(p[:,perm],y[:,perm],s[:,perm]))

    def test_threshold_shapes_and_empty(self):
        for threshold in ([.5,.5],float('nan'),2):
            with self.assertRaises(ValueError):par_metrics([[.5]],[[1]],threshold)
        with self.assertRaises(ValueError):par_metrics(np.empty((1,0)),np.empty((1,0)))
