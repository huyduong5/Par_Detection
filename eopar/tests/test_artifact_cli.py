import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from eopar.train.runtime import digest


class TestArtifactCLI(unittest.TestCase):
    def test_natural_rejects_prediction_split_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'prediction.npz'
            np.savez_compressed(p,names=np.array(['a']),attributes=np.array(['Hat']),
                clean=np.array([[.7]]),labels=np.array([[1]]),obs=np.array([[.9]]))
            p.with_suffix('.json').write_text(json.dumps(dict(artifact_state='complete',predictions_sha256=digest(p),split='test')))
            annotation=root/'a.csv';annotation.write_text('image,split,attribute,natural_occlusion_confirmed,observability\na,val,Hat,yes,visible\n')
            output=root/'result.json'
            result=subprocess.run([sys.executable,'-m','eopar.eval.natural','--predictions',str(p),
                '--annotations',str(annotation),'--split','val','--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('split mismatch',result.stderr)
            self.assertFalse(output.exists())

    def test_bootstrap_rejects_nan_before_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'bad.npz'
            np.savez_compressed(p,names=np.array(['a']),attributes=np.array(['Hat']),
                clean=np.array([[float('nan')]]),occ=np.array([[.5]]),labels=np.array([[1]]))
            output=root/'result.json'
            result=subprocess.run([sys.executable,'-m','eopar.eval.bootstrap','--baseline',str(p),
                '--candidate',str(p),'--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('nonfinite',result.stderr)
            self.assertFalse(output.exists())
