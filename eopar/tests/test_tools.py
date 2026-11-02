import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from eopar.scripts.annotations import agreement


class TestResearchTools(unittest.TestCase):
    def test_calibration_refuses_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'test.json', Path(tmp) / 'thresholds.json'
            source.write_text(json.dumps({'split': 'test'}))
            result = subprocess.run([sys.executable, '-m', 'eopar.eval.calibrate',
                                    '--validation-json', str(source), '--output', str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_validation_calibration_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'val.json', Path(tmp) / 'thresholds.json'
            source.write_text(json.dumps({'split': 'val', 'checkpoint_sha256': 'synthetic_test_hash'}))
            np.savez_compressed(source.with_suffix('.npz'), clean=np.array([[.2], [.8]]),
                                labels=np.array([[0], [1]]), attributes=np.array(['Hat']),
                                names=np.array(['a', 'b']))
            from eopar.train.runtime import digest
            source.write_text(json.dumps({'split':'val','checkpoint_sha256':'synthetic_test_hash',
                'artifact_state':'complete','predictions_sha256':digest(source.with_suffix('.npz'))}))
            result = subprocess.run([sys.executable, '-m', 'eopar.eval.calibrate',
                                    '--validation-json', str(source), '--output', str(out)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out.read_text())
            self.assertEqual(data['fitted_on'], 'val')
            self.assertEqual(data['checkpoint_sha256'], 'synthetic_test_hash')
            self.assertEqual(data['names'], ['a', 'b'])

    def test_annotation_agreement_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'{i}.csv' for i in range(2)]
            for path in paths:
                with path.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['image', 'attribute', 'observability'])
                    w.writeheader()
                    w.writerow(dict(image='a', attribute='Hat', observability='visible'))
                    w.writerow(dict(image='b', attribute='Hat', observability='unobservable'))
            result = agreement(*paths)
            self.assertEqual(result['compared'], 2)
            self.assertEqual(result['cohen_kappa'], 1.)
            paths[0].write_text('image,attribute,observability\na,Hat,\n')
            self.assertIsNone(agreement(*paths)['agreement'])

    def test_annotation_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.csv'
            path.write_text('image,attribute,observability\na,Hat,visible\na,Hat,visible\n')
            with self.assertRaises(ValueError):
                agreement(path, path)


if __name__ == '__main__':
    unittest.main()
