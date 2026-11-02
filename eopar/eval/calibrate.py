"""Fit thresholds ONLY using a validation evaluation artifact and its NPZ."""
import argparse
import json
from pathlib import Path
import numpy as np
from eopar.eval.metrics import par_metrics
from eopar.train.runtime import digest, write_json
from eopar.eval.artifacts import validate_arrays


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--validation-json', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    meta_path = Path(args.validation_json)
    meta = json.loads(meta_path.read_text())
    if meta.get('split') != 'val':
        raise ValueError('Threshold fitting requires split=val; test fitting is forbidden')
    data_path = meta_path.with_suffix('.npz')
    if meta.get('artifact_state') != 'complete' or meta.get('predictions_sha256') != digest(data_path):
        raise ValueError('Validation artifact is incomplete or checksum changed')
    a = np.load(data_path, allow_pickle=False)
    validate_arrays(a)
    values, labels = a['clean'], a['labels']
    thresholds = []
    for i in range(values.shape[1]):
        grid = np.linspace(.01, .99, 99)
        # Fixed grid; deterministic first maximum; no test observations.
        scores = [par_metrics(values[:, i:i+1], labels[:, i:i+1], float(t))['mA'] for t in grid]
        thresholds.append(float(grid[int(np.argmax(scores))]))
    if Path(args.output).exists(): raise FileExistsError(args.output)
    write_json(args.output, dict(thresholds=thresholds, fitted_on='val',
        validation_sha256=digest(data_path), attributes=a['attributes'].tolist(),
        names=a['names'].tolist(), checkpoint_sha256=meta['checkpoint_sha256'],
        note='Separate calibrated decoding; standard tables remain fixed 0.5'))


if __name__ == '__main__':
    main()
