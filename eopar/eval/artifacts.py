"""Validate artifacts before publishing the JSON completion marker."""
import os
import uuid
from pathlib import Path
import numpy as np


def validate_arrays(a):
    names, attrs = a['names'].tolist(), a['attributes'].tolist()
    if not names or not attrs or len(names) != len(set(names)) or len(attrs) != len(set(attrs)):
        raise ValueError('Empty or duplicate names/attributes')
    shape = (len(names), len(attrs))
    for key in ('clean', 'occ', 'labels', 'obs', 'obs_occ', 'vis_target', 'vis_valid', 'affected', 'intact'):
        if key not in a:
            if key in ('clean', 'labels'):
                raise ValueError(f'Missing array {key}')
            continue
        value = a[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid shape or nonfinite array {key}')
        if (value < 0).any() or (value > 1).any():
            raise ValueError(f'Array outside [0,1]: {key}')
        if key == 'labels' and not np.isin(value, [0, 1]).all():
            raise ValueError('Nonbinary labels')
    return shape


def save_predictions(path, arrays):
    validate_arrays(arrays)
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_thresholds(fitted, attributes, checkpoint_hash, names=None, split=None):
    values = np.asarray(fitted.get('thresholds', []), dtype=float)
    if fitted.get('fitted_on') != 'val' or fitted.get('attributes') != list(attributes):
        raise ValueError('Invalid calibration provenance/attributes')
    if values.shape != (len(attributes),) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Invalid calibrated thresholds')
    if fitted.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Calibration belongs to another checkpoint')
    if not fitted.get('validation_sha256') or not fitted.get('names'):
        raise ValueError('Missing validation provenance')
    if split == 'test' and set(fitted['names']) & set(names):
        raise ValueError('Calibration/test image overlap')
    return values
