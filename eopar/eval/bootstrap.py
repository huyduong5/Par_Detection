"""Paired bootstrap on saved predictions; resample clusters when supplied."""
import argparse
import json
from pathlib import Path
import numpy as np
from eopar.eval.metrics import par_metrics
from eopar.eval.artifacts import validate_arrays


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--clusters', help='JSON map image name -> identity/camera cluster')
    p.add_argument('--repeats', type=int, default=1000)
    args = p.parse_args()
    a, b = [np.load(x, allow_pickle=False) for x in (args.baseline, args.candidate)]
    validate_arrays(a);validate_arrays(b)
    if 'occ' not in a or 'occ' not in b:
        raise ValueError('Both clean and occluded arrays are required')
    for key in ('names', 'attributes', 'labels'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'Paired arrays disagree on {key}')
    if args.repeats < 100:
        raise ValueError('Use at least 100 bootstrap samples')
    if args.clusters:
        mapping = json.loads(Path(args.clusters).read_text())
        clusters = np.asarray([mapping[str(n)] for n in a['names']])
    else:
        clusters = np.arange(len(a['names']))
    unique = np.unique(clusters)
    blocks = [np.flatnonzero(clusters == v) for v in unique]
    rng = np.random.default_rng(42)
    deltas = {'clean_mA': [], 'occluded_mA': []}
    for _ in range(args.repeats):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        for key, field in [('clean_mA', 'clean'), ('occluded_mA', 'occ')]:
            deltas[key].append(par_metrics(b[field][idx], b['labels'][idx])['mA'] -
                               par_metrics(a[field][idx], a['labels'][idx])['mA'])
    result = dict(seed=42, repeats=args.repeats, unit='cluster' if args.clusters else 'image',
                  warning='Image bootstrap does not establish identity independence' if not args.clusters else None,
                  delta_ci95={k: np.quantile(v, [.025, .975]).tolist() for k, v in deltas.items()})
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
