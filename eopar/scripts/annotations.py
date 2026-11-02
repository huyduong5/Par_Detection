"""Prepare blinded candidate sheets or compare independent observability labels."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

STATES = ('visible', 'partial', 'unobservable', 'uncertain')


def agreement(first, second):
    def read(path):
        rows = list(csv.DictReader(open(path, encoding='utf-8')))
        keys = [(r['image'], r['attribute']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate annotation keys')
        if any(r.get('observability') not in ('', *STATES) for r in rows):
            raise ValueError('Unknown annotation state')
        return {k: r['observability'] for k, r in zip(keys, rows) if r.get('observability')}
    a, b = read(first), read(second)
    keys = sorted(a.keys() & b.keys())
    matrix = np.zeros((4, 4), dtype=int)
    for k in keys:
        matrix[STATES.index(a[k]), STATES.index(b[k])] += 1
    n = int(matrix.sum())
    observed = float(np.trace(matrix) / n) if n else None
    expected = float((matrix.sum(0) * matrix.sum(1)).sum() / n**2) if n else None
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    return dict(compared=n, agreement=observed, cohen_kappa=kappa, confusion=matrix.tolist(),
                states=STATES, disagreement_keys=[list(k) for k in keys if a[k] != b[k]])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--candidates', help='CSV: image,split,attribute; no attribute labels/predictions')
    p.add_argument('--first'); p.add_argument('--second')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.candidates:
        rows = list(csv.DictReader(open(args.candidates, encoding='utf-8')))
        partitions = {}
        for r in rows:
            if r['split'] not in ('val', 'test'):
                raise ValueError('Natural audit must exclude training images')
            if r['image'] in partitions and partitions[r['image']] != r['split']:
                raise ValueError('Image appears in both validation and test')
            partitions[r['image']] = r['split']
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['image', 'split', 'attribute', 'natural_occlusion_confirmed',
                                             'observability', 'annotator', 'notes'])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in ('image', 'split', 'attribute')})
    elif args.first and args.second:
        out.write_text(json.dumps(agreement(args.first, args.second), indent=2))
    else:
        p.error('Use --candidates or both --first and --second')


if __name__ == '__main__':
    main()
