"""Collect available historical JSON, report absent seeds without inventing values."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.root) / 'aq_visibility/output'
    result = {'protocol': 'historical legacy metrics; not a new standardized evaluation', 'groups': {}}
    for variant in ('C3_MS', 'C4b_MS'):
        rows = []
        for seed in (42, 43, 44):
            name = (('C4b_MS_bs16' if variant == 'C4b_MS' else variant) if seed == 42 else f'{variant}_s{seed}')
            path = root / ('seed44_20261031' if seed == 44 else 'eval_json') / f'{name}_test.json'
            if path.is_file():
                data = json.loads(path.read_text())
                rows.append(dict(seed=seed, state='available', path=str(path), clean=data['clean'],
                                 occluded=data.get('occluded')))
            else:
                rows.append(dict(seed=seed, state='not_available', expected_path=str(path)))
        available = [r for r in rows if r['state'] == 'available']
        summary = {}
        for key in ('mA', 'f1'):
            values = [r['clean'][key] for r in available]
            summary[key] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                               sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        result['groups'][variant] = dict(rows=rows, summary=summary)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
