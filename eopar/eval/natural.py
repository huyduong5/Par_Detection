"""Evaluate human observability labels against saved SINGLE-IMAGE predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def binary_auc(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    if y.shape != s.shape or y.ndim != 1 or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid AUC inputs')
    positive, negative = y == 1, y == 0
    if not positive.any() or not negative.any():
        return None
    # Mann-Whitney statistic with average ranks for ties.
    order = np.argsort(s, kind='stable')
    sorted_s = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1]
    ends = np.r_[starts[1:], len(order)]
    ranks = np.empty(len(order), float)
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + 1 + b) / 2
    n, m = int(positive.sum()), int(negative.sum())
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * m))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--annotations', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    a = np.load(args.predictions, allow_pickle=False)
    from eopar.eval.artifacts import validate_arrays
    from eopar.train.runtime import digest
    metadata_path=Path(args.predictions).with_suffix('.json')
    metadata=json.loads(metadata_path.read_text())
    if metadata.get('artifact_state')!='complete' or metadata.get('predictions_sha256')!=digest(args.predictions):
        raise ValueError('Natural evaluation requires a complete, checksummed prediction artifact')
    if metadata.get('split')!=args.split:
        raise ValueError('Prediction/annotation split mismatch')
    validate_arrays(a)
    if 'obs' not in a:
        raise ValueError('No observability predictions')
    names = list(a['names']); attrs = list(a['attributes'])
    lookup = {str(n): i for i, n in enumerate(names)}
    attr_lookup = {str(n): i for i, n in enumerate(attrs)}
    labels, scores, counts, seen = [], [], {}, set()
    with open(args.annotations, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r['split'] != args.split:
            continue
        if r.get('natural_occlusion_confirmed', '').lower() not in ('1', 'true', 'yes'):
            raise ValueError('Natural occlusion must be confirmed by a human')
        key = r['image'], r['attribute']
        if key in seen:
            raise ValueError('Use adjudicated labels; duplicate keys found')
        seen.add(key)
        if key[0] not in lookup or key[1] not in attr_lookup:
            raise ValueError(f'Annotation missing from prediction: {key}')
        state = r['observability']
        if state not in ('visible', 'partial', 'unobservable', 'uncertain'):
            raise ValueError('Incomplete annotation')
        counts[state] = counts.get(state, 0) + 1
        if state in ('visible', 'unobservable'):
            labels.append(int(state == 'visible'))
            scores.append(float(a['obs'][lookup[key[0]], attr_lookup[key[1]]]))
    report = dict(split=args.split, counts=counts, binary_observability_auc=binary_auc(labels, scores),
                  evaluated_binary_pairs=len(scores),
                  partial_uncertain_policy='Reported as counts; excluded from binary visibility AUC')
    if not seen:
        raise ValueError('No matching annotations')
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
