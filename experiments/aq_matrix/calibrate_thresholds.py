#!/usr/bin/env python3
"""Post-hoc decoding from saved probabilities (no retraining).

Thresholds are chosen on the val split only, then applied to test:
  fixed0.5   : standard protocol (threshold 0.5)
  thr_val    : per-attribute threshold maximising val balanced accuracy
  thr_val+grp: thr_val, then one-hot decoding inside mutually exclusive groups
               (argmax of p - t), only for groups that are exclusive in train labels
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    'metrics_aq', REPO / 'aq_visibility' / 'metrics' / 'metrics_aq.py')
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
compute_metrics, per_attr_label_accuracy = _m.compute_metrics, _m.per_attr_label_accuracy

CANDIDATE_GROUPS = (
    ('Fat', 'Normal', 'Thin'),
    ('Child', 'Adult', 'Elderly'),
    ('Front', 'Back', 'Side'),
)
GRID = np.concatenate([np.linspace(0.01, 0.1, 10), np.linspace(0.12, 0.98, 44)])


def balanced_acc(pred, y):
    pos, neg = y == 1, y == 0
    return ((pred[pos] == 1).sum() / max(pos.sum(), 1) + (pred[neg] == 0).sum() / max(neg.sum(), 1)) / 2


def fit_thresholds(p_val, y_val):
    t = np.full(p_val.shape[1], 0.5)
    for i in range(p_val.shape[1]):
        scores = [balanced_acc((p_val[:, i] >= g).astype(np.int8), y_val[:, i]) for g in GRID]
        t[i] = GRID[int(np.argmax(scores))]
    return t


def exclusive_groups(attrs, y_train_like, min_rate=0.99):
    groups = []
    for g in CANDIDATE_GROUPS:
        if not all(a in attrs for a in g):
            continue
        idx = [attrs.index(a) for a in g]
        rate = float((y_train_like[:, idx].sum(1) == 1).mean())
        groups.append((g, idx, rate, rate >= min_rate))
    return groups


def decode(p, t, groups=()):
    pred = (p >= t).astype(np.float32)
    for _, idx, _, ok in groups:
        if not ok:
            continue
        margin = p[:, idx] - t[idx]
        onehot = np.zeros_like(margin)
        onehot[np.arange(len(margin)), margin.argmax(1)] = 1
        pred[:, idx] = onehot
    return pred


def metrics_from_pred(pred, y):
    # compute_metrics thresholds at 0.5, so binary predictions pass through unchanged.
    return compute_metrics(pred, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True, help='Has <VAR>_val.npz and <VAR>_test.npz')
    ap.add_argument('--variants', required=True)
    ap.add_argument('--output_md', required=True)
    ap.add_argument('--output_json', default='')
    args = ap.parse_args()
    pdir = Path(args.pred_dir)

    results, notes = {}, []
    for var in [v.strip() for v in args.variants.split(',') if v.strip()]:
        fv, ft = pdir / f'{var}_val.npz', pdir / f'{var}_test.npz'
        if not (fv.is_file() and ft.is_file()):
            notes.append(f'{var}: missing predictions')
            continue
        val, test = np.load(fv), np.load(ft)
        attrs = [str(a) for a in test['attributes']]
        yv, yt = val['labels'].astype(np.float32), test['labels'].astype(np.float32)
        t = fit_thresholds(val['probs_clean'], yv)
        groups = exclusive_groups(attrs, yv)
        res = {'thresholds': dict(zip(attrs, map(float, t))),
               'groups': [{'attrs': list(g), 'exclusive_rate_val': r, 'used': ok}
                          for g, _, r, ok in groups]}
        for name, tt, gg in (('fixed0.5', np.full(len(attrs), 0.5), ()),
                             ('thr_val', t, ()),
                             ('thr_val+grp', t, groups)):
            entry = {'clean': metrics_from_pred(decode(test['probs_clean'], tt, gg), yt)}
            if 'probs_occ' in test.files:
                entry['occluded'] = metrics_from_pred(decode(test['probs_occ'], tt, gg), yt)
            entry['per_attr_mA'] = dict(zip(attrs, map(float, per_attr_label_accuracy(
                decode(test['probs_clean'], tt, gg), yt))))
            res[name] = entry
        results[var] = res

    L = ['# Post-hoc threshold calibration (no retraining)', '',
         'Thresholds fitted on **val** only (per attribute, max balanced accuracy), applied to **test**. '
         '`fixed0.5` is the standard protocol number; calibrated rows must be reported as a separate decoding.', '',
         '| Variant | decoding | mA | F1 | occ_mA | occ_F1 |',
         '|---------|----------|----|----|--------|--------|']
    for var, res in results.items():
        for name in ('fixed0.5', 'thr_val', 'thr_val+grp'):
            e = res[name]
            o = e.get('occluded', {})
            L.append(f"| {var} | {name} | {e['clean']['mA']:.4f} | {e['clean']['f1']:.4f} | "
                     f"{o.get('mA', float('nan')):.4f} | {o.get('f1', float('nan')):.4f} |")
    if results:
        first = next(iter(results.values()))
        L += ['', '## Exclusive groups (label check on val)', '']
        for g in first['groups']:
            L.append(f"- {', '.join(g['attrs'])}: exactly-one rate {g['exclusive_rate_val']:.4f} "
                     f"→ {'used' if g['used'] else 'not used'}")
        L += ['', '## Attributes most changed by calibration (first variant, clean mA)', '',
              '| Attribute | fixed0.5 | thr_val | thr_val+grp | threshold |',
              '|-----------|----------|---------|-------------|-----------|']
        var0 = next(iter(results))
        r0 = results[var0]
        diffs = sorted(r0['fixed0.5']['per_attr_mA'],
                       key=lambda a: r0['thr_val+grp']['per_attr_mA'][a] - r0['fixed0.5']['per_attr_mA'][a],
                       reverse=True)[:12]
        for a in diffs:
            L.append(f"| {a} | {r0['fixed0.5']['per_attr_mA'][a]:.4f} | {r0['thr_val']['per_attr_mA'][a]:.4f} | "
                     f"{r0['thr_val+grp']['per_attr_mA'][a]:.4f} | {r0['thresholds'][a]:.2f} |")
        L.append(f'\n_(table above: {var0})_')
    if notes:
        L += ['', '## Notes', ''] + [f'- {n}' for n in notes]
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {out} ({len(results)} variants)')
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
