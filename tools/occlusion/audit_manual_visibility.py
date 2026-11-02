#!/usr/bin/env python3
"""Đối chiếu nhãn visibility tay (CSV) với synthetic occlusion pkl."""

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_manual_csv(path):
    """CSV columns: imgname, region, visibility (0-1) OR attr, visibility."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manual_csv',
        default=str(ROOT / 'data/occlusion/manual_visibility.csv'),
    )
    parser.add_argument(
        '--occlusion_pkl',
        default=str(ROOT / 'data/occlusion/occlusion_pairs_test.pkl'),
    )
    parser.add_argument('--region_only', action='store_true', help='Compare region-level mean vis')
    parser.add_argument(
        '--output_json',
        default=str(ROOT / 'data/occlusion/manual_audit_report.json'),
    )
    args = parser.parse_args()

    with open(args.occlusion_pkl, 'rb') as f:
        occ = pickle.load(f)
    by_img = occ.get('by_imgname', {})

    manual = load_manual_csv(args.manual_csv)
    agree = []
    missing = 0
    details = []

    for row in manual:
        name = row['imgname']
        rec = by_img.get(name) or by_img.get(Path(name).name)
        if rec is None:
            missing += 1
            continue
        manual_v = float(row['visibility'])
        if args.region_only:
            vt = np.array(rec['vis_target'])
            vv = np.array(rec['vis_valid'])
            if vv.sum() < 1:
                continue
            synth_v = float((vt * vv).sum() / vv.sum())
        else:
            attr = row.get('attr')
            if not attr:
                continue
            attrs = occ.get('attributes', [])
            if attr not in attrs:
                missing += 1
                continue
            i = attrs.index(attr)
            synth_v = float(rec['vis_target'][i])
        ok = abs(manual_v - synth_v) < 0.35
        agree.append(ok)
        details.append({
            'imgname': name,
            'manual': manual_v,
            'synth': synth_v,
            'agree': bool(ok),
        })

    n = len(agree)
    report = {
        'compared': n,
        'missing': missing,
        'agreement_at_0.35': float(np.mean(agree)) if n else None,
        'note': 'Holdout audit only — not used in train loss.',
        'details': details,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if n == 0:
        print('No comparable rows. Wrote', out)
        return
    print(f'Compared {n} rows, missing={missing}, agreement@0.35={np.mean(agree):.3f}')
    print('Wrote', out)


if __name__ == '__main__':
    main()
