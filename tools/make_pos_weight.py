#!/usr/bin/env python3
"""Fixed per-attribute pos_weight from train label frequencies (for use_fixed_pos_weight).

mode=sqrt: sqrt(neg/pos). Damps the extreme ratios (Paper bag ≈ 590 → ≈ 24) while
still up-weighting rare attributes and down-weighting near-always-positive ones (Normal).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl_path', default=str(ROOT / 'MSP60k/SUBMIT/dataset_random.pkl'))
    ap.add_argument('--mode', choices=['sqrt', 'ratio'], default='sqrt')
    ap.add_argument('--output', default=str(ROOT / 'data/attr_pos_weight_sqrt.json'))
    args = ap.parse_args()

    with open(args.pkl_path, 'rb') as f:
        info = pickle.load(f)
    idx = info.partition['train']
    idx = idx[0] if isinstance(idx, list) else idx
    y = np.asarray(info.label)[idx].astype(np.float64)
    pos = y.sum(0).clip(min=1)
    neg = (len(y) - y.sum(0)).clip(min=1)
    ratio = neg / pos
    pw = np.sqrt(ratio) if args.mode == 'sqrt' else ratio
    attrs = [str(a) for a in info.attributes]
    payload = {
        'mode': args.mode,
        'num_train': int(len(y)),
        'attributes': attrs,
        'pos_freq': (y.mean(0)).tolist(),
        'pos_weight': pw.tolist(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    order = np.argsort(pw)
    print(f'Wrote {args.output} mode={args.mode}')
    print('lowest :', [(attrs[i], round(float(pw[i]), 3)) for i in order[:3]])
    print('highest:', [(attrs[i], round(float(pw[i]), 2)) for i in order[-5:]])


if __name__ == '__main__':
    main()
