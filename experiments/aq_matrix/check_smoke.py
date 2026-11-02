#!/usr/bin/env python3
"""Validate a smoke run of run_ext.sh: logs show the right loss terms, JSON has all keys."""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_JSON = ('clean', 'occluded', 'visibility', 'consistency',
                 'region_head', 'region_upper', 'region_lower', 'region_shoes', 'per_attr_mA')
EXPECT_CONS = {'C4b': True, 'C4b_MS': True, 'C3_MS': False}
EXPECT_MS = {'C4b': False, 'C4b_MS': True, 'C3_MS': True}


def check(out_root, var):
    errs = []
    log = out_root / 'ext_logs' / f'train_{var}.log'
    text = log.read_text(encoding='utf-8', errors='replace') if log.is_file() else ''
    # Log is appended across retries; judge only the last attempt.
    header = 'CLIMP-PAR v9 Training'
    if header in text:
        text = text[text.rindex(header):]
    if not text:
        errs.append('missing train log')
    if 'Traceback' in text:
        errs.append('Traceback in train log')
    if 'L_vis=' not in text:
        errs.append('no L_vis in log')
    has_cons = 'L_consist[sg_bce]=' in text
    if has_cons != EXPECT_CONS[var]:
        errs.append(f'L_consist[sg_bce] present={has_cons}, expected {EXPECT_CONS[var]}')
    if f'use_multiscale={EXPECT_MS[var]}' not in text:
        errs.append(f'use_multiscale={EXPECT_MS[var]} not in log')
    if not (out_root / var / 'best_model.pth').is_file():
        errs.append('no best_model.pth')

    jp = out_root / 'eval_json' / f'{var}_test.json'
    if not jp.is_file():
        errs.append('no eval JSON')
    else:
        d = json.load(open(jp, encoding='utf-8'))
        missing = [k for k in REQUIRED_JSON if k not in d]
        if missing:
            errs.append(f'JSON missing {missing}')
        if len(d.get('per_attr_mA') or {}) != 57:
            errs.append('per_attr_mA != 57 attrs')
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--variants', default='C4b,C3_MS,C4b_MS')
    args = ap.parse_args()
    out_root = Path(args.out_root)
    failed = False
    for var in [v.strip() for v in args.variants.split(',') if v.strip()]:
        errs = check(out_root, var)
        print(f'SMOKE {var}: {"PASS" if not errs else "FAIL " + "; ".join(errs)}')
        failed |= bool(errs)
    print('SMOKE_ALL_PASS' if not failed else 'SMOKE_HAS_FAIL')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
