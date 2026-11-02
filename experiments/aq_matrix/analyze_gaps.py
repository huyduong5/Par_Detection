#!/usr/bin/env python3
"""Phase 6: per-region gap analysis; gate multi-scale recommendation."""

import argparse
import json
from pathlib import Path


def clean_ma(d):
    if 'clean' in d:
        return d['clean'].get('mA')
    return d.get('mA')


def region_ma(d, reg):
    return (d.get(f'region_{reg}') or {}).get('mA')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output_md', required=True)
    ap.add_argument('--shoes_gap_threshold', type=float, default=0.04,
                    help='If C4 shoes_mA lags C0 by more than this, recommend multi-scale')
    ap.add_argument('--variants',
                    default='C0,C1,C2,C3,C4,C4_reaudit,C4b,C3_MS,C4b_MS')
    args = ap.parse_args()
    root = Path(args.root)
    json_dir = root / 'aq_visibility' / 'output' / 'eval_json'

    data = {}
    for var in [v.strip() for v in args.variants.split(',') if v.strip()]:
        p = json_dir / f'{var}_test.json'
        if p.is_file():
            data[var] = json.load(open(p))

    lines = [
        '# AQ Visibility — gap analysis (Phase 6)',
        '',
        'Multi-scale is **deferred** unless shoes/small-attr regions lag clearly.',
        '',
    ]
    if not data:
        lines += [
            '_No eval JSON yet. After `run_all.sh`, re-run this script._',
            '',
            '## Gate rule',
            '',
            f'- Recommend multi-scale if `(C0.shoes_mA - C4.shoes_mA) > {args.shoes_gap_threshold}`.',
            '- Otherwise keep single-scale VMamba tokens (32).',
        ]
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'Wrote stub {args.output_md}')
        return

    lines += [
        '| Variant | overall_mA | head | upper | lower | shoes |',
        '|---------|------------|------|-------|-------|-------|',
    ]
    for var, d in data.items():
        lines.append(
            f"| {var} | {clean_ma(d) or float('nan'):.4f} | "
            f"{region_ma(d, 'head') or float('nan'):.4f} | "
            f"{region_ma(d, 'upper') or float('nan'):.4f} | "
            f"{region_ma(d, 'lower') or float('nan'):.4f} | "
            f"{region_ma(d, 'shoes') or float('nan'):.4f} |"
        )

    recommend = False
    detail = 'insufficient C0/C4 region metrics'
    c4_key = 'C4_reaudit' if 'C4_reaudit' in data else 'C4'
    if 'C0' in data and c4_key in data:
        s0 = region_ma(data['C0'], 'shoes')
        s4 = region_ma(data[c4_key], 'shoes')
        if s0 is not None and s4 is not None:
            gap = s0 - s4
            detail = f'C0.shoes_mA={s0:.4f}, {c4_key}.shoes_mA={s4:.4f}, gap={gap:.4f}'
            recommend = gap > args.shoes_gap_threshold

    lines += [
        '',
        '## Multi-scale decision',
        '',
        f'- Evidence: {detail}',
        f'- Threshold: {args.shoes_gap_threshold}',
        f'- **Decision: {"ENABLE multi-scale (Phase 7)" if recommend else "KEEP single-scale (deferred)"}**',
        '',
    ]

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {out}; recommend_multiscale={recommend}')


if __name__ == '__main__':
    main()
