#!/usr/bin/env python3
"""Aggregate C0–C5 eval JSON → docs/aq_visibility/results.md (measured only)."""

import argparse
import json
from pathlib import Path


KEYS = ('mA', 'accuracy', 'precision', 'recall', 'f1')


def extract(d):
    if not isinstance(d, dict):
        return {}
    if 'clean' in d and isinstance(d['clean'], dict):
        base = d['clean']
    else:
        base = {k: d[k] for k in KEYS if k in d}
    out = {k: base.get(k) for k in KEYS}
    if 'occluded' in d and isinstance(d['occluded'], dict):
        out['occ_mA'] = d['occluded'].get('mA')
    if 'visibility' in d:
        out['vis_mae'] = d['visibility'].get('vis_mae')
        out['vis_acc'] = d['visibility'].get('vis_acc')
    if 'consistency' in d:
        out['consist_gap'] = d['consistency'].get('consistency_gap')
    for reg in ('head', 'upper', 'lower', 'shoes'):
        rk = f'region_{reg}'
        if rk in d and isinstance(d[rk], dict):
            out[f'{reg}_mA'] = d[rk].get('mA')
    return out


def fmt(x):
    return f'{x:.4f}' if isinstance(x, (int, float)) else '—'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output_md', required=True)
    ap.add_argument('--variants',
                    default='C0,C1,C2,C3,C4,C4_reaudit,C4b,C3_MS,C4b_MS,C5')
    args = ap.parse_args()
    root = Path(args.root)
    json_dir = root / 'aq_visibility' / 'output' / 'eval_json'
    rows = []
    for var in [v.strip() for v in args.variants.split(',') if v.strip()]:
        p = json_dir / f'{var}_test.json'
        if not p.is_file():
            continue
        m = extract(json.load(open(p)))
        rows.append((var, m))

    lines = [
        '# AQ Visibility — results',
        '',
        'Protocol: `docs/aq_visibility/protocol.md`. Metrics from measured JSON only.',
        '',
        '| Variant | mA | Acc | Prec | Rec | F1 | occ_mA | vis_mae | consist_gap |',
        '|---------|-----|-----|------|-----|-----|--------|---------|-------------|',
    ]
    for var, m in rows:
        lines.append(
            f"| {var} | {fmt(m.get('mA'))} | {fmt(m.get('accuracy'))} | "
            f"{fmt(m.get('precision'))} | {fmt(m.get('recall'))} | {fmt(m.get('f1'))} | "
            f"{fmt(m.get('occ_mA'))} | {fmt(m.get('vis_mae'))} | {fmt(m.get('consist_gap'))} |"
        )

    lines += [
        '',
        '## Region mA (clean)',
        '',
        '| Variant | head | upper | lower | shoes |',
        '|---------|------|-------|-------|-------|',
    ]
    for var, m in rows:
        lines.append(
            f"| {var} | {fmt(m.get('head_mA'))} | {fmt(m.get('upper_mA'))} | "
            f"{fmt(m.get('lower_mA'))} | {fmt(m.get('shoes_mA'))} |"
        )

    lines += [
        '',
        '## Source JSON',
        '',
    ]
    for var, _ in rows:
        lines.append(f'- `aq_visibility/output/eval_json/{var}_test.json`')
    if not rows:
        lines.append('_No eval JSON found yet. Run `experiments/aq_matrix/run_all.sh`._')

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {out} ({len(rows)} variants)')


if __name__ == '__main__':
    main()
