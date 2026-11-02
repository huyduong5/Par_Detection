#!/usr/bin/env python3
"""Small-attribute gate: flag weak small-object attrs from per_attr_mA in eval JSON."""

import argparse
import json
from pathlib import Path

SMALL_ATTRS = (
    'Glasses', 'Mask', 'Gloves',
    'Leather shoes', 'Casual shoes', 'Boots', 'Sandals', 'Other shoes',
    'Plastic bag', 'Paper bag', 'Smoking', 'Making a phone call',
)
DEFAULT_MARGIN = 0.05


def overall_ma(d):
    if 'clean' in d and isinstance(d['clean'], dict):
        return d['clean'].get('mA')
    return d.get('mA')


def weak_small_attrs(d, margin=DEFAULT_MARGIN, small_attrs=SMALL_ATTRS):
    """Return [(attr, mA)] for small attrs with mA < overall_mA - margin."""
    per_attr = d.get('per_attr_mA') or {}
    ref = overall_ma(d)
    if ref is None or not per_attr:
        return []
    out = [(a, per_attr[a]) for a in small_attrs
           if a in per_attr and per_attr[a] < ref - margin]
    return sorted(out, key=lambda x: x[1])


def render_md(data, margin=DEFAULT_MARGIN):
    lines = [
        '# Small-attribute gate',
        '',
        f'Weak = small attr with `mA_attr < overall_mA - {margin}`. '
        'Source: `per_attr_mA` in eval JSON.',
        '',
    ]
    present = [v for v, d in data.items() if d.get('per_attr_mA')]
    if not present:
        lines.append('_No eval JSON with per_attr_mA yet._')
        return '\n'.join(lines) + '\n'

    lines += ['| Attribute | ' + ' | '.join(present) + ' |',
              '|-----------|' + '|'.join(['------'] * len(present)) + '|']
    for attr in SMALL_ATTRS:
        cells = []
        for v in present:
            val = data[v]['per_attr_mA'].get(attr)
            cells.append(f'{val:.4f}' if isinstance(val, (int, float)) else '—')
        lines.append(f'| {attr} | ' + ' | '.join(cells) + ' |')
    lines.append('| **overall mA** | ' + ' | '.join(
        f'{overall_ma(data[v]):.4f}' for v in present) + ' |')

    lines += ['', '## Weak small attributes per variant', '']
    for v in present:
        weak = weak_small_attrs(data[v], margin=margin)
        txt = ', '.join(f'{a} ({m:.4f})' for a, m in weak) if weak else 'none'
        lines.append(f'- **{v}**: {txt}')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output_md', required=True)
    ap.add_argument('--variants', default='C0,C3,C4_reaudit,C4b,C3_MS,C4b_MS')
    ap.add_argument('--margin', type=float, default=DEFAULT_MARGIN)
    args = ap.parse_args()
    json_dir = Path(args.root) / 'aq_visibility' / 'output' / 'eval_json'
    data = {}
    for v in [x.strip() for x in args.variants.split(',') if x.strip()]:
        p = json_dir / f'{v}_test.json'
        if p.is_file():
            data[v] = json.load(open(p, encoding='utf-8'))
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(data, margin=args.margin), encoding='utf-8')
    print(f'Wrote {out} ({len(data)} variants)')


if __name__ == '__main__':
    main()
