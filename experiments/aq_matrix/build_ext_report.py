#!/usr/bin/env python3
"""Extension-matrix report from eval JSON only: consistency L1 vs sg_bce, single vs multi-scale."""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('small_attr_gate', HERE / 'small_attr_gate.py')
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

MAIN = ('C3', 'C4_reaudit', 'C4b', 'C4b_bs16', 'C3_MS', 'C4b_MS', 'C4b_MS_bs16',
        'C3_s43', 'C3_MS_s43', 'C4b_s43', 'C4b_MS_s43', 'C4b_MS_fpw')
SEED_PAIRS = (('42', 'C3', 'C3_MS'), ('43', 'C3_s43', 'C3_MS_s43'))
# Same config, two seeds: spread of the main metrics.
SEED_REPEATS = (('C3', 'C3', 'C3_s43'), ('C3_MS', 'C3_MS', 'C3_MS_s43'),
                ('C4b (16x4)', 'C4b_bs16', 'C4b_s43'),
                ('C4b_MS (16x4)', 'C4b_MS_bs16', 'C4b_MS_s43'))
PAIRS = (
    ('Consistency L1 vs sg_bce (16x4)', 'C4_reaudit', 'C4b_bs16'),
    ('Batch 8x8 vs 16x4 (sg_bce)', 'C4b_bs16', 'C4b'),
    ('Multi-scale on C3', 'C3', 'C3_MS'),
    ('Multi-scale on C4b (16x4)', 'C4b_bs16', 'C4b_MS_bs16'),
    ('sg_bce on C3_MS (16x4)', 'C3_MS', 'C4b_MS_bs16'),
    ('Multi-scale on C4b (8x8)', 'C4b', 'C4b_MS'),
    ('Fixed sqrt pos_weight on C4b_MS', 'C4b_MS_bs16', 'C4b_MS_fpw'),
)


def fmt(x, signed=False):
    if not isinstance(x, (int, float)):
        return '—'
    return f'{x:+.4f}' if signed else f'{x:.4f}'


def row(d):
    c = d.get('clean', d)
    return {
        'mA': c.get('mA'), 'acc': c.get('accuracy'), 'prec': c.get('precision'),
        'rec': c.get('recall'), 'f1': c.get('f1'),
        'occ': (d.get('occluded') or {}).get('mA'),
        'vis': (d.get('visibility') or {}).get('vis_mae'),
        'gap': (d.get('consistency') or {}).get('consistency_gap'),
        **{r: (d.get(f'region_{r}') or {}).get('mA') for r in ('head', 'upper', 'lower', 'shoes')},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output_md', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    jdir = root / 'aq_visibility' / 'output' / 'eval_json'
    data = {v: json.load(open(jdir / f'{v}_test.json', encoding='utf-8'))
            for v in MAIN if (jdir / f'{v}_test.json').is_file()}
    rows = {v: row(d) for v, d in data.items()}

    L = ['# Extension matrix — consistency v2 and multi-scale', '',
         'All numbers come from `aq_visibility/output/eval_json/<VAR>_test.json` (MSP60k test).', '',
         '## Main table', '',
         '| Variant | batch×accum | mA | Acc | Prec | Rec | F1 | occ_mA | vis_mae | consist_gap |',
         '|---------|-------------|----|-----|------|-----|----|--------|---------|-------------|']
    for v in MAIN:
        if v not in rows:
            continue
        r = rows[v]
        cfg = root / 'aq_visibility' / 'output' / v / 'batch.cfg'
        bs = cfg.read_text().strip().replace(' ', '×') if cfg.is_file() else '16×4'
        L.append(f"| {v} | {bs} | {fmt(r['mA'])} | {fmt(r['acc'])} | {fmt(r['prec'])} | "
                 f"{fmt(r['rec'])} | {fmt(r['f1'])} | {fmt(r['occ'])} | {fmt(r['vis'])} | {fmt(r['gap'])} |")

    L += ['', '## Region mA (clean)', '',
          '| Variant | head | upper | lower | shoes |', '|---------|------|-------|-------|-------|']
    for v in MAIN:
        if v in rows:
            r = rows[v]
            L.append(f"| {v} | {fmt(r['head'])} | {fmt(r['upper'])} | {fmt(r['lower'])} | {fmt(r['shoes'])} |")

    L += ['', '## Paired deltas (B − A)', '',
          '| Comparison | A | B | Δ mA | Δ occ_mA | Δ consist_gap | Δ shoes |',
          '|------------|---|---|------|----------|---------------|---------|']
    for name, a, b in PAIRS:
        if a in rows and b in rows:
            ra, rb = rows[a], rows[b]
            d = {k: (rb[k] - ra[k]) if isinstance(ra[k], float) and isinstance(rb[k], float) else None
                 for k in ('mA', 'occ', 'gap', 'shoes')}
            L.append(f"| {name} | {a} | {b} | {fmt(d['mA'], True)} | {fmt(d['occ'], True)} | "
                     f"{fmt(d['gap'], True)} | {fmt(d['shoes'], True)} |")

    seed_rows = [(s, a, b) for s, a, b in SEED_PAIRS if a in rows and b in rows]
    if seed_rows:
        L += ['', '## Seed check: multi-scale gain (C3_MS − C3)', '',
              '| Seed | Δ mA | Δ occ_mA | Δ shoes | Δ F1 |', '|------|------|----------|---------|------|']
        acc = {'mA': [], 'occ': [], 'shoes': [], 'f1': []}
        for s, a, b in seed_rows:
            d = {k: rows[b][k] - rows[a][k] for k in acc}
            for k in acc:
                acc[k].append(d[k])
            L.append(f"| {s} | {fmt(d['mA'], True)} | {fmt(d['occ'], True)} | "
                     f"{fmt(d['shoes'], True)} | {fmt(d['f1'], True)} |")
        if len(seed_rows) > 1:
            mean = {k: sum(v) / len(v) for k, v in acc.items()}
            L.append(f"| mean | {fmt(mean['mA'], True)} | {fmt(mean['occ'], True)} | "
                     f"{fmt(mean['shoes'], True)} | {fmt(mean['f1'], True)} |")

    rep = [(n, a, b) for n, a, b in SEED_REPEATS if a in rows and b in rows]
    if rep:
        L += ['', '## Seed repeats (seed 42 / seed 43, mean)', '',
              '| Config | mA | occ_mA | consist_gap | shoes |', '|--------|----|--------|-------------|-------|']
        for n, a, b in rep:
            cells = []
            for k in ('mA', 'occ', 'gap', 'shoes'):
                x, y = rows[a][k], rows[b][k]
                cells.append(f'{fmt(x)} / {fmt(y)} ({fmt((x + y) / 2)})'
                             if isinstance(x, float) and isinstance(y, float) else '—')
            L.append(f'| {n} | ' + ' | '.join(cells) + ' |')

    L += ['', '## Small attributes (per-attribute mA)', '']
    L.append(gate.render_md({v: data[v] for v in MAIN if v in data}).split('\n', 2)[2])
    if 'C3' in data and 'C3_MS' in data:
        pa, pb = data['C3'].get('per_attr_mA', {}), data['C3_MS'].get('per_attr_mA', {})
        L += ['', '### C3 → C3_MS on small attributes', '',
              '| Attribute | C3 | C3_MS | Δ |', '|-----------|----|-------|---|']
        for a in gate.SMALL_ATTRS:
            if a in pa and a in pb:
                L.append(f'| {a} | {fmt(pa[a])} | {fmt(pb[a])} | {fmt(pb[a] - pa[a], True)} |')
        L.append('')

    missing = [v for v in MAIN if v not in data]
    if missing:
        L += ['', f'_Missing eval JSON: {", ".join(missing)}_', '']
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'Wrote {out} ({len(data)} variants)')


if __name__ == '__main__':
    main()
