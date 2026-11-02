#!/usr/bin/env python3
"""Run C0–C4 on the same 2 test images and save a comparison folder.

Usage (on GPU host):
  python aq_visibility/tools/compare_c0_c4_images.py \\
    --out_dir aq_visibility/output/compare_5c_images \\
    --seed 42
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parents[1]


def _load_pkl(pkl_path: Path):
    import pickle
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def _pick_two_images(info, seed: int, indices=None):
    test_idx = list(info.partition['test'])
    attrs = list(info.attributes)
    labels = np.asarray(info.label)
    rng = np.random.RandomState(seed)

    if indices is not None and len(indices) == 2:
        chosen = [int(x) for x in indices]
    else:
        # Prefer samples with enough positive attrs, and different pos-count bands.
        scored = []
        for gi in test_idx:
            pos = int(labels[gi].sum())
            if 4 <= pos <= 18:
                scored.append((pos, gi))
        scored.sort(key=lambda x: x[0])
        if len(scored) < 2:
            chosen = [int(x) for x in rng.choice(test_idx, size=2, replace=False)]
        else:
            lo = scored[len(scored) // 5][1]
            hi = scored[(4 * len(scored)) // 5][1]
            if lo == hi:
                hi = scored[-1][1]
            chosen = [int(lo), int(hi)]

    samples = []
    for k, gi in enumerate(chosen, start=1):
        name = info.image_name[gi]
        gt = labels[gi].astype(np.float32)
        samples.append({
            'id': f'img{k:02d}',
            'global_idx': gi,
            'imgname': name,
            'gt': gt,
            'gt_positive': [attrs[i] for i in range(len(attrs)) if gt[i] >= 0.5],
        })
    return samples, attrs


def _get_transforms(h, w, which='aq'):
    if which == 'aq':
        sys.path.insert(0, str(PKG))
        from dataset.clip_dataset import get_clip_transforms
        return get_clip_transforms(h, w)[1]
    sys.path.insert(0, str(REPO / '_my_model_8'))
    from dataset.clip_dataset import get_clip_transforms
    return get_clip_transforms(h, w)[1]


def _domain_zeros(device, n=32, d=768):
    return torch.zeros(1, n, d, device=device)


def _load_base_model(ckpt_path, device):
    """C0/C1: _my_model_8 CLIMPPAR."""
    m8 = str(REPO / '_my_model_8')
    # Isolate from aq_visibility package name collisions
    for k in list(sys.modules):
        if k == 'config' or k.startswith('config.') or k == 'model' or k.startswith('model.') \
                or k == 'dataset' or k.startswith('dataset.') or k == 'losses' or k.startswith('losses.'):
            del sys.modules[k]
    if m8 not in sys.path:
        sys.path.insert(0, m8)
    from config import CLIMPConfig, get_kaggle_config
    from model import CLIMPPAR

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = get_kaggle_config()
    if 'config' in ckpt and isinstance(ckpt['config'], dict):
        import dataclasses
        valid = {f.name for f in dataclasses.fields(CLIMPConfig)}
        filtered = {k: v for k, v in ckpt['config'].items() if k in valid}
        cfg = CLIMPConfig(**{**dataclasses.asdict(cfg), **filtered})
    model = CLIMPPAR(cfg)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device).eval()
    return model, cfg, 'base'


def _load_aq_model(ckpt_path, variant, device, num_attrs):
    """C2–C4: aq_visibility CLIMPPAR."""
    for k in list(sys.modules):
        if k == 'config' or k.startswith('config.') or k == 'model' or k.startswith('model.') \
                or k == 'dataset' or k.startswith('dataset.') or k == 'losses' or k.startswith('losses.'):
            del sys.modules[k]
    if str(PKG) not in sys.path:
        sys.path.insert(0, str(PKG))
    from config import get_thesis_variant
    from model import CLIMPPAR

    cfg = get_thesis_variant(variant)
    model = CLIMPPAR(cfg, num_attrs=num_attrs)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device).eval()
    return model, cfg, 'aq'


@torch.no_grad()
def _predict(model, kind, img_path, transform, attrs, device, threshold=0.5):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    dt = _domain_zeros(device)
    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        if kind == 'aq' and hasattr(model, 'predict_attributes'):
            out = model.predict_attributes(x, dt, attrs)
            probs = out['probs'][0].float().cpu().numpy()
            vis = out['visibility'][0].float().cpu().numpy() if 'visibility' in out else None
        else:
            logits, feats = model(x, dt, attrs)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()
            vis = None
            if isinstance(feats, dict) and feats.get('vis_pred') is not None:
                vis = feats['vis_pred'][0].float().cpu().numpy()

    rows = []
    for i, name in enumerate(attrs):
        p = float(probs[i])
        row = {
            'attr': name,
            'prob': round(p, 4),
            'pred': int(p >= threshold),
        }
        if vis is not None:
            row['vis'] = round(float(vis[i]), 4)
        rows.append(row)
    return rows, probs, vis


def _per_image_metrics(probs, gt, threshold=0.5):
    pred = (probs >= threshold).astype(np.float32)
    gt = gt.astype(np.float32)
    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    tn = float(((1 - pred) * (1 - gt)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    acc = (tp + tn) / max(len(gt), 1)
    # sample-level balanced attr agreement
    correct = int((pred == gt).sum())
    return {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'n_correct_attrs': correct,
        'n_attrs': int(len(gt)),
        'n_pred_pos': int(pred.sum()),
        'n_gt_pos': int(gt.sum()),
    }


def render_one(raw_img, attribute_names, gt_label, probs, threshold, title, save_path):
    """Layout: image left, GT (green) + Prediction (blue) panels with ✔/✖ FP."""
    preds_binary = (probs >= threshold).astype(np.int32)
    gt_label = gt_label.astype(np.int32)

    gt_attrs = [attr for i, attr in enumerate(attribute_names) if gt_label[i] == 1]
    pred_attrs = []
    for i, attr in enumerate(attribute_names):
        if preds_binary[i] == 1:
            status = '✔' if gt_label[i] == 1 else '✖ (FP)'
            pred_attrs.append(f'{attr}: {probs[i]:.1%} {status}')
    missed_attrs = []
    for i, attr in enumerate(attribute_names):
        if gt_label[i] == 1 and preds_binary[i] == 0:
            missed_attrs.append(f'{attr}: {probs[i]:.1%} ✖ (FN)')

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.imshow(raw_img)
    ax.axis('off')

    gt_text = 'Ground Truth:\n' + (
        '\n'.join(['- ' + a for a in gt_attrs]) if gt_attrs else '- None')
    pred_text = f'Prediction (Probs >= {int(threshold * 100)}%):\n'
    pred_text += '\n'.join(['- ' + a for a in pred_attrs]) if pred_attrs else '- None'
    if missed_attrs:
        pred_text += '\n\nMissed:\n' + '\n'.join(['- ' + a for a in missed_attrs])

    ax.text(
        1.05, 0.95, gt_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
    )
    ax.text(
        1.55, 0.95, pred_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
    )
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _write_markdown(out_dir: Path, samples, attrs, all_results, names):
    lines = [
        '# So sánh C0–C4 trên 2 ảnh',
        '',
        f'Ảnh: `{samples[0]["imgname"]}` · `{samples[1]["imgname"]}`',
        '',
    ]
    for s in samples:
        sid = s['id']
        lines += [
            f'## {sid} — `{s["imgname"]}`',
            '',
            f'GT+ ({len(s["gt_positive"])}): ' + ', '.join(s['gt_positive']),
            '',
            '| Variant | Acc | Prec | Rec | F1 | #Pred+ | TP | FP | FN |',
            '|---------|-----|------|-----|----|--------|----|----|----|',
        ]
        for var in names:
            m = all_results[var][sid]['metrics']
            lines.append(
                f"| {var} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                f"{m['recall']:.4f} | {m['f1']:.4f} | {m['n_pred_pos']} | "
                f"{m['tp']} | {m['fp']} | {m['fn']} |"
            )
        lines += [
            '',
            '### Attr-level (GT+ và mọi pred khác nhau giữa các C)',
            '',
            '| Attr | GT | ' + ' | '.join(names) + ' |',
            '|------|----|' + '|'.join(['------'] * len(names)) + '|',
        ]
        # collect attrs of interest
        interesting = set(s['gt_positive'])
        for var in names:
            for row in all_results[var][sid]['preds']:
                if row['pred'] == 1:
                    interesting.add(row['attr'])
        # also attrs where variants disagree
        for attr in attrs:
            preds_v = [next(r['pred'] for r in all_results[v][sid]['preds'] if r['attr'] == attr)
                       for v in names]
            if len(set(preds_v)) > 1:
                interesting.add(attr)

        for attr in attrs:
            if attr not in interesting:
                continue
            gt = int(s['gt'][attrs.index(attr)] >= 0.5)
            cells = []
            for var in names:
                row = next(r for r in all_results[var][sid]['preds'] if r['attr'] == attr)
                mark = f"{row['prob']:.2f}"
                if row['pred'] != gt:
                    mark = f"**{mark}**"
                cells.append(mark)
            lines.append(f"| {attr} | {gt} | " + ' | '.join(cells) + ' |')
        lines.append('')

        # visibility for C3/C4 if present
        for var in ('C3', 'C4'):
            if var not in all_results:
                continue
            vis_rows = [r for r in all_results[var][sid]['preds'] if 'vis' in r]
            if not vis_rows:
                continue
            lines += [f'### {var} visibility (top-low / top-high)', '']
            by_vis = sorted(vis_rows, key=lambda r: r['vis'])
            lines.append('Lowest vis: ' + ', '.join(f"{r['attr']}={r['vis']:.2f}" for r in by_vis[:8]))
            lines.append('Highest vis: ' + ', '.join(f"{r['attr']}={r['vis']:.2f}" for r in by_vis[-8:]))
            lines.append('')

    (out_dir / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=str(REPO / 'aq_visibility' / 'output' / 'compare_5c_images'))
    ap.add_argument('--pkl_path', default=str(REPO / 'MSP60k' / 'SUBMIT' / 'dataset_random.pkl'))
    ap.add_argument('--img_dir', default=str(REPO / 'MSP60k' / 'SUBMIT' / 'images'))
    ap.add_argument('--ckpt_root', default=str(REPO / 'aq_visibility' / 'output'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--indices', nargs=2, type=int, default=None,
                    help='Optional two global indices into dataset_random.pkl')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--variants', nargs='+', default=['C0', 'C1', 'C2', 'C3', 'C4'])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_out = out_dir / 'images'
    img_out.mkdir(parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = _load_pkl(Path(args.pkl_path))
    samples, attrs = _pick_two_images(info, args.seed, args.indices)

    meta = {
        'seed': args.seed,
        'threshold': args.threshold,
        'variants': args.variants,
        'images': [],
    }
    for s in samples:
        src = Path(args.img_dir) / s['imgname']
        if not src.exists():
            # sometimes nested
            cand = list(Path(args.img_dir).rglob(os.path.basename(s['imgname'])))
            if not cand:
                raise FileNotFoundError(src)
            src = cand[0]
        dst = img_out / f"{s['id']}_{os.path.basename(s['imgname'])}"
        shutil.copy2(src, dst)
        s['path'] = str(dst)
        s['src'] = str(src)
        meta['images'].append({
            'id': s['id'],
            'imgname': s['imgname'],
            'global_idx': s['global_idx'],
            'path': str(dst),
            'gt_positive': s['gt_positive'],
        })
        print(f"Selected {s['id']}: {s['imgname']}  GT+={len(s['gt_positive'])}")

    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    all_results = {}
    names = args.variants

    for var in names:
        ckpt = Path(args.ckpt_root) / var / 'best_model.pth'
        if not ckpt.is_file():
            print(f'SKIP {var}: missing {ckpt}')
            continue
        print(f'\n=== {var}  {ckpt} ===')
        if var in ('C0', 'C1'):
            model, cfg, kind = _load_base_model(str(ckpt), device)
            tf = _get_transforms(cfg.img_height, cfg.img_width, which='base')
        else:
            model, cfg, kind = _load_aq_model(str(ckpt), var, device, num_attrs=len(attrs))
            tf = _get_transforms(cfg.img_height, cfg.img_width, which='aq')

        var_dir = out_dir / var
        var_dir.mkdir(parents=True, exist_ok=True)
        all_results[var] = {}

        for sample_k, s in enumerate(samples, start=1):
            rows, probs, vis = _predict(
                model, kind, s['path'], tf, attrs, device, threshold=args.threshold)
            metrics = _per_image_metrics(probs, s['gt'], threshold=args.threshold)
            pred_pos = [r['attr'] for r in rows if r['pred'] == 1]
            payload = {
                'variant': var,
                'image_id': s['id'],
                'imgname': s['imgname'],
                'metrics': metrics,
                'gt_positive': s['gt_positive'],
                'pred_positive': pred_pos,
                'preds': rows,
            }
            all_results[var][s['id']] = payload
            with open(var_dir / f"{s['id']}.json", 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            # Same naming/layout as c2_strong_sample1_*.jpg.png
            safe_name = s['imgname'].replace('/', '_')
            viz_name = f'{var.lower()}_sample{sample_k}_{safe_name}.png'
            title = f'{var} | idx={s["global_idx"]} | {s["imgname"]}'
            raw_img = Image.open(s['path']).convert('RGB')
            render_one(
                raw_img, attrs, s['gt'], probs, args.threshold, title,
                str(var_dir / viz_name),
            )
            # also copy to folder root for easy browse
            shutil.copy2(var_dir / viz_name, out_dir / viz_name)
            print(f"  {s['id']}: F1={metrics['f1']:.3f} Acc={metrics['accuracy']:.3f} "
                  f"Pred+={metrics['n_pred_pos']} GT+={metrics['n_gt_pos']} → {viz_name}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Aggregate JSON + markdown
    with open(out_dir / 'comparison.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    _write_markdown(out_dir, samples, attrs, all_results, [v for v in names if v in all_results])

    print(f'\nDone → {out_dir}')
    print('Files: comparison.md, comparison.json, images/, C0..C4/, {c*}_sample*.png')


if __name__ == '__main__':
    main()
