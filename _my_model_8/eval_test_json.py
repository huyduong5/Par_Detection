#!/usr/bin/env python3
"""Evaluate v8 Ref on test and write JSON metrics."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_kaggle_config
from model import CLIMPPAR
from losses import CLIMPPARLoss
from dataset.clip_dataset import PARDataset, get_clip_transforms
from train import compute_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pkl_path', default='')
    parser.add_argument('--img_dir', default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--region_map', default='')
    args = parser.parse_args()

    config = get_kaggle_config()
    if args.pkl_path:
        config.pkl_path = args.pkl_path
    if args.img_dir:
        config.img_dir = args.img_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, tf = get_clip_transforms(config.img_height, config.img_width)
    ds = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        split='test',
        transform=tf,
        domain_tokens_path=config.domain_tokens_path,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    attrs = ds.attributes

    config.vmamba_pretrained = False
    model = CLIMPPAR(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(clean)
    criterion = CLIMPPARLoss(config=config)

    model.eval()
    preds, labels, total_loss = [], [], 0.0
    for images, y, _, domain in tqdm(loader, desc='C0 test'):
        images, y, domain = images.to(device), y.to(device), domain.to(device)
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            logits, feat = model(images, domain, attrs)
            loss = criterion(logits, y, feat)
        total_loss += float(loss.item())
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(y.float().cpu().numpy())

    p = np.concatenate(preds, 0)
    l = np.concatenate(labels, 0)
    metrics = compute_metrics(p, l)
    metrics['loss'] = total_loss / max(len(loader), 1)
    metrics['val_best_mA'] = float(ckpt.get('best_mA', float('nan'))) if isinstance(ckpt, dict) else float('nan')
    metrics['num_test'] = int(l.shape[0])

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    region_map_path = args.region_map or os.path.join(
        repo, 'data', 'occlusion', 'body_region_map.json')
    if os.path.isfile(region_map_path):
        with open(region_map_path, encoding='utf-8') as f:
            region_map = json.load(f)
        for reg in ('head', 'upper', 'lower', 'shoes'):
            idx = [i for i, a in enumerate(attrs) if region_map.get(a) == reg]
            if idx:
                metrics[f'region_{reg}'] = compute_metrics(p[:, idx], l[:, idx])

    pred_bin = (p >= 0.5)
    per_attr = {}
    for i, attr in enumerate(attrs):
        pos = l[:, i] == 1
        neg = ~pos
        acc_pos = (pred_bin[pos, i]).sum() / max(pos.sum(), 1)
        acc_neg = (~pred_bin[neg, i]).sum() / max(neg.sum(), 1)
        per_attr[str(attr)] = float((acc_pos + acc_neg) / 2.0)
    metrics['per_attr_mA'] = per_attr

    print(json.dumps(metrics, indent=2))
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
