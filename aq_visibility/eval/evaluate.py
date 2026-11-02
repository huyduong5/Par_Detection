# evaluate.py — clean + occluded + visibility + consistency metrics

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, PKG_DIR)

from config import get_kaggle_config, get_aq_config_variant
from model import CLIMPPAR
from losses import CLIMPPARLoss
from dataset.clip_dataset import PARDataset, get_clip_transforms
from dataset.occlusion_dataset import OcclusionPairDataset, collate_occlusion_batch
from dataset.region_utils import load_body_region_map
from metrics.metrics_aq import (
    compute_metrics,
    consistency_gap,
    per_attr_ma_dict,
    per_region_attr_metrics,
    visibility_metrics,
)


@torch.no_grad()
def run_eval(model, loader, criterion, device, config, attribute_names, use_occ_image=False):
    model.eval()
    all_preds_clean, all_labels = [], []
    all_preds_occ = []
    all_vis_p, all_vis_t, all_vis_v = [], [], []
    all_consist = []
    total_loss = 0.0

    for batch in tqdm(loader, desc='eval'):
        if isinstance(batch, dict):
            images = batch['img_clean'].to(device)
            labels = batch['labels'].to(device)
            domain_tokens = batch['domain_token'].to(device)
            images_occ = batch['img_occ'].to(device)
            vis_target = batch['vis_target'].to(device)
            vis_valid = batch['vis_valid'].to(device)
            consist_mask = batch['consist_mask'].to(device)
        else:
            images, labels, _, domain_tokens = batch
            images = images.to(device)
            labels = labels.to(device)
            domain_tokens = domain_tokens.to(device)
            images_occ = vis_target = vis_valid = consist_mask = None

        with torch.amp.autocast('cuda', enabled=config.use_amp):
            logits, features = model(images, domain_tokens, attribute_names)
            vt, vv = vis_target, vis_valid
            if images_occ is not None and config.use_visibility:
                _, f_occ = model(images_occ, domain_tokens, attribute_names)
                if 'vis_pred' in f_occ:
                    features = dict(features)
                    features['vis_pred'] = f_occ['vis_pred']
            loss, _ = criterion(logits, labels, features, vis_target=vt, vis_valid_mask=vv)
        total_loss += loss.item()

        all_preds_clean.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if images_occ is not None:
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                logits_o, _ = model(
                    images_occ if use_occ_image else images_occ,
                    domain_tokens,
                    attribute_names,
                )
            all_preds_occ.append(torch.sigmoid(logits_o).cpu().numpy())

        if features.get('vis_pred') is not None and vis_target is not None:
            all_vis_p.append(features['vis_pred'].cpu().numpy())
            all_vis_t.append(vis_target.cpu().numpy())
            all_vis_v.append(vis_valid.cpu().numpy())

        if images_occ is not None and consist_mask is not None:
            p_c = torch.sigmoid(logits).cpu().numpy()
            p_o = torch.sigmoid(logits_o).cpu().numpy()
            all_consist.append((p_c, p_o, consist_mask.cpu().numpy()))

    preds = np.concatenate(all_preds_clean, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    out = {'clean': compute_metrics(preds, labels), 'loss': total_loss / max(len(loader), 1)}

    if all_preds_occ:
        preds_occ = np.concatenate(all_preds_occ, axis=0)
        out['occluded'] = compute_metrics(preds_occ, labels)
        out['_preds_occ'] = preds_occ

    if all_vis_p:
        vp = np.concatenate(all_vis_p, axis=0)
        vt = np.concatenate(all_vis_t, axis=0)
        vv = np.concatenate(all_vis_v, axis=0)
        out['visibility'] = visibility_metrics(vp, vt, vv)

    if all_consist:
        p_c = np.concatenate([x[0] for x in all_consist], axis=0)
        p_o = np.concatenate([x[1] for x in all_consist], axis=0)
        m = np.concatenate([x[2] for x in all_consist], axis=0)
        out['consistency'] = consistency_gap(p_c, p_o, m)

    out['_preds_clean'] = preds
    out['_labels'] = labels
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config_variant', default='C2')
    parser.add_argument('--split', default='test')
    parser.add_argument('--pkl_path', default='')
    parser.add_argument('--img_dir', default='')
    parser.add_argument('--occlusion_pkl_path', default='')
    parser.add_argument('--occlusion_img_dir', default='')
    parser.add_argument('--output_json', default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Smoke only: evaluate the first N samples (0 = all)')
    parser.add_argument('--save_preds', default='',
                        help='Optional .npz with clean/occluded probabilities and labels')
    args = parser.parse_args()

    config = get_aq_config_variant(args.config_variant)
    if args.pkl_path:
        config.pkl_path = args.pkl_path
    if args.img_dir:
        config.img_dir = args.img_dir
    if args.occlusion_pkl_path:
        config.occlusion_pkl_path = args.occlusion_pkl_path
        # Keep images next to pkl under data/occlusion/
        if not args.occlusion_img_dir:
            config.occlusion_img_dir = os.path.join(
                os.path.dirname(os.path.abspath(args.occlusion_pkl_path)),
                'occlusion_images',
            )
    if args.occlusion_img_dir:
        config.occlusion_img_dir = args.occlusion_img_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, tf = get_clip_transforms(config.img_height, config.img_width)
    domain_path = config.domain_tokens_path if config.use_domain_tokens else None

    if config.use_occlusion_pairs:
        occ_path = config.occlusion_pkl_path.replace('_train.pkl', f'_{args.split}.pkl')
        if not os.path.exists(occ_path):
            occ_path = config.occlusion_pkl_path
        ds = OcclusionPairDataset(
            pkl_path=config.pkl_path,
            img_dir=config.img_dir,
            split=args.split,
            transform=tf,
            domain_tokens_path=domain_path,
            occlusion_pkl_path=occ_path,
            occ_img_dir=config.occlusion_img_dir,
            return_occ=True,
        )
        collate = collate_occlusion_batch
    else:
        ds = PARDataset(
            pkl_path=config.pkl_path,
            img_dir=config.img_dir,
            split=args.split,
            transform=tf,
            domain_tokens_path=domain_path,
        )
        collate = None

    attrs = list(ds.attributes)
    eval_ds = ds
    if args.max_samples > 0:
        eval_ds = Subset(ds, list(range(min(args.max_samples, len(ds)))))
    loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )
    model = CLIMPPAR(config, num_attrs=len(attrs)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'load_state_dict missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print('missing keys:', missing[:30])
    critical_prefixes = ('query_decoder', 'query_heads', 'ms_tokenizer')
    if getattr(config, 'use_attr_queries', False):
        bad = [k for k in missing if k.startswith(critical_prefixes)]
        if bad:
            raise RuntimeError(
                f'Checkpoint missing {len(bad)} query weights, e.g. {bad[:8]}')
    criterion = CLIMPPARLoss(config)

    metrics = run_eval(model, loader, criterion, device, config, attrs)
    region_map = load_body_region_map(config.body_region_map_path)
    preds = metrics.pop('_preds_clean')
    labels = metrics.pop('_labels')
    preds_occ = metrics.pop('_preds_occ', None)
    if args.save_preds:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_preds)), exist_ok=True)
        payload = {'probs_clean': preds.astype(np.float32), 'labels': labels.astype(np.uint8),
                   'attributes': np.array(attrs)}
        if preds_occ is not None:
            payload['probs_occ'] = preds_occ.astype(np.float32)
        np.savez_compressed(args.save_preds, **payload)
        print(f'Saved predictions to {args.save_preds}')
    for reg in ('head', 'upper', 'lower', 'shoes'):
        metrics[f'region_{reg}'] = per_region_attr_metrics(preds, labels, attrs, region_map, reg)
    metrics['per_attr_mA'] = per_attr_ma_dict(preds, labels, attrs)
    metrics['num_test'] = int(labels.shape[0])
    metrics['variant'] = args.config_variant

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
