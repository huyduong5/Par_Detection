#!/usr/bin/env python3
"""
Tạo cặp occlusion synthetic sau split cố định (dataset_random.pkl).
Pose-lite: ước lượng uncertain khi band quá nhỏ / ảnh méo → vis_valid=0.

Output mặc định: data/occlusion/occlusion_pairs_{split}.pkl
"""

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / 'aq_visibility'))
from dataset.region_utils import (  # noqa: E402
    EVIDENCE_REGIONS,
    REGION_IDS,
    load_body_region_map,
    protocol_consist_mask,
    vertical_region_box,
)


def occlude_region(img: Image.Image, region: str, fill_mode='noise') -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = vertical_region_box(w, h, region)
    if x1 <= x0 or y1 <= y0:
        return img.copy()
    out = img.copy()
    crop = out.crop((x0, y0, x1, y1))
    arr = np.array(crop)
    if fill_mode == 'gray':
        arr[:] = 128
    else:
        rng = np.random.default_rng()
        arr = rng.integers(0, 256, size=arr.shape, dtype=np.uint8)
    out.paste(Image.fromarray(arr), (x0, y0))
    return out


def pose_lite_uncertain(w, h, region, min_side=32, min_band_frac=0.08):
    """
    Heuristic uncertainty without keypoints:
    - image too small
    - occluded band area fraction too small / degenerate
    Returns (bbox_ok, uncertain_reason)
    """
    if min(w, h) < min_side:
        return False, 'min_side'
    x0, y0, x1, y1 = vertical_region_box(w, h, region)
    bw, bh = max(0, x1 - x0), max(0, y1 - y0)
    if bw < 4 or bh < 4:
        return False, 'degenerate_box'
    frac = (bw * bh) / float(max(w * h, 1))
    if frac < min_band_frac:
        return False, 'thin_band'
    aspect = w / float(max(h, 1))
    if aspect < 0.2 or aspect > 1.5:
        return False, 'extreme_aspect'
    return True, ''


def vis_targets_for_occlusion(attributes, region_map, occluded_region, bbox_ok=True):
    n = len(attributes)
    vis_target = np.full(n, np.nan, dtype=np.float32)
    vis_valid = np.zeros(n, dtype=np.float32)
    affected = np.zeros(n, dtype=bool)

    for i, attr in enumerate(attributes):
        reg = region_map.get(attr, 'global')
        if reg not in EVIDENCE_REGIONS:
            continue
        if not bbox_ok:
            continue
        vis_valid[i] = 1.0
        if reg == occluded_region:
            vis_target[i] = 0.15
            affected[i] = True
        else:
            vis_target[i] = 0.85

    vis_target = np.nan_to_num(vis_target, nan=0.5)
    consist_mask = protocol_consist_mask(
        attributes, region_map, occluded_region, bbox_ok=bbox_ok)
    return vis_target, vis_valid, affected, consist_mask


def process_split(args, split, dataset_info, attributes, region_map):
    idx = dataset_info.partition[split]
    if isinstance(idx, list):
        idx = idx[0]
    img_names = [dataset_info.image_name[i] for i in idx]

    out_dir = Path(args.occ_img_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    by_imgname = {}
    region_counts = {r: 0 for r in EVIDENCE_REGIONS}
    uncertain_count = 0
    rng = random.Random(args.seed)

    sample_names = img_names
    if args.max_per_split > 0:
        sample_names = rng.sample(img_names, min(args.max_per_split, len(img_names)))

    img_root = args.img_dir or getattr(dataset_info, 'root', '')

    for name in tqdm(sample_names, desc=f'build {split}'):
        region = rng.choice(EVIDENCE_REGIONS)
        src = os.path.join(img_root, name)
        if not os.path.exists(src):
            continue
        try:
            img = Image.open(src).convert('RGB')
        except OSError:
            continue
        w, h = img.size
        bbox_ok, reason = pose_lite_uncertain(
            w, h, region, min_side=args.min_side, min_band_frac=args.min_band_frac
        )
        if not bbox_ok:
            uncertain_count += 1
            if args.skip_uncertain:
                continue

        occ = occlude_region(img, region, fill_mode=args.fill_mode)
        rel_name = name.replace('/', '__')
        occ_path = out_dir / rel_name
        occ_path.parent.mkdir(parents=True, exist_ok=True)
        occ.save(occ_path)

        vis_target, vis_valid, affected, consist_mask = vis_targets_for_occlusion(
            attributes, region_map, region, bbox_ok=bbox_ok
        )
        by_imgname[name] = {
            'pair_id': f'{name}::{region}',
            'region': region,
            'region_id': REGION_IDS[region],
            'occ_rel_path': str(occ_path.relative_to(args.occ_img_dir))
            if args.store_relative
            else str(occ_path),
            'vis_target': vis_target.tolist(),
            'vis_valid': vis_valid.tolist(),
            'attr_affected_mask': affected.tolist(),
            'consist_mask': consist_mask.tolist(),
            'uncertain': (not bbox_ok),
            'uncertain_reason': reason,
        }
        region_counts[region] += 1

    payload = {
        'split': split,
        'by_imgname': by_imgname,
        'region_counts': region_counts,
        'num_pairs': len(by_imgname),
        'uncertain_skipped_or_invalid': uncertain_count,
        'attributes': list(attributes),
    }
    out_pkl = Path(args.output_dir) / f'occlusion_pairs_{split}.pkl'
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(payload, f)
    print(
        f'Wrote {out_pkl} ({len(by_imgname)} pairs) counts={region_counts} '
        f'uncertain={uncertain_count}'
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', default=str(ROOT / 'MSP60k/SUBMIT/dataset_random.pkl'))
    parser.add_argument('--img_dir', default='')
    parser.add_argument(
        '--region_map', default=str(ROOT / 'data/occlusion/body_region_map.json'))
    parser.add_argument('--output_dir', default=str(ROOT / 'data/occlusion'))
    parser.add_argument(
        '--occ_img_dir', default=str(ROOT / 'data/occlusion/occlusion_images'))
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--max_per_split', type=int, default=0, help='0 = all images')
    parser.add_argument('--min_side', type=int, default=32)
    parser.add_argument('--min_band_frac', type=float, default=0.08)
    parser.add_argument('--skip_uncertain', action='store_true',
                        help='Skip images marked uncertain instead of writing vis_valid=0')
    parser.add_argument('--fill_mode', choices=['noise', 'gray'], default='noise')
    parser.add_argument('--store_relative', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.pkl_path, 'rb') as f:
        dataset_info = pickle.load(f)

    if hasattr(dataset_info, 'attributes'):
        attributes = list(dataset_info.attributes)
    elif hasattr(dataset_info, 'attr_name'):
        attributes = list(dataset_info.attr_name)
    else:
        attributes = list(dataset_info.attr_words)

    region_map = load_body_region_map(args.region_map)
    for split in args.splits.split(','):
        process_split(args, split.strip(), dataset_info, attributes, region_map)


if __name__ == '__main__':
    main()
