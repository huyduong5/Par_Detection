#!/usr/bin/env python3
"""Train C0 (base) or C1 (base + region occ aug) via _my_model_8 CLIMPPAR."""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, '_my_model_8'))
sys.path.insert(0, os.path.join(REPO, 'aq_visibility'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['C0', 'C1'], required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--pkl_path', type=str, default='')
    parser.add_argument('--img_dir', type=str, default='')
    parser.add_argument('--occ_aug_prob', type=float, default=0.5)
    parser.add_argument('--no_ddp', action='store_true', default=True)
    args = parser.parse_args()

    # Patch _my_model_8 transforms for C1 before importing train
    if args.variant == 'C1':
        from dataset import occ_aug as aq_occ_aug
        import dataset.clip_dataset as m8_ds

        def _c1_transforms(height, width):
            return aq_occ_aug.get_c1_transforms(height, width, occ_aug_prob=args.occ_aug_prob)

        m8_ds.get_clip_transforms = _c1_transforms
        print(f'[C1] Using RegionOcclusionAug p={args.occ_aug_prob}')

    from config import get_kaggle_config
    import train as m8_train

    out = args.output_dir or os.path.join(REPO, 'aq_visibility', 'output', args.variant)
    os.makedirs(out, exist_ok=True)
    config = get_kaggle_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_ddp=False,
        output_dir=out,
        use_domain_tokens=False,
    )
    if args.pkl_path:
        config.pkl_path = args.pkl_path
    if args.img_dir:
        config.img_dir = args.img_dir

    print(f'Training {args.variant} → {out}')
    m8_train.main(rank=0, world_size=1, config=config)


if __name__ == '__main__':
    main()
