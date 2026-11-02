"""Controlled occlusion augmentation for C1 (base + occ aug).

Uses vertical body-region bands (head/upper/lower/shoes) instead of
generic RandomErasing, so the ablation is distinct from default transforms.
"""

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .region_utils import EVIDENCE_REGIONS, vertical_region_box


class RegionOcclusionAug:
    """Erase one random evidence region with probability `p` (tensor CHW)."""

    def __init__(self, p=0.5, fill=0.0):
        self.p = p
        self.fill = fill

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        C, H, W = tensor.shape
        region = random.choice(list(EVIDENCE_REGIONS))
        x0, y0, x1, y1 = vertical_region_box(W, H, region)
        if x1 <= x0 or y1 <= y0:
            return tensor
        out = tensor.clone()
        out[:, y0:y1, x0:x1] = self.fill
        return out


def get_c1_transforms(height, width, occ_aug_prob=0.5):
    """Train transforms for C1: no RandomErasing; use RegionOcclusionAug instead."""
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    normalize = T.Normalize(mean=clip_mean, std=clip_std)

    from .clip_dataset import PadToFixedSize

    train_transform = T.Compose([
        PadToFixedSize(height, width, fill=0),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        normalize,
        RegionOcclusionAug(p=occ_aug_prob, fill=0.0),
    ])
    valid_transform = T.Compose([
        PadToFixedSize(height, width, fill=0),
        T.ToTensor(),
        normalize,
    ])
    return train_transform, valid_transform
