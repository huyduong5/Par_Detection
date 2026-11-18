"""Deterministic person-band intervention, before padding and after shared flip.

Heuristic bands are *synthetic proxy* evidence regions, never natural labels.
No RandomErasing or independent random crops can alter pair alignment.
"""
import hashlib
import json
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from .geometry import REGIONS, BANDS, read_geometry, bands_for

MEAN = torch.tensor([.48145466, .4578275, .40821073])[:, None, None]
STD = torch.tensor([.26862954, .26130258, .27577711])[:, None, None]


def seed_for(seed, split, epoch, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{split}:{epoch}:{name}'.encode()).digest()[:8], 'big')


def region_masks(attributes, mapping, region, valid=True):
    known = torch.tensor([mapping.get(a) in REGIONS for a in attributes])
    affected = torch.tensor([mapping.get(a) == region for a in attributes]) & known
    if not valid:
        known[:] = False
        affected[:] = False
    return known.float(), affected.float(), (known & ~affected).float()


def encode(image, size):
    # Exactly preserve the baseline's thumbnail+center-padding validation geometry.
    image = image.convert('RGB').copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    tensor = torch.from_numpy(np.array(canvas).copy()).permute(2, 0, 1).float() / 255
    return (tensor - MEAN) / STD


class InterventionDataset(Dataset):
    def __init__(self, base, region_map, split='train', seed=42, size=448,
                 two_types=False, family='solid', severity=1., augment=True, geometry_manifest=None):
        if split not in ('train', 'val', 'test') or family not in ('solid', 'noise'):
            raise ValueError('Unsupported split/family')
        if not 0 < severity <= 1:
            raise ValueError('severity must be in (0,1]')
        self.base, self.split, self.seed, self.size = base, split, seed, size
        with open(region_map, encoding='utf-8') as handle:
            self.mapping = json.load(handle)
        self.attributes = list(base.attributes)
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError('Duplicate attribute names')
        if set(self.attributes) - set(self.mapping):
            raise ValueError('Region map is missing attributes')
        if any(self.mapping[a] not in (*REGIONS, 'global', 'none') for a in self.attributes):
            raise ValueError('Unknown region name')
        self.two_types, self.family, self.severity = two_types, family, severity
        self.augment = augment
        self.geometry = None
        self.geometry_records = None
        if geometry_manifest:
            _, self.geometry_records = read_geometry(geometry_manifest)
            self.geometry = {name: r['valid_regions'] for name, r in self.geometry_records.items() if r['split'] == split}
        self.epoch = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, labels, name, tokens = self.base[index]
        if not isinstance(image, Image.Image):
            raise TypeError('Base dataset must have transform=None')
        rng = random.Random(seed_for(self.seed, self.split, self.epoch if self.split == 'train' else 0, name))
        image = image.convert('RGB').copy()
        if image.height < 8 or image.width < 1:
            raise ValueError(f'Image too small for four evidence bands: {name} {image.size}')
        valid_regions = list(REGIONS) if self.geometry is None else self.geometry.get(str(name), [])
        if self.geometry_records is not None and str(name) in self.geometry_records:
            row = self.geometry_records[str(name)]
            if row['split'] != self.split or row['bands'] != bands_for(*image.size):
                raise ValueError(f'Geometry image split/size differs: {name}')
        if self.split == 'train' and self.augment:
            if rng.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(.8, 1.2))
        # Preserve the diagnostic RNG path when no reviewed manifest was provided.
        target_region = rng.randrange(4) if self.geometry is None else (
            REGIONS.index(rng.choice(valid_regions)) if valid_regions else 0)
        outside_requested = self.two_types and rng.random() < .5
        outside = outside_requested and len(valid_regions) >= 2
        region_index = ((target_region + rng.randrange(1, 4)) % 4 if self.geometry is None else
                        REGIONS.index(rng.choice([g for g in valid_regions if g != REGIONS[target_region]]))) if outside else target_region
        region = REGIONS[region_index]
        y0, y1 = BANDS[region_index]
        w, h = image.size
        lo, hi = int(h * y0), max(int(h * y0) + 1, int(h * y1))
        hi = min(h, lo + max(1, round((hi - lo) * self.severity)))
        occ = image.copy()
        if self.family == 'solid':
            patch = Image.new('RGB', (w, hi - lo), tuple(rng.randrange(256) for _ in range(3)))
        else:
            nrng = np.random.default_rng(rng.randrange(2**32))
            patch = Image.fromarray(nrng.integers(0, 256, (hi - lo, w, 3), dtype=np.uint8))
        occ.paste(patch, (0, lo))
        known, affected, intact = region_masks(self.attributes, self.mapping, region)
        geometry_verified=False
        if self.geometry is not None:
            # Manifest asserts validity of these exact fixed bands for this crop,
            # not simply whether a named body part exists somewhere in the image.
            geometry_verified=region in valid_regions
            eligible=torch.tensor([self.mapping.get(a) in valid_regions for a in self.attributes])
            if not geometry_verified:eligible.zero_()
            known*=eligible;affected*=eligible;intact*=eligible
        # Existing legacy visibility proxy: soft values, valid only for evidence attrs.
        visibility = torch.where(affected.bool(), torch.tensor(.85 - .7 * self.severity), torch.tensor(.85))
        rank_weights = affected * self.severity
        if outside:
            # Explicit outside intervention supervises only the selected target region.
            intact *= torch.tensor([self.mapping.get(a) == REGIONS[target_region] for a in self.attributes])
            rank_weights.zero_()
        no_intervention = self.geometry is not None and (not valid_regions or (outside_requested and not outside))
        if no_intervention:
            occ = image.copy()  # Classification only; never corrupt an unverified region.
            geometry_verified = False
        if self.geometry is not None and outside_requested and not outside:
            # Do not silently replace an unavailable outside pair with an inside objective.
            known.zero_(); affected.zero_(); intact.zero_(); rank_weights.zero_()
        label = torch.as_tensor(labels, dtype=torch.float32)
        if not torch.all((label == 0) | (label == 1)):
            raise ValueError('This adapter requires fully binary labels; add an explicit missing-label adapter')
        return dict(img_clean=encode(image, self.size), img_occ=encode(occ, self.size),
                    labels=label, domain_token=torch.as_tensor(tokens), imgname=name,
                    vis_target=visibility, vis_valid=known, affected_mask=affected,
                    rank_weights=rank_weights,
                    intact_mask=intact, intervention_type='none' if no_intervention or (self.geometry is not None and outside_requested and not outside) else 'outside' if outside else 'inside',
                    region=region, target_region=REGIONS[target_region], severity=self.severity,
                    geometry_verified=geometry_verified, outside_requested=outside_requested,
                    supervision_active=bool(known.sum().item()))
