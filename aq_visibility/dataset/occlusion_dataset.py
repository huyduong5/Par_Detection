import os
import pickle
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .clip_dataset import PARDataset
from .region_utils import load_body_region_map, protocol_consist_mask


class OcclusionPairDataset(PARDataset):
    """
    MSP60k split + synthetic occlusion pairs (clean / occluded).
    Mỗi sample: clean image; optional occ image + visibility / consistency masks.
    """

    def __init__(
        self,
        pkl_path,
        occlusion_pkl_path,
        img_dir=None,
        occ_img_dir=None,
        split='train',
        transform=None,
        domain_tokens_path=None,
        return_occ=True,
    ):
        super().__init__(
            pkl_path=pkl_path,
            img_dir=img_dir,
            split=split,
            transform=transform,
            domain_tokens_path=domain_tokens_path,
        )
        assert os.path.exists(occlusion_pkl_path), occlusion_pkl_path
        with open(occlusion_pkl_path, 'rb') as f:
            occ_data = pickle.load(f)

        self.occ_by_img = occ_data.get('by_imgname', {})
        self.default_vis_valid = occ_data.get('default_vis_valid', None)
        # Allow occ images on test when pairs exist (needed for occ/consist metrics).
        self.return_occ = bool(return_occ)
        map_path = os.path.join(
            os.path.dirname(os.path.abspath(occlusion_pkl_path)),
            'body_region_map.json',
        )
        self.region_map = load_body_region_map(map_path) if os.path.isfile(map_path) else None
        # occ_rel_path in pkl is relative to this dir (e.g. train/xxx.jpg)
        if occ_img_dir:
            self.occ_img_dir = occ_img_dir
        else:
            self.occ_img_dir = os.path.join(
                os.path.dirname(os.path.abspath(occlusion_pkl_path)),
                'occlusion_images',
            )

    def _load_rgb(self, path_or_name, root=None):
        if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
            return Image.open(path_or_name).convert('RGB')
        search_root = root if root is not None else self.root_path
        imgpath = os.path.join(search_root, path_or_name)
        if os.path.exists(imgpath):
            return Image.open(imgpath).convert('RGB')
        if os.path.exists(path_or_name):
            return Image.open(path_or_name).convert('RGB')
        raise FileNotFoundError(
            f'Image not found: {path_or_name} (tried {imgpath})'
        )

    def __getitem__(self, index):
        imgname = self.img_ids[index]
        gt_label = self.labels[index].astype(np.float32)
        img_pil = self._load_rgb(imgname)

        if self.transform is not None:
            img_clean = self.transform(img_pil)
        else:
            img_clean = img_pil

        domain_token = torch.zeros(32, 768)
        if self.domain_tokens is not None:
            domain_name = imgname.split('/')[0] if '/' in imgname else ''
            if domain_name in self.domain_tokens:
                domain_token = self.domain_tokens[domain_name]['mean']

        n_attr = self.attr_num
        vis_target = np.full(n_attr, 0.5, dtype=np.float32)
        vis_valid = np.zeros(n_attr, dtype=np.float32)
        # Unpaired: no evidence that anything is intact under occlusion.
        consist_mask = np.zeros(n_attr, dtype=np.float32)
        region_id = -1
        pair_id = imgname
        img_occ = img_clean

        rec = self.occ_by_img.get(imgname)
        if rec is None:
            rec = self.occ_by_img.get(os.path.basename(imgname))

        if rec is not None:
            pair_id = rec.get('pair_id', pair_id)
            region_id = int(rec.get('region_id', -1))
            if 'vis_target' in rec:
                vis_target = np.array(rec['vis_target'], dtype=np.float32)
            if 'vis_valid' in rec:
                vis_valid = np.array(rec['vis_valid'], dtype=np.float32)
            occ_region = rec.get('region', '')
            uncertain = bool(rec.get('uncertain', False))
            if self.region_map is not None and occ_region:
                # Recompute so stored masks that included global attrs are corrected.
                consist_mask = protocol_consist_mask(
                    self.attributes, self.region_map, occ_region, bbox_ok=not uncertain)
            elif 'consist_mask' in rec:
                consist_mask = np.array(rec['consist_mask'], dtype=np.float32)
            elif 'attr_affected_mask' in rec:
                affected = np.array(rec['attr_affected_mask'], dtype=bool)
                consist_mask = (~affected).astype(np.float32)

            if self.return_occ and rec.get('occ_rel_path'):
                occ_pil = self._load_rgb(rec['occ_rel_path'], root=self.occ_img_dir)
                if self.transform is not None:
                    img_occ = self.transform(occ_pil)
                else:
                    img_occ = occ_pil
        elif self.default_vis_valid is not None:
            vis_valid = np.array(self.default_vis_valid, dtype=np.float32)

        return {
            'img_clean': img_clean,
            'img_occ': img_occ,
            'labels': torch.from_numpy(gt_label),
            'domain_token': domain_token,
            'imgname': imgname,
            'vis_target': torch.from_numpy(vis_target),
            'vis_valid': torch.from_numpy(vis_valid),
            'consist_mask': torch.from_numpy(consist_mask),
            'region_id': region_id,
            'pair_id': pair_id,
        }


def collate_occlusion_batch(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ('imgname', 'pair_id'):
            out[k] = [b[k] for b in batch]
        elif k == 'region_id':
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
