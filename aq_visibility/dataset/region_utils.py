import json
import os
from typing import Dict, List, Tuple

REGION_IDS = {'head': 0, 'upper': 1, 'lower': 2, 'shoes': 3, 'global': 4, 'none': 4}
EVIDENCE_REGIONS = ('head', 'upper', 'lower', 'shoes')


def load_body_region_map(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def attrs_for_region(region_map: Dict[str, str], region: str) -> List[str]:
    return [a for a, r in region_map.items() if r == region]


def protocol_consist_mask(attributes: List[str], region_map: Dict[str, str],
                          occluded_region: str, bbox_ok: bool = True):
    """1 only for evidence-region attrs whose region is intact.

    Global / non-evidence attrs stay 0. Occluded evidence region stays 0.
    Uncertain boxes (bbox_ok=False) stay all 0 so they drop out of L_consist.
    """
    import numpy as np
    mask = np.zeros(len(attributes), dtype=np.float32)
    if not bbox_ok or occluded_region not in EVIDENCE_REGIONS:
        return mask
    for i, attr in enumerate(attributes):
        reg = region_map.get(attr, 'global')
        if reg in EVIDENCE_REGIONS and reg != occluded_region:
            mask[i] = 1.0
    return mask


def build_attr_index_masks(attributes: List[str], region_map: Dict[str, str]):
    """Per-region boolean masks length len(attributes)."""
    masks = {reg: [] for reg in EVIDENCE_REGIONS}
    for attr in attributes:
        reg = region_map.get(attr, 'global')
        for r in EVIDENCE_REGIONS:
            masks[r].append(reg == r)
    import numpy as np
    return {k: np.array(v, dtype=bool) for k, v in masks.items()}


def vertical_region_box(w: int, h: int, region: str) -> Tuple[int, int, int, int]:
    """Heuristic person-centric vertical bands (x0,y0,x1,y1) in pixel coords."""
    margins = {
        'head': (0.0, 0.0, 1.0, 0.18),
        'upper': (0.0, 0.18, 1.0, 0.55),
        'lower': (0.0, 0.55, 1.0, 0.85),
        'shoes': (0.0, 0.85, 1.0, 1.0),
    }
    if region not in margins:
        return 0, 0, w, h
    x0r, y0r, x1r, y1r = margins[region]
    return int(x0r * w), int(y0r * h), int(x1r * w), int(y1r * h)


def validate_region_map(attributes: List[str], region_map_path: str) -> None:
    region_map = load_body_region_map(region_map_path)
    missing = [a for a in attributes if a not in region_map]
    orphans = [k for k in region_map.keys() if k not in attributes]
    if missing:
        raise ValueError(f"Attributes missing from region map: {missing}")
    if orphans:
        raise ValueError(f"Orphan keys in region map: {orphans}")
