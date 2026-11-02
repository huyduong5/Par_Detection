"""Lazy exports so protocol tests can import region_utils without torch."""

__all__ = [
    'PARDataset',
    'get_clip_transforms',
    'OcclusionPairDataset',
    'collate_occlusion_batch',
    'RegionOcclusionAug',
    'get_c1_transforms',
]


def __getattr__(name):
    if name in ('PARDataset', 'get_clip_transforms'):
        from .clip_dataset import PARDataset, get_clip_transforms
        return {'PARDataset': PARDataset, 'get_clip_transforms': get_clip_transforms}[name]
    if name in ('OcclusionPairDataset', 'collate_occlusion_batch'):
        from .occlusion_dataset import OcclusionPairDataset, collate_occlusion_batch
        return {
            'OcclusionPairDataset': OcclusionPairDataset,
            'collate_occlusion_batch': collate_occlusion_batch,
        }[name]
    if name in ('RegionOcclusionAug', 'get_c1_transforms'):
        from .occ_aug import RegionOcclusionAug, get_c1_transforms
        return {
            'RegionOcclusionAug': RegionOcclusionAug,
            'get_c1_transforms': get_c1_transforms,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
