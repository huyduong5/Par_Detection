# Thesis AQ + visibility configs (C0–C5). Contrastive / CLIP fuse / KD off.

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIMPConfig:
    dataset_name: str = 'MSP60k'
    pkl_path: str = ''
    img_dir: str = ''

    vmamba_variant: str = 'small'
    vmamba_pretrained: bool = True
    vmamba_pretrained_path: Optional[str] = None

    mamba_model: str = 'mamba-130m'
    mamba_freeze: bool = True
    max_text_length: int = 0

    embed_dim: int = 768
    domain_tokens_path: str = ''

    img_height: int = 448
    img_width: int = 448

    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    grad_accum_steps: int = 4

    use_amp: bool = True
    gradient_checkpointing: bool = True
    use_ddp: bool = False

    loss_type: str = 'weighted_bce_asl'
    global_contrastive_weight: float = 0.0
    finegrained_contrastive_weight: float = 0.0
    temperature_init: float = 0.07
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    attr_freq_path: str = ''
    use_fixed_pos_weight: bool = False

    use_attr_queries: bool = True
    use_visibility: bool = False
    use_domain_tokens: bool = False
    use_spatial_cross_attn: bool = True
    aq_num_layers: int = 2
    lambda_v: float = 0.5
    lambda_c: float = 0.0
    vis_uncertain_ignore: bool = True

    unfreeze_vision: bool = False
    unfreeze_vision_after_epoch: int = 0

    # Disabled in thesis matrix (kept for API compat)
    use_clip_vision: bool = False
    clip_model_id: str = 'openai/clip-vit-base-patch32'
    clip_img_size: int = 224
    clip_freeze: bool = True
    clip_lr: float = 1e-5
    vision_fuse_mode: str = 'gate'
    kd_teacher_ckpt: str = ''
    kd_temperature: float = 2.0
    lambda_kd: float = 0.0

    # Optional background ablation (C5)
    use_background: bool = False
    background_ckpt: str = ''

    occlusion_pkl_path: str = ''
    occlusion_img_dir: str = ''
    body_region_map_path: str = ''
    use_occlusion_pairs: bool = False
    consistency_on_unoccluded_only: bool = True
    # 'l1': |σ(clean)−σ(occ)| (C4). 'sg_bce': KL(sg(p_clean) || p_occ) (C4b).
    consistency_type: str = 'l1'

    # Multi-scale: extra tokens from an intermediate VMamba stage → AQ decoder only.
    use_multiscale: bool = False
    ms_stage_index: int = 1
    ms_pool_h: int = 16
    ms_pool_w: int = 8

    # C1: controlled occlusion augmentation on clean base
    use_occ_aug: bool = False
    occ_aug_prob: float = 0.5

    seed: int = 42
    device: str = 'cuda'
    output_dir: str = ''
    log_interval: int = 20
    save_interval: int = 5
    variant_id: str = ''
    # Smoke runs only (0 = full epoch / full val).
    max_train_batches: int = 0
    max_val_batches: int = 0

    def __post_init__(self):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(pkg_dir)

        if not self.pkl_path:
            self.pkl_path = os.path.join(repo_root, 'MSP60k', 'SUBMIT', 'dataset_random.pkl')
        if not self.img_dir:
            self.img_dir = os.path.join(repo_root, 'MSP60k', 'SUBMIT', 'images')
        if not self.domain_tokens_path:
            self.domain_tokens_path = os.path.join(
                repo_root, '_my_model_8', 'domain_tokens', 'domain_tokens.pt')
        if not self.body_region_map_path:
            self.body_region_map_path = os.path.join(
                repo_root, 'data', 'occlusion', 'body_region_map.json')
        if not self.occlusion_pkl_path:
            self.occlusion_pkl_path = os.path.join(
                repo_root, 'data', 'occlusion', 'occlusion_pairs_train.pkl')
        if not self.occlusion_img_dir:
            self.occlusion_img_dir = os.path.join(
                repo_root, 'data', 'occlusion', 'occlusion_images')
        if not self.attr_freq_path:
            self.attr_freq_path = os.path.join(repo_root, 'data', 'attr_freq.json')
        if not self.background_ckpt:
            self.background_ckpt = os.path.join(
                repo_root, '_my_model_8', 'domain_tokens', 'resnet18_background_best.pth')
        if not self.output_dir:
            tag = self.variant_id or 'default'
            self.output_dir = os.path.join(pkg_dir, 'output', tag)
        os.makedirs(self.output_dir, exist_ok=True)


def get_default_config(**kwargs):
    return CLIMPConfig(**kwargs)


def get_kaggle_config(**kwargs):
    return CLIMPConfig(**kwargs)


def get_thesis_variant(variant: str, **kwargs):
    """C0–C5 thesis presets. C0/C1 are meta (base package); C2–C5 use this package."""
    variant = variant.upper().replace('-', '_')
    common = dict(
        use_domain_tokens=False,
        use_spatial_cross_attn=True,
        global_contrastive_weight=0.0,
        finegrained_contrastive_weight=0.0,
        use_clip_vision=False,
        lambda_kd=0.0,
        seed=42,
        epochs=30,
        use_ddp=False,
    )

    if variant in ('C0', 'C0_BASE'):
        # Meta: train via _my_model_8; stored for aggregate tagging.
        common.update(
            variant_id='C0',
            use_attr_queries=False,
            use_visibility=False,
            lambda_v=0.0,
            lambda_c=0.0,
            use_occlusion_pairs=False,
            use_occ_aug=False,
            use_background=False,
        )
    elif variant in ('C1', 'C1_BASE_OCCAUG'):
        common.update(
            variant_id='C1',
            use_attr_queries=False,
            use_visibility=False,
            lambda_v=0.0,
            lambda_c=0.0,
            use_occlusion_pairs=False,
            use_occ_aug=True,
            occ_aug_prob=0.5,
            use_background=False,
        )
    elif variant == 'C2':
        common.update(
            variant_id='C2',
            use_attr_queries=True,
            use_visibility=False,
            lambda_v=0.0,
            lambda_c=0.0,
            use_occlusion_pairs=False,
            use_background=False,
        )
    elif variant == 'C3':
        common.update(
            variant_id='C3',
            use_attr_queries=True,
            use_visibility=True,
            lambda_v=0.5,
            lambda_c=0.0,
            use_occlusion_pairs=True,
            use_background=False,
        )
    elif variant == 'C4':
        common.update(
            variant_id='C4',
            use_attr_queries=True,
            use_visibility=True,
            lambda_v=0.5,
            lambda_c=0.2,
            use_occlusion_pairs=True,
            use_background=False,
        )
    elif variant == 'C4_REAUDIT':
        common.update(
            variant_id='C4_reaudit',
            use_attr_queries=True,
            use_visibility=True,
            lambda_v=0.5,
            lambda_c=0.2,
            use_occlusion_pairs=True,
            use_background=False,
        )
    elif variant in ('C4B', 'C3_MS', 'C4B_MS'):
        sg = variant in ('C4B', 'C4B_MS')
        common.update(
            variant_id={'C4B': 'C4b', 'C3_MS': 'C3_MS', 'C4B_MS': 'C4b_MS'}[variant],
            use_attr_queries=True,
            use_visibility=True,
            lambda_v=0.5,
            lambda_c=0.2 if sg else 0.0,
            consistency_type='sg_bce' if sg else 'l1',
            use_occlusion_pairs=True,
            use_multiscale=variant.endswith('_MS'),
            use_background=False,
        )
    elif variant == 'C4B_MS_FPW':
        # C4b_MS with a dataset-level pos_weight instead of the per-batch one.
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        common.update(
            variant_id='C4b_MS_fpw',
            use_attr_queries=True,
            use_visibility=True,
            lambda_v=0.5,
            lambda_c=0.2,
            consistency_type='sg_bce',
            use_occlusion_pairs=True,
            use_multiscale=True,
            use_background=False,
            use_fixed_pos_weight=True,
            attr_freq_path=os.path.join(os.path.dirname(pkg_dir), 'data', 'attr_pos_weight_sqrt.json'),
        )
    elif variant in ('C5', 'C5_BG'):
        common.update(
            variant_id='C5',
            use_attr_queries=True,
            use_visibility=True,
            lambda_v=0.5,
            lambda_c=0.2,
            use_occlusion_pairs=True,
            use_background=True,
        )
    else:
        raise ValueError(
            f'Unknown thesis variant {variant}; expected C0–C5, C4_reaudit, '
            'C4b, C3_MS, C4b_MS')

    common.update(kwargs)
    return CLIMPConfig(**common)


# Back-compat alias used by train.py
def get_aq_config_variant(variant: str, **kwargs):
    return get_thesis_variant(variant, **kwargs)
