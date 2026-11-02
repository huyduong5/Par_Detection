# losses.py — v9: L_attr + λ_v L_vis + λ_c L_consist (+ optional KD)

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        return -(loss * one_sided_w).mean()


class CLIMPPARLoss(nn.Module):
    """Classification + optional contrastive (v8) + visibility + consistency (v9) + KD."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        if config is not None:
            self.loss_type = getattr(config, 'loss_type', 'weighted_bce_asl')
            self.alpha = getattr(config, 'global_contrastive_weight', 0.1)
            self.beta = getattr(config, 'finegrained_contrastive_weight', 0.1)
            init_temp = getattr(config, 'temperature_init', 0.07)
            self.lambda_v = getattr(config, 'lambda_v', 0.0)
            self.lambda_c = getattr(config, 'lambda_c', 0.0)
            self.use_visibility = getattr(config, 'use_visibility', False)
            self.lambda_kd = getattr(config, 'lambda_kd', 0.0)
            self.kd_temperature = getattr(config, 'kd_temperature', 2.0)
            self.consistency_on_unoccluded_only = getattr(
                config, 'consistency_on_unoccluded_only', True)
            self.consistency_type = getattr(config, 'consistency_type', 'l1')
            gamma_neg = getattr(config, 'asl_gamma_neg', 4.0)
            gamma_pos = getattr(config, 'asl_gamma_pos', 1.0)
            asl_clip = getattr(config, 'asl_clip', 0.05)
            self.use_fixed_pos_weight = getattr(config, 'use_fixed_pos_weight', False)
            freq_path = getattr(config, 'attr_freq_path', '')
        else:
            self.loss_type = 'weighted_bce_asl'
            self.alpha = 0.1
            self.beta = 0.1
            init_temp = 0.07
            self.lambda_v = 0.0
            self.lambda_c = 0.0
            self.use_visibility = False
            self.lambda_kd = 0.0
            self.kd_temperature = 2.0
            self.consistency_on_unoccluded_only = True
            self.consistency_type = 'l1'
            gamma_neg, gamma_pos, asl_clip = 4.0, 1.0, 0.05
            self.use_fixed_pos_weight = False
            freq_path = ''

        if self.loss_type in ('asl', 'weighted_bce_asl'):
            self.asl_loss = AsymmetricLoss(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=asl_clip
            )

        if self.use_fixed_pos_weight and not (freq_path and os.path.exists(freq_path)):
            raise FileNotFoundError(f'use_fixed_pos_weight=True but no pos_weight file: {freq_path!r}')
        if self.use_fixed_pos_weight:
            with open(freq_path) as f:
                payload = json.load(f)
            pw = torch.tensor(payload['pos_weight'], dtype=torch.float32)
            self.register_buffer('fixed_pos_weight', pw, persistent=False)
        else:
            self.fixed_pos_weight = None

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))

    def forward(
        self,
        logits,
        labels,
        features=None,
        vis_target=None,
        vis_valid_mask=None,
        logits_occ=None,
        consist_mask=None,
        teacher_logits=None,
    ):
        if self.loss_type == 'bce':
            base_loss = F.binary_cross_entropy_with_logits(logits, labels)
        elif self.loss_type == 'asl':
            base_loss = self.asl_loss(logits, labels)
        elif self.loss_type == 'weighted_bce_asl':
            base_loss = self.weighted_bce_loss(logits, labels) + self.asl_loss(logits, labels)
        else:
            base_loss = self.weighted_bce_loss(logits, labels)

        total = base_loss
        log_parts = {'L_attr': base_loss.detach()}

        if features is not None and (self.alpha > 0 or self.beta > 0):
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
            if self.alpha > 0:
                g = self.global_contrastive_loss(
                    features['image_features'], features['full_features'], logit_scale
                )
                total = total + self.alpha * g
                log_parts['L_global'] = g.detach()
            if self.beta > 0:
                fg = self.finegrained_contrastive_loss(
                    features['image_features'], features['diff_features'], labels, logit_scale
                )
                total = total + self.beta * fg
                log_parts['L_fine'] = fg.detach()

        if self.use_visibility and self.lambda_v > 0 and features is not None:
            vis_pred = features.get('vis_pred')
            if vis_pred is not None and vis_target is not None and vis_valid_mask is not None:
                vloss = self.visibility_loss(vis_pred, vis_target, vis_valid_mask)
                total = total + self.lambda_v * vloss
                log_parts['L_vis'] = vloss.detach()

        if self.lambda_c > 0 and logits_occ is not None and consist_mask is not None:
            # Protocol: consist_mask already marks intact evidence-region attrs.
            mask = consist_mask
            if self.consistency_type == 'sg_bce':
                closs = self.consistency_loss_sg(logits, logits_occ, mask)
            elif self.consistency_type == 'l1':
                closs = self.consistency_loss(logits, logits_occ, mask)
            else:
                raise ValueError(f'Unknown consistency_type {self.consistency_type!r}')
            total = total + self.lambda_c * closs
            log_parts['L_consist'] = closs.detach()

        if self.lambda_kd > 0 and teacher_logits is not None:
            kd = self.kd_loss(logits, teacher_logits, self.kd_temperature)
            total = total + self.lambda_kd * kd
            log_parts['L_kd'] = kd.detach()

        log_parts['_total'] = total
        return total, log_parts

    def weighted_bce_loss(self, logits, labels):
        if self.fixed_pos_weight is not None:
            pos_weight = self.fixed_pos_weight.to(device=logits.device, dtype=logits.dtype)
        else:
            with torch.no_grad():
                pos_count = labels.sum(dim=0).clamp(min=1)
                neg_count = (labels.shape[0] - pos_count).clamp(min=1)
                pos_weight = (neg_count / pos_count).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    @staticmethod
    def kd_loss(student_logits, teacher_logits, temperature=2.0):
        t = max(float(temperature), 1e-6)
        # Soft-target BCE is unsafe under AMP/fp16 — run in fp32 with logits.
        with torch.amp.autocast('cuda', enabled=False):
            s_logits = (student_logits / t).float()
            with torch.no_grad():
                soft = torch.sigmoid((teacher_logits / t).float())
            return F.binary_cross_entropy_with_logits(s_logits, soft) * (t * t)

    @staticmethod
    def visibility_loss(vis_pred, vis_target, vis_valid_mask):
        mask = vis_valid_mask.float()
        if mask.sum() < 1:
            return vis_pred.sum() * 0.0
        # BCE on probabilities is unsafe under AMP/fp16 — run in fp32.
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(
                vis_pred.float(), vis_target.float(), reduction='none'
            )
        return (bce * mask.float()).sum() / mask.sum()

    @staticmethod
    def consistency_loss(logits_clean, logits_occ, consist_mask):
        """L1 on sigmoid probs where evidence should be stable."""
        mask = consist_mask.float()
        if mask.sum() < 1:
            return logits_clean.sum() * 0.0
        p_clean = torch.sigmoid(logits_clean)
        p_occ = torch.sigmoid(logits_occ)
        diff = (p_clean - p_occ).abs()
        return (diff * mask).sum() / mask.sum()

    @staticmethod
    def consistency_loss_sg(logits_clean, logits_occ, consist_mask):
        """KL(Bern(sg(p_clean)) || Bern(p_occ)) on intact evidence attrs.

        Clean is the teacher (no gradient). Computed as BCE minus target entropy,
        so the gradient equals soft-target BCE and the value is 0 when p_occ == p_clean.
        """
        mask = consist_mask.float()
        if mask.sum() < 1:
            return logits_occ.sum() * 0.0
        with torch.amp.autocast('cuda', enabled=False):
            z_occ = logits_occ.float()
            with torch.no_grad():
                z_clean = logits_clean.detach().float()
                target = torch.sigmoid(z_clean)
                entropy = F.binary_cross_entropy_with_logits(
                    z_clean, target, reduction='none')
            bce = F.binary_cross_entropy_with_logits(z_occ, target, reduction='none')
            kl = bce - entropy
            return (kl * mask).sum() / mask.sum()

    def global_contrastive_loss(self, image_features, text_features, logit_scale):
        B = image_features.shape[0]
        device = image_features.device
        image_global = F.normalize(image_features.mean(dim=1), dim=-1)
        text_global = F.normalize(text_features, dim=-1)
        logits_per_image = logit_scale * image_global @ text_global.T
        labels = torch.arange(B, dtype=torch.long, device=device)
        return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_image.T, labels)) / 2

    def finegrained_contrastive_loss(self, image_features, attribute_features, labels, logit_scale):
        img_feat_norm = F.normalize(image_features, dim=-1)
        attr_feat_norm = F.normalize(attribute_features, dim=-1)
        sim_matrix = torch.bmm(img_feat_norm, attr_feat_norm.transpose(1, 2)) * logit_scale
        max_sim, _ = torch.max(sim_matrix, dim=1)
        with torch.no_grad():
            pos_count = labels.sum(dim=0).clamp(min=1)
            neg_count = (labels.shape[0] - pos_count).clamp(min=1)
            pos_weight = neg_count / pos_count
        return F.binary_cross_entropy_with_logits(max_sim, labels, pos_weight=pos_weight)
