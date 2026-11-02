"""Wrap the baseline without modifying its state or classification output."""
import torch
from torch import nn


class EOPAR(nn.Module):
    def __init__(self, baseline, embed_dim=768, enabled=True):
        super().__init__()
        self.baseline = baseline
        self.enabled = enabled
        self.observability_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Linear(embed_dim // 2, 1),
        ) if enabled else None
        # Query logits never consume z_out/t_out from this legacy block. Keep its
        # computation for parity verification, but exclude disconnected weights.
        if getattr(baseline, 'use_attr_queries', False):
            for p in baseline.cross_mamba.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Frozen stochastic backbones must not randomize the evidence teacher.
        for encoder_name in ('vision_encoder', 'text_encoder'):
            encoder = getattr(self.baseline, encoder_name, None)
            backbone = getattr(encoder, 'backbone', None)
            if backbone is not None and not any(p.requires_grad for p in backbone.parameters()):
                backbone.eval()
        return self

    def forward(self, images, domain_tokens, attribute_names):
        logits, features = self.baseline(images, domain_tokens, attribute_names)
        obs = self.observability_head(features['query_feats']).squeeze(-1) if self.enabled else None
        return {'attr_logits': logits, 'observability_logits': obs,
                'legacy_visibility': features.get('vis_pred')}

    @torch.no_grad()
    def predict(self, images, domain_tokens, attribute_names):
        if self.training:
            raise RuntimeError('Call eval() before predict()')
        out = self(images, domain_tokens, attribute_names)
        return {'attribute_probability': out['attr_logits'].sigmoid(),
                'observability_probability': (out['observability_logits'].sigmoid()
                    if out['observability_logits'] is not None else None)}


def load_baseline(baseline, checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    baseline.load_state_dict(payload.get('model_state_dict', payload), strict=True)
