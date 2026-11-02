"""Multi-scale tokens from an intermediate VMamba stage for the attribute-query decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def vmamba_stage_channels(base_dim: int, stage_index: int, num_stages: int = 4) -> int:
    """Channels of ``backbone.layers[stage_index]`` output (v1 downsample sits at stage end)."""
    return base_dim * 2 ** min(stage_index + 1, num_stages - 1)


class MultiScaleTokenizer(nn.Module):
    """Pool a (B, C, H, W) stage map to ms_h*ms_w tokens, project to d_model.

    ``fuse`` concatenates them after the main 32 tokens. Each branch gets a
    learnable scale embedding so the decoder can tell the two resolutions apart.
    """

    def __init__(self, in_channels, d_model, pool_hw=(16, 8)):
        super().__init__()
        self.pool_hw = tuple(pool_hw)
        self.norm = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(in_channels, d_model)
        self.scale_emb_main = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale_emb_ms = nn.Parameter(torch.randn(d_model) * 0.02)

    @property
    def num_tokens(self):
        return self.pool_hw[0] * self.pool_hw[1]

    def tokens(self, stage_feat):
        x = F.adaptive_avg_pool2d(stage_feat, self.pool_hw)       # (B, C, h, w)
        x = x.flatten(2).transpose(1, 2)                           # (B, h*w, C)
        return self.proj(self.norm(x)) + self.scale_emb_ms

    def fuse(self, main_tokens, stage_feat):
        ms = self.tokens(stage_feat.to(main_tokens.dtype))
        return torch.cat([main_tokens + self.scale_emb_main, ms], dim=1)
