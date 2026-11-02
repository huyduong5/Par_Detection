"""CLIP vision tokens for fusion with VMamba (Phase 3 / OpenPAR-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPVisionAdapter(nn.Module):
    def __init__(self, model_id='openai/clip-vit-base-patch32', embed_dim=768, freeze=True):
        super().__init__()
        from transformers import CLIPModel

        clip = CLIPModel.from_pretrained(model_id)
        self.vision = clip.vision_model
        self.visual_projection = clip.visual_projection
        self.in_dim = clip.config.projection_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, embed_dim),
        )
        self.image_size = int(getattr(clip.config.vision_config, 'image_size', 224))
        # CLIP mean/std for on-the-fly renormalize from ImageNet/CLIP-pad tensors is hard;
        # we expect caller to pass images already CLIP-normalized when possible.
        # Fallback: interpolate spatial size only.
        if freeze:
            for p in self.vision.parameters():
                p.requires_grad = False
            for p in self.visual_projection.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        images: (B,3,H,W) — will be resized to CLIP native size.
        Returns token sequence (B, N, embed_dim) including CLS as first token projected.
        Runs CLIP vision in fp32 to avoid AMP overflow / NaNs.
        """
        # Cast under autocast can push CLIP ViT into NaN; force fp32 here.
        with torch.amp.autocast('cuda', enabled=False):
            x = images.float()
            if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
                x = F.interpolate(
                    x, size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False,
                )
            out = self.vision(pixel_values=x)
            tokens = out.last_hidden_state
            tokens_proj = self.visual_projection(tokens)
            return self.proj(tokens_proj)


class VisionTokenFusion(nn.Module):
    """Fuse VMamba tokens with CLIP tokens to a common length for AQ."""

    def __init__(self, embed_dim=768, mode='gate'):
        super().__init__()
        self.mode = mode
        if mode == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.clip_pool = nn.AdaptiveAvgPool1d(32)  # match typical VMamba L=32
            self.vm_proj = nn.Identity()
        elif mode == 'concat':
            self.proj = nn.Linear(embed_dim * 2, embed_dim)
            self.clip_pool = nn.AdaptiveAvgPool1d(32)
        else:
            raise ValueError(mode)

    def forward(self, vmamba_tokens, clip_tokens):
        # vmamba: (B, Lv, D), clip: (B, Lc, D)
        clip_t = clip_tokens.transpose(1, 2)  # (B, D, Lc)
        clip_aligned = self.clip_pool(clip_t).transpose(1, 2)  # (B, 32, D)
        # align vmamba length
        if vmamba_tokens.shape[1] != clip_aligned.shape[1]:
            vm = F.adaptive_avg_pool1d(vmamba_tokens.transpose(1, 2), clip_aligned.shape[1]).transpose(1, 2)
        else:
            vm = vmamba_tokens
        if self.mode == 'gate':
            g = self.gate(torch.cat([vm, clip_aligned], dim=-1))
            # Bias toward VMamba early (stable init); clamp for AMP safety
            out = g * vm + (1.0 - g) * clip_aligned
            return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        out = self.proj(torch.cat([vm, clip_aligned], dim=-1))
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
