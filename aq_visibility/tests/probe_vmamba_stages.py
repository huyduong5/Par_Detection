#!/usr/bin/env python3
"""Print output shape of each VMamba-S stage for a 448x448 input (run on GPU host)."""

import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from model.vision_encoder import VMambaVisionEncoder  # noqa: E402


@torch.no_grad()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = VMambaVisionEncoder(variant='small', embed_dim=768, pretrained=False).to(device).eval()
    x = torch.randn(1, 3, 448, 448, device=device)
    h = enc.backbone.patch_embed(x)
    print('patch_embed', tuple(h.shape), 'channel_first=', enc.backbone.channel_first)
    for i, layer in enumerate(enc.backbone.layers):
        h = layer(h)
        shape = tuple(h.shape)
        c = shape[1] if enc.backbone.channel_first else shape[-1]
        hw = shape[2:] if enc.backbone.channel_first else shape[1:3]
        print(f'layers[{i}] out shape={shape} C={c} HW={tuple(hw)} stride={448 // hw[0]}')
    feats, stage = enc(x, return_stage=1)
    print('forward tokens', tuple(feats.shape), 'return_stage=1', tuple(stage.shape))


if __name__ == '__main__':
    main()
