#!/usr/bin/env python3
"""Smoke: unit shapes + optional 1-batch forward (requires GPU + weights)."""

import argparse
import os
import sys

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import get_aq_config_variant
from model.attribute_query_decoder import AttributeQueryDecoder, PerQueryHeads
from losses import CLIMPPARLoss


def test_decoder_shapes():
    B, L, D, N = 2, 32, 768, 57
    dec = AttributeQueryDecoder(N, D)
    heads = PerQueryHeads(D, use_visibility=True)
    v = torch.randn(B, L, D)
    t = torch.randn(B, N, D)
    q = dec(v, t)
    assert q.shape == (B, N, D)
    logits, vis = heads(q)
    assert logits.shape == (B, N) and vis.shape == (B, N)
    print('OK decoder shapes')


def test_loss():
    cfg = get_aq_config_variant('C4')
    crit = CLIMPPARLoss(cfg)
    logits = torch.randn(4, 57)
    labels = (torch.rand(4, 57) > 0.5).float()
    vis_t = torch.rand(4, 57)
    vis_v = torch.ones(4, 57)
    features = {
        'vis_pred': torch.sigmoid(torch.randn(4, 57)),
        'image_features': torch.randn(4, 32, 768),
        'full_features': torch.randn(4, 768),
        'diff_features': torch.randn(4, 57, 768),
    }
    loss, parts = crit(
        logits, labels, features,
        vis_target=vis_t, vis_valid_mask=vis_v,
        logits_occ=logits + 0.1, consist_mask=torch.ones(4, 57),
    )
    assert loss.ndim == 0 and 'L_vis' in parts
    print('OK loss C4')


def test_forward_full():
    from model import CLIMPPAR

    cfg = get_aq_config_variant('C3')
    cfg.use_amp = False
    attrs = [f'attr_{i}' for i in range(57)]
    model = CLIMPPAR(cfg, num_attrs=57)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    x = torch.randn(1, 3, cfg.img_height, cfg.img_width, device=device)
    dom = torch.zeros(1, 32, 768, device=device)
    logits, feat = model(x, dom, attrs)
    assert logits.shape == (1, 57)
    print(f'OK full forward on {device}, logits {tuple(logits.shape)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full model forward (heavy)')
    args = parser.parse_args()
    test_decoder_shapes()
    test_loss()
    if args.full:
        test_forward_full()
    else:
        print('Skip full forward (pass --full to run VMamba+Mamba)')
