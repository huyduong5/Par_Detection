#!/usr/bin/env python3
"""GPU check for multi-scale variants (run on sol1).

1. Build C3_MS with real VMamba + Mamba text; forward clean and occluded (B=2).
2. Backbone frozen, ms_tokenizer trainable.
3. Load C3 best_model.pth with strict=False: only ms_tokenizer.* may be missing.
4. One AMP train step at --batch_size, report peak VRAM.
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
sys.path.insert(0, str(PKG))

from config import get_thesis_variant  # noqa: E402
from losses import CLIMPPARLoss  # noqa: E402
from model import CLIMPPAR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='C4b_MS')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--c3_ckpt', default=str(REPO / 'aq_visibility/output/C3/best_model.pth'))
    args = ap.parse_args()
    device = torch.device('cuda')

    cfg = get_thesis_variant(args.variant, output_dir=str(REPO / 'aq_visibility/output/_smoke/check'))
    with open(cfg.pkl_path, 'rb') as f:
        attrs = list(pickle.load(f).attributes)
    model = CLIMPPAR(cfg, num_attrs=len(attrs)).to(device)

    bb_trainable = sum(p.requires_grad for p in model.vision_encoder.backbone.parameters())
    ms_trainable = all(p.requires_grad for p in model.ms_tokenizer.parameters())
    print(f'backbone trainable params: {bb_trainable} (expect 0)')
    print(f'ms_tokenizer trainable: {ms_trainable} (expect True)')
    assert bb_trainable == 0 and ms_trainable

    ck = torch.load(args.c3_ckpt, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck.get('model_state_dict', ck).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    non_ms = [k for k in missing if not k.startswith('ms_tokenizer.')]
    print(f'load C3 -> {args.variant}: missing={len(missing)} non_ms_missing={len(non_ms)} '
          f'unexpected={len(unexpected)}')
    assert not non_ms, non_ms[:10]
    assert not unexpected, unexpected[:10]

    model.eval()
    dom = torch.zeros(2, 32, cfg.embed_dim, device=device)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        x = torch.randn(2, 3, cfg.img_height, cfg.img_width, device=device)
        logits, feats = model(x, dom, attrs)
        logits_o, feats_o = model(x.flip(-1), dom, attrs)
    print('logits', tuple(logits.shape), 'aq tokens', feats.get('ms_num_tokens'),
          'vis', tuple(feats['vis_pred'].shape))
    assert logits.shape == (2, len(attrs)) and feats.get('ms_num_tokens') == 32 + 128

    model.train()
    crit = CLIMPPARLoss(config=cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    # Small init scale: the default 2**16 often overflows on step 1 (step then
    # skipped by design), which would make the finite-gradient check meaningless.
    scaler = torch.amp.GradScaler('cuda', init_scale=2.0 ** 10)
    torch.cuda.reset_peak_memory_stats()
    B = args.batch_size
    x = torch.randn(B, 3, cfg.img_height, cfg.img_width, device=device)
    x_o = torch.randn_like(x)
    dom = torch.zeros(B, 32, cfg.embed_dim, device=device)
    y = (torch.rand(B, len(attrs), device=device) > 0.8).float()
    vt = torch.rand(B, len(attrs), device=device)
    vv = torch.ones(B, len(attrs), device=device)
    with torch.amp.autocast('cuda', enabled=True):
        lg, ft = model(x, dom, attrs)
        lg_o, ft_o = model(x_o, dom, attrs)
        ft = dict(ft)
        ft['vis_pred'] = ft_o['vis_pred']
        loss, parts = crit(lg, y, ft, vis_target=vt, vis_valid_mask=vv,
                           logits_occ=lg_o, consist_mask=vv)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nonfinite = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    scaler.step(opt)
    scaler.update()
    grad_ms = float(model.ms_tokenizer.proj.weight.grad.abs().sum())
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f'train step loss={float(loss):.4f} parts={ {k: round(float(v), 4) for k, v in parts.items()} }')
    print(f'non-finite grads: {len(nonfinite)} {nonfinite[:5]}')
    print(f'ms grad sum={grad_ms:.4f}  peak VRAM batch {B}: alloc {peak:.2f} GiB, reserved {reserved:.2f} GiB')
    assert not nonfinite and grad_ms > 0
    print('CHECK_MS_OK')


if __name__ == '__main__':
    main()
