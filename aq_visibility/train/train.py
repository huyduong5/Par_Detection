# train.py
# Script huấn luyện CLIMP-PAR v9 (AQ + visibility + consistency)
# Pipeline: 3-branch (Text + Vision + Cross-Modal Mamba) với Spatial Cross-Attention
# Hỗ trợ: DDP, AMP (fp16), Gradient Accumulation
# v8.1 fixes: scheduler step-level, eval DDP gather, deprecated autocast

import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

# Package root = aq_visibility/; repo root = parent of package
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TRAIN_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, REPO_ROOT)
# Aliases used by KD / legacy paths
PROJECT_DIR = PKG_DIR
ROOT_DIR = REPO_ROOT

from config import CLIMPConfig, get_kaggle_config, get_aq_config_variant
from model import CLIMPPAR
from losses import CLIMPPARLoss
from dataset.clip_dataset import PARDataset, get_clip_transforms
from dataset.occlusion_dataset import OcclusionPairDataset, collate_occlusion_batch
from metrics.metrics_aq import compute_metrics, visibility_metrics, consistency_gap


# ============================================================================
# Setup
# ============================================================================
def set_seed(seed):
    """Đặt random seed cho tính reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp(rank, world_size):
    """Khởi tạo process group cho DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Dọn dẹp DDP."""
    dist.destroy_process_group()


def is_main_process(rank, use_ddp):
    """Kiểm tra có phải process chính không."""
    return (not use_ddp) or (rank == 0)


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config,
                    epoch, device, attribute_names, scheduler=None, teacher=None):
    """
    Huấn luyện 1 epoch.
    
    v8.1: scheduler.step() được gọi sau mỗi optimizer step (không phải mỗi epoch).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    rank = device if config.use_ddp else 0
    optimizer.zero_grad()
    
    # Forward occluded image whenever visibility or consistency needs it (C3/C4).
    use_occ = getattr(config, 'use_occlusion_pairs', False) and (
        getattr(config, 'use_visibility', False) or getattr(config, 'lambda_c', 0) > 0
    )

    max_batches = int(getattr(config, 'max_train_batches', 0) or 0)
    cons_tag = getattr(config, 'consistency_type', 'l1')

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        if isinstance(batch, dict):
            images = batch['img_clean'].to(device)
            labels = batch['labels'].to(device)
            domain_tokens = batch['domain_token'].to(device)
            vis_target = batch['vis_target'].to(device)
            vis_valid = batch['vis_valid'].to(device)
            consist_mask = batch['consist_mask'].to(device)
            images_occ = batch['img_occ'].to(device) if use_occ else None
        else:
            images, labels, _, domain_tokens = batch
            images = images.to(device)
            labels = labels.to(device)
            domain_tokens = domain_tokens.to(device)
            vis_target = vis_valid = consist_mask = images_occ = None

        with autocast('cuda', enabled=config.use_amp):
            actual_model = model.module if config.use_ddp else model
            logits, features = actual_model(images, domain_tokens, attribute_names)
            logits_occ = None
            if images_occ is not None and (use_occ or config.use_visibility):
                logits_occ, features_occ = actual_model(images_occ, domain_tokens, attribute_names)
                if config.use_visibility and 'vis_pred' in features_occ:
                    features = dict(features)
                    features['vis_pred'] = features_occ['vis_pred']
            elif use_occ and images_occ is not None:
                logits_occ, _ = actual_model(images_occ, domain_tokens, attribute_names)

            teacher_logits = None
            if teacher is not None and getattr(config, 'lambda_kd', 0) > 0:
                with torch.no_grad():
                    teacher_logits = teacher(images)

            loss, log_parts = criterion(
                logits,
                labels,
                features,
                vis_target=vis_target,
                vis_valid_mask=vis_valid,
                logits_occ=logits_occ,
                consist_mask=consist_mask,
                teacher_logits=teacher_logits,
            )
            loss = loss / config.grad_accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # [FIX] Scheduler step sau mỗi optimizer step (không phải mỗi epoch)
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * config.grad_accum_steps
        num_batches += 1
        
        # Log
        if is_main_process(rank, config.use_ddp) and (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            extra = ''
            if 'L_vis' in log_parts:
                extra += f" | L_vis={log_parts['L_vis']:.4f}"
            if 'L_consist' in log_parts:
                extra += f" | L_consist[{cons_tag}]={log_parts['L_consist']:.4f}"
            if 'L_kd' in log_parts:
                extra += f" | L_kd={log_parts['L_kd']:.4f}"
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f}{extra} | LR: {optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)
    
    # Flush remaining gradients nếu batch cuối không chia hết cho grad_accum_steps
    if num_batches % config.grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, config, device, attribute_names, rank=0):
    """
    Đánh giá trên tập validation/test.
    Also logs visibility / consistency when occlusion pairs are available.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_vis_pred, all_vis_tgt, all_vis_valid = [], [], []
    all_p_clean, all_p_occ, all_consist = [], [], []
    use_occ = getattr(config, 'use_occlusion_pairs', False) and (
        getattr(config, 'use_visibility', False) or getattr(config, 'lambda_c', 0) > 0
    )
    max_val = int(getattr(config, 'max_val_batches', 0) or 0)
    val_batches = 0

    for val_idx, batch in enumerate(dataloader):
        if max_val and val_idx >= max_val:
            break
        if isinstance(batch, dict):
            images = batch['img_clean'].to(device)
            labels = batch['labels'].to(device)
            domain_tokens = batch['domain_token'].to(device)
            vis_target = batch['vis_target'].to(device)
            vis_valid = batch['vis_valid'].to(device)
            consist_mask = batch['consist_mask'].to(device)
            images_occ = batch['img_occ'].to(device) if use_occ else None
        else:
            images, labels, _, domain_tokens = batch
            images = images.to(device)
            labels = labels.to(device)
            domain_tokens = domain_tokens.to(device)
            vis_target = vis_valid = consist_mask = images_occ = None

        with autocast('cuda', enabled=config.use_amp):
            actual_model = model.module if config.use_ddp else model
            logits, features = actual_model(images, domain_tokens, attribute_names)
            logits_occ = None
            if images_occ is not None:
                logits_occ, features_occ = actual_model(images_occ, domain_tokens, attribute_names)
                if config.use_visibility and 'vis_pred' in features_occ:
                    features = dict(features)
                    features['vis_pred'] = features_occ['vis_pred']
            loss, _ = criterion(
                logits, labels, features,
                vis_target=vis_target, vis_valid_mask=vis_valid,
                logits_occ=logits_occ, consist_mask=consist_mask,
            )

        total_loss += loss.item()
        val_batches += 1
        probs = torch.sigmoid(logits)
        all_preds.append(probs)
        all_labels.append(labels)
        if features.get('vis_pred') is not None and vis_target is not None:
            all_vis_pred.append(features['vis_pred'].float().cpu())
            all_vis_tgt.append(vis_target.float().cpu())
            all_vis_valid.append(vis_valid.float().cpu())
        if logits_occ is not None and consist_mask is not None:
            all_p_clean.append(probs.float().cpu())
            all_p_occ.append(torch.sigmoid(logits_occ).float().cpu())
            all_consist.append(consist_mask.float().cpu())
    
    # Concat local predictions
    all_preds = torch.cat(all_preds, dim=0)   # (N_local, 57)
    all_labels = torch.cat(all_labels, dim=0) # (N_local, 57)
    
    # [FIX] Gather từ tất cả ranks khi dùng DDP
    if config.use_ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        
        # Gather sizes trước (mỗi rank có thể có số sample khác nhau)
        local_size = torch.tensor([all_preds.shape[0]], device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)
        
        # Pad để all_gather đồng nhất shape
        if all_preds.shape[0] < max_size:
            pad_size = max_size - all_preds.shape[0]
            all_preds = torch.cat([all_preds, torch.zeros(pad_size, all_preds.shape[1], device=device)])
            all_labels = torch.cat([all_labels, torch.zeros(pad_size, all_labels.shape[1], device=device)])
        
        # Gather
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
        dist.all_gather(gathered_preds, all_preds)
        dist.all_gather(gathered_labels, all_labels)
        
        # Trim padding và concat
        final_preds = []
        final_labels = []
        for i in range(world_size):
            actual_size = all_sizes[i].item()
            final_preds.append(gathered_preds[i][:actual_size])
            final_labels.append(gathered_labels[i][:actual_size])
        
        all_preds = torch.cat(final_preds, dim=0)
        all_labels = torch.cat(final_labels, dim=0)
        
        # Tính loss trung bình qua các ranks
        total_loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item() / world_size
    
    # Chuyển sang numpy
    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    avg_loss = total_loss / max(val_batches, 1)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    if all_vis_pred:
        metrics.update(visibility_metrics(
            torch.cat(all_vis_pred).numpy(),
            torch.cat(all_vis_tgt).numpy(),
            torch.cat(all_vis_valid).numpy(),
        ))
    if all_p_occ:
        metrics.update(consistency_gap(
            torch.cat(all_p_clean).numpy(),
            torch.cat(all_p_occ).numpy(),
            torch.cat(all_consist).numpy(),
        ))
    return metrics


# ============================================================================
# Main
# ============================================================================
def main(rank=0, world_size=1, config=None):
    """Main training function."""
    
    if config is None:
        config = get_kaggle_config()
    
    # DDP setup
    if config.use_ddp and world_size > 1:
        setup_ddp(rank, world_size)
        device = rank
    else:
        config.use_ddp = False
        device = config.device
    
    set_seed(config.seed + rank)
    
    main_proc = is_main_process(rank, config.use_ddp)
    if main_proc:
        print("=" * 60)
        print("CLIMP-PAR v9 Training (Attribute Queries + Visibility)")
        print(f"  use_attr_queries={config.use_attr_queries} use_visibility={config.use_visibility}")
        print(f"  lambda_v={config.lambda_v} lambda_c={config.lambda_c} "
              f"consistency_type={getattr(config, 'consistency_type', 'l1')}")
        print(f"  use_occlusion_pairs={config.use_occlusion_pairs} seed={config.seed}")
        print(f"  use_multiscale={getattr(config, 'use_multiscale', False)} "
              f"ms_stage_index={getattr(config, 'ms_stage_index', -1)} "
              f"ms_pool={getattr(config, 'ms_pool_h', 0)}x{getattr(config, 'ms_pool_w', 0)}")
        print(f"  Device: {'DDP' if config.use_ddp else 'Single GPU'} "
              f"(world_size={world_size})")
        print(f"  VMamba Vision: {config.vmamba_variant} "
              f"(unfreeze={getattr(config, 'unfreeze_vision', False)})")
        print(f"  CLIP fuse: {getattr(config, 'use_clip_vision', False)} "
              f"AQ layers={getattr(config, 'aq_num_layers', 2)}")
        print(f"  Mamba Text: {config.mamba_model} (backbone frozen)")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Image size: {config.img_height}x{config.img_width}")
        print(f"  Batch size (per GPU): {config.batch_size}")
        print(f"  Grad accum: {config.grad_accum_steps}")
        print(f"  Effective batch: {config.batch_size * world_size * config.grad_accum_steps}")
        print(f"  AMP: {config.use_amp} (fp16)")
        print(f"  Epochs: {config.epochs}")
        print(f"  Loss: {config.loss_type} contrastive=({config.global_contrastive_weight},"
              f"{config.finegrained_contrastive_weight}) kd={getattr(config, 'lambda_kd', 0)}")
        print(f"  fixed_pos_weight={getattr(config, 'use_fixed_pos_weight', False)} "
              f"({getattr(config, 'attr_freq_path', '')})")
        print(f"  Domain tokens: {config.domain_tokens_path}")
        print("=" * 60)
    
    # ========================
    # 1. Dataset & DataLoader
    # ========================
    train_transform, val_transform = get_clip_transforms(config.img_height, config.img_width)
    
    domain_path = config.domain_tokens_path if config.use_domain_tokens else None
    DatasetCls = OcclusionPairDataset if config.use_occlusion_pairs else PARDataset
    ds_kw = dict(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        transform=train_transform,
        domain_tokens_path=domain_path,
    )
    if config.use_occlusion_pairs:
        occ_train = config.occlusion_pkl_path.replace('_train.pkl', '_train.pkl')
        if not os.path.exists(occ_train):
            occ_train = config.occlusion_pkl_path
        train_dataset = DatasetCls(
            **ds_kw,
            split='train',
            occlusion_pkl_path=occ_train,
            occ_img_dir=config.occlusion_img_dir,
        )
        val_occ = occ_train.replace('_train.pkl', '_val.pkl')
        val_dataset = DatasetCls(
            pkl_path=config.pkl_path,
            img_dir=config.img_dir,
            split='val',
            transform=val_transform,
            domain_tokens_path=domain_path,
            occlusion_pkl_path=val_occ if os.path.exists(val_occ) else occ_train,
            occ_img_dir=config.occlusion_img_dir,
        )
    else:
        train_dataset = PARDataset(**ds_kw, split='train')
        val_dataset = PARDataset(
            pkl_path=config.pkl_path,
            img_dir=config.img_dir,
            split='val',
            transform=val_transform,
            domain_tokens_path=domain_path,
        )
    
    # Lấy danh sách attribute names
    attribute_names = train_dataset.attributes
    
    if main_proc:
        print(f"\nDataset: {config.dataset_name}")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Attributes ({len(attribute_names)}): {attribute_names[:5]}...")
    
    # Sampler cho DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if config.use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
        if config.use_ddp else None
    
    collate_fn = collate_occlusion_batch if config.use_occlusion_pairs else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # ========================
    # 2. Model
    # ========================
    model = CLIMPPAR(config, num_attrs=len(attribute_names))
    init_from = getattr(config, '_init_from', '') or ''
    if init_from and os.path.exists(init_from):
        ckpt = torch.load(init_from, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if main_proc:
            print(f'Init from {init_from} missing={len(missing)} unexpected={len(unexpected)}')
    model = model.to(device)
    
    if config.use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if main_proc:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"\nModel parameters:")
        print(f"  Total:     {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M")
        print(f"  Frozen:    {frozen_params / 1e6:.1f}M")
        
        # Chi tiết các module có thể học
        print(f"\n  Trainable modules:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.numel() / 1e3:.1f}K")
    
    # ========================
    # 3. Loss, Optimizer, Scheduler
    # ========================
    criterion = CLIMPPARLoss(config=config).to(device)

    # KD / DeepMAR teacher removed from thesis package (no _my_model_baseline).
    teacher = None
    if getattr(config, 'lambda_kd', 0) > 0 and main_proc:
        print('WARNING: lambda_kd>0 but KD teacher is not available in aq_visibility; ignoring.')

    # Param groups: lower LR for CLIP adapter if present
    actual = model.module if config.use_ddp else model
    clip_params, other_params = [], []
    for name, p in actual.named_parameters():
        if not p.requires_grad:
            continue
        if 'clip_adapter' in name:
            clip_params.append(p)
        else:
            other_params.append(p)
    param_groups = [{'params': other_params, 'lr': config.lr}]
    if clip_params:
        param_groups.append({
            'params': clip_params,
            'lr': getattr(config, 'clip_lr', 1e-5),
        })
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    # [FIX] Cosine annealing with warmup — tính theo optimizer steps
    total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
    warmup_steps = len(train_loader) * config.warmup_epochs // config.grad_accum_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP GradScaler (fp16 cho T4)
    scaler = GradScaler('cuda', enabled=config.use_amp)
    
    # ========================
    # 4. Resume from checkpoint (nếu có)
    # ========================
    start_epoch = 0
    best_mA = 0.0
    resume_path = os.path.join(config.output_dir, 'latest_checkpoint.pth')
    if os.path.exists(resume_path):
        if main_proc:
            print(f"\nResuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        save_model = model.module if config.use_ddp else model
        save_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_mA = ckpt.get('best_mA', 0.0)
        if main_proc:
            print(f"  Resumed at epoch {start_epoch}, best mA: {best_mA:.4f}")
    
    # ========================
    # 5. Training Loop
    # ========================
    if main_proc:
        print(f"\nBắt đầu huấn luyện từ epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, config.epochs):
        if config.use_ddp:
            train_sampler.set_epoch(epoch)

        # Delayed unfreeze of VMamba backbone
        thaw_ep = int(getattr(config, 'unfreeze_vision_after_epoch', 0))
        if getattr(config, 'unfreeze_vision', False) and epoch == thaw_ep:
            actual = model.module if config.use_ddp else model
            actual.set_vision_trainable(True)
            # rebuild optimizer param groups with newly thawed params
            clip_params, other_params = [], []
            for name, p in actual.named_parameters():
                if not p.requires_grad:
                    continue
                if 'clip_adapter' in name:
                    clip_params.append(p)
                else:
                    other_params.append(p)
            groups = [{'params': other_params, 'lr': config.lr}]
            if clip_params:
                groups.append({'params': clip_params, 'lr': getattr(config, 'clip_lr', 1e-5)})
            optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
            if main_proc:
                print(f'  Unfroze VMamba backbone at epoch {epoch+1}', flush=True)
        
        start_time = time.time()
        
        # Train — [FIX] truyền scheduler vào để step mỗi optimizer step
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            config, epoch, device, attribute_names, scheduler=scheduler,
            teacher=teacher,
        )
        
        epoch_time = time.time() - start_time
        
        # [FIX] Eval trên TẤT CẢ ranks, gather predictions
        metrics = evaluate(
            model, val_loader, criterion, config, device, attribute_names, rank=rank
        )
        
        if main_proc:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config.epochs} — {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {metrics['loss']:.4f}")
            print(f"  mA:         {metrics['mA']:.4f}")
            print(f"  Accuracy:   {metrics['accuracy']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1:         {metrics['f1']:.4f}")
            if 'vis_mae' in metrics:
                print(f"  Vis MAE:    {metrics['vis_mae']:.4f}  Vis Acc: {metrics.get('vis_acc', float('nan')):.4f}")
            if 'consistency_gap' in metrics:
                print(f"  Consist gap:{metrics['consistency_gap']:.4f}")
            print(f"{'='*60}\n")
            
            # Save best model
            if metrics['mA'] > best_mA:
                best_mA = metrics['mA']
                save_path = os.path.join(config.output_dir, 'best_model.pth')
                save_model = model.module if config.use_ddp else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mA': best_mA,
                    'metrics': metrics,
                    'config': vars(config),
                }, save_path)
                print(f"  ★ Best model saved! mA={best_mA:.4f}")
            
            # Latest checkpoint (để resume)
            save_model = model.module if config.use_ddp else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_mA': best_mA,
            }, os.path.join(config.output_dir, 'latest_checkpoint.pth'))
            
            # Periodic checkpoint
            if (epoch + 1) % config.save_interval == 0:
                save_path = os.path.join(config.output_dir, f'checkpoint_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_mA': best_mA,
                }, save_path)
    
    if main_proc:
        print(f"\nTraining complete! Best mA: {best_mA:.4f}")
    
    if config.use_ddp:
        cleanup_ddp()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIMP-PAR v9 Training')
    parser.add_argument('--config_variant', type=str, default='', help='C2, C3, or C4 preset')
    parser.add_argument('--occlusion_pkl_path', type=str, default='', help='occlusion_pairs_train.pkl')
    parser.add_argument('--occlusion_img_dir', type=str, default='', help='occlusion_images root')
    parser.add_argument('--pkl_path', type=str, default='', help='Path to dataset pkl')
    parser.add_argument('--img_dir', type=str, default='', help='Path to image directory')
    parser.add_argument('--domain_tokens_path', type=str, default='', help='Path to domain_tokens.pt')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=-1, help='DataLoader workers')
    parser.add_argument('--no_ddp', action='store_true', help='Disable DDP (single GPU)')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--kd_teacher_ckpt', type=str, default='', help='DeepMAR teacher ckpt')
    parser.add_argument('--lambda_kd', type=float, default=-1.0, help='KD weight')
    parser.add_argument('--attr_freq_path', type=str, default='', help='attr_freq.json')
    parser.add_argument('--init_from', type=str, default='', help='Init weights from ckpt')
    parser.add_argument('--max_train_batches', type=int, default=0, help='Smoke only')
    parser.add_argument('--max_val_batches', type=int, default=0, help='Smoke only')
    parser.add_argument('--grad_accum_steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()
    
    if args.config_variant:
        config = get_aq_config_variant(args.config_variant)
    else:
        config = get_kaggle_config()

    # Override from args
    if args.pkl_path:
        config.pkl_path = args.pkl_path
    if args.img_dir:
        config.img_dir = args.img_dir
    if args.domain_tokens_path:
        config.domain_tokens_path = args.domain_tokens_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.num_workers >= 0:
        config.num_workers = args.num_workers
    if args.no_ddp:
        config.use_ddp = False
    if args.no_amp:
        config.use_amp = False
    if args.output_dir:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
    if args.occlusion_pkl_path:
        config.occlusion_pkl_path = args.occlusion_pkl_path
        if not args.occlusion_img_dir:
            config.occlusion_img_dir = os.path.join(
                os.path.dirname(os.path.abspath(args.occlusion_pkl_path)),
                'occlusion_images',
            )
    if args.occlusion_img_dir:
        config.occlusion_img_dir = args.occlusion_img_dir
    if args.kd_teacher_ckpt:
        config.kd_teacher_ckpt = args.kd_teacher_ckpt
    if args.lambda_kd >= 0:
        config.lambda_kd = args.lambda_kd
    if args.attr_freq_path:
        config.attr_freq_path = args.attr_freq_path
        config.use_fixed_pos_weight = True
    if args.max_train_batches:
        config.max_train_batches = args.max_train_batches
    if args.max_val_batches:
        config.max_val_batches = args.max_val_batches
    if args.grad_accum_steps:
        config.grad_accum_steps = args.grad_accum_steps
    if args.seed >= 0:
        config.seed = args.seed
    # stash init path on config for main()
    config._init_from = args.init_from or ''
    
    # Phát hiện số GPU
    world_size = torch.cuda.device_count()
    
    if config.use_ddp and world_size > 1:
        print(f"Launching DDP with {world_size} GPUs...")
        torch.multiprocessing.spawn(
            main,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        config.use_ddp = False
        main(rank=0, world_size=1, config=config)
