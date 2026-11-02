"""python -m eopar.train.run --config eopar/configs/P3.json --output ..."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from eopar.datasets.interventions import InterventionDataset
from eopar.eval.metrics import evaluate_arrays
from eopar.losses.objective import objective
from eopar.models.adapter import EOPAR, load_baseline
from eopar.train.runtime import (ROOT, atomic_save, capture_rng, manifest,
                                restore_rng, source_hashes, write_json, verify_pretrained,
                                digest, environment_identity)


def load_config(path, output=None, resume=False):
    config = json.loads(Path(path).read_text())
    required = {'variant', 'pkl_path', 'img_dir', 'region_map', 'init_checkpoint', 'num_attrs',
                'size', 'seed', 'batch_size', 'grad_accum', 'epochs', 'schedule_epochs', 'workers',
                'lr', 'lambda_obs', 'lambda_rank', 'margin', 'max_train_batches', 'max_val_batches'}
    optional = {'pretrained_lock', 'eval_batch_size', 'eval_amp', 'augment', 'eval_exclusions', 'gradient_checkpointing', 'geometry_manifest'}
    if required - config.keys() or config.keys() - required - optional:
        raise ValueError(f'Config fields: missing={required-config.keys()}, unknown={config.keys()-required-optional}')
    config.setdefault('pretrained_lock', 'eopar/configs/pretrained_lock.json')
    config.setdefault('eval_batch_size', config['batch_size'])
    config.setdefault('eval_amp', True)
    config.setdefault('augment', True)
    config.setdefault('gradient_checkpointing', True)
    for key in ('pkl_path', 'img_dir', 'region_map', 'init_checkpoint', 'pretrained_lock', 'eval_exclusions', 'geometry_manifest'):
        if config.get(key) and not Path(config[key]).is_absolute():
            config[key] = str(ROOT / config[key])
        if config.get(key) and not (resume and key == 'init_checkpoint') and not Path(config[key]).exists():
            raise FileNotFoundError(config[key])
    if not Path(config['img_dir']).is_dir() or any(not Path(config[key]).is_file() for key in ('pkl_path', 'region_map', 'pretrained_lock')):
        raise ValueError('Expected image directory and regular metadata/lock files')
    if output:
        config['output'] = str(Path(output).resolve())
    if config['variant'] not in ('B0', 'B1', 'P1', 'P2', 'P3'):
        raise ValueError('Unknown variant')
    for key in ('batch_size', 'eval_batch_size', 'grad_accum', 'epochs', 'schedule_epochs', 'num_attrs', 'size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('workers', 'seed', 'max_train_batches', 'max_val_batches'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('lr', 'lambda_obs', 'lambda_rank', 'margin'):
        if isinstance(config[key], bool) or not isinstance(config[key], (float, int)) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if config['lr'] == 0 or not 0 < config['margin'] < 1 or config['size'] < 32 or config['size'] % 32:
        raise ValueError('Require lr>0, margin in (0,1), image size a multiple of 32')
    if any(type(config[key]) is not bool for key in ('eval_amp', 'augment', 'gradient_checkpointing')):
        raise ValueError('eval_amp/augment must be boolean')
    if config['epochs'] > config['schedule_epochs']:
        raise ValueError('Training cannot exceed fixed schedule horizon')
    return config


def build(config, output, initialize=True):
    # Heavy CUDA dependencies are lazy: CPU tests never import VMamba/Mamba.
    from aq_visibility.config import get_aq_config_variant
    verify_pretrained(config['pretrained_lock'])
    from eopar.models.backbone import CLIMPPAR
    from aq_visibility.losses.climpar_loss import CLIMPPARLoss
    preset = 'C3_MS' if config['variant'] == 'B0' else 'C4b_MS'
    base_config = get_aq_config_variant(preset, output_dir=str(output),
                                      unfreeze_vision=False, use_ddp=False,
                                      gradient_checkpointing=config.get('gradient_checkpointing', True),
                                      img_height=config['size'], img_width=config['size'])
    model = CLIMPPAR(base_config, num_attrs=config['num_attrs'])
    if initialize and config.get('init_checkpoint'):
        load_baseline(model, config['init_checkpoint'])
    wrapped = EOPAR(model, base_config.embed_dim, config['variant'].startswith('P'))
    return wrapped, CLIMPPARLoss(base_config)


def make_dataset(config, split, family='solid', severity=1.):
    from aq_visibility.dataset.clip_dataset import PARDataset
    base = PARDataset(config['pkl_path'], config['img_dir'], split, transform=None)
    if config.get('eval_exclusions') and split != 'train':
        excluded = set(json.loads(Path(config['eval_exclusions']).read_text())[split])
        from torch.utils.data import Subset
        original = base
        base = Subset(original, [i for i, name in enumerate(original.img_ids) if name not in excluded])
        base.attributes = original.attributes
    return InterventionDataset(base, config['region_map'], split, config['seed'],
                               config['size'], config['variant'] == 'P3' and split == 'train',
                               family, severity, augment=config.get('augment', True),
                               geometry_manifest=config.get('geometry_manifest'))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, attributes, limit=0, prediction_path=None, amp=True):
    model.eval()
    arrays = {key: [] for key in ('clean', 'occ', 'labels', 'obs', 'obs_occ',
                                   'vis_target', 'vis_valid', 'affected', 'intact')}
    names, times = [], []
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        b = to_device(batch, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.monotonic()
        with autocast(device.type, enabled=device.type == 'cuda' and amp):
            clean = model(b['img_clean'], b['domain_token'], attributes)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.monotonic() - start)
        with autocast(device.type, enabled=device.type == 'cuda' and amp):
            occ = model(b['img_occ'], b['domain_token'], attributes)
        for key, value in [('clean', clean['attr_logits'].sigmoid()),
                           ('occ', occ['attr_logits'].sigmoid()), ('labels', b['labels']),
                           ('vis_target', b['vis_target']), ('vis_valid', b['vis_valid']),
                           ('affected', b['affected_mask']), ('intact', b['intact_mask'])]:
            arrays[key].append(value.cpu().float().numpy())
        if clean['observability_logits'] is not None:
            arrays['obs'].append(clean['observability_logits'].float().sigmoid().cpu().numpy())
            arrays['obs_occ'].append(occ['observability_logits'].float().sigmoid().cpu().numpy())
        names.extend(batch['imgname'])
    if not names:
        raise ValueError('Empty evaluation dataset/limit')
    a = {k: np.concatenate(v) for k, v in arrays.items() if v}
    result = dict(clean=evaluate_arrays(a['clean'], a['labels'], a.get('obs')),
                  occluded=evaluate_arrays(a['occ'], a['labels'], a.get('obs_occ')),
                  samples=len(names), timing=dict(clean_forward_seconds=sum(times),
                  warmup_included=True, batch_count=len(times), precision='amp_fp16' if device.type == 'cuda' and amp else 'fp32'))
    mask = a['intact'].astype(bool)
    result['consistency_gap'] = float(np.abs(a['clean'] - a['occ'])[mask].mean()) if mask.any() else None
    if 'obs_occ' in a:
        valid, affected = a['vis_valid'].astype(bool), a['affected'].astype(bool)
        result['synthetic_observability'] = dict(
            proxy_mae=float(np.abs(a['obs_occ'] - a['vis_target'])[valid].mean()) if valid.any() else None,
            mean=float(a['obs_occ'].mean()), std=float(a['obs_occ'].std()),
            drop_on_affected=float((a['obs'] - a['obs_occ'])[affected].mean()) if affected.any() else None,
            note='Proxy band targets, not naturally annotated observability')
    if prediction_path:
        from eopar.eval.artifacts import save_predictions
        save_predictions(prediction_path, dict(**a, names=np.asarray(names), attributes=np.asarray(attributes)))
    return result


def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, config, device, epoch):
    model.train()
    loader.dataset.epoch = epoch
    total = min(len(loader), config.get('max_train_batches', 0) or len(loader))
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for i, batch in enumerate(loader):
        if i >= total:
            break
        b = to_device(batch, device)
        # Each loss is a microbatch mean. Weight a short final batch by examples.
        group_start = (i // config['grad_accum']) * config['grad_accum']
        group_size = min(config['grad_accum'], total - group_start)
        group_samples = sum(min(loader.batch_size, len(loader.dataset) - j * loader.batch_size)
                            for j in range(group_start, group_start + group_size))
        with autocast(device.type, enabled=device.type == 'cuda'):
            clean = model(b['img_clean'], b['domain_token'], loader.dataset.attributes)
            occ = model(b['img_occ'], b['domain_token'], loader.dataset.attributes)
        # Loss calculations are fp32 even when model forward uses AMP.
        clean = {k: v.float() if torch.is_tensor(v) else v for k, v in clean.items()}
        occ = {k: v.float() if torch.is_tensor(v) else v for k, v in occ.items()}
        loss, parts = objective(clean, occ, b, criterion, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss epoch={epoch} batch={i}')
        scaler.scale(loss * (len(b['labels']) / group_samples)).backward()
        if (i + 1) % config['grad_accum'] == 0 or i + 1 == total:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if i % 20 == 0 or i + 1 == total:
            print(json.dumps(dict(epoch=epoch + 1, batch=i + 1, batches=total,
                                  loss=float(loss.detach()), parts=parts,
                                  elapsed=time.monotonic() - started)), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--diagnostic-proxy', action='store_true',
                        help='Only bounded <=4-batch <=2-epoch integration diagnostics; never a pilot')
    parser.add_argument('--verification-run',action='store_true',
                        help='Bounded verification only: <=1 full epoch or <=2 four-batch epochs')
    parser.add_argument('--gate-report')
    args = parser.parse_args()
    if args.stop_after < 0:
        parser.error('--stop-after must be nonnegative')
    c = load_config(args.config, args.output, resume=args.resume)
    if args.verification_run and args.diagnostic_proxy:
        raise ValueError('Choose one verification mode')
    requested_end=min(c['epochs'],args.stop_after or c['epochs'])
    if args.diagnostic_proxy:
        if c['epochs']>2 or not 0<c['max_train_batches']<=4 or not 0<c['max_val_batches']<=2:
            raise ValueError('Proxy diagnostics must be bounded, not a full-data training run')
    elif c['variant'].startswith('P') and not c.get('geometry_manifest'):
        raise ValueError('P1/P2/P3 pilot blocked: fixed-band anatomical validity must be verified first')
    if args.verification_run:
        bounded=c['epochs']<=2 and 0<c['max_train_batches']<=4 and 0<c['max_val_batches']<=2
        if requested_end>1 and not bounded:
            raise ValueError('Verification cannot run an unbounded multi-epoch training job')
    elif not args.diagnostic_proxy:
        if not args.gate_report or requested_end>5:
            raise ValueError('Pilot requires a matching gate report and explicit end <=5; full training is not authorized')
        from eopar.verification.gate import validate_gate
        validate_gate(args.gate_report,c['variant'],args.config)
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists() and not args.resume:
        raise FileExistsError('Existing run: use --resume or choose a new output')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator().manual_seed(c['seed'])
    train, val = make_dataset(c, 'train'), make_dataset(c, 'val')
    if c['variant'].startswith('P') and not args.diagnostic_proxy:
        for dataset in (train,val):
            base=dataset.base
            names=base.img_ids if hasattr(base,'img_ids') else [base.dataset.img_ids[i] for i in base.indices]
            regions=set(r for name in names for r in (dataset.geometry or {}).get(str(name),[]))
            if regions!={'head','upper','lower','shoes'}:
                raise ValueError('Verified geometry must cover all four evidence regions')
    if len(train.attributes) != c['num_attrs']:
        raise ValueError('Attribute count mismatch')
    train_loader = DataLoader(train, batch_size=c['batch_size'], shuffle=True,
                              num_workers=c['workers'], pin_memory=device.type == 'cuda',
                              generator=generator)
    val_loader = DataLoader(val, batch_size=c['eval_batch_size'], num_workers=c['workers'])
    model, criterion = build(c, out, initialize=not args.resume)
    model, criterion = model.to(device), criterion.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=c['lr'], weight_decay=.05)
    n_batches = min(len(train_loader), c.get('max_train_batches', 0) or len(train_loader))
    steps = math.ceil(n_batches / c['grad_accum'])
    warmup, horizon = 2 * steps, c['schedule_epochs'] * steps
    def lr_multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, horizon - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = GradScaler(device.type, enabled=device.type == 'cuda')
    start, best = 0, -1.
    if args.resume:
        state = torch.load(out / 'latest.pth', map_location='cpu', weights_only=False)
        if state['config'] != c or state['source_hashes'] != source_hashes() or state['environment'] != environment_identity():
            raise ValueError('Resume config/source mismatch: preserve original source and config')
        if state['split_sha256'] != digest(c['pkl_path']) or state['region_sha256'] != digest(c['region_map']):
            raise ValueError('Resume data changed')
        if state.get('geometry_sha256') != (digest(c['geometry_manifest']) if c.get('geometry_manifest') else None):
            raise ValueError('Resume geometry changed')
        if state['attributes'] != val.attributes:
            raise ValueError('Resume attribute order changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start, best = state['epoch'], state['best_mA']
        restore_rng(state['rng'], generator)
    else:
        write_json(out / 'manifest.json', dict(manifest(c, model), diagnostic_proxy=args.diagnostic_proxy,
                                              verification_run=args.verification_run))
    end = min(c['epochs'], args.stop_after or c['epochs'])
    if end <= start:
        raise ValueError('No epochs to run; checkpoint is already at/after requested end')
    completed = start
    for epoch in range(start, end):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, c, device, epoch)
        metrics = evaluate(model, val_loader, device, val.attributes, c.get('max_val_batches', 0), amp=c['eval_amp'])
        score = metrics['clean']['par']['mA']
        improved = score > best
        best = max(best, score)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                       epoch=epoch + 1, best_mA=best, config=c, attributes=val.attributes,
                       environment=environment_identity(), split_sha256=digest(c['pkl_path']),
                       region_sha256=digest(c['region_map']),
                       geometry_sha256=digest(c['geometry_manifest']) if c.get('geometry_manifest') else None,
                       rng=capture_rng(generator), source_hashes=source_hashes())
        atomic_save(payload, out / 'latest.pth')
        if improved:
            atomic_save(payload, out / 'best.pth')
        write_json(out / f'val_epoch{epoch + 1:02d}.json', metrics)
        print(f'EPOCH_DONE {epoch + 1} val_mA={score:.6f} best={best:.6f}', flush=True)
        completed = epoch + 1
    write_json(out / 'status.json', dict(completed_epochs=completed,
               planned_epochs=c['epochs'], state='pilot_complete' if args.stop_after and args.stop_after < c['epochs'] else 'train_complete'))


if __name__ == '__main__':
    main()
