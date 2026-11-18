"""Matched C3/C3_MS classification runs for original benchmark images."""
import argparse
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from eopar.datasets.interventions import encode, seed_for
from eopar.train.runtime import (ROOT, atomic_save, capture_rng, digest, environment_identity,
                                 restore_rng, verify_pretrained, write_json)
from eopar.train.run import load_config
from experiments.par_quality_recovery.amp import audit_steps, train_epoch
from .metrics import scores


class CleanDataset(Dataset):
    def __init__(self, manifest, split, seed):
        self.rows = [r for r in manifest['rows'] if r['split']==split]
        if len(self.rows)!=manifest['counts'][split] or not self.rows:
            raise ValueError('Manifest split count differs')
        self.root = Path(manifest['image_root'])
        self.attributes = manifest['attributes']
        self.split,self.seed,self.epoch = split,seed,0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        path = self.root/row['path']
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Unsafe image path')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError(f"Changed image {row['id']}")
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert('RGB')
        if self.split=='train':
            rng = random.Random(seed_for(self.seed,'train',self.epoch,row['id']))
            if rng.random()<.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(.8,1.2))
        labels = torch.tensor(np.where(row['valid'],row['label'],-1),dtype=torch.float32)
        return dict(img_clean=encode(image,448),labels=labels,domain_token=torch.zeros(32,768),
                    imgname=row['id'],scale=1)


class MaskedClassification(nn.Module):
    def forward(self, logits, target):
        valid = target>=0
        y = target.clamp(0,1)
        with torch.no_grad():
            positive = (y*valid).sum(0).clamp(min=1)
            negative = (valid.sum(0)-positive).clamp(min=1)
        bce = F.binary_cross_entropy_with_logits(logits,y,pos_weight=negative/positive,reduction='none')
        p = logits.sigmoid()
        q = (1-p+.05).clamp(max=1)
        asl = -(y*p.clamp_min(1e-8).log()+(1-y)*q.clamp_min(1e-8).log())*(1-(p*y+q*(1-y))).pow(y+4*(1-y))
        loss = ((bce+asl)*valid).sum()/valid.sum().clamp(min=1)
        return loss,{'L_attr':loss.detach()}


def build_model(variant, attrs, out):
    from aq_visibility.config import get_aq_config_variant
    from eopar.models.backbone import CLIMPPAR
    from eopar.models.adapter import EOPAR
    config = get_aq_config_variant("C3_MS",output_dir=str(out),unfreeze_vision=False,use_ddp=False,
                                   gradient_checkpointing=True,img_height=448,img_width=448)
    model = EOPAR(CLIMPPAR(config,num_attrs=len(attrs)),config.embed_dim,False)
    # Initialize common parameters identically before removing multiscale from C3.
    if variant=='C3':
        model.baseline.use_multiscale = False
        model.baseline.ms_tokenizer = None
        config.use_multiscale = False
        config.variant_id = 'C3'
    for p in model.baseline.query_heads.vis_head.parameters():
        p.requires_grad_(False)
    return model.cuda()


@torch.no_grad()
def predict(model, loader, limit=0):
    model.eval()
    probabilities, labels, ids = [], [], []
    for i,batch in enumerate(loader):
        if limit and i>=limit:
            break
        with torch.amp.autocast('cuda'):
            result = model(batch['img_clean'].cuda(),batch['domain_token'].cuda(),loader.dataset.attributes)
        probabilities.append(result['attr_logits'].float().sigmoid().cpu().numpy())
        labels.append(batch['labels'].numpy())
        ids.extend(batch['imgname'])
    y = np.concatenate(labels)
    return dict(clean=np.concatenate(probabilities),labels=np.maximum(y,0).astype(np.uint8),valid=y>=0,
                names=np.asarray(ids),attributes=np.asarray(loader.dataset.attributes))


def evaluate(arrays):
    return scores(arrays['clean'],arrays['labels'],arrays['valid'])


def same_resume(state, model, opt, scheduler, scaler, generator):
    audit_steps(state)
    model.load_state_dict(state['model'],strict=True)
    opt.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'],generator)


def run(a):
    out = a.output
    out.mkdir(parents=True,exist_ok=True)
    source = json.loads(a.manifest.read_text())
    attrs = source['attributes']
    cfg = load_config(ROOT/'eopar/configs/B0.json')
    from experiments.par_quality_train.run import identity
    evidence = dict(dataset=source['dataset'],variant=a.variant,seed=a.seed,batch=a.batch,accumulation=64//a.batch,
        image_size=448,epochs=30,lr=5e-5,manifest_sha256=digest(a.manifest),source_hashes=identity(),
        pretrained=verify_pretrained(cfg['pretrained_lock']),environment=environment_identity(),
        clean=True,threshold=.5)
    evidence['benchmark_source'] = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    evidence['amp_recovery_sha256'] = digest(ROOT/'experiments/par_quality_recovery/amp.py')
    if a.action=='train':
        gate = json.loads(a.gate.read_text())
        protocol = {k:v for k,v in evidence.items() if k!='seed'}
        if gate['state']!='PASS' or gate['protocol']!=protocol:
            raise ValueError('Smoke evidence mismatch')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    generator = torch.Generator().manual_seed(a.seed)
    model = build_model(a.variant,attrs,out)
    loaded_pretrained = {}
    for encoder in (model.baseline.vision_encoder,model.baseline.text_encoder):
        loaded_pretrained.update(encoder.pretrained_files)
    if loaded_pretrained!=evidence['pretrained']:
        raise ValueError('Loaded pretrained files differ from the lock')

    def loader(split):
        return DataLoader(CleanDataset(source,split,a.seed),batch_size=a.batch,shuffle=split=='train',
                          generator=generator if split=='train' else None,num_workers=4,pin_memory=True)

    if a.action=='test':
        status = json.loads((out/'status.json').read_text())
        checkpoint = out/'best.pth'
        ckpt_hash = digest(checkpoint)
        if status['state']!='TRAIN_COMPLETE' or status.get('best_sha256')!=ckpt_hash:
            raise ValueError('Training incomplete or checkpoint changed')
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['evidence']!=evidence or state['attributes']!=attrs:
            raise ValueError('Checkpoint provenance differs')
        model.load_state_dict(state['model'],strict=True)
        result = predict(model,loader('test'))
        dedup = ~np.isin(result['names'],source['test_dedup_exclusions'])
        if len(result['names'])!=source['counts']['test'] or digest(checkpoint)!=ckpt_hash:
            raise ValueError('Test count or checkpoint changed')
        expected = [r['id'] for r in source['rows'] if r['split']=='test']
        if result['names'].tolist()!=expected:
            raise ValueError('Test IDs differ from locked manifest')
        from eopar.eval.artifacts import save_predictions
        save_predictions(out/'test.npz',result)
        clean = {k:v[dedup] if k!='attributes' else v for k,v in result.items()}
        write_json(out/'test.json',dict(state='COMPLETE',metrics=evaluate(result),
            dedup_metrics=evaluate(clean),dedup_samples=int(dedup.sum()),samples=len(dedup),
            epoch=state['epoch'],evidence=evidence,checkpoint_sha256=ckpt_hash,predictions_sha256=digest(out/'test.npz')))
        return
    train,val = loader('train'),loader('val')
    accumulation = evidence['accumulation']
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=5e-5,weight_decay=.05)
    steps = math.ceil(len(train)/accumulation)

    def multiplier(step):
        if a.action=='smoke':
            return 1.
        if step<2*steps:
            return step/(2*steps)
        return .5*(1+math.cos(math.pi*min(1.,(step-2*steps)/(28*steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    scaler = torch.amp.GradScaler('cuda')
    criterion = MaskedClassification()
    latest = out/'latest.pth'
    start,best = 0,-1.
    if latest.exists() and a.action=='train':
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['evidence']!=evidence or state['attributes']!=attrs:
            raise ValueError('Resume provenance differs')
        same_resume(state,model,optimizer,scheduler,scaler,generator)
        start,best = state['epoch'],state['best']
        # The latest checkpoint is authoritative if interrupted between writes.
        if state['improved']:
            best_path = out/'best.pth'
            selected = torch.load(best_path,map_location='cpu',weights_only=False) if best_path.exists() else {}
            if selected.get('epoch')!=start:
                atomic_save(state,best_path)
        del state
    end = 1 if a.action=='smoke' else a.stop_after
    for epoch in range(start,end):
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        train_stats = train_epoch(model,criterion,train,optimizer,scheduler,scaler,
            {'grad_accum':accumulation},torch.device('cuda'),epoch,limit=2*accumulation if a.action=='smoke' else 0)
        validation = predict(model,val,limit=4 if a.action=='smoke' else 0)
        val_stats = evaluate(validation)
        improved = val_stats['mA']>best
        best = max(best,val_stats['mA'])
        state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(),rng=capture_rng(generator),epoch=epoch+1,best=best,
                     improved=improved,evidence=evidence,attributes=attrs)
        audit_steps(state)
        atomic_save(state,latest)
        if improved and a.action=='train':
            atomic_save(state,out/'best.pth')
        peak = torch.cuda.max_memory_reserved()
        write_json(out/f'epoch{epoch+1:02d}.json',dict(train=train_stats,validation=val_stats,
                   peak_reserved_bytes=peak,seconds=time.monotonic()-begin))
        write_json(out/'status.json',dict(state='RUNNING',completed_epochs=epoch+1))
        if a.action=='smoke':
            loaded = torch.load(latest,map_location='cpu',weights_only=False)
            same_resume(loaded,model,optimizer,scheduler,scaler,generator)
            replay = predict(model,val,limit=4)
            if not np.array_equal(validation['clean'],replay['clean']):
                raise AssertionError('Checkpoint prediction mismatch')
            write_json(out/'smoke.json',dict(state='PASS',protocol={k:v for k,v in evidence.items() if k!='seed'},
                peak_reserved_bytes=peak,epoch_estimate_seconds=train_stats['seconds']*len(train)/train_stats['batches']))
    if a.action=='train':
        final = max(start,end)
        status = dict(state='TRAIN_COMPLETE' if final==30 else 'EPOCH_CHECK',completed_epochs=final)
        if final==30:
            status['best_sha256'] = digest(out/'best.pth')
        write_json(out/'status.json',status)


if __name__=='__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path)
    p.add_argument('--variant',choices=('C3','C3_MS'),required=True)
    p.add_argument('--action',choices=('smoke','train','test'),required=True)
    p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    p.add_argument('--batch',type=int,choices=(8,16),default=16)
    p.add_argument('--stop-after',type=int,choices=(1,30),default=30)
    run(p.parse_args())
