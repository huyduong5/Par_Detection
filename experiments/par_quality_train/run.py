"""Bounded classification controls, no anatomical/observability supervision."""
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from eopar.datasets.interventions import encode, seed_for
from eopar.train.run import build, load_config, to_device
from eopar.train.runtime import ROOT, source_hashes, environment_identity, digest, verify_pretrained, atomic_save, capture_rng, restore_rng, write_json, manifest
from eopar.eval.metrics import evaluate_arrays
from experiments.par_resolution.core import degraded


def identity():
    result=source_hashes()
    for p in sorted((ROOT/'experiments/par_quality_train').rglob('*.py')):
        result[str(p.relative_to(ROOT))]=digest(p)
    result['experiments/par_resolution/core.py']=digest(ROOT/'experiments/par_resolution/core.py')
    return result


def specification(mode):
    if mode not in ('clean','mixed'):raise ValueError('Unknown classification control')
    return dict(name='quality_classification_control_v1',mode=mode,seed=42,batch_size=16,grad_accum=4,
                size=448,lr=5e-5,schedule_epochs=30,pilot_epochs=5,workers=4,
                train_scales=[1] if mode=='clean' else [1,2,4,8],
                geometry_supervision=False,observability=False,loss='microbatch weighted BCE + ASL',
                validation='clean official validation only; threshold 0.5; no test evaluation',
                augmentation='shared deterministic flip/brightness; independent uniform scale RNG')


class QualityDataset(Dataset):
    def __init__(self,base,mode,split,size=448,seed=42):
        self.base,self.mode,self.split,self.size,self.seed=base,mode,split,size,seed
        self.attributes=list(base.attributes);self.epoch=0
        specification(mode)
        if split not in ('train','val'):raise ValueError('No test access in this training experiment')
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        image,labels,name,tokens=self.base[i]
        if not isinstance(image,Image.Image):raise TypeError('Expected decoded PIL image')
        image=image.convert('RGB').copy();scale=1
        if self.split=='train':
            rng=random.Random(seed_for(self.seed,self.split,self.epoch,name))
            if rng.random()<.5:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image=ImageEnhance.Brightness(image).enhance(rng.uniform(.8,1.2))
            if self.mode=='mixed':
                scale=random.Random(seed_for(self.seed,'quality-scale',self.epoch,name)).choice((1,2,4,8))
                image,_=degraded(image,scale)
        y=torch.as_tensor(np.asarray(labels).copy(),dtype=torch.float32)
        if not torch.isfinite(y).all() or not ((y==0)|(y==1)).all():raise ValueError('Invalid PAR labels')
        return dict(img_clean=encode(image,self.size),labels=y,domain_token=torch.as_tensor(tokens),imgname=str(name),scale=scale)


def datasets(mode,c):
    from aq_visibility.dataset.clip_dataset import PARDataset
    return tuple(QualityDataset(PARDataset(c['pkl_path'],c['img_dir'],split,transform=None),mode,split)
                 for split in ('train','val'))


def model_and_loss(c,out):
    model,criterion=build(c,out)
    # This control learns only attribute classification, not heuristic visibility.
    criterion.lambda_v=0.;criterion.lambda_c=0.;criterion.use_visibility=False
    for p in model.baseline.query_heads.vis_head.parameters():p.requires_grad_(False)
    return model,criterion


def classification_loss(model,criterion,b,attributes,device):
    with autocast(device.type,enabled=device.type=='cuda'):
        outputs=model(b['img_clean'],b['domain_token'],attributes)
    loss,parts=criterion(outputs['attr_logits'].float(),b['labels'])
    if set(parts)-{'L_attr','_total'}:raise AssertionError('Unexpected auxiliary loss in classification control')
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite classification objective')
    return loss


def train_epoch(model,criterion,loader,opt,scheduler,scaler,spec,device,epoch,limit=0):
    model.train();loader.dataset.epoch=epoch;opt.zero_grad(set_to_none=True)
    total=min(len(loader),limit or len(loader));start=time.monotonic();losses=[];scales={};skips=0
    for i,batch in enumerate(loader):
        if i>=total:break
        b=to_device(batch,device)
        group_start=(i//spec['grad_accum'])*spec['grad_accum']
        group_size=min(spec['grad_accum'],total-group_start)
        group_samples=sum(min(loader.batch_size,len(loader.dataset)-j*loader.batch_size) for j in range(group_start,group_start+group_size))
        loss=classification_loss(model,criterion,b,loader.dataset.attributes,device)
        scaler.scale(loss*len(b['labels'])/group_samples).backward()
        for scale in batch['scale'].tolist():scales[str(scale)]=scales.get(str(scale),0)+1
        if (i+1)%spec['grad_accum']==0 or i+1==total:
            scaler.unscale_(opt)
            for name,p in model.named_parameters():
                if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                    raise FloatingPointError('Missing/nonfinite trainable gradient: '+name)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous=scaler.get_scale();scaler.step(opt);scaler.update()
            if scaler.get_scale()>=previous:scheduler.step()
            else:skips+=1
            opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if i%50==0 or i+1==total:print(json.dumps(dict(epoch=epoch+1,batch=i+1,total=total,loss=losses[-1],seconds=time.monotonic()-start)),flush=True)
    if not losses:raise ValueError('Empty train epoch')
    return dict(loss_mean=float(np.mean(losses)),batches=len(losses),scales=scales,scaler_skips=skips,seconds=time.monotonic()-start)


@torch.no_grad()
def validate(model,loader,device,limit=0):
    model.eval();predictions=[];labels=[]
    for i,batch in enumerate(loader):
        if limit and i>=limit:break
        if (batch['scale']!=1).any():raise AssertionError('Validation must remain clean')
        b=to_device(batch,device)
        with autocast(device.type,enabled=device.type=='cuda'):out=model(b['img_clean'],b['domain_token'],loader.dataset.attributes)
        predictions.append(out['attr_logits'].float().sigmoid().cpu().numpy());labels.append(b['labels'].cpu().numpy())
    if not predictions:raise ValueError('Empty validation')
    return dict(samples=sum(len(x) for x in labels),metrics=evaluate_arrays(np.concatenate(predictions),np.concatenate(labels)))


def run(mode,action,out,cpu_report,resume=False):
    spec=specification(mode);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=load_config(ROOT/'eopar/configs/B0.json');c.update(augment=False)
    config_hash=digest(ROOT/'eopar/configs/B0.json');sources=identity();env=environment_identity()
    pre=verify_pretrained(c['pretrained_lock']);split_hash=digest(c['pkl_path'])
    report=json.loads(Path(cpu_report).read_text())
    if report['state']!='PASS' or report['source_hashes']!=sources:raise ValueError('CPU quality verification stale/missing')
    if report['environment']!=env:raise ValueError('Quality verification environment changed')
    gate_path=out.parent/(mode+'_smoke')/'smoke_gate.json'
    evidence=dict(spec=spec,source_hashes=sources,environment=env,pretrained=pre,split_sha256=split_hash,config_hash=config_hash)
    if action=='pilot':
        gate=json.loads(gate_path.read_text())
        if gate.get('state')!='PASS' or gate.get('evidence')!=evidence:raise ValueError('Classification smoke gate missing/stale')
    if (out/'manifest.json').exists() and not resume:raise FileExistsError('Use resume; refusing output overwrite')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    device=torch.device('cuda');generator=torch.Generator().manual_seed(42)
    train,val=datasets(mode,c)
    loader=DataLoader(train,batch_size=16,shuffle=True,num_workers=4,generator=generator,pin_memory=True)
    val_loader=DataLoader(val,batch_size=16,num_workers=4,pin_memory=True)
    model,criterion=model_and_loss(c,out);model.to(device);criterion.to(device)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=spec['lr'],weight_decay=.05)
    steps=math.ceil(len(loader)/4)
    def multiplier(step):
        if action=='smoke':return 1.
        if step<2*steps:return step/max(1,2*steps)
        return .5*(1+math.cos(math.pi*min(1.,(step-2*steps)/(28*steps))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,multiplier);scaler=GradScaler('cuda')
    start=0;best=-1.
    if resume:
        state=torch.load(out/'latest.pth',map_location='cpu',weights_only=False)
        if state['evidence']!=evidence or state['attributes']!=val.attributes:raise ValueError('Resume provenance mismatch')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);restore_rng(state['rng'],generator);start=state['epoch'];best=state['best']
    else:write_json(out/'manifest.json',dict(manifest(c,model),classification_control=evidence,action=action))
    end=1 if action=='smoke' else 5
    if start>=end:raise ValueError('No epochs remaining')
    for epoch in range(start,end):
        torch.cuda.reset_peak_memory_stats()
        training=train_epoch(model,criterion,loader,opt,scheduler,scaler,spec,device,epoch,limit=105 if action=='smoke' else 0)
        metrics=validate(model,val_loader,device,limit=105 if action=='smoke' else 0)
        score=metrics['metrics']['par']['mA'];improved=score>best;best=max(score,best)
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=capture_rng(generator),epoch=epoch+1,best=best,evidence=evidence,attributes=val.attributes)
        atomic_save(payload,out/'latest.pth')
        if improved:atomic_save(payload,out/'best.pth')
        write_json(out/f'epoch{epoch+1:02d}.json',dict(training=training,validation=metrics,peak_reserved=torch.cuda.max_memory_reserved()))
        write_json(out/'status.json',dict(state='RUNNING',completed_epochs=epoch+1,maximum_epochs=end))
        if action=='smoke':
            before=validate(model,val_loader,device,limit=2)
            loaded=torch.load(out/'latest.pth',map_location='cpu',weights_only=False)
            model.load_state_dict(loaded['model'],strict=True);opt.load_state_dict(loaded['optimizer']);scheduler.load_state_dict(loaded['scheduler']);scaler.load_state_dict(loaded['scaler'])
            after=validate(model,val_loader,device,limit=2)
            if after!=before:raise AssertionError('Checkpoint inference is unstable')
            peak=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
            if peak>.85*total:raise MemoryError('Insufficient 15% VRAM headroom')
            write_json(out/'smoke_gate.json',dict(state='PASS',evidence=evidence,peak_reserved=peak,headroom=1-peak/total,train_batches=105,val_batches=105,classification_only=True))
    write_json(out/'status.json',dict(state='SMOKE_COMPLETE' if action=='smoke' else 'PILOT_COMPLETE',completed_epochs=end,maximum_epochs=end,not_eopar=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['clean','mixed'],required=True);p.add_argument('--action',choices=['smoke','pilot'],required=True);p.add_argument('--output',required=True);p.add_argument('--cpu-report',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a.mode,a.action,a.output,a.cpu_report,a.resume)
    except Exception as e:
        path=Path(a.output);path.mkdir(parents=True,exist_ok=True);status=path/'status.json'
        previous=json.loads(status.read_text()) if status.exists() else {}
        write_json(status,dict(previous,state='FAILED',error=repr(e)));raise
