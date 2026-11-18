import json,time
import torch
from torch.amp import autocast
from eopar.train.run import to_device


def loss_for(model,criterion,b,attributes,device,amp=True):
    with autocast(device.type,enabled=device.type=='cuda' and amp):out=model(b['img_clean'],b['domain_token'],attributes)
    loss,parts=criterion(out['attr_logits'].float(),b['labels'])
    if set(parts)-{'L_attr','_total'}:raise AssertionError('Auxiliary loss in classification control')
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite forward classification loss')
    return loss


def update(model,opt,scheduler,scaler):
    scaler.unscale_(opt)
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    missing=[n for n,p in params if p.grad is None]
    if missing:raise FloatingPointError('Missing trainable gradient: '+str(missing[:8]))
    checks=[torch.isfinite(p.grad).all() for _,p in params]
    finite=bool(torch.stack(checks).all().item())
    bad=[]
    if not finite:
        bad=[n for (n,_),ok in zip(params,checks) if not ok.item()]
        if not scaler.is_enabled():raise FloatingPointError('Nonfinite FP32 gradient: '+str(bad[:8]))
    else:
        torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)
    old=scaler.get_scale();step_before=scheduler.last_epoch
    scaler.step(opt);scaler.update();new=scaler.get_scale()
    skipped=new<old
    if not finite and not skipped:raise AssertionError('GradScaler did not skip nonfinite gradients')
    if not skipped:scheduler.step()
    if skipped and scheduler.last_epoch!=step_before:raise AssertionError('LR advanced on skipped AMP step')
    opt.zero_grad(set_to_none=True)
    return dict(skipped=skipped,scale_before=old,scale_after=new,bad_parameters=bad[:8],scheduler_step=scheduler.last_epoch)


def next_streak(previous,skipped):
    value=previous+1 if skipped else 0
    if value>=8:raise FloatingPointError('Eight consecutive AMP overflows; requires FP32 diagnosis')
    return value


def audit_steps(state):
    expected=int(state['scheduler']['last_epoch'])
    counts=[int(s['step'].item() if torch.is_tensor(s['step']) else s['step']) for s in state['optimizer']['state'].values() if 'step' in s]
    if not counts or set(counts)!={expected}:raise ValueError('Optimizer/scheduler update counts disagree')
    return expected


def train_epoch(model,criterion,loader,opt,scheduler,scaler,spec,device,epoch,limit=0):
    model.train();loader.dataset.epoch=epoch;opt.zero_grad(set_to_none=True)
    total=min(len(loader),limit or len(loader));start=time.monotonic();losses=[];counts={};skips=0;streak=0;updates=0
    for i,batch in enumerate(loader):
        if i>=total:break
        b=to_device(batch,device)
        group_start=(i//spec['grad_accum'])*spec['grad_accum'];group_size=min(spec['grad_accum'],total-group_start)
        group_samples=sum(min(loader.batch_size,len(loader.dataset)-j*loader.batch_size) for j in range(group_start,group_start+group_size))
        loss=loss_for(model,criterion,b,loader.dataset.attributes,device)
        scaler.scale(loss*len(b['labels'])/group_samples).backward()
        for value in batch['scale'].tolist():counts[str(value)]=counts.get(str(value),0)+1
        if (i+1)%spec['grad_accum']==0 or i+1==total:
            result=update(model,opt,scheduler,scaler);streak=next_streak(streak,result['skipped']);skips+=int(result['skipped']);updates+=int(not result['skipped'])
            if result['skipped']:print(json.dumps(dict(event='AMP_STEP_SKIPPED',epoch=epoch+1,batch=i+1,**result)),flush=True)
        losses.append(float(loss.detach()))
        if i%50==0 or i+1==total:print(json.dumps(dict(epoch=epoch+1,batch=i+1,total=total,loss=losses[-1],seconds=time.monotonic()-start,scale=scaler.get_scale(),amp_skips=skips,lr=opt.param_groups[0]['lr'])),flush=True)
    if not losses:raise ValueError('Empty train epoch')
    return dict(loss_mean=sum(losses)/len(losses),batches=len(losses),scales=counts,scaler_skips=skips,optimizer_updates=updates,seconds=time.monotonic()-start)
