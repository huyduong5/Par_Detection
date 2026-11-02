"""Real-backbone verification. Run only on an idle GPU, never beside live jobs.

This performs bounded verification epochs, not model-selection/full training.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from eopar.losses.objective import objective
from eopar.train import run as training
from eopar.train.run import build,load_config,make_dataset,to_device,train_epoch,evaluate
from eopar.train.runtime import ROOT,source_hashes,environment_identity,write_json,digest


def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def finite_gradients(model):
    missing=[]; groups={}
    for name,param in model.named_parameters():
        group='.'.join(name.split('.')[:3])
        record=groups.setdefault(group,dict(trainable=0,gradients=0,norm=0.))
        if not param.requires_grad:
            if param.grad is not None: raise AssertionError('Frozen gradient: '+name)
            continue
        record['trainable']+=param.numel()
        if param.grad is None: missing.append(name);continue
        if not torch.isfinite(param.grad).all(): raise AssertionError('Nonfinite gradient: '+name)
        record['gradients']+=param.numel();record['norm']+=float(param.grad.float().norm())
    if missing: raise AssertionError('Unexplained disconnected parameters: '+repr(missing))
    return groups


def forward_loss(model,criterion,b,c,amp=True):
    with autocast('cuda',enabled=amp):
        clean=model(b['img_clean'],b['domain_token'],make_dataset.attributes)
        occ=model(b['img_occ'],b['domain_token'],make_dataset.attributes)
    clean={k:v.float() if torch.is_tensor(v) else v for k,v in clean.items()}
    occ={k:v.float() if torch.is_tensor(v) else v for k,v in occ.items()}
    return objective(clean,occ,b,criterion,c),clean


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--variant',choices=['B1','P3'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--diagnostics-only',action='store_true')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0')
    report=dict(state='RUNNING',physical_gpu=args.physical_gpu,variant=args.variant,
                source_hashes=source_hashes(),environment=environment_identity(),checks={})
    path=out/f'gpu{args.physical_gpu}.json'
    def mark(name,evidence):
        report['checks'][name]=dict(state='PASS',evidence=evidence);write_json(path,report)
    try:
        assert torch.cuda.is_available()
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        u=torch.randn(1,4,8,device=device,requires_grad=True)
        delta=torch.rand_like(u,requires_grad=True);a=-torch.ones(4,2,device=device)
        b=torch.randn(1,2,8,device=device);cc=torch.randn_like(b)
        result=selective_scan_fn(u,delta,a,b,cc)
        result.sum().backward();assert torch.isfinite(u.grad).all()
        mark('cuda_extensions',dict(torch=torch.__version__,cuda=torch.version.cuda))
        variants={};gradient_report={}
        for variant in ('B0','B1','P1','P2','P3'):
            seed();c=load_config(ROOT/f'eopar/configs/{variant}.json');c['augment']=False
            ds=make_dataset(c,'val');make_dataset.attributes=ds.attributes
            batch=to_device(next(iter(DataLoader(ds,batch_size=2))),device)
            model,crit=build(c,out/f'build_{variant}');model.to(device);crit.to(device);model.train()
            (loss,parts),output=forward_loss(model,crit,batch,c)
            assert output['attr_logits'].shape==(2,57) and torch.isfinite(loss)
            loss.backward();gradient_report[variant]=finite_gradients(model)
            opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=c['lr'])
            opt.step();opt.zero_grad(set_to_none=True)
            metrics=evaluate(model,DataLoader(ds,batch_size=2),device,ds.attributes,limit=1,
                             prediction_path=out/f'{variant}_smoke_predictions.npz')
            saved=out/f'{variant}_smoke_state.pth'
            torch.save(model.state_dict(),saved);model.load_state_dict(torch.load(saved,weights_only=True),strict=True)
            variants[variant]=dict(loss=float(loss.detach()),parts=parts,clean_mA=metrics['clean']['par']['mA'])
            del opt,model,crit,batch,loss,output;gc.collect();torch.cuda.empty_cache()
        mark('variants',variants);mark('gradient_audit',gradient_report)

        seed();c=load_config(ROOT/f'eopar/configs/{args.variant}.json');c['augment']=False
        ds=make_dataset(c,'val');make_dataset.attributes=ds.attributes
        batch=to_device(next(iter(DataLoader(ds,batch_size=2))),device)
        model,crit=build(c,out/'mechanisms');model.to(device).eval();crit.to(device)
        from aq_visibility.config import get_aq_config_variant
        from aq_visibility.model.climp_par import CLIMPPAR as Original
        original=Original(get_aq_config_variant('C4b_MS',output_dir=str(out/'original')),57).to(device).eval()
        original.load_state_dict(model.baseline.state_dict(),strict=True)
        with torch.no_grad():
            reference=original(batch['img_clean'],batch['domain_token'],ds.attributes)[0]
            actual=model(batch['img_clean'],batch['domain_token'],ds.attributes)['attr_logits']
        torch.testing.assert_close(reference,actual,atol=1e-6,rtol=1e-5)
        mark('parity',dict(max_absolute_error=float((reference-actual).abs().max())))
        del original,reference;gc.collect();torch.cuda.empty_cache()
        base=model.baseline
        def logits():
            with torch.no_grad():return model(batch['img_clean'],batch['domain_token'],ds.attributes)['attr_logits']
        effects={}
        with patch.object(base.cross_mamba,'forward',lambda x,t:(torch.zeros_like(x),torch.zeros_like(t))):
            effects['cross_mamba']=float((logits()-actual).abs().max())
        assert effects['cross_mamba']==0.,effects
        encode=base.encode_text
        def no_text(*a,**k):
            x,y=encode(*a,**k);return torch.zeros_like(x),torch.zeros_like(y)
        with patch.object(base,'encode_text',no_text):effects['text']=float((logits()-actual).abs().max())
        with patch.object(base.ms_tokenizer,'fuse',lambda x,stage:x):effects['multiscale']=float((logits()-actual).abs().max())
        assert effects['text']>1e-7 and effects['multiscale']>1e-7,effects
        mark('branch_effects',dict(effects=effects,cross_mamba_policy='Frozen/excluded; no AQ-logit contribution claimed'))

        model.eval();model.zero_grad(set_to_none=True)
        (fp_loss,_),_=forward_loss(model,crit,batch,c,False)
        fp_loss.backward();param=next(p for p in model.parameters() if p.requires_grad)
        fp_grad=param.grad.detach().clone();model.zero_grad(set_to_none=True)
        (amp_loss,_),_=forward_loss(model,crit,batch,c,True);amp_loss.backward()
        error=float((fp_loss-amp_loss).abs()/fp_loss.abs().clamp_min(1e-6))
        gradient_error=float((fp_grad-param.grad).norm()/fp_grad.norm().clamp_min(1e-6))
        assert error<.02 and gradient_error<.05,(error,gradient_error)
        # Actual trainer must skip optimizer and scheduler on AMP overflow.
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=c['lr'])
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        scaler=GradScaler('cuda');before=param.detach().clone()
        cfg=dict(c,max_train_batches=1,grad_accum=1)
        def overflow(*a,**k):return param.flatten()[0]*1e38,{}
        with patch.object(training,'objective',overflow):
            train_epoch(model,DataLoader(ds,batch_size=1),crit,opt,sch,scaler,cfg,device,0)
        assert sch.last_epoch==0 and torch.equal(before,param.detach())
        assert scaler.get_scale()<65536
        mark('amp',dict(relative_loss_error=error,relative_gradient_error=gradient_error,overflow_skipped=True))
        del base,encode,param,before,actual
        del opt,sch,scaler,fp_loss,amp_loss,fp_grad,batch;model.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache()

        # Repeated fixed 16-image set: production objective and real backbone.
        small=Subset(ds,range(16));small.attributes=ds.attributes;small.epoch=0
        loader=DataLoader(small,batch_size=2,shuffle=False,num_workers=0)
        fixed_batches=list(loader)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=c['lr'])
        losses=[];obs=[]
        model.train()
        for iteration in range(160):
            b=to_device(fixed_batches[iteration % len(fixed_batches)],device)
            (loss,_),output=forward_loss(model,crit,b,c)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
            if output['observability_logits'] is not None:obs.append(float(output['observability_logits'].sigmoid().std()))
            del loss,output,b
        assert np.mean(losses[-10:])<np.mean(losses[:10])*.8
        if obs:assert np.mean(obs[-10:])>.01
        mark('overfit',dict(initial_loss=float(np.mean(losses[:10])),final_loss=float(np.mean(losses[-10:])),
             obs_std=float(np.mean(obs[-10:])) if obs else None,
             note='Fixed deterministic 16-image learnability; not scientific performance'))
        del opt,model,crit;gc.collect();torch.cuda.empty_cache()

        # Fresh-process production main checkpoint/resume; limited real batches.
        cfg=load_config(ROOT/f'eopar/configs/{args.variant}.json');cfg.pop('output',None)
        cfg.update(batch_size=2,eval_batch_size=2,grad_accum=2,epochs=2,workers=0,
                   augment=False,max_train_batches=4,max_val_batches=2)
        config=out/f'resume_gpu{args.physical_gpu}.json';write_json(config,cfg)
        for folder,extra in [('continuous',[]),('resumed',['--stop-after','1']),('resumed',['--resume'])]:
            command=[sys.executable,'-m','eopar.train.run','--config',str(config),
                     '--output',str(out/f'gpu{args.physical_gpu}_{folder}'),'--diagnostic-proxy',*extra]
            with (out/f'gpu{args.physical_gpu}_resume.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        a,b=[torch.load(out/f'gpu{args.physical_gpu}_{folder}/latest.pth',map_location='cpu',weights_only=False)
             for folder in ('continuous','resumed')]
        maximum=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
        for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=1e-4,rtol=1e-3)
        assert a['scheduler']==b['scheduler'] and a['epoch']==b['epoch']==2
        mark('resume',dict(max_absolute_parameter_error=maximum,atol=1e-4,rtol=1e-3,epochs=2))
        del a,b;gc.collect()

        seed();c=load_config(ROOT/f'eopar/configs/{args.variant}.json')
        train=make_dataset(c,'train');val=make_dataset(c,'val');make_dataset.attributes=train.attributes
        model,crit=build(c,out/'memory');model.to(device);crit.to(device)
        profiles=[]
        flags=[(module,module.use_checkpoint) for module in model.modules() if hasattr(module,'use_checkpoint')]
        assert flags,'No checkpointing switches found; cannot validate this matrix'
        for size in (2,4,8,16):
            for checkpoint in (False,True):
                for module,_ in flags: module.use_checkpoint=checkpoint
                for amp in (False,True):
                    for mode in ('train','validation'):
                        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                        try:
                            if mode=='validation':
                                evaluate(model,DataLoader(val,batch_size=size,num_workers=0),device,val.attributes,limit=4,amp=amp)
                            else:
                                model.train()
                                # Two warmup and two measured batches, no weight update.
                                for index,raw in enumerate(DataLoader(train,batch_size=size,num_workers=0)):
                                    if index==4:break
                                    b=to_device(raw,device);model.zero_grad(set_to_none=True)
                                    (loss,_),result=forward_loss(model,crit,b,c,amp)
                                    if not torch.isfinite(loss):raise FloatingPointError('Profile loss nonfinite')
                                    loss.backward();del b,loss,result
                            profiles.append(dict(batch=size,mode=mode,amp=amp,checkpointing=checkpoint,
                                state='PASS',peak_reserved=torch.cuda.max_memory_reserved()))
                        except torch.cuda.OutOfMemoryError:
                            profiles.append(dict(batch=size,mode=mode,amp=amp,checkpointing=checkpoint,state='OOM_NOT_APPROVED'))
                        finally:
                            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        for module,value in flags:module.use_checkpoint=value
        # Production batch 16: >100 batches and >15% headroom, no silent fallback.
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=c['lr'])
        cfg=dict(c,max_train_batches=105)
        train_epoch(model,DataLoader(train,batch_size=16,num_workers=4),crit,opt,None,GradScaler('cuda'),cfg,device,0)
        train_peak=torch.cuda.max_memory_reserved()
        allocated_before=torch.cuda.memory_allocated()
        evaluate(model,DataLoader(val,batch_size=16,num_workers=4),device,val.attributes,limit=105,amp=c['eval_amp'])
        peak=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
        allocated_after=torch.cuda.memory_allocated()
        assert peak<=.85*total,(peak,total)
        assert allocated_after<=allocated_before+64*1024**2,(allocated_before,allocated_after)
        mark('memory_100_batches',dict(train_batches=105,val_batches=105,batch_size=16,train_peak=train_peak,
             peak_reserved=peak,total_memory=total,headroom=1-peak/total,profiles=profiles,
             allocated_before_validation=allocated_before,allocated_after_validation=allocated_after))
        del opt,model,crit,flags,module;gc.collect();torch.cuda.empty_cache()
        if args.diagnostics_only:
            report.update(state='DIAGNOSTICS_PASS_BLOCKED_DATA',
                blocked='Anatomical validity of fixed-band supervision failed visual review; no full verification epoch or pilot')
            write_json(path,report);return
        config=ROOT/f'eopar/configs/{args.variant}.json'
        # Full single verification epoch retains a 30-epoch schedule horizon.
        with (out/f'gpu{args.physical_gpu}_full_epoch.log').open('w') as log:
            subprocess.run([sys.executable,'-m','eopar.train.run','--config',str(config),
                '--output',str(out/f'full_epoch_{args.variant}'),'--stop-after','1','--verification-run'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        mark('full_epoch_'+args.variant,dict(output=str(out/f'full_epoch_{args.variant}')))
        if report['source_hashes']!=source_hashes():raise ValueError('Source changed during verification')
        report['state']='PASS';write_json(path,report)
    except BaseException as exc:
        report.update(state='FAIL',error=repr(exc));write_json(path,report);raise


if __name__=='__main__':main()
