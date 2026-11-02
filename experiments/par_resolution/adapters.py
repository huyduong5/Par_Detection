"""Dedicated, strict adapters. Old AQ checkpoints are not EOPAR results."""
import dataclasses
import importlib
from pathlib import Path
import torch
from .core import ROOT,sha


def load(entry,index,output):
    checkpoint=Path(entry['checkpoint'])
    if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
    if sha(checkpoint)!=entry['checkpoint_sha256']:raise ValueError('Checkpoint changed after registry lock')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    saved=payload['config']
    if not isinstance(saved,dict):raise ValueError('Expected explicit saved configuration')
    family=entry['family'];attrs=index['attributes']
    if payload.get('attributes') is not None and list(payload['attributes'])!=attrs:raise ValueError('Checkpoint attribute order mismatch')
    if family=='eopar':
        from eopar.train.run import build
        model,_=build(saved,output,initialize=False)
        model.load_state_dict(payload['model'],strict=True)
        config=model.baseline.config
        tfmodule=importlib.import_module('aq_visibility.dataset.clip_dataset')
    else:
        package='_my_model_8' if family=='base8' else 'aq_visibility' if family=='aq' else None
        if package is None:raise ValueError('Unknown checkpoint adapter')
        cfgmodule=importlib.import_module(package+'.config')
        keys={f.name for f in dataclasses.fields(cfgmodule.CLIMPConfig)}
        args={k:v for k,v in saved.items() if k in keys}
        args.update(output_dir=str(output),use_ddp=False)
        config=cfgmodule.CLIMPConfig(**args)
        # C0 training attached this flag dynamically, outside its dataclass.
        config.use_domain_tokens=bool(saved.get('use_domain_tokens',False))
        bootstrap=entry.get('vision_bootstrap')
        if bootstrap is None or sha(bootstrap['path'])!=bootstrap['sha256']:raise ValueError('Missing/changed local vision bootstrap weights')
        config.vmamba_pretrained_path=bootstrap['path']
        if config.dataset_name!='MSP60k' or Path(config.pkl_path).resolve()!=Path(index['pkl_path']).resolve():
            raise ValueError('Checkpoint dataset binding differs from benchmark')
        if sha(config.pkl_path)!=index['protocol_split_sha256']:raise ValueError('Label pickle checksum differs')
        cls=importlib.import_module(package+'.model.climp_par').CLIMPPAR
        model=cls(config) if family=='base8' else cls(config,num_attrs=len(attrs))
        state=payload['model_state_dict']
        if state and all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
        model.load_state_dict(state,strict=True)
        tfmodule=importlib.import_module(package+'.dataset.clip_dataset')
    if sha(config.pkl_path)!=index['protocol_split_sha256']:raise ValueError('Label checksum mismatch')
    _,transform=tfmodule.get_clip_transforms(config.img_height,config.img_width)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,config,transform


def forward(model,family,images,tokens,attributes):
    value=model(images,tokens,attributes)
    if family=='eopar':return value['attr_logits'],value['observability_logits']
    logits,features=value
    # Legacy visibility is not silently renamed to EOPAR observability.
    return logits,None
