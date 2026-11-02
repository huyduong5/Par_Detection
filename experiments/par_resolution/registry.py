"""Inventory completed historical checkpoints without touching running jobs."""
import argparse
import json
from pathlib import Path
import torch
from .core import ROOT,atomic_json,read_source,sha

CANDIDATES=[('C0','C0','base8'),('C3','C3','aq'),('C3_s43','C3','aq'),('C3_MS','C3_MS','aq'),('C3_MS_s43','C3_MS','aq'),('C4b_bs16','C4b','aq'),('C4b_s43','C4b','aq'),('C4b_MS_bs16','C4b_MS','aq'),('C4b_MS_s43','C4b_MS','aq')]

def source_fingerprint():
    files={}
    for folder in ('experiments/par_resolution','aq_visibility','_my_model_8','VMamba','mamba','eopar/eval','eopar/models','eopar/train','eopar/losses','eopar/datasets'):
        for p in sorted((ROOT/folder).rglob('*.py')):
            if not {'__pycache__','.git','output','outputs','releases','build'} & set(p.relative_to(ROOT).parts):files[str(p.relative_to(ROOT))]=sha(p)
    import hashlib
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files

def inventory(output):
    attrs,_=read_source(ROOT/'MSP60k/SUBMIT/dataset_random.pkl');models=[];missing=[]
    for directory,group,family in CANDIDATES:
        folder=ROOT/'aq_visibility/output'/directory;checkpoint=folder/'best_model.pth'
        finished=(folder/'checkpoint_epoch30.pth').exists() or (folder/'DONE').exists()
        if not checkpoint.exists() or not finished:
            missing.append(dict(id=directory,reason='No fixed completed checkpoint'));continue
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=payload['config']
        if not isinstance(config,dict):raise ValueError('Checkpoint config must be a mapping')
        pkl=Path(config['pkl_path'])
        if not pkl.is_absolute():pkl=ROOT/pkl
        cpattrs,_=read_source(pkl)
        if cpattrs!=attrs:raise ValueError('Checkpoint label order differs')
        if group.startswith('C4b') and (config['batch_size']!=16 or config['grad_accum_steps']!=4):
            raise ValueError(f'{directory}: expected matched 16x4')
        models.append(dict(id=directory,group=group,family=family,seed=int(config.get('seed',42)),
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epoch=payload['epoch'],
            completion_evidence=str(folder/('checkpoint_epoch30.pth' if (folder/'checkpoint_epoch30.pth').exists() else 'DONE')),
            saved_config=config,pkl_sha256=sha(pkl),attributes=attrs,
            attribute_binding='Saved training pickle path and current 57-attribute order; historical checkpoints contain no training-time label checksum. This limitation is explicit.',
            degradation_exposure='Not verified; no claim of unseen or zero-shot degradation'))
        del payload
        print('Registered',directory,flush=True)
    fingerprint,files=source_fingerprint()
    from eopar.train.runtime import environment_identity
    value=dict(schema=1,models=models,missing=missing,source_fingerprint=fingerprint,source_files=files,environment=environment_identity(),
        external_methods={x:'NOT_EVALUATED: compatible MSP60K 57-attribute checkpoint and label order not yet verified' for x in ('MambaPAR','PromptPAR','LLM-PAR','FOCUS')},
        eopar='NOT_EVALUATED: no eligible tested trained checkpoint; training remains blocked',
        title='Baseline comparison under resolution degradation')
    atomic_json(output,value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args();inventory(a.output)
