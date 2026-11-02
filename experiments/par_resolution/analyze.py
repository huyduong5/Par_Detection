"""Rebuild measured tables from hashed prediction artifacts; never invent rows."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .core import atomic_json,sha
from eopar.eval.metrics import par_metrics

def paired_bootstrap(clean,degraded,labels,draws=1000,seed=42):
    if clean.shape!=degraded.shape or clean.shape!=labels.shape:raise ValueError('Paired arrays must match')
    rng=np.random.default_rng(seed);values=[];n=len(labels)
    # Attribute confusion counts, weighted by an image-level bootstrap draw.
    y=labels==1;p0=clean>.5;p1=degraded>.5
    counts=np.concatenate([(p0&y),(~p0&~y),(p1&y),(~p1&~y),y,~y],axis=1).astype(float)
    for _ in range(draws):
        weights=np.bincount(rng.integers(0,n,n),minlength=n)
        tp0,tn0,tp1,tn1,pos,neg=np.split(weights@counts,6)
        values.append(float(50*np.mean((tp1-tp0)/np.maximum(pos,1)+(tn1-tn0)/np.maximum(neg,1))))
    return dict(delta_mA_pp=float(100*(par_metrics(degraded,labels)['mA']-par_metrics(clean,labels)['mA'])),
        ci95_pp=np.quantile(values,[.025,.975]).tolist(),draws=draws,seed=seed,
        resampling_unit='image',limitation='Identity/camera metadata unavailable; image bootstrap is not seed variability')

def analyze(root,output,draws=1000):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];attributes=[];strata=[];details={}
    for marker in sorted(root.glob('*/test_complete.json')):
        folder=marker.parent;levels={};artifacts={}
        for scale,tag in ((1,'clean'),(2,'x2'),(4,'x4'),(8,'x8')):
            p=folder/f'test_{tag}.json';result=json.loads(p.read_text());npz=p.with_suffix('.npz')
            if sha(npz)!=result['predictions_sha256']:raise ValueError('Prediction digest differs')
            with np.load(npz,allow_pickle=False) as f:data={k:f[k] for k in f.files}
            if par_metrics(data['clean'],data['labels'])!=result['par']:raise ValueError('Recomputed metrics differ')
            if levels:
                reference=artifacts[1]
                if not np.array_equal(reference['names'],data['names']) or not np.array_equal(reference['labels'],data['labels']):raise ValueError('Unpaired scale predictions')
            levels[scale]=result;artifacts[scale]=data
            for i,name in enumerate(data['attributes']):attributes.append(dict(model=result['model_id'],seed=result['seed'],scale=scale,attribute=str(name),mA=100*result['par']['per_attribute_mA'][i],positive=result['par']['positive_support'][i],negative=result['par']['negative_support'][i]))
            for group,size in (('native',np.minimum(data['width'],data['height'])),('intermediate',np.minimum(data['low_width'],data['low_height']))):
                for lower,upper in ((1,16),(16,32),(32,64),(64,128),(128,1000000)):
                    mask=(size>=lower)&(size<upper)
                    if mask.any():strata.append(dict(model=result['model_id'],seed=result['seed'],scale=scale,group=group,minimum=lower,maximum=upper,n=int(mask.sum()),mA=100*par_metrics(data['clean'][mask],data['labels'][mask])['mA']))
        first=levels[1];row=dict(model=first['model_id'],group=first.get('group',first['model_id']),seed=first['seed'])
        for s in (1,2,4,8):
            for key in ('mA','instance_accuracy','instance_precision','instance_recall','instance_f1_mean'):row[f'{key}_x{s}']=100*levels[s]['par'][key]
        row['drop_x8_pp']=row['mA_x1']-row['mA_x8'];row['mean_degraded_mA']=np.mean([row[f'mA_x{s}'] for s in (2,4,8)])
        rows.append(row)
        details[first['model_id']]={str(s):paired_bootstrap(artifacts[1]['clean'],artifacts[s]['clean'],artifacts[1]['labels'],draws) for s in (2,4,8)} if draws else {}
    def write(name,values):
        if values:
            with (output/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    write('comparison.csv',rows);write('per_attribute.csv',attributes);write('resolution_strata.csv',strata)
    aggregate=[]
    for group in sorted({r['group'] for r in rows}):
        members=[r for r in rows if r['group']==group];entry=dict(group=group,seeds=len(members))
        for key in members[0]:
            if key not in ('model','group','seed'):
                values=[r[key] for r in members];entry[key+'_mean']=float(np.mean(values));entry[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
        aggregate.append(entry)
    write('seed_summary.csv',aggregate)
    atomic_json(output/'paired_bootstrap.json',details)
    lines=['# Baseline comparison — resolution degradation','', 'All scores are measured from saved predictions; threshold > 0.5. Drops are percentage points.','', '| Method | seed | mA clean | mA ×2 | mA ×4 | mA ×8 | drop ×8 | mean-instance F1 ×8 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['seed']} | {r['mA_x1']:.2f} | {r['mA_x2']:.2f} | {r['mA_x4']:.2f} | {r['mA_x8']:.2f} | {r['drop_x8_pp']:.2f} | {r['instance_f1_mean_x8']:.2f} |")
    if not rows:lines+=['','No completed test evaluations yet.']
    lines+=['','External methods and EOPAR are not evaluated until compatible, verified checkpoints exist.','Clean is the decoded reference, not a claim of high-resolution ground truth.','Image bootstrap confidence intervals and between-seed standard deviations are separate.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        for r in rows:ax.plot([1,2,4,8],[r[f'mA_x{s}'] for s in (1,2,4,8)],marker='o',label=f"{r['model']} (s{r['seed']})")
        ax.set(xlabel='Resolution reduction factor',ylabel='mA (%)',xticks=[1,2,4,8]);ax.legend(fontsize=7);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(output/'resolution_curve.png',dpi=180);plt.close(fig)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args();analyze(a.results,a.output,a.bootstrap)
