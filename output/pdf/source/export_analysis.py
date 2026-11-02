"""Export paired attribute and native-resolution diagnostics from frozen validation predictions."""
from pathlib import Path
import csv,json
import numpy as np
from metric_reference import par_metrics
HERE=Path(__file__).resolve().parent
folder=HERE/'data/resolution_validation_C3'
a={}
for scale,tag in [(1,'clean'),(2,'x2'),(4,'x4'),(8,'x8')]:
    with np.load(folder/f'val_{tag}.npz',allow_pickle=False) as f:a[scale]={k:f[k] for k in f.files}
rows=[]
for j,name in enumerate(a[1]['attributes']):
    clean=par_metrics(a[1]['clean'],a[1]['labels']);row=dict(attribute=str(name),positive=clean['positive_support'][j],negative=clean['negative_support'][j])
    for s in [1,2,4,8]:row[f'mA_x{s}_percent']=100*par_metrics(a[s]['clean'],a[s]['labels'])['per_attribute_mA'][j]
    row['drop_x8_pp']=row['mA_x1_percent']-row['mA_x8_percent'];rows.append(row)
with (HERE/'attribute_resolution.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
strata=[]
for scale in [1,2,4,8]:
    d=a[scale]
    for group,size in [('native',np.minimum(d['width'],d['height'])),('intermediate',np.minimum(d['low_width'],d['low_height']))]:
        for lo,hi in [(1,16),(16,32),(32,64),(64,128),(128,1000000)]:
            mask=(size>=lo)&(size<hi)
            if mask.any():strata.append(dict(scale=scale,group=group,min_dimension_ge=lo,min_dimension_lt=hi,n=int(mask.sum()),mA_percent=100*par_metrics(d['clean'][mask],d['labels'][mask])['mA']))
with (HERE/'resolution_strata.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=strata[0].keys());w.writeheader();w.writerows(strata)
print('Exported 57 attributes with positive/negative supports and resolution strata; validation only.')
