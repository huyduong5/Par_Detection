import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.par_resolution.core import atomic_json,sha
from .metrics import paired_bootstrap,scores

METRICS = ('mA','instance_accuracy','instance_precision','instance_recall','instance_f1_mean','macro_ap')


def report(root):
    rows,details,ci,groups = [],[],[],[]
    for dataset in ('pa100k','rapv1','peta'):
        for seed in (42,43,44):
            paired = []
            for model in ('C3','C3_MS'):
                run = root/dataset/f'{model}_s{seed}'
                if not (run/'test.json').exists():
                    continue
                meta = json.loads((run/'test.json').read_text())
                if meta['state']!='COMPLETE' or sha(run/'test.npz')!=meta['predictions_sha256']:
                    raise ValueError('Incomplete or stale predictions')
                if sha(run/'best.pth')!=meta['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed')
                with np.load(run/'test.npz',allow_pickle=False) as file:
                    a = {k:file[k] for k in file.files}
                m = scores(a['clean'],a['labels'],a['valid'])
                if m!=meta['metrics'] or (meta['evidence']['dataset'],meta['evidence']['variant'],meta['evidence']['seed'])!=(dataset,model,seed):
                    raise ValueError('Prediction scores or provenance differ')
                for subset in ('official','dedup'):
                    entry = m if subset=='official' else meta['dedup_metrics']
                    rows.append(dict(dataset=dataset,model=model,seed=seed,epoch=meta['epoch'],subset=subset,
                        samples=meta['samples'] if subset=='official' else meta['dedup_samples'],
                        checkpoint_sha256=meta['checkpoint_sha256'],prediction_sha256=meta['predictions_sha256'],
                        **{k:None if entry[k] is None else 100*entry[k] for k in METRICS}))
                details.extend(dict(dataset=dataset,model=model,seed=seed,attribute=name,
                    mA=m['per_attribute_mA'][i],ap=m['per_attribute_ap'][i],positive=m['positive_support'][i],
                    negative=m['negative_support'][i]) for i,name in enumerate(a['attributes']))
                paired.append(a)
            if len(paired)==2:
                l,r = paired
                if any(not np.array_equal(l[k],r[k]) for k in ('names','labels','valid','attributes')):
                    raise ValueError('Cannot compare unpaired test predictions')
                ci.append(dict(dataset=dataset,training_seed=seed,
                               **paired_bootstrap(l['clean'],r['clean'],l['labels'],l['valid'])))
        for model in ('C3','C3_MS'):
            for subset in ('official','dedup'):
                group = [r for r in rows if (r['dataset'],r['model'],r['subset'])==(dataset,model,subset)]
                if not group:
                    continue
                aggregate = dict(dataset=dataset,model=model,subset=subset,n_seeds=len(group))
                for k in METRICS:
                    values = [r[k] for r in group if r[k] is not None]
                    aggregate[k+'_mean'] = float(np.mean(values)) if values else None
                    aggregate[k+'_std'] = float(np.std(values,ddof=1)) if len(values)>1 else None
                groups.append(aggregate)
    for name,data in [('test_metrics',rows),('per_attribute',details),('seed_summary',groups)]:
        if data:
            with (root/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
                writer = csv.DictWriter(handle,fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    atomic_json(root/'bootstrap.json',ci)
    lines = ['# Benchmark PAR trên ảnh chuẩn','','C3 và C3_MS dùng cùng preprocessing, 30 epoch và ngưỡng p > 0,5.',
             'F1 tính theo từng ảnh. Bootstrap ghép cặp theo ảnh vì manifest hiện dùng ID ảnh. RAPv1/PETA dùng partition 0.','']
    if not rows:
        lines += ['Đang chờ prediction test hoàn tất.']
    else:
        lines += ['| Dataset | Model | Số seed | mA (%) | Mean-instance F1 (%) | Macro AP (%) |',
                  '|---|---|---:|---:|---:|---:|']
        for group in groups:
            if group['subset']!='official':
                continue
            cells = []
            for key in ('mA','instance_f1_mean','macro_ap'):
                m,s = group[key+'_mean'],group[key+'_std']
                cells.append('—' if m is None else f'{m:.2f}'+('' if s is None else f' ± {s:.2f}'))
            lines.append('| '+ ' | '.join([group['dataset'],group['model'],str(group['n_seeds']),*cells])+' |')
        lines += ['','[Test chính thức và loại trùng](test_metrics.csv) · [Theo thuộc tính](per_attribute.csv) · '
                  '[Theo seed](seed_summary.csv) · [Bootstrap](bootstrap.json)']
    (root/'REPORT_CLEAN_BENCHMARK.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    report(parser.parse_args().output)
