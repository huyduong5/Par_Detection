"""Lock clean-image splits against the original annotation and decoded RGB assets."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.par_resolution.core import atomic_json, sha

COUNTS = {'pa100k': (80000,10000,10000,26), 'rapv1': (29941,3327,8317,51),
          'peta': (9500,1900,7600,35)}
PETA_ORDER = [10,18,19,30,15,7,9,11,14,21,26,29,32,33,34,6,8,12,25,27,31,13,23,24,28,
              4,5,17,20,22,0,1,2,3,16]


def strings(items):
    return [str(v) for v in np.asarray(items,dtype=object).ravel()]


def validation_partition(indices, labels, count, seed=42):
    """Use the benchmark trainval pool for a deterministic multilabel holdout."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    split = MultilabelStratifiedShuffleSplit(n_splits=1,test_size=count,random_state=seed)
    train,val = next(split.split(indices,(labels[indices]==1).astype(int)))
    if len(val)!=count:
        # Iterative stratification can vary the realized size. Keep its order of
        # membership and use a seeded tie-break when correcting the quota.
        rng = np.random.default_rng(seed)
        rng.shuffle(train)
        rng.shuffle(val)
        if len(val)<count:
            missing = count-len(val)
            val,train = np.r_[val,train[:missing]],train[missing:]
        else:
            train,val = np.r_[train,val[count:]],val[:count]
    return indices[np.sort(train)],indices[np.sort(val)]


def annotation(dataset, path):
    from scipy.io import loadmat
    raw = loadmat(path, simplify_cells=True)
    if dataset=='pa100k':
        attrs = strings(raw['attributes'])
        result = {part:(strings(raw[f'{part}_images_name']),np.asarray(raw[f'{part}_label']))
                  for part in ('train','val','test')}
        return attrs,result
    original = loadmat(path)
    if dataset=='rapv1':
        rap = original['RAP_annotation'][0][0]
        names = [x[0][0] for x in rap[5]]
        attrs = [x[0][0] for x in rap[3]][:51]
        labels = rap[1][:,:51]
        fold = rap[0][0][0][0][0]
        trainval,test = (np.asarray(x).ravel().astype(int)-1 for x in (fold[0],fold[1]))
        train,val = validation_partition(trainval,labels,3327)
    else:
        peta = original['peta'][0][0]
        names = [f'{i+1:05}.png' for i in range(19000)]
        attrs = [peta[1][i][0][0] for i in PETA_ORDER]
        labels = peta[0][:,4:][:,PETA_ORDER]
        fold = peta[3][0][0][0][0]
        train,val,test = (np.asarray(x).ravel().astype(int)-1 for x in (fold[0],fold[1],fold[2]))
    parts = dict(train=train,val=val,test=test)
    joined = np.concatenate(list(parts.values()))
    if len(joined)!=len(names) or sorted(joined.tolist())!=list(range(len(names))):
        raise ValueError('Official partition does not cover image indices exactly once')
    return [str(a) for a in attrs],{part:([str(names[i]) for i in ids],labels[ids]) for part,ids in parts.items()}


def image_index(root):
    paths = defaultdict(list)
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
            paths[p.name].append(p)
    return paths


def prepare(dataset, annotations, root, out):
    attrs, parts = annotation(dataset,annotations)
    *counts,num_attrs = COUNTS[dataset]
    if len(attrs)!=num_attrs or len(set(attrs))!=len(attrs):
        raise ValueError('Attribute order/count invalid')
    by_basename = image_index(root)
    tasks = []
    seen = set()
    for (split,(ids,labels)),count in zip(parts.items(),counts,strict=True):
        if (len(ids)!=count or labels.shape!=(count,num_attrs)
                or not np.isin(labels,[-1,0,1,2]).all()):
            raise ValueError(f'Unexpected annotation shape or label values for {split}')
        for name,label in zip(ids,labels,strict=True):
            if name in seen or Path(name).name!=name or len(by_basename[name])!=1:
                raise ValueError(f'Missing, ambiguous or repeated image: {name}')
            seen.add(name)
            tasks.append((split,name,by_basename[name][0],label))

    def inspect(item):
        split,name,path,label = item
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Image escapes dataset root')
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.convert('RGB').load()
        valid = np.isin(label,[0,1])
        return dict(split=split,id=name,path=str(path.relative_to(root)),label=np.where(valid,label,0).astype(int).tolist(),
                    valid=valid.astype(int).tolist(),sha256=hashlib.sha256(data).hexdigest())

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(inspect,tasks))
    out.mkdir(parents=True,exist_ok=True)
    overlap = defaultdict(set)
    for row in rows:
        overlap[row['sha256']].add(row['split'])
    excluded = [r['id'] for r in rows if r['split']=='test' and len(overlap[r['sha256']])>1]
    support = {}
    for split in parts:
        y = np.asarray([r['label'] for r in rows if r['split']==split])
        mask = np.asarray([r['valid'] for r in rows if r['split']==split],bool)
        support[split] = {'positive':(mask&(y==1)).sum(0).tolist(),
                          'negative':(mask&(y==0)).sum(0).tolist(),'unknown':(~mask).sum(0).tolist()}
    atomic_json(out/'manifest.json',dict(dataset=dataset,attributes=attrs,
        counts=dict(Counter(r['split'] for r in rows)),image_root=str(root.resolve()),
        annotation_sha256=sha(annotations),annotation=str(annotations.resolve()),
        partition=None if dataset=='pa100k' else 0,support=support,
        cross_split_duplicate_hashes=sum(len(s)>1 for s in overlap.values()),test_dedup_exclusions=excluded,rows=rows))
    print(dataset,dict(Counter(r['split'] for r in rows)),len(excluded),sha(out/'manifest.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',choices=COUNTS,required=True)
    for flag in ('annotation','images','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    prepare(a.dataset,a.annotation,a.images,a.output)
