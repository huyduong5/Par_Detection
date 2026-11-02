"""Materialize val/test clean/x2/x4/x8, resume safely, publish manifest last."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from .core import ROOT,SCALES,atomic_json,degraded,protocol,read_source,safe_relative,sha,write_png


def stratified(rows,n=128,seed=42):
    groups={}
    areas=np.array([r['width']*r['height'] for r in rows]);edges=np.quantile(areas,[.25,.5,.75])
    for r in rows:
        key=(int(np.searchsorted(edges,r['width']*r['height'])),r['image_id'].split('_')[0])
        groups.setdefault(key,[]).append(r['image_id'])
    rng=np.random.default_rng(seed)
    for values in groups.values():rng.shuffle(values)
    keys=sorted(groups,key=lambda k:(k[1],k[0]));chosen=[]
    while len(chosen)<min(n,len(rows)):
        for key in keys:
            if groups[key]:chosen.append(groups[key].pop())
            if len(chosen)>=min(n,len(rows)):break
    return chosen


def generate(pkl_path,images,output,exclusions=None,resume=False,repair=False):
    output=Path(output).resolve();images=Path(images).resolve()
    if output.is_relative_to(images) or images.is_relative_to(output):raise ValueError('Benchmark and source image roots must be disjoint')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.generation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=protocol(pkl_path);spec_path=output/'protocol.json'
        if spec_path.exists():
            if json.loads(spec_path.read_text())!=spec:raise ValueError('Existing protocol differs; use a new benchmark version')
            if not resume:raise FileExistsError('Existing benchmark: --resume to verify/reuse')
        else:atomic_json(spec_path,spec)
        attrs,source=read_source(pkl_path)
        exclusions=json.loads(Path(exclusions).read_text()) if exclusions else {'val':[],'test':[]}
        for split in ('val','test'):
            if set(exclusions.get(split,[]))-{r['image_id'] for r in source[split]}:raise ValueError('Unknown exclusion ID')
        started=time.monotonic();index=dict(state='building',protocol_sha256=sha(spec_path),attributes=attrs,
            pkl_path=str(Path(pkl_path).resolve()),images_path=str(images),splits={},source_images=0)
        atomic_json(output/'manifests/index.json',index)
        visual=[];clipped={}
        for split in ('val','test'):
            manifest=output/f'manifests/{split}.jsonl';tmp=manifest.with_suffix('.jsonl.tmp')
            official=[];dedup=[];split_rows=[]
            clipped[split]={str(s):0 for s in SCALES}
            with tmp.open('w',encoding='utf-8') as sink:
                for i,record in enumerate(source[split]):
                    name=record['image_id'];rel=safe_relative(name);path=(images/str(rel)).resolve()
                    if not path.is_relative_to(images):raise ValueError('Source image escapes image root')
                    raw=path.read_bytes()
                    with Image.open(io.BytesIO(raw)) as im:rgb=im.convert('RGB')
                    w,h=rgb.size;versions={}
                    for scale in SCALES:
                        im,low=degraded(rgb,scale);level='clean' if scale==1 else f'x{scale}'
                        dest=Path(split)/level/(str(rel)+'.png')
                        digest=write_png(output/dest,im,repair)
                        clamp=w//scale<1 or h//scale<1
                        clipped[split][str(scale)]+=int(clamp)
                        versions[str(scale)]=dict(path=dest.as_posix(),sha256=digest,width=w,height=h,
                            low_width=low[0],low_height=low[1],clamped_to_one=clamp)
                    row=dict(record,width=w,height=h,source_path=str(path),source_sha256=hashlib.sha256(raw).hexdigest(),versions=versions)
                    sink.write(json.dumps(row,ensure_ascii=False)+'\n');split_rows.append(row);official.append(name)
                    if name not in set(exclusions.get(split,[])):dedup.append(name)
                    if i%1000==0:
                        atomic_json(output/'generation_status.json',dict(state='building',split=split,processed=i+1,total=len(source[split]),seconds=time.monotonic()-started))
                        print(split,i+1,'/',len(source[split]),flush=True)
                sink.flush();os.fsync(sink.fileno())
            os.replace(tmp,manifest)
            atomic_json(output/f'manifests/{split}_official_ids.json',official)
            atomic_json(output/f'manifests/{split}_dedup_ids.json',dedup)
            index['splits'][split]=dict(path=manifest.relative_to(output).as_posix(),sha256=sha(manifest),count=len(official),dedup_count=len(dedup),
                official_ids_sha256=sha(output/f'manifests/{split}_official_ids.json'),dedup_ids_sha256=sha(output/f'manifests/{split}_dedup_ids.json'))
            index['source_images']+=len(official)
            if split=='val':
                smoke=stratified(split_rows);atomic_json(output/'manifests/val_smoke128_ids.json',smoke)
                preview_ids=stratified(split_rows,64)
                atomic_json(output/'manifests/preview64_ids.json',preview_ids)
                wanted=set(preview_ids);visual=[r for r in split_rows if r['image_id'] in wanted]
        preview=output/'previews';preview.mkdir(exist_ok=True)
        for page in range((len(visual)+7)//8):
            canvas=Image.new('RGB',(4*192,8*220),'white');draw=ImageDraw.Draw(canvas)
            for row,record in enumerate(visual[page*8:page*8+8]):
                for col,scale in enumerate(SCALES):
                    with Image.open(output/record['versions'][str(scale)]['path']) as im:
                        im.thumbnail((192,192));canvas.paste(im,(col*192+(192-im.width)//2,row*220))
                    draw.text((col*192,row*220+192),f"{record['image_id'][:20]} x{scale}",fill='black')
            canvas.save(preview/f'comparison_{page:02d}.png')
        index.update(state='complete',clipped_dimensions=clipped,generated_images=4*index['source_images'],seconds=time.monotonic()-started,
            val_smoke_sha256=sha(output/'manifests/val_smoke128_ids.json'),
            exclusion_policy='Official images retained; separate pre-existing duplicate exclusion IDs')
        atomic_json(output/'manifests/index.json',index)
        atomic_json(output/'generation_status.json',dict(state='complete',images=index['generated_images'],seconds=index['seconds']))
        return index


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--pkl',default=str(ROOT/'MSP60k/SUBMIT/dataset_random.pkl'))
    p.add_argument('--images',default=str(ROOT/'MSP60k/SUBMIT/images'))
    p.add_argument('--output',default=str(ROOT/'data/robustness/par_resolution_v1'))
    p.add_argument('--exclusions',default=str(ROOT/'eopar/research/annotation/deduplicated_eval_exclusions.json'))
    p.add_argument('--resume',action='store_true');p.add_argument('--repair',action='store_true')
    a=p.parse_args();generate(a.pkl,a.images,a.output,a.exclusions,a.resume,a.repair)


if __name__=='__main__':main()
