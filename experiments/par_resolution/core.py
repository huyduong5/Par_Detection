import hashlib
import io
import json
import os
import pickle
import uuid
from pathlib import Path, PurePosixPath
import PIL
from PIL import Image
import numpy as np

SCALES=(1,2,4,8)
ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def safe_relative(name):
    p=PurePosixPath(str(name))
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in str(name):
        raise ValueError(f'Unsafe image ID: {name}')
    return p


def degraded(image,scale):
    if type(scale) is not int or scale not in SCALES:raise ValueError('scale must be 1/2/4/8')
    rgb=image.convert('RGB').copy()
    w,h=rgb.size
    low=(max(1,w//scale),max(1,h//scale))
    if scale==1:return rgb,low
    # No additional Gaussian kernel, noise, recompression, crop or EXIF transpose.
    lr=rgb.resize(low,Image.Resampling.BICUBIC,reducing_gap=None)
    return lr.resize((w,h),Image.Resampling.BICUBIC,reducing_gap=None),low


def protocol(pkl_path):
    return dict(version='par_resolution_v1',scales=list(SCALES),pillow=PIL.__version__,
        jpeg_decoder=getattr(Image.core,'jpeglib_version',None),rgb_policy='PIL convert RGB; same orientation as baseline; no EXIF transpose',
        resample='Pillow BICUBIC',reducing_gap=None,rounding='max(1,floor(dimension/scale))',
        restore='original decoded width,height before model preprocessing',format='PNG',compress_level=6,
        cascading=False,extra_blur=False,extra_noise=False,super_resolution=False,
        split_sha256=sha(pkl_path),threshold=.5,checkpoint_selection='best clean validation mA; fixed before test',
        hypotheses=['Multiscale may retain fine detail','Small attributes may degrade more',
                    'Observability may track evidence loss; not assumed true'],
        terminology='Resolution-degradation robustness; NOT ZSSR; no unseen-degradation claim')


def read_source(pkl_path):
    with open(pkl_path,'rb') as f:data=pickle.load(f)
    attrs=next((list(getattr(data,k)) for k in ('attributes','attr_name','attr_words') if hasattr(data,k)),None)
    if not attrs or len(attrs)!=len(set(attrs)):raise ValueError('Invalid attribute definitions')
    result={}
    for split in ('val','test'):
        idx=data.partition[split]
        if isinstance(idx,list):idx=idx[0]
        rows=[]
        for i in np.asarray(idx).ravel():
            name=str(data.image_name[i]);safe_relative(name)
            label=np.asarray(data.label[i])
            if label.shape!=(len(attrs),) or not np.isin(label,[0,1]).all():raise ValueError('Expected binary PAR labels')
            rows.append(dict(image_id=name,labels=label.astype(int).tolist()))
        if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs within split')
        result[split]=rows
    if {r['image_id'] for r in result['val']} & {r['image_id'] for r in result['test']}:
        raise ValueError('Validation/test name overlap')
    return [str(a) for a in attrs],result


def write_png(path,image,repair=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        try:
            with Image.open(path) as old:
                old.load();same=old.format=='PNG' and old.mode=='RGB' and old.size==image.size and np.array_equal(np.asarray(old),np.asarray(image))
        except Exception:same=False
        if same:return sha(path)
        if not repair:raise ValueError(f'Existing output corrupt/different: {path}; explicit --repair required')
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        image.save(tmp,format='PNG',compress_level=6,optimize=False)
        with Image.open(tmp) as check:
            check.load()
            if check.size!=image.size or not np.array_equal(np.asarray(check),np.asarray(image)):raise ValueError('PNG roundtrip mismatch')
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)
    return sha(path)


def load_manifest(path,verify=True):
    path=Path(path);index=json.loads(path.read_text())
    if index.get('state')!='complete':raise ValueError('Benchmark is not complete')
    root=path.parent.parent
    if sha(root/'protocol.json')!=index['protocol_sha256']:raise ValueError('Protocol hash mismatch')
    rows=[]
    for split in ('val','test'):
        p=root/index['splits'][split]['path']
        if verify and sha(p)!=index['splits'][split]['sha256']:raise ValueError('Manifest hash mismatch')
        with p.open() as f:rows.extend(dict(json.loads(line),split=split) for line in f)
    return root,index,rows
