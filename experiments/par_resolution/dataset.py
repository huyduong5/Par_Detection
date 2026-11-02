import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .core import load_manifest,sha,SCALES


class ResolutionDataset(Dataset):
    def __init__(self,degradation_manifest,split,scale,transform,subset=None,domain_tokens_path=None,embed_dim=768):
        if split not in ('val','test') or scale not in SCALES:raise ValueError('Invalid split/scale')
        self.root,self.index,rows=load_manifest(degradation_manifest)
        self.attributes=self.index['attributes'];self.split=split;self.scale=scale;self.transform=transform
        rows=[r for r in rows if r['split']==split]
        if subset is not None:
            ids=json.loads(Path(subset).read_text())
            if len(set(ids))!=len(ids) or set(ids)-{r['image_id'] for r in rows}:raise ValueError('Invalid subset IDs')
            keep=set(ids);rows=[r for r in rows if r['image_id'] in keep]
        if not rows:raise ValueError('Empty evaluation selection')
        self.rows=rows;self.embed_dim=embed_dim
        self.domain_tokens=torch.load(domain_tokens_path,map_location='cpu',weights_only=False) if domain_tokens_path else None

    def __len__(self):return len(self.rows)

    def token(self,row):
        value=torch.zeros(32,self.embed_dim)
        if self.domain_tokens is not None:
            name=row['image_id'].split('/')[0] if '/' in row['image_id'] else Path(row['source_path']).parent.name
            if name in self.domain_tokens:value=self.domain_tokens[name]['mean']
            else:
                for key,record in self.domain_tokens.items():
                    if key!='_metadata' and key in row['source_path']:value=record['mean'];break
        return value

    def __getitem__(self,i):
        row=self.rows[i];version=row['versions'][str(self.scale)];path=self.root/version['path']
        if not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('Output escapes benchmark root')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=version['sha256']:raise ValueError(f'Image checksum mismatch: {path}')
        with Image.open(io.BytesIO(raw)) as im:
            if im.format!='PNG' or im.size!=(row['width'],row['height']):raise ValueError('Image format/geometry mismatch')
            image=im.convert('RGB')
        if self.transform:image=self.transform(image)
        return dict(image=image,labels=torch.tensor(row['labels'],dtype=torch.float32),
                    names=row['image_id'],domain=self.token(row))
