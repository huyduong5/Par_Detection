"""Stage the PA-100K and PETA members of the publisher's OpenPAR archive."""
import argparse
import mmap
from pathlib import Path
import shutil
import struct
import zipfile
import zlib

from experiments.par_resolution.core import atomic_json,sha
from .data import prepare

MEMBERS = {
    'pa100k': ('PA100K/data.zip','PA100K/annotation.zip'),
    'peta': ('PETA/images.zip','PETA/PETA.mat'),
}
SOURCE = 'https://github.com/Event-AHU/OpenPAR#one-step-to-download-the-six-benchmark-datasets'


def stored_member(archive,name,target):
    """Read a complete stored member before the outer central directory arrives."""
    name_bytes = name.encode()
    with archive.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as data:
        pos = data.find(name_bytes)
        while pos>=0:
            if data[pos-30:pos-26]==b'PK\x03\x04':
                break
            pos = data.find(name_bytes,pos+1)
        if pos<0:
            raise ValueError(f'Member header unavailable: {name}')
        _,_,flags,method,_,_,_,_,_,length,extra = struct.unpack('<4s5H3I2H',data[pos-30:pos])
        if length!=len(name_bytes) or flags!=8 or method!=0:
            raise ValueError('Expected stored member with data descriptor')
        start = pos+length+extra
        end = data.find(b'PK\x07\x08',start)
        while end>=0 and end+16<=len(data):
            crc,size,uncompressed = struct.unpack('<III',data[end+4:end+16])
            if size==uncompressed==end-start:
                break
            end = data.find(b'PK\x07\x08',end+4)
        if end<0 or end+16>len(data):
            raise ValueError(f'Member data incomplete: {name}')
        partial = target.with_suffix(target.suffix+'.partial')
        observed = 0
        with partial.open('wb') as output:
            for offset in range(start,end,8*1024**2):
                chunk = data[offset:min(end,offset+8*1024**2)]
                observed = zlib.crc32(chunk,observed)
                output.write(chunk)
        if observed!=crc:
            raise ValueError(f'Archive CRC differs: {name}')
        partial.replace(target)


def unzip(archive,output):
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive member')
            path = output/relative
            if entry.is_dir():
                path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with source.open(entry) as src,path.open('wb') as dst:
                    shutil.copyfileobj(src,dst)


def acquire(archive,output,dataset):
    root = output/dataset
    assets = root/'source'
    assets.mkdir(parents=True,exist_ok=True)
    hashes = {}
    for name in MEMBERS[dataset]:
        target = assets/Path(name).name
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as package,package.open(name) as src,target.open('wb') as dst:
                shutil.copyfileobj(src,dst)
        else:
            stored_member(archive,name,target)
        hashes[name] = sha(target)
        if target.suffix=='.zip':
            unzip(target,assets/target.stem)
    annotation = list(assets.rglob('annotation.mat' if dataset=='pa100k' else 'PETA.mat'))
    if len(annotation)!=1:
        raise ValueError(f'Expected exactly one MAT annotation: {annotation}')
    images = assets/('data' if dataset=='pa100k' else 'images')
    prepare(dataset,annotation[0],images,root)
    atomic_json(root/'acquisition.json',dict(source=SOURCE,archive_members=hashes,
                annotation_sha256=sha(annotation[0]),manifest_sha256=sha(root/'manifest.json')))


if __name__=='__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',choices=MEMBERS,required=True)
    a = p.parse_args()
    acquire(a.archive,a.output,a.dataset)
