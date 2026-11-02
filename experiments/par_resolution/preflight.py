"""Strict CPU deserialization and real architecture loading; never uses a GPU."""
import argparse
import json
import gc
from pathlib import Path
from .core import ROOT,atomic_json,sha
from .adapters import load
from .registry import source_fingerprint

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--registry',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    reg=json.loads(Path(a.registry).read_text());entries=[]
    for group in ('C0','C3','C3_MS','C4b','C4b_MS'):
        entry=next((e for e in reg['models'] if e['group']==group),None)
        if entry is None:continue
        index=dict(attributes=entry['attributes'],pkl_path=entry['saved_config']['pkl_path'],protocol_split_sha256=entry['pkl_sha256'])
        model,config,tf=load(entry,index,Path(a.output).parent/'preflight_config')
        entries.append(dict(model=entry['id'],state='PASS',parameters=sum(p.numel() for p in model.parameters()),all_frozen=all(not p.requires_grad for p in model.parameters()),eval=not model.training))
        print('Strict real-model load PASS',entry['id'],flush=True);del model;gc.collect()
    atomic_json(a.output,dict(state='PASS',models=entries,source_fingerprint=source_fingerprint()[0],registry_sha256=sha(a.registry),note='CPU strict weight loading only; no GPU forward claimed'))

if __name__=='__main__':main()
