"""Test is inaccessible until CPU tests and actual validation smoke pass."""
import json
from pathlib import Path
from .core import atomic_json,load_manifest,sha
from .registry import source_fingerprint

def validate(path,registry,manifest):
    gate=json.loads(Path(path).read_text())
    root,index,_=load_manifest(manifest)
    fingerprint,_=source_fingerprint()
    expected=dict(registry_sha256=sha(registry),benchmark_index_sha256=sha(manifest),source_fingerprint=fingerprint)
    if gate.get('state')!='PASS' or any(gate.get(k)!=v for k,v in expected.items()):raise ValueError('Test gate missing, stale or failed')
    for path,digest in gate['evidence'].items():
        if sha(path)!=digest:raise ValueError('Gate evidence changed')
    return gate

def seal(path,registry,manifest,smoke,cpu_report):
    root,index,_=load_manifest(manifest);report=json.loads(Path(cpu_report).read_text())
    if report.get('state')!='PASS':raise ValueError('CPU verification failed')
    fingerprint,_=source_fingerprint();reg=json.loads(Path(registry).read_text())
    if fingerprint!=reg['source_fingerprint']:raise ValueError('Runtime source changed after inventory')
    if report.get('source_fingerprint')!=fingerprint:raise ValueError('CPU tests belong to another source revision')
    evidence={str(Path(cpu_report).resolve()):sha(cpu_report)}
    for tag in ('clean','x2','x4','x8'):
        p=Path(smoke)/f'val_{tag}.json';result=json.loads(p.read_text())
        if result.get('artifact_state')!='complete' or result['samples']!=min(128,index['splits']['val']['count']):raise ValueError('Incomplete validation smoke')
        if result['clean_reference_parity']['state']!='PASS' or result['threshold']!=.5:raise ValueError('Invalid validation parity/threshold')
        if result['benchmark_index_sha256']!=sha(manifest) or result['subset_sha256']!=index['val_smoke_sha256']:raise ValueError('Wrong smoke dataset')
        pred=p.with_suffix('.npz')
        if sha(pred)!=result['predictions_sha256']:raise ValueError('Smoke predictions changed')
        evidence[str(p.resolve())]=sha(p);evidence[str(pred.resolve())]=sha(pred)
    atomic_json(path,dict(state='PASS',registry_sha256=sha(registry),benchmark_index_sha256=sha(manifest),source_fingerprint=fingerprint,
        evidence=evidence,selection='Fixed existing best clean-validation checkpoints; threshold 0.5; no test tuning',
        visual_review='Operator unit tests passed; preview contact sheets retained for inspection. No model selection from previews.'))
