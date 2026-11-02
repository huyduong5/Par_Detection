"""Freeze evaluation sources in an isolated release, then test that exact release."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .core import ROOT,atomic_json,sha
from .registry import source_fingerprint
from eopar.train.runtime import environment_identity

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--registry',required=True);p.add_argument('--release',required=True);p.add_argument('--reports',required=True);a=p.parse_args()
    release=Path(a.release).resolve();reports=Path(a.reports).resolve();reports.mkdir(parents=True,exist_ok=True)
    if release.exists():raise FileExistsError('Release already exists; create a new release instead of mutating it')
    fingerprint,files=source_fingerprint()
    for rel,digest in files.items():
        dest=release/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        if sha(dest)!=digest:raise ValueError('Source changed while freezing')
    for rel in ('eopar/__init__.py',):
        if (ROOT/rel).exists():shutil.copy2(ROOT/rel,release/rel)
    reg=json.loads(Path(a.registry).read_text());reg.update(source_fingerprint=fingerprint,source_files=files,environment=environment_identity(),release=str(release))
    for entry in reg['models']:
        package='_my_model_8' if entry['family']=='base8' else 'aq_visibility'
        bootstrap=ROOT/package/'checkpoints'/f"vmamba_{entry['saved_config'].get('vmamba_variant','small')}_pretrained.pth"
        entry['vision_bootstrap']=dict(path=str(bootstrap),sha256=sha(bootstrap))
    registry=reports/'checkpoint_registry.json';atomic_json(registry,reg)
    # Triton imports query the driver even for CPU construction. Do not hide
    # devices here; preflight never calls .cuda(), .to('cuda') or forward().
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (reports/'cpu_tests.log').open('w') as log:
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s','experiments/par_resolution/tests','-v'],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(reports/'cpu_verification.json',dict(state='PASS' if test.returncode==0 else 'FAIL',source_fingerprint=fingerprint,
        log=str(reports/'cpu_tests.log'),log_sha256=sha(reports/'cpu_tests.log'),scope='Operator, generator, dataset, mock-model production evaluator, resume/artifact/gate, bootstrap and GPU-busy handling. Real GPU inference pending.'))
    if test.returncode:raise SystemExit(test.returncode)
    with (reports/'preflight.log').open('w') as log:
        test=subprocess.run([sys.executable,'-u','-m','experiments.par_resolution.preflight','--registry',str(registry),'--output',str(reports/'real_model_cpu_preflight.json')],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT)
    if test.returncode:raise SystemExit(test.returncode)
    atomic_json(reports/'release_ready.json',dict(state='PASS',release=str(release),source_fingerprint=fingerprint,registry_sha256=sha(registry),
        cpu_report_sha256=sha(reports/'cpu_verification.json'),real_model_cpu_preflight_sha256=sha(reports/'real_model_cpu_preflight.json'),
        decision='Eligible to wait for validation smoke only; test requires the separate validation gate. No training authorized by this release.'))

if __name__=='__main__':main()
