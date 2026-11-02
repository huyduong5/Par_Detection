"""Wait for idle GPUs, run verification, and authorize only proven pilots."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from eopar.scripts.queue import idle
from eopar.train.runtime import ROOT,source_hashes,environment_identity,digest,write_json
from eopar.verification.gate import REQUIRED_GPU


def assemble(out):
    cpu=json.loads((out/'cpu.json').read_text())
    data=json.loads((out/'data.json').read_text())
    visual=out/'visual_review.json'
    gpu_paths=[out/f'gpu{i}.json' for i in (0,1)]
    if not visual.is_file() or any(not p.is_file() for p in gpu_paths):return False
    if json.loads(visual.read_text()).get('state')!='PASS':return False
    gpus=[json.loads(p.read_text()) for p in gpu_paths]
    sources,env=source_hashes(),environment_identity()
    if cpu['state']!='PASS' or cpu['source_hashes']!=sources or cpu['environment']!=env:return False
    if data['state'] not in ('PASS','PASS_WITH_DISCLOSED_DUPLICATES'):return False
    if any(g['state']!='PASS' or g['source_hashes']!=sources or g['environment']!=env for g in gpus):return False
    checks={}
    for name in REQUIRED_GPU:
        if name=='both_devices': checks[name]='PASS'
        else:
            matching=[g['checks'][name]['state'] for g in gpus if name in g['checks']]
            checks[name]='PASS' if matching and all(s=='PASS' for s in matching) else 'MISSING'
    if set(checks.values())!={'PASS'}:return False
    config=json.loads((ROOT/'eopar/configs/P3.json').read_text())
    inputs={str(ROOT/config[k]):digest(ROOT/config[k]) for k in ('pkl_path','region_map')}
    lock=ROOT/'eopar/configs/pretrained_lock.json';inputs[str(lock)]=digest(lock)
    inputs.update(json.loads(lock.read_text()))
    images=out/'image_files.json'
    if digest(images)!=data.get('image_files_sha256'):return False
    inputs[str(images)]=digest(images)
    inputs.update(json.loads(images.read_text()))
    report=dict(state='PASS',authorization='pilot_only',release_root=str(ROOT.resolve()),
        source_hashes=sources,environment=env,cpu='PASS',data='PASS',gpu_checks=checks,
        config_hashes={v:digest(ROOT/f'eopar/configs/{v}.json') for v in ('B1','P3')},input_hashes=inputs,
        duplicate_policy='Official benchmark unchanged; supplementary deduplication required in scientific reporting')
    write_json(out/'pilot_gate.json',report)
    return True


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--wait-hours',type=float,default=72.)
    p.add_argument('--launch-pilots-on-pass',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    status=out/f'worker{args.gpu}.json';variant='B1' if args.gpu==0 else 'P3';child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write_json(status,dict(state='stopped',gpu=args.gpu));raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with Path(f'/tmp/eopar-{os.getuid()}-gpu{args.gpu}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            cpu=json.loads((out/'cpu.json').read_text());data=json.loads((out/'data.json').read_text())
            if cpu['state']!='PASS' or cpu['source_hashes']!=source_hashes() or data['state']=='FAIL':
                raise ValueError('CPU/data verification is stale or failed')
            deadline=time.monotonic()+args.wait_hours*3600
            while True:
                write_json(status,dict(state='waiting_for_idle_gpu_for_verification',gpu=args.gpu,pid=os.getpid(),timestamp=time.time()))
                if idle(args.gpu,22000):
                    time.sleep(10)
                    if idle(args.gpu,22000):break
                if time.monotonic()>deadline:
                    write_json(status,dict(state='wait_timeout',gpu=args.gpu));return
                time.sleep(30)
            write_json(status,dict(state='verification_running',gpu=args.gpu))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            visual=json.loads((out/'visual_review.json').read_text())
            diagnostics=visual.get('state')!='PASS'
            command=[sys.executable,'-m','eopar.verification.gpu','--physical-gpu',str(args.gpu),
                     '--variant',variant,'--output',str(out)]
            if diagnostics:command.append('--diagnostics-only')
            with (out/f'gpu{args.gpu}.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                rc=child.wait()
            if rc:raise RuntimeError(f'GPU verification failed, exit={rc}')
            write_json(status,dict(state='diagnostics_done_pilot_blocked_by_data' if diagnostics else 'verification_pass_waiting_for_other_gpu',gpu=args.gpu))
        except BaseException as exc:
            if not isinstance(exc,SystemExit):write_json(status,dict(state='failed',gpu=args.gpu,error=repr(exc)))
            raise
    # Release GPU lock before queuing any pilot. A single atomic claim avoids
    # duplicate launches when both supervisors finish simultaneously.
    if assemble(out) and args.launch_pilots_on_pass:
        try:(out/'pilot_launch_claim').mkdir()
        except FileExistsError:return
        for gpu,v in ((0,'B1'),(1,'P3')):
            with (out/f'pilot_queue{gpu}.log').open('w') as log:
                subprocess.Popen([sys.executable,'-m','eopar.scripts.queue','--gpu',str(gpu),'--variant',v,
                    '--gate-report',str(out/'pilot_gate.json'),'--run-pilot','--output',str(out/f'approved_pilot_{v}')],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)


if __name__=='__main__':main()
