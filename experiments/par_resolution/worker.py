"""Evaluation-only worker: wait for complete data and an idle GPU, never kill other jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .core import ROOT,atomic_json,sha
from .gate import seal,validate
from .registry import source_fingerprint

def idle(gpu,need_mb=18000):
    try:
        row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        memory,util=map(int,row.split(','))
        apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        return memory>=need_mb and util<=5 and not apps
    except (ValueError,OSError,subprocess.SubprocessError):return False

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--registry',required=True);p.add_argument('--manifest',required=True);p.add_argument('--cpu-report',required=True)
    p.add_argument('--output',required=True);p.add_argument('--wait-hours',type=float,default=72);p.add_argument('--poll-seconds',type=float,default=60)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);status=out/f'worker{a.gpu}.json'
    reg=json.loads(Path(a.registry).read_text());entries=reg['models'][a.gpu::2];deadline=time.monotonic()+3600*a.wait_hours;child=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    def stop(signum,frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        atomic_json(status,dict(state='stopped',gpu=a.gpu,pid=os.getpid()));raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(command,logfile):
        nonlocal child
        with logfile.open('a') as f:
            child=subprocess.Popen([sys.executable,'-u','-m',*command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    def evaluation(entry,split,dest,subset=None,gate=None):
        command=['experiments.par_resolution.evaluate','--registry',a.registry,'--model-id',entry['id'],'--degradation-manifest',a.manifest,'--split',split,'--output',str(dest),'--batch-size','8','--workers','2']
        if subset:command+=['--subset',str(subset)]
        if gate:command+=['--test-gate',str(gate)]
        run(command,out/f"{entry['id']}_{split}_{'smoke' if subset else 'full'}.log")
    lockfile=Path('/tmp')/f'eopar-{os.getuid()}-gpu{a.gpu}.lock'
    gate=out/'test_gate.json';manifest=Path(a.manifest)
    try:
        for entry in entries:
            while True:
                if time.monotonic()>deadline:raise TimeoutError('GPU/data wait timeout')
                if source_fingerprint()[0]!=reg['source_fingerprint']:raise ValueError('Registered runtime source changed')
                data_ready=manifest.exists() and json.loads(manifest.read_text()).get('state')=='complete'
                atomic_json(status,dict(state='waiting_for_idle_gpu' if data_ready else 'waiting_for_dataset',gpu=a.gpu,model=entry['id'],pid=os.getpid(),timestamp=time.time()))
                if data_ready:
                    with lockfile.open('a') as lock:
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:pass
                        else:
                            if idle(a.gpu):
                                time.sleep(5)
                                if idle(a.gpu):
                                    with (out/'.test_gate.lock').open('a') as gate_lock:
                                        try:fcntl.flock(gate_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                                        except BlockingIOError:pass
                                        else:
                                            if not gate.exists():
                                                smoke_entry=next((e for e in reg['models'] if e['id']=='C3_MS'),reg['models'][0]);smoke=out/'validation_smoke'
                                                atomic_json(status,dict(state='validation_smoke_running',gpu=a.gpu,model=smoke_entry['id'],pid=os.getpid()))
                                                evaluation(smoke_entry,'val',smoke,manifest.parent/'val_smoke128_ids.json')
                                                seal(gate,a.registry,a.manifest,smoke,a.cpu_report)
                                    if gate.exists():
                                        validate(gate,a.registry,a.manifest)
                                        atomic_json(status,dict(state='evaluation_running',gpu=a.gpu,model=entry['id'],pid=os.getpid()))
                                        evaluation(entry,'val',out/'full_validation'/entry['id'])
                                        evaluation(entry,'test',out/'results'/entry['id'],gate=gate)
                                        break
                time.sleep(a.poll_seconds)
        # Analyze under a shared lock; a second worker can refresh once more after finishing.
        with (out/'.analysis.lock').open('a') as analysis_lock:
            fcntl.flock(analysis_lock,fcntl.LOCK_EX)
            run(['experiments.par_resolution.analyze','--results',str(out/'results'),'--output',str(out/'analysis')],out/f'analysis_gpu{a.gpu}.log')
        atomic_json(status,dict(state='complete',gpu=a.gpu,pid=os.getpid(),models=[e['id'] for e in entries]))
    except Exception as exc:
        atomic_json(status,dict(state='failed',gpu=a.gpu,pid=os.getpid(),error=repr(exc)));raise

if __name__=='__main__':main()
