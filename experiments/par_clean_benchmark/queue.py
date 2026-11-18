"""Resume clean benchmark jobs only after real-data smoke and a first epoch per variant."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.par_resolution.core import atomic_json
from .report import report

MODELS = ('C3','C3_MS')


def job(root, manifest, variant, seed, action, batch, stop=30):
    dataset = manifest.parent.name
    folder = root/dataset/(f'smoke_{variant}_b{batch}' if action=='smoke' else f'{variant}_s{seed}')
    folder.mkdir(parents=True,exist_ok=True)
    gate = root/dataset/f'smoke_{variant}_b{batch}/smoke.json'
    deadline = time.monotonic()+24*3600
    while time.monotonic()<deadline:
        table = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                      '--format=csv,noheader,nounits'],text=True,timeout=10)
        for line in table.splitlines():
            gpu,free,util = map(int,line.split(','))
            if free<20000 or util>10:
                continue
            with Path(f'/tmp/eopar-{os.getuid()}-gpu{gpu}.lock').open('a') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CC='/usr/bin/gcc-12',CXX='/usr/bin/g++-12',
                           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                command = [sys.executable,'-u','-m','experiments.par_clean_benchmark.train',
                           '--manifest',str(manifest),'--variant',variant,'--action',action,'--output',str(folder),
                           '--seed',str(seed),'--batch',str(batch),'--gate',str(gate),'--stop-after',str(stop)]
                atomic_json(folder/'worker.json',dict(state='RUNNING',gpu=gpu,action=action))
                with (folder/f'{action}.log').open('w') as log:
                    child = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(folder/'worker.json',dict(state='COMPLETE' if child.returncode==0 else 'FAILED',
                                                     gpu=gpu,action=action,returncode=child.returncode))
                return child.returncode,folder/f'{action}.log'
        time.sleep(30)
    raise TimeoutError('No available GPU for 24 hours')


def run(out, manifests):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for manifest in manifests:
                dataset = manifest.parent.name
                atomic_json(out/'queue_status.json',dict(state='SMOKE',dataset=dataset,pid=os.getpid()))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for batch in (16,8):
                        smoke = [pool.submit(job,out,manifest,m,42,'smoke',batch) for m in MODELS]
                        smoke_results = [x.result() for x in smoke]
                        errors = [log for code,log in smoke_results if code]
                        if not errors:
                            break
                        if batch==8 or not all('out of memory' in log.read_text().lower() for log in errors):
                            raise RuntimeError(f'Smoke failed: {errors}')
                    atomic_json(out/'queue_status.json',dict(state='EPOCH_1',dataset=dataset,batch=batch,pid=os.getpid()))
                    first = [pool.submit(job,out,manifest,m,42,'train',batch,1) for m in MODELS]
                    for x in first:
                        code,log = x.result()
                        if code:
                            raise RuntimeError(f'First epoch failed: {log}')
                    atomic_json(out/'queue_status.json',dict(state='TRAIN',dataset=dataset,batch=batch,pid=os.getpid()))
                    for seed in (42,43,44):
                        trains = [pool.submit(job,out,manifest,m,seed,'train',batch) for m in MODELS]
                        for x in trains:
                            code,log = x.result()
                            if code:
                                raise RuntimeError(f'Train failed: {log}')
                        tests = [pool.submit(job,out,manifest,m,seed,'test',batch) for m in MODELS]
                        for x in tests:
                            code,log = x.result()
                            if code:
                                raise RuntimeError(f'Test failed: {log}')
                        report(out)
            atomic_json(out/'queue_status.json',dict(state='COMPLETE',pid=os.getpid()))
        except Exception as e:
            atomic_json(out/'queue_status.json',dict(state='FAILED',error=repr(e),pid=os.getpid()))
            raise


if __name__=='__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('manifests',nargs='+',type=Path)
    args = p.parse_args()
    run(args.output,args.manifests)
