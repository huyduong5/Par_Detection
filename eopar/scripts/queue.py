"""Wait for an idle physical GPU, then smoke-test and run a validation-only pilot.

Uses a cooperative per-GPU flock and rechecks resource availability. It cannot
reserve a shared GPU against other users; it never kills existing processes.
No long novel-model run is launched before the research/data gates are reviewed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

from eopar.train.runtime import ROOT, write_json
from eopar.verification.gate import validate_gate


def idle(gpu, need_mb):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
          '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    memory, util = (int(v.strip()) for v in row.split(','))
    apps = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    return memory >= need_mb and util <= 5 and not apps


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--variant', choices=['B0', 'B1', 'P1', 'P2', 'P3'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--run-pilot', action='store_true')
    p.add_argument('--gate-report')
    p.add_argument('--wait-hours', type=float, default=72.)
    args = p.parse_args()
    if args.run_pilot:
        if not args.gate_report:
            p.error('--run-pilot requires --gate-report from completed verification')
        validate_gate(args.gate_report, args.variant, ROOT / f'eopar/configs/{args.variant}.json')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'queue_status.json'
    lock_path = Path('/tmp') / f'eopar-{os.getuid()}-gpu{args.gpu}.lock'
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another EOPAR worker owns this GPU lock')
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            write_json(status, dict(state='waiting_for_idle_gpu', gpu=args.gpu,
                                   variant=args.variant, pid=os.getpid(), timestamp=time.time()))
            if idle(args.gpu, 18000):
                time.sleep(10)
                if idle(args.gpu, 18000):
                    break
            if time.monotonic() > deadline:
                write_json(status, dict(state='wait_timeout', gpu=args.gpu))
                return
            time.sleep(60)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        child = None
        def stop(signum, frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            write_json(status, dict(state='stopped', gpu=args.gpu))
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        def run(command, logfile):
            nonlocal child
            with (out / logfile).open('a') as log:
                child = subprocess.Popen([sys.executable, '-m', *command], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait()
                if code: raise subprocess.CalledProcessError(code, command)
        try:
            write_json(status, dict(state='smoke_running', gpu=args.gpu, variant=args.variant))
            smoke = json.loads((ROOT / f'eopar/configs/{args.variant}.json').read_text())
            smoke.update(batch_size=2, eval_batch_size=2, grad_accum=2, epochs=1, max_train_batches=4, max_val_batches=2, workers=0)
            write_json(out / 'smoke_config.json', smoke)
            run(['eopar.train.run', '--config', str(out / 'smoke_config.json'),
                 '--output', str(out / 'smoke'), '--diagnostic-proxy'], 'smoke.log')
            if args.run_pilot:
                validate_gate(args.gate_report, args.variant, ROOT / f'eopar/configs/{args.variant}.json')
                write_json(status, dict(state='pilot_running', gpu=args.gpu, variant=args.variant))
                run(['eopar.train.run', '--config', f'eopar/configs/{args.variant}.json',
                     '--output', str(out / 'pilot'), '--stop-after', '5', '--gate-report', args.gate_report], 'pilot.log')
                run(['eopar.eval.run', '--checkpoint', str(out / 'pilot/best.pth'),
                     '--split', 'val', '--family', 'noise',
                     '--output', str(out / 'pilot/val_unseen_noise.json')], 'eval.log')
            write_json(status, dict(state='done', gpu=args.gpu, variant=args.variant,
                                   next='Review validation evidence and novelty before full training'))
        except Exception as exc:
            write_json(status, dict(state='failed', error=str(exc), gpu=args.gpu))
            raise


if __name__ == '__main__':
    main()
