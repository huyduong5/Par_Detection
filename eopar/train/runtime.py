import hashlib
import json
import os
import random
import subprocess
import sys
import importlib.metadata
import importlib.util
import uuid
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    paths = [ROOT / 'eopar/__init__.py']
    for folder in ('models', 'datasets', 'losses', 'train', 'eval', 'scripts', 'verification'):
        paths += list((ROOT / 'eopar' / folder).rglob('*.py'))
    paths += [p for p in (ROOT / 'aq_visibility').rglob('*.py')
              if not {'output', 'tests', '__pycache__'} & set(p.parts)]
    paths += [p for p in (ROOT / 'VMamba').rglob('*.py') if not {'.git', '__pycache__'} & set(p.parts)]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def capture_rng(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def atomic_save(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def command(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f'unavailable: {e}'


def environment_identity():
    packages = {}
    for name in ('torch', 'torchvision', 'triton', 'mamba-ssm', 'causal-conv1d', 'transformers', 'numpy', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    extensions={}
    for name in ('selective_scan_cuda','selective_scan_cuda_oflex','causal_conv1d_cuda'):
        spec=importlib.util.find_spec(name)
        if spec and spec.origin and Path(spec.origin).is_file():
            extensions[name]=digest(spec.origin)
    return dict(python=sys.version, packages=packages, torch_cuda=torch.version.cuda,
                extensions=extensions,driver=command(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']))


def verify_pretrained(lock_path):
    records = json.loads(Path(lock_path).read_text())
    if not records or not isinstance(records, dict):
        raise ValueError('Empty pretrained lock')
    for name, expected in records.items():
        if digest(name) != expected:
            raise ValueError(f'Pretrained checksum mismatch: {name}')
    return records


def manifest(config, model=None):
    pretrained = {}
    if model is not None:
        for encoder in (model.baseline.vision_encoder, model.baseline.text_encoder):
            pretrained.update(encoder.pretrained_files)
    locked = verify_pretrained(config['pretrained_lock'])
    if pretrained != locked:
        raise ValueError('Actually loaded pretrained files differ from the lock')
    return dict(config=config, git_commit=command(['git', 'rev-parse', 'HEAD']),
                source_hashes=source_hashes(), split_sha256=digest(config['pkl_path']),
                region_map_sha256=digest(config['region_map']),
                geometry_sha256=digest(config['geometry_manifest']) if config.get('geometry_manifest') else None,
                pretrained_hashes=pretrained, environment=environment_identity(),
                initialized_from_sha256=digest(config['init_checkpoint']) if config.get('init_checkpoint') else None,
                packages=command([sys.executable, '-m', 'pip', 'freeze']),
                gpu=command(['nvidia-smi']), torch_version=torch.__version__)


def write_json(path, payload):
    path = Path(path)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
