"""Fail-closed pilot authorization bound to a tested release, never full training."""
import json
from pathlib import Path
from eopar.train.runtime import ROOT, source_hashes, environment_identity, digest, verify_pretrained

REQUIRED_GPU = ('variants', 'parity', 'gradient_audit', 'branch_effects', 'amp', 'resume',
                'overfit', 'memory_100_batches', 'full_epoch_B1', 'full_epoch_P3', 'both_devices')


def validate_gate(path, variant, config_path):
    report = json.loads(Path(path).read_text())
    if report.get('state') != 'PASS' or report.get('authorization') != 'pilot_only':
        raise ValueError('Pilot is not authorized by verification')
    if report.get('release_root') != str(ROOT.resolve()):
        raise ValueError('Gate belongs to another source snapshot')
    if report.get('source_hashes') != source_hashes() or report.get('environment') != environment_identity():
        raise ValueError('Runtime source/environment changed after verification')
    if report.get('config_hashes', {}).get(variant) != digest(config_path):
        raise ValueError('Pilot config was not verified')
    for key in REQUIRED_GPU:
        if report.get('gpu_checks', {}).get(key) != 'PASS':
            raise ValueError(f'Mandatory GPU check missing: {key}')
    for name, sha in report.get('input_hashes', {}).items():
        if digest(name) != sha: raise ValueError(f'Verified input changed: {name}')
    if not report.get('input_hashes') or report.get('cpu') != 'PASS' or report.get('data') != 'PASS':
        raise ValueError('Missing CPU/data evidence')
    return report
