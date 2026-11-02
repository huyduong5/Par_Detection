"""Audit partition names/indices and optional byte-identical image leakage."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--pkl', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--hash-images', action='store_true')
    args = p.parse_args()
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    indices, names, missing = {}, {}, []
    for split in ('train', 'val', 'test'):
        idx = data.partition[split]
        if isinstance(idx, list):
            idx = idx[0]
        indices[split] = set(map(int, np.asarray(idx).ravel()))
        names[split] = {str(data.image_name[i]) for i in indices[split]}
        missing.extend(str(Path(args.images) / n) for n in names[split] if not (Path(args.images) / n).is_file())
    overlap = {f'{a}:{b}': sorted(names[a] & names[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    duplicates = []
    if args.hash_images:
        seen = {}
        for split in names:
            for name in sorted(names[split]):
                path = Path(args.images) / name
                if not path.is_file():
                    continue
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha in seen and seen[sha][0] != split:
                    duplicates.append([seen[sha], [split, name]])
                else:
                    seen[sha] = [split, name]
    labels = np.asarray(data.label)
    report = dict(split_counts={k: len(v) for k, v in names.items()}, name_overlap=overlap,
                  missing_images=missing, label_values=np.unique(labels).tolist(),
                  byte_hash_audit_performed=args.hash_images, cross_split_identical_bytes=duplicates,
                  limitation='Identity/camera leakage and near-duplicates require metadata/perceptual audit')
    Path(args.output).write_text(json.dumps(report, indent=2))
    if missing or any(overlap.values()) or duplicates:
        raise SystemExit('AUDIT_FAILED: see report; do not silently alter the official split')


if __name__ == '__main__':
    main()
