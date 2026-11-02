"""Evaluate one EOPAR checkpoint, saving predictions and explicit metrics."""
import argparse
from pathlib import Path
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from eopar.eval.metrics import par_metrics
from eopar.eval.artifacts import validate_thresholds
from eopar.train.run import build, make_dataset, evaluate
from eopar.train.runtime import digest, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--family', choices=['solid', 'noise'], default='solid')
    p.add_argument('--severity', type=float, default=1.)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--exclude', help='Supplementary deduplicated evaluation manifest; never alters the official split')
    p.add_argument('--thresholds', help='Optional validation-only threshold artifact; standard metrics remain separate')
    args = p.parse_args()
    path = Path(args.output)
    if path.exists() or path.with_suffix('.npz').exists():
        raise FileExistsError('Evaluation output already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    c = state['config']
    if state.get('split_sha256') != digest(c['pkl_path']) or state.get('region_sha256') != digest(c['region_map']):
        raise ValueError('Checkpoint dataset provenance mismatch')
    if args.exclude:
        c = dict(c, eval_exclusions=str(Path(args.exclude).resolve()))
    # Avoid depending on an external warmstart file after a complete checkpoint exists.
    model, _ = build(c, path.parent, initialize=False)
    model.load_state_dict(state['model'], strict=True)
    ds = make_dataset(c, args.split, args.family, args.severity)
    if state['attributes'] != ds.attributes:
        raise ValueError('Attribute order mismatch')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    fitted = None
    if args.thresholds:
        fitted = json.loads(Path(args.thresholds).read_text())
        validate_thresholds(fitted, ds.attributes, digest(args.checkpoint),
                            ds.base.img_ids if hasattr(ds.base, 'img_ids') else [], args.split)
    loader = DataLoader(ds, batch_size=c['eval_batch_size'], num_workers=c['workers'])
    result = evaluate(model, loader, device, ds.attributes, args.max_batches,
                      path.with_suffix('.npz'), amp=c['eval_amp'])
    if args.thresholds:
        a = np.load(path.with_suffix('.npz'), allow_pickle=False)
        validate_thresholds(fitted, ds.attributes, digest(args.checkpoint), a['names'].tolist(), args.split)
        result['calibrated_decoding'] = {
            'thresholds_sha256': digest(args.thresholds),
            'clean': par_metrics(a['clean'], a['labels'], fitted['thresholds']),
            'occluded': par_metrics(a['occ'], a['labels'], fitted['thresholds'])}
    result.update(split=args.split, family=args.family, severity=args.severity,
                  artifact_state='complete', predictions_sha256=digest(path.with_suffix('.npz')),
                  split_sha256=digest(c['pkl_path']), attributes=ds.attributes,
                  eval_exclusions_sha256=digest(args.exclude) if args.exclude else None,
                  checkpoint_sha256=digest(args.checkpoint),
                  selection='checkpoint chosen by clean validation mA',
                  peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)
    write_json(path, result)


if __name__ == '__main__':
    main()
