"""Export blinded human-screening candidates and supplementary duplicate exclusions."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--pkl', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--region-map', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(args.audit).read_text())
    if not audit.get('byte_hash_audit_performed'):
        raise ValueError('Full byte audit required first')
    exclusions = {'val': set(), 'test': set()}
    for first, second in audit['cross_split_identical_bytes']:
        # Auditor walks train, val, test: retain earlier partition, exclude later.
        exclusions[second[0]].add(second[1])
    payload = {key: sorted(value) for key, value in exclusions.items()}
    (out / 'deduplicated_eval_exclusions.json').write_text(json.dumps(payload, indent=2))
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    mapping = json.loads(Path(args.region_map).read_text())
    attributes = list(getattr(data, 'attributes', getattr(data, 'attr_name', [])))
    attributes = [a for a in attributes if mapping.get(a) in ('head', 'upper', 'lower', 'shoes')]
    if not attributes:
        raise ValueError('No mapped evidence attributes')
    rng = np.random.default_rng(42)
    counts = {}
    with (out / 'candidates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'split', 'attribute'])
        writer.writeheader()
        for split, count in [('val', 400), ('test', 1200)]:
            indices = data.partition[split]
            if isinstance(indices, list):
                indices = indices[0]
            names = sorted({str(data.image_name[i]) for i in indices} - exclusions[split])
            chosen = rng.choice(names, size=min(count, len(names)), replace=False)
            counts[split] = len(chosen)
            for name in chosen:
                for attr in attributes:
                    writer.writerow(dict(image=name, split=split, attribute=attr))
    (out / 'screening_manifest.json').write_text(json.dumps(dict(
        seed=42, candidate_images=counts, target_human_accepted={'val': 200, 'test': 600},
        human_natural_occlusion_confirmed=False,
        note='Candidates only; human screen/stratification/adjudication and final lock still required'), indent=2))


if __name__ == '__main__':
    main()
