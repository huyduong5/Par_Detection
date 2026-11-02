"""Full decode/byte audit and 64 deterministic human-readable intervention pairs."""
import argparse
import hashlib
import io
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from eopar.datasets.interventions import MEAN, STD
from eopar.train.run import load_config, make_dataset
from eopar.train.runtime import digest, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    c = load_config(args.config)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with open(c['pkl_path'], 'rb') as f:
        data = pickle.load(f)
    started = time.monotonic()
    seen, duplicates, errors, names, counts, sizes = {}, [], [], {}, {}, Counter()
    image_files={}
    for split in ('train', 'val', 'test'):
        idx = data.partition[split]
        if isinstance(idx, list): idx = idx[0]
        names[split] = [str(data.image_name[i]) for i in np.asarray(idx).ravel()]
        counts[split] = len(names[split])
        for number, name in enumerate(names[split]):
            try:
                raw = (Path(c['img_dir']) / name).read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                image_files[str((Path(c['img_dir'])/name).resolve())]=sha
                if sha in seen and seen[sha][0] != split:
                    duplicates.append([seen[sha], [split, name]])
                else:
                    seen[sha] = [split, name]
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.height < 8 or image.width < 1:
                        raise ValueError(f'Invalid band geometry {image.size}')
                    sizes[str(image.size)] += 1
            except Exception as exc:
                errors.append(dict(split=split, image=name, error=str(exc)))
            if number % 5000 == 0:
                print(split, number, flush=True)
    exclusions = {'val': [], 'test': []}
    for first, second in duplicates:
        exclusions[second[0]].append(second[1])
    exclusions = {k: sorted(set(v)) for k, v in exclusions.items()}
    prior = Path(c['region_map']).parent.parent.parent / 'eopar/research/annotation/deduplicated_eval_exclusions.json'
    # Release snapshots do not contain old research artifacts; regenerate, then
    # compare with canonical evidence if a path was explicitly present.
    write_json(out / 'deduplicated_eval_exclusions.json', exclusions)
    write_json(out / 'image_files.json', image_files)
    labels = np.asarray(data.label)
    overlap = {f'{a}:{b}': sorted(set(names[a]) & set(names[b]))
               for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    report = dict(state='PASS_WITH_DISCLOSED_DUPLICATES' if not errors and not any(overlap.values()) else 'FAIL',
        images=counts, errors=errors, cross_split_byte_duplicates=duplicates, name_overlap=overlap,
        labels_binary=bool(np.isin(labels, [0, 1]).all()), split_sha256=digest(c['pkl_path']),
        region_sha256=digest(c['region_map']), image_manifest_sha256=hashlib.sha256(
            json.dumps(sorted((v[0], v[1], k) for k, v in seen.items())).encode()).hexdigest(),
        seconds=time.monotonic() - started, size_distribution=dict(sizes),
        note='Official splits retained; byte duplicates excluded only in supplementary evaluation. Identity/near-duplicate audit remains open.')
    report['image_files_sha256']=digest(out/'image_files.json')
    if not report['labels_binary']: report['state'] = 'FAIL'
    write_json(out / 'data.json', report)
    ds = make_dataset(c, 'val')
    metadata = []
    for page in range(8):
        canvas = Image.new('RGB', (2 * 192, 8 * 220), 'white')
        draw = ImageDraw.Draw(canvas)
        for row in range(8):
            index = page * 8 + row
            ds.severity = (.25, .5, .75, 1.)[index % 4]
            sample = ds[index]
            for col, key in enumerate(('img_clean', 'img_occ')):
                pixels = ((sample[key] * STD + MEAN).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                tile = Image.fromarray(pixels).resize((192, 192))
                canvas.paste(tile, (col * 192, row * 220))
            draw.text((2, row * 220 + 192), f"{index}: {sample['region']} sev={sample['severity']}", fill='black')
            metadata.append(dict(index=index, image=sample['imgname'], region=sample['region'],
                                 severity=sample['severity'], affected=sample['affected_mask'].tolist(),
                                 intact=sample['intact_mask'].tolist()))
        canvas.save(out / f'pairs_{page:02d}.png')
    write_json(out / 'visual_pairs.json', metadata)
    if report['state'] == 'FAIL': raise SystemExit(1)


if __name__ == '__main__': main()
