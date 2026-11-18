"""Human-reviewed fixed-band geometry, distinct from observability labels.

Pose suggestions and provisional UI exports are never runtime manifests.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

REGIONS = ('head', 'upper', 'lower', 'shoes')
BANDS = ((0., .18), (.18, .55), (.55, .85), (.85, 1.))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bands_for(width, height):
    return {r: [0, int(height * lo), width, min(height, max(int(height * lo) + 1, int(height * hi)))]
            for r, (lo, hi) in zip(REGIONS, BANDS)}


def unique_records(records):
    result = {}
    for r in records:
        name = r['image_id']
        if name in result:
            raise ValueError(f'Duplicate image record: {name}')
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe image ID')
        result[name] = r
    return result


def read_geometry(path):
    doc = json.loads(Path(path).read_text())
    if doc.get('schema') != 'eopar.geometry.runtime.v1' or doc.get('state') != 'HUMAN_REVIEWED':
        raise ValueError('Requires compiled human-reviewed geometry; provisional/legacy labels rejected')
    if not doc.get('review_evidence') or not doc.get('split_sha256'):
        raise ValueError('Missing review provenance')
    records = unique_records(doc['records'])
    for name, r in records.items():
        if r['split'] not in ('train', 'val') or r['bands'] != bands_for(r['width'], r['height']):
            raise ValueError(f'Invalid split/coordinates: {name}')
        if (not r.get('reviewer') or len(r.get('source_sha256', '')) != 64 or
                not isinstance(r['valid_regions'], list) or
                len(r['valid_regions']) != len(set(r['valid_regions'])) or
                set(r['valid_regions']) - set(REGIONS)):
            raise ValueError(f'Invalid geometry provenance/regions: {name}')
        answers = r.get('answers', {})
        if (set(answers) != set(REGIONS) or set(answers.values()) - {'valid', 'invalid', 'uncertain'} or
                set(r['valid_regions']) != {g for g in REGIONS if answers[g] == 'valid'}):
            raise ValueError(f'Compiled regions differ from adjudicated answers: {name}')
    return doc, records


def verify_sources(path, image_dir, split_sha256, split_names):
    doc, records = read_geometry(path)
    if doc['split_sha256'] != split_sha256:
        raise ValueError('Geometry split checksum changed')
    for evidence, expected in doc['review_evidence'].items():
        if sha(evidence) != expected:
            raise ValueError(f'Geometry review evidence changed: {evidence}')
    for name, row in records.items():
        if name not in split_names[row['split']]:
            raise ValueError(f'Geometry image belongs to another split: {name}')
        image = Path(image_dir) / name
        if sha(image) != row['source_sha256']:
            raise ValueError(f'Geometry source image changed: {name}')
        from PIL import Image
        with Image.open(image) as im:
            if im.size != (row['width'], row['height']):
                raise ValueError(f'Geometry dimensions changed: {name}')
    return doc


def coverage(records, split):
    rows = [r for r in records.values() if r['split'] == split]
    counts = Counter(g for r in rows for g in r['valid_regions'])
    return dict(images=len(rows), images_with_any=sum(bool(r['valid_regions']) for r in rows),
                outside_eligible_images=sum(len(r['valid_regions']) >= 2 for r in rows),
                valid_images_by_region={g: counts[g] for g in REGIONS})


def require_coverage(records):
    result = {s: coverage(records, s) for s in ('train', 'val')}
    for split, minimum in (('train', 100), ('val', 25)):
        if any(n < minimum for n in result[split]['valid_images_by_region'].values()):
            raise ValueError(f'Insufficient verified geometry coverage: {split}; minimum {minimum}/region')
        if result[split]['outside_eligible_images'] < minimum:
            raise ValueError(f'Insufficient verified outside-intervention pairs: {split}; minimum {minimum} images')
    return result
