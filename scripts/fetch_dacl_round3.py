"""Acquire a frozen, labelled real-photo subset from the author-linked DACL mirror.

Dataset author CC BY-NC 4.0 terms prevail over the mirror's contradictory card.
No inference, image ranking by model output, full archive, or testdev labels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/construction/dacl10k"
COMMIT = "7aa4a3e6fd8818d94bdf51b9d02ea0443808f68b"
BASE = f"https://huggingface.co/datasets/Voxel51/dacl10k/resolve/{COMMIT}/"
LABELS = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']
PRIORITY = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'WConccor']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classes(row):
    return {p['label'] for p in row.get('semseg_ground_truth_filled', {}).get('polylines', [])}


def ordered(rows):
    return sorted(rows, key=lambda row: hashlib.sha256(('dacl-round3-20261020:' + row['name']).encode()).hexdigest())


def download(url, target):
    if target.exists():
        return
    target.parent.mkdir(exist_ok=True, parents=True)
    temporary = target.with_suffix(target.suffix + '.partial')
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as incoming, temporary.open('wb') as outgoing:
                while block := incoming.read(1024 * 1024):
                    outgoing.write(block)
            temporary.replace(target)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--negatives-only', action='store_true')
    args = parser.parse_args()
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    DATA.mkdir(exist_ok=True, parents=True)
    download(BASE + 'samples.json', DATA / 'samples.json')
    if digest(DATA / 'samples.json') != '329e842ee83c700c3717b629ea199faa7a9d3451b2d5fff5923b5f44e8abc588':
        raise RuntimeError('Pinned mirror annotation manifest changed')
    samples = json.loads((DATA / 'samples.json').read_text(encoding='utf-8'))['samples']
    no_target = lambda row: bool(classes(row)) and not classes(row).intersection({'Crack', 'ACrack', 'Spalling'})
    train = ordered([r for r in samples if r['tags'] == ['train'] and no_target(r)])
    validation = ordered([r for r in samples if r['tags'] == ['validation'] and no_target(r)])
    chosen = {'negative_train': train[:20], 'negative_calibration': train[20:30], 'negative_heldout': validation[:10]}
    reserved = {r['name'] for group in chosen.values() for r in group}
    evaluation = {}
    for label in PRIORITY:
        candidates = ordered([r for r in samples if r['tags'] == ['validation'] and label in classes(r) and r['name'] not in reserved])
        for row in candidates[:8]:
            evaluation[row['name']] = row
    chosen['damage_diagnostic'] = sorted(evaluation.values(), key=lambda row: row['name'])
    selection = {'dataset': 'dacl10k v2', 'author_source': 'https://github.com/phiyodr/dacl10k-toolkit',
                 'mirror': 'Voxel51/dacl10k', 'mirror_commit': COMMIT,
                 'license': 'CC BY-NC 4.0 (author terms); mirror card incorrectly says CC BY 4.0',
                 'seed_rule': 'SHA256(dacl-round3-20261020:original_name) ascending before viewing pixels/inference',
                 'negative_scope': 'Complete published polygons contain no Crack, ACrack or Spalling; other damage can be present. Not a healthy building or physical-condition negative.',
                 'diagnostic_scope': 'First8officialvalidation photos per14damageclasses, union deduplicated by original filename. Publisher model-training overlap is not independently established. No testdev images used.',
                 'group_scope': 'Original image separated; physical bridge IDs are not supplied in this mirror, so no bridge-group independence claim.',
                 'selections': {k: [r['name'] for r in group] for k, group in chosen.items()}}
    split_path = DATA / 'round3-selection-before-inference.json'
    if split_path.exists() and json.loads(split_path.read_text(encoding='utf-8')) != selection:
        raise RuntimeError('Frozen subset changed; do not silently replace')
    split_path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    records = []
    seen = {}
    for partition, group in chosen.items():
        if args.negatives_only and partition == 'damage_diagnostic':
            continue
        for row in group:
            path = Path(row['filepath'])
            if path.is_absolute() or '..' in path.parts:
                raise RuntimeError('Unsafe mirror image path')
            target = DATA / 'images' / partition / path.name
            download(BASE + row['filepath'], target)
            if target.stat().st_size != row['metadata']['size_bytes']:
                raise RuntimeError('Image byte size differs from pinned publisher-mirror metadata')
            image_sha = digest(target)
            if image_sha in seen:
                raise RuntimeError(f'Exact image duplicate across selected observations: {seen[image_sha]} / {row["name"]}')
            seen[image_sha] = row['name']
            annotation = {'source_image': row['name'], 'source_file': row['filepath'], 'source_split': row['tags'][0],
                          'image_width': row['metadata']['width'], 'image_height': row['metadata']['height'],
                          'labels': sorted(classes(row)), 'polygon_format': 'Normalized xy coordinates; multiple labels can overlap',
                          'polygons': row['semseg_ground_truth_filled']['polylines']}
            label = DATA / 'annotations' / partition / (row['name'] + '.json')
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(json.dumps(annotation, indent=2) + '\n', encoding='utf-8')
            records.append({'partition': partition, 'image': str(target.relative_to(DATA)).replace('\\', '/'),
                            'annotation': str(label.relative_to(DATA)).replace('\\', '/'),
                            'source_name': row['name'], 'labels': annotation['labels'], 'image_sha256': image_sha,
                            'annotation_sha256': digest(label), 'source_url': BASE + row['filepath']})
        print(partition, len(group), 'downloaded', flush=True)
    manifest = {**selection, 'selection_sha256': digest(split_path), 'records': records,
                'counts': {key: sum(r['partition'] == key for r in records) for key in chosen}}
    manifest_path = DATA / ('round3-negative-manifest.json' if args.negatives_only else 'round3-asset-manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(manifest_path), 'counts': manifest['counts'], 'sha256': digest(manifest_path)}, indent=2))


if __name__ == '__main__':
    main()
