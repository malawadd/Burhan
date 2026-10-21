"""Freeze one unmarked photograph derivative per SDNET source, real polygons and duplicate groups."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, shutil
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'datasets/construction/expansion-public-v1/sdnet2025'
OUT = BASE / 'prepared'
CLASSES = ['visibly_unseated_connection', 'fastener_absence_appearance']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def main():
    if (OUT / 'manifest.json').exists(): raise ValueError('Preserve frozen prepared manifest; use a new version to change splits')
    rows = []
    for filename, directory in [('Loosened.json', 'Resized images 640-640'), ('Missing.json', 'Resized- 640-640')]:
        annotation = next((BASE / 'original').rglob(filename))
        data = json.loads(annotation.read_text())
        mapping = {c['id']: c['name'] for c in data['categories']}
        by_image = defaultdict(list)
        for a in data['annotations']:
            label = mapping[a['category_id']]
            category = 0 if label == 'Loosen' else 1 if label == 'Missing' else None
            if category is None: raise ValueError('Unexpected actual annotation label: ' + label)
            polygons = a.get('segmentation', [])
            if not polygons or any(len(p) < 6 or len(p) % 2 for p in polygons): raise ValueError('Actual polygon required')
            x, y, w, h = a['bbox']
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > 640.01 or y + h > 640.01: raise ValueError('Invalid source box')
            by_image[a['image_id']].append({'class_id': category, 'label': CLASSES[category],
                'bbox': [x, y, w, h], 'polygons': polygons, 'publisher_annotation_id': a['id']})
        images_dir = annotation.parent.parent / directory
        for image in data['images']:
            path = images_dir / image['file_name']
            if not path.exists(): raise FileNotFoundError(path)
            rows.append({'id': Path(image['file_name']).stem, 'source_image': path.relative_to(ROOT).as_posix(),
                         'source_sha256': sha(path), 'annotation_sha256': sha(annotation),
                         'publisher_label': filename, 'boxes': by_image[image['id']]})
    for path in sorted((BASE / 'original/Dataset/Fixed/640-640').glob('*.jpg')):
        rows.append({'id': 'fixed-' + path.stem.replace(' ', '_'), 'source_image': path.relative_to(ROOT).as_posix(),
                     'source_sha256': sha(path), 'publisher_label': 'Fixed', 'boxes': []})
    if len(rows) != 826: raise ValueError('Expected exactly 826 original photograph derivatives')
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    hashes, thumbs = [], []
    for row in rows:
        image = cv2.imread(str(ROOT / row['source_image']))
        if image is None or image.shape[:2] != (640, 640): raise ValueError('Expected publisher 640x640 unmarked derivative')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        hashes.append(np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes())
        thumbs.append(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))
    # Conservative cross-class duplicate/near-duplicate graph. This is NOT a claim
    # that different graph components depict independent physical connections.
    for i in range(len(rows)):
        for j in range(i):
            distance = (int.from_bytes(hashes[i], 'big') ^ int.from_bytes(hashes[j], 'big')).bit_count()
            difference = float(np.mean(np.abs(thumbs[i] - thumbs[j])))
            if rows[i]['source_sha256'] == rows[j]['source_sha256'] or distance <= 8 or difference <= 8:
                parents[root(i)] = root(j)
    groups = defaultdict(list)
    for i in range(len(rows)): groups[root(i)].append(i)
    counts = Counter()
    for indices in groups.values():
        group_id = min(rows[i]['source_sha256'] for i in indices)
        bucket = int(hashlib.sha256(('sdnet-expansion-v1:' + group_id).encode()).hexdigest()[:8], 16) % 100
        split = 'train' if bucket < 70 else 'calibration' if bucket < 85 else 'test'
        for i in indices:
            row = rows[i]; row.update({'split': split, 'duplicate_group': group_id})
            target = OUT / 'images' / split / (row['id'] + '.jpg')
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / row['source_image'], target)
            row['image'] = target.relative_to(ROOT).as_posix()
            yolo = OUT / 'labels' / split / (row['id'] + '.txt'); yolo.parent.mkdir(parents=True, exist_ok=True)
            yolo.write_text(''.join(f"{b['class_id']} {(b['bbox'][0]+b['bbox'][2]/2)/640:.8f} {(b['bbox'][1]+b['bbox'][3]/2)/640:.8f} {b['bbox'][2]/640:.8f} {b['bbox'][3]/640:.8f}\n" for b in row['boxes']))
            mask = np.zeros((640, 640), np.uint8)
            for b in row['boxes']:
                for polygon in b['polygons']:
                    cv2.fillPoly(mask, [np.rint(np.asarray(polygon).reshape(-1, 2)).astype(np.int32)], b['class_id'] + 1)
            mask_path = OUT / 'masks' / split / (row['id'] + '.png'); mask_path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(mask_path), mask)
            row['mask'] = mask_path.relative_to(ROOT).as_posix(); row['mask_sha256'] = sha(mask_path)
            counts[split] += 1
    yaml = OUT / 'dataset.yaml'
    yaml.write_text(f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/calibration\nnames:\n  0: {CLASSES[0]}\n  1: {CLASSES[1]}\n")
    report = {'schema_version': 1, 'classes': CLASSES, 'license': 'CC BY 4.0',
              'source': 'https://data.mendeley.com/datasets/cxnj6cp4rb/1', 'records': rows,
              'photos': len(rows), 'split_counts': dict(counts), 'duplicate_groups': len(groups),
              'instances': dict(Counter(b['label'] for r in rows for b in r['boxes'])),
              'originals_policy': 'One unmarked publisher640 derivative per826sourcephoto; overlays excluded; RAW originals lack verifiedname mapping, not mixed.',
              'scene_independence': 'NOT VERIFIED: publisher provides no physical connection/site IDs. Graph groups only exact/near duplicates; research-only evaluation.',
              'split_protocol': 'Before model: union exactSHA or64bitdHash<=8 or32pxmeanabsolutegraydifference<=8; componenthash70/15/15.',
              'private_videos_used': False, 'prepared_script_sha256': sha(Path(__file__)), 'dataset_yaml_sha256': sha(yaml)}
    write(OUT / 'manifest.json', report)
    print(json.dumps({k: report[k] for k in ['photos','split_counts','duplicate_groups','instances']}), flush=True)


if __name__ == '__main__': main()
