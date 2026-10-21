"""Prepare real-mask TRAIN views and box-only camera records; never loads CAL pixels."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_model import prepare
from edge_runtime.profiles.construction_preprocessing import camera_exposure

OUT = ROOT / 'datasets/construction/round4-pixel-domain/prepared-mixed-v1'
INSEG = ROOT / 'datasets/construction/round4-pixel-domain/cubit-inseg'
DACL = ROOT / 'datasets/construction/dacl10k/round3-positive-contingency'
NEG = ROOT / 'datasets/construction/dacl10k'
CAMERA = ROOT / 'datasets/construction/evcivil-round4-real-lowlight-gain/native-input-manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def source_polygons(record, family):
    if family == 'inseg':
        p = ROOT / record['label_path']
        if sha(p) != record['label_sha256']:
            raise ValueError('Acquired source polygon file changed')
        rows = []
        for i, line in enumerate(p.read_text().splitlines()):
            values = line.split()
            points = np.asarray(list(map(float, values[1:])), np.float64).reshape(-1, 2)
            if int(values[0]) not in {0, 1} or len(points) < 3 or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
                raise ValueError('Invalid actual normalized instance polygon')
            rows.append({'id': str(i), 'class_id': int(values[0]), 'parts': [points]})
        return rows
    p = DACL / record['original_annotation']
    if sha(p) != record['original_annotation_sha256']:
        raise ValueError('Complete DACL source annotation changed')
    annotation = json.loads(p.read_text())
    rows = []
    for i, polygon in enumerate(annotation['polygons']):
        if polygon['label'] not in {'Crack', 'ACrack', 'Spalling'}:
            continue
        parts = [np.asarray(points, np.float64) for points in polygon['points']]
        if any(len(points) < 3 or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1) for points in parts):
            raise ValueError('Invalid actual DACL semantic polygon')
        rows.append({'id': str(i), 'class_id': 1 if polygon['label'] == 'Spalling' else 0, 'parts': parts})
    return rows


def view_geometry(raw, polygons):
    """Full source and one deterministic TRAIN-only closeup; no image/prediction selection."""
    h, w = raw.shape[:2]
    yield 'full', (0, 0, w, h)
    # Prefer the most spatially extensive annotated target; deterministic index
    # breaks ties. The nearest real boundary vertex keeps a curved target in view.
    def extent(row):
        points = np.concatenate(row['parts']) * [w, h]
        return float(np.prod(points.max(0) - points.min(0)))
    anchor = max(polygons, key=extent)
    points = np.concatenate(anchor['parts']) * [w, h]
    center = (points.min(0) + points.max(0)) / 2
    vertex = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
    scale = min(160, w // 4, h // 3)
    cw, ch = 4 * scale, 3 * scale  # at most640×480 native; no invented pixels
    left = max(0, min(w - cw, round(vertex[0] - cw / 2)))
    top = max(0, min(h - ch, round(vertex[1] - ch / 2)))
    if (left, top, cw, ch) != (0, 0, w, h):
        yield 'native-closeup', (left, top, cw, ch)


def actual_view(raw, polygons, crop):
    x, y, w, h = crop
    view = raw[y:y + h, x:x + w].copy()
    gained, gain = camera_exposure(view, 'low_light_gain')
    tensor, transform = prepare(gained, 'crack', size=640)
    ratio, _, px, py = transform
    rw, rh = round(w * ratio), round(h * ratio)
    classes, boxes, masks, mask_details = [], [], [], []
    original_h, original_w = raw.shape[:2]
    for row in polygons:
        actual = np.zeros((h, w), np.uint8)
        for part in row['parts']:
            pts = np.rint(part * [original_w, original_h] - [x, y]).astype(np.int32)
            cv2.fillPoly(actual, [pts], 1)
        ys, xs = np.where(actual)
        if not len(xs):
            continue
        left, right, top, bottom = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        box = [(left + right) / 2 * ratio + px, (top + bottom) / 2 * ratio + py,
               (right - left) * ratio, (bottom - top) * ratio]
        # Source-pixel coverage is area-resampled. A thin source polygon is not
        # widened to a rectangle or a minimum stroke; fractional targets remain.
        resized = cv2.resize(actual.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA)
        padded = np.zeros((640, 640), np.float32)
        padded[py:py + rh, px:px + rw] = resized
        fractional = padded
        if not fractional.max() > 0:
            raise ValueError('Actual visible source mask vanished during fractional conversion')
        classes.append(row['class_id'])
        boxes.append(np.asarray(box, np.float32) / 640)
        masks.append(fractional)
        mask_details.append({'source_polygon_id': row['id'], 'source_positive_pixels_in_crop': len(xs),
            'soft_target_coverage_sum_640': float(fractional.sum()), 'soft_target_max': float(fractional.max()),
            'source_coverage_pixels_ge_half_640': int(np.sum(fractional >= .5)),
            'source_crop_mask_bbox_xyxy': [left, top, right, bottom]})
    return tensor[0], np.asarray(classes, np.int64), np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(masks, np.float32), gain, mask_details


def main():
    cv2.setNumThreads(1)
    if OUT.exists():
        raise ValueError('Preserve prepared dataset attempt; no silent overwrite')
    assets = json.loads((INSEG / 'image-assets.json').read_text())
    dacl = json.loads((DACL / 'asset-manifest.json').read_text())
    negative = json.loads((NEG / 'round3-negative-manifest.json').read_text())
    camera = json.loads(CAMERA.read_text())
    OUT.mkdir(parents=True)
    files = OUT / 'samples'
    files.mkdir()
    sources = [INSEG / 'image-assets.json', INSEG / 'image-selection-before-pixels.json', DACL / 'asset-manifest.json',
               NEG / 'round3-negative-manifest.json', CAMERA, Path(__file__), ROOT / 'edge_runtime/profiles/construction_model.py',
               ROOT / 'edge_runtime/profiles/construction_preprocessing.py']
    write_new(OUT / 'preparation-before-pixels.json', {'classes': ['crack', 'spalling'],
        'source_pins': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        'supervision': 'Actual native polygon masks only; publisher camera boxes never become masks. Other unannotated class ignored.',
        'geometry': 'Full source plus one deterministic polygon-boundary-anchored TRAIN native crop≤640×480; nativegain once then serving640letterbox. Actual640mask coverage; prototype upsampled in training loss.',
        'role': 'TRAIN preparation only; no calibration/heldout/private pixels loaded.'})
    records = []
    seen_photo_hashes = set()
    for family, photo_records in [('inseg', assets['records']), ('dacl', dacl['records'])]:
        for record in photo_records:
            image = ROOT / record['image_path'] if family == 'inseg' else DACL / record['original_image']
            expected_hash = record['image_sha256'] if family == 'inseg' else record['original_image_sha256']
            if sha(image) != expected_hash:
                raise ValueError('Original acquired TRAIN image changed')
            if expected_hash in seen_photo_hashes:
                raise ValueError('Exact duplicate acquired photo; preserve attempt for explicit quarantine')
            seen_photo_hashes.add(expected_hash)
            raw = cv2.imread(str(image))
            if raw is None:
                raise ValueError('Original public photograph decode failed')
            polygons = source_polygons(record, family)
            if not polygons:
                raise ValueError('Expected target-positive photo has no actual polygon')
            # CUBIT absence is conservatively unknown; DACL has complete
            # nineteen-class semantic annotation, permitting target absence.
            known = [int(any(p['class_id'] == c for p in polygons)) for c in range(2)] if family == 'inseg' else [1, 1]
            for view_name, crop in view_geometry(raw, polygons):
                tensor, classes, boxes, masks, gain, details = actual_view(raw, polygons, crop)
                if not len(classes):
                    raise ValueError('Declared source-anchored positive view contains no annotated target')
                uid = f'{family}-{image.stem}-{view_name}'
                path = files / (uid + '.npz')
                if path.exists():
                    raise ValueError('Duplicate prepared photo-view identity')
                np.savez_compressed(path, image_rgb_u8=np.rint(tensor * 255).astype(np.uint8), cls=classes, bboxes=boxes, masks=masks)
                records.append({'id': uid, 'source_family': family, 'source_photo_sha256': expected_hash,
                    'raw_source_image': image.relative_to(ROOT).as_posix(), 'crop_xywh': crop, 'gain': gain,
                    'supervision_kind': 'actual_masks', 'supervised_classes': known,
                    'mask_origin': 'publisher_instance_polygon' if family == 'inseg' else 'publisher_semantic_polygon_region',
                    'sample': path.relative_to(ROOT).as_posix(), 'sample_sha256': sha(path), 'classes': classes.tolist(),
                    'mask_details': details, 'split': 'train'})
    for row in camera:
        # Only actual recorded-camera TRAIN frames; no DACL/LCSD photo rows and
        # no CAL/heldout pixels. Native raw images precede shared serving gain.
        if row['split'] != 'train' or not Path(row['raw_source_image']).name.startswith(('dset_', 'night_dec27-', '31_crack_')):
            continue
        source = ROOT / row['raw_source_image']
        label = ROOT / row['label']
        if sha(source) != row['raw_source_file_sha256'] or sha(label) != row['label_sha256']:
            raise ValueError('Protected original camera TRAIN pair changed')
        raw = cv2.imread(str(source))
        gained, gain = camera_exposure(raw, 'low_light_gain')
        tensor, transform = prepare(gained, 'crack', size=640)
        classes, boxes = [], []
        h, w = raw.shape[:2]
        ratio, _, px, py = transform
        for line in label.read_text().splitlines():
            old_class, x, y, bw, bh = map(float, line.split())
            if old_class not in {0, 1}:
                raise ValueError('Camera source class mapping changed')
            classes.append(1 - int(old_class))  # OLD0spall/1crack → AUTHOR0crack/1spall
            boxes.append([(x * w * ratio + px) / 640, (y * h * ratio + py) / 640,
                          bw * w * ratio / 640, bh * h * ratio / 640])
        if not classes:
            raise ValueError('Unlabelled camera frames are not verified healthy negatives')
        uid = 'camera-' + source.stem + '-' + sha(source)[:10]
        path = files / (uid + '.npz')
        if path.exists():
            raise ValueError('Duplicate recorded-camera view identity')
        np.savez_compressed(path, image_rgb_u8=np.rint(tensor[0] * 255).astype(np.uint8), cls=np.asarray(classes, np.int64), bboxes=np.asarray(boxes, np.float32))
        records.append({'id': uid, 'source_family': 'evcivil', 'raw_source_image': row['raw_source_image'],
            'source_raw_sha256': row['raw_source_file_sha256'], 'source_label': row['label'], 'source_label_sha256': row['label_sha256'],
            'gain': gain, 'supervision_kind': 'publisher_boxes', 'supervised_classes': [int(c in classes) for c in range(2)],
            'sample': path.relative_to(ROOT).as_posix(), 'sample_sha256': sha(path), 'classes': classes, 'split': 'train'})
    for row in negative['records']:
        if row['partition'] != 'negative_train':
            continue
        if {'Crack', 'ACrack', 'Spalling'} & set(row['labels']):
            raise ValueError('Supposed target negative contains a target annotation')
        source, annotation = NEG / row['image'], NEG / row['annotation']
        if sha(source) != row['image_sha256'] or sha(annotation) != row['annotation_sha256']:
            raise ValueError('Complete TRAIN negative source changed')
        raw = cv2.imread(str(source))
        gained, gain = camera_exposure(raw, 'low_light_gain')
        tensor, _ = prepare(gained, 'crack', size=640)
        uid = 'negative-' + row['source_name']
        path = files / (uid + '.npz')
        if path.exists():
            raise ValueError('Duplicate verified-negative identity')
        np.savez_compressed(path, image_rgb_u8=np.rint(tensor[0] * 255).astype(np.uint8), cls=np.zeros(0, np.int64), bboxes=np.zeros((0, 4), np.float32))
        records.append({'id': uid, 'source_family': 'dacl_target_negative', 'raw_source_image': source.relative_to(ROOT).as_posix(),
            'source_photo_sha256': row['image_sha256'], 'gain': gain, 'supervision_kind': 'verified_target_negative',
            'supervised_classes': [1, 1], 'sample': path.relative_to(ROOT).as_posix(), 'sample_sha256': sha(path), 'classes': [], 'split': 'train'})
    counts = Counter(r['source_family'] for r in records)
    if counts['evcivil'] != 990 or counts['dacl_target_negative'] != 20:
        raise ValueError('Frozen existing TRAIN scope changed')
    write_new(OUT / 'training-manifest.json', {'classes': ['crack', 'spalling'], 'counts': dict(counts),
        'new_inseg_photographs': len(assets['records']), 'dacl_photographs': len(dacl['records']),
        'camera_split_manifest_sha256': sha(CAMERA), 'preparation_protocol_sha256': sha(OUT / 'preparation-before-pixels.json'),
        'records': records, 'calibration': 'Original122 untouched and not loaded by this preparer.',
        'limits': 'Photo/native-crop views share source identities; semantic regions are not native independent instances. No private source or video simulation.'})
    print('PREPARED', dict(counts), sha(OUT / 'training-manifest.json'), flush=True)


if __name__ == '__main__':
    main()
