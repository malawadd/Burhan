"""Prepare source-pinned crack TRAIN pairs only; no model, CAL or heldout pixels."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_preprocessing import camera_exposure

OUT = ROOT / 'output/optimization/construction-pixel-supervision-v1'
LCSD = ROOT / 'datasets/construction/lcsd-low-light-future-r4'
DACL = ROOT / 'datasets/construction/dacl10k'
POS = DACL / 'round3-positive-contingency'
PINS = {
    LCSD / 'acceptance-v1.json': '28b13decfb62bd695dcc5a68af3cc7cbf5874191108e9184342d943d218e355f',
    POS / 'asset-manifest.json': '2dc962b359bba28ba8eb1871184c2c2c5d8aa841cf4540d393fccd150276c360',
    DACL / 'samples.json': '329e842ee83c700c3717b629ea199faa7a9d3451b2d5fff5923b5f44e8abc588',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lcsd_target(mask):
    if mask.ndim != 2 or not set(np.unique(mask)) <= {0, 2, 255} or not np.any(mask == 2):
        raise ValueError('Expected directly verified LCSD0background/2crack/255ignore')
    return (mask == 2).astype(np.uint8), (mask != 255).astype(np.uint8)


def rasterize_cracks(annotation):
    width, height = annotation['image_width'], annotation['image_height']
    mask = Image.new('1', (width, height))
    draw = ImageDraw.Draw(mask)
    count = 0
    for polygon in annotation['polygons']:
        if polygon['label'] not in {'Crack', 'ACrack'}:
            continue
        for contour in polygon['points']:
            points = [(float(x) * width, float(y) * height) for x, y in contour]
            if len(points) >= 3:
                if not np.isfinite(points).all():
                    raise ValueError('Nonfinite source polygon')
                draw.polygon(points, fill=1)
                count += 1
    result = np.asarray(mask, dtype=np.uint8)
    if bool(count) != bool(np.any(result)):
        raise ValueError('Crack polygons versus raster occupancy disagree; never infer negative')
    return result, count


def prepare_pair(frame, target, valid):
    if frame.dtype != np.uint8 or frame.shape[:2] != target.shape or valid.shape != target.shape:
        raise ValueError('Raw image and pixel annotation geometry differ')
    corrected, gain = camera_exposure(frame, 'low_light_gain')  # Exactly once, full native frame.
    image = cv2.resize(corrected, (384, 384), interpolation=cv2.INTER_LINEAR)
    truth = cv2.resize(target, (384, 384), interpolation=cv2.INTER_NEAREST)
    validity = cv2.resize(valid, (384, 384), interpolation=cv2.INTER_NEAREST)
    if np.any(target) and not np.any(truth * validity):
        raise ValueError('Positive GT vanished at serving geometry; reconcile, never relabel negative')
    return image, truth, validity, gain


def source_rows():
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError(f'Protected source changed: {path}')
    accepted = json.loads((LCSD / 'acceptance-v1.json').read_text())['accepted_records']
    if len(accepted) != 41:
        raise ValueError('All41 accepted TRAIN-transfer pairs required')
    rows = []
    for row in accepted:
        rows.append({'name': row['source_id'], 'dataset': 'LCSD', 'image': LCSD / row['image'],
                     'annotation': LCSD / row['mask'], 'image_sha256': row['original_image_sha256'],
                     'annotation_sha256': row['original_mask_sha256'], 'source_author_partition': 'VAL/TEST repurposed OUR_TRAIN',
                     'expected_crack_positive': True})
    samples = {row['name']: row for row in json.loads((DACL / 'samples.json').read_text())['samples'] if 'name' in row}
    selected = json.loads((DACL / 'round3-selection-before-inference.json').read_text())['selections']
    assets = json.loads((DACL / 'round3-asset-manifest.json').read_text())['records']
    positive = json.loads((POS / 'asset-manifest.json').read_text())['records']
    for row in positive:
        rows.append({'name': row['source_name'], 'dataset': 'DACL', 'image': POS / row['original_image'],
                     'annotation': POS / row['original_annotation'], 'image_sha256': row['original_image_sha256'],
                     'annotation_sha256': row['original_annotation_sha256'], 'source_author_partition': 'train'})
    negatives = [r for r in assets if r['partition'] == 'negative_train']
    if len(negatives) != 20 or {r['source_name'] for r in negatives} != set(selected['negative_train']):
        raise ValueError('Expected exactly original20 TRAIN target negatives')
    for row in negatives:
        rows.append({'name': row['source_name'], 'dataset': 'DACL', 'image': DACL / row['image'],
                     'annotation': DACL / row['annotation'], 'image_sha256': row['image_sha256'],
                     'annotation_sha256': row['annotation_sha256'], 'source_author_partition': 'train',
                     'expected_crack_positive': False})
    dacl = [r for r in rows if r['dataset'] == 'DACL']
    forbidden = set(selected['negative_calibration'] + selected['negative_heldout'] + selected['damage_diagnostic'])
    if len(dacl) != 100 or len({r['name'] for r in dacl}) != 100 or forbidden & {r['name'] for r in dacl}:
        raise ValueError('DACL TRAIN identities duplicate/overlap reserved inputs')
    for row in dacl:
        sample = samples[row['name']]
        annotation = json.loads(row['annotation'].read_text())
        if sample['tags'] != ['train'] or annotation['source_split'] != 'train':
            raise ValueError('Only official TRAIN annotations are permitted')
        if annotation['polygons'] != sample['semseg_ground_truth_filled']['polylines']:
            raise ValueError('Saved annotation differs from COMPLETE pinned publisher polygons')
        if (annotation['image_width'], annotation['image_height']) != (sample['metadata']['width'], sample['metadata']['height']):
            raise ValueError('Annotation/native source geometry differs')
    if len({r['image_sha256'] for r in rows}) != 141:
        raise ValueError('Exact source-file duplicates must be reconciled before preparation')
    return rows


def main():
    cv2.setNumThreads(1)
    rows = source_rows()
    target = OUT / 'prepared-train'
    if target.exists():
        raise ValueError('Refusing to replace prepared inputs')
    target.mkdir(parents=True)
    records = []
    for row in rows:
        for field in ['image', 'annotation']:
            if sha(row[field]) != row[field + '_sha256']:
                raise ValueError('Raw source bytes differ from pinned manifest')
        frame = cv2.imread(str(row['image']))
        if frame is None:
            raise ValueError('Invalid raw source image')
        if row['dataset'] == 'LCSD':
            truth, valid = lcsd_target(np.array(Image.open(row['annotation'])))
            polygon_count = None
        else:
            annotation = json.loads(row['annotation'].read_text())
            truth, polygon_count = rasterize_cracks(annotation)
            valid = np.ones_like(truth)
        if 'expected_crack_positive' in row and bool(truth.any()) != row['expected_crack_positive']:
            raise ValueError('Claimed positive/negative contradicts actual crack annotation')
        image, label, validity, gain = prepare_pair(frame, truth, valid)
        path = target / (row['name'] + '.npz')
        np.savez_compressed(path, bgr=image, target=label, valid=validity)
        record = {k: str(v.relative_to(ROOT)) if isinstance(v, Path) else v for k, v in row.items()}
        record.update(prepared=str(path.relative_to(ROOT)), prepared_sha256=sha(path),
                      native_shape=list(truth.shape), native_positive_pixels=int(truth.sum()),
                      prepared_positive_pixels=int(label.sum()), ignore_pixels=int((validity == 0).sum()),
                      native_crack_polygon_contours=polygon_count, gain=gain,
                      prepared_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                      prepared_target_sha256=hashlib.sha256(label.tobytes()).hexdigest(),
                      role='lcsd_positive' if row['dataset'] == 'LCSD' else 'dacl_positive' if truth.any() else 'dacl_crack_negative')
        records.append(record)
    for row in rows:
        if sha(row['image']) != row['image_sha256'] or sha(row['annotation']) != row['annotation_sha256']:
            raise ValueError('Raw source changed during preparation')
    manifest = {'prepared_utc': datetime.now(timezone.utc).isoformat(), 'records': records,
                'preparation_source_sha256': sha(Path(__file__)),
                'shared_preprocessing_sha256': sha(ROOT / 'edge_runtime/profiles/construction_preprocessing.py'),
                'source_manifest_pins': {str(p.relative_to(ROOT)): digest for p, digest in PINS.items()},
                'geometry': 'Full original native frame gain exactly once, then serving-matched384square INTER_LINEAR; GT/valid INTER_NEAREST. No crop, box target, morphology or image synthesis.',
                'annotation_scope': 'Crack-only; LCSD0/2 explicit with255ignore, DACL complete officialTRAIN filled Crack/ACrack polygons checked against pinned samples metadata. Spall is unsupported, never inferred absent from LCSD.',
                'fit_started': False, 'gpu_inference_started': False}
    (OUT / 'prepared-train-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'manifest_sha256': sha(OUT / 'prepared-train-manifest.json'), 'counts': {role: sum(r['role'] == role for r in records) for role in sorted({r['role'] for r in records})}}, indent=2))


if __name__ == '__main__':
    main()
