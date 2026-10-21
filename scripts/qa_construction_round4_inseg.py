"""No-NN integrity and fixed source-order RAW/GT review sheets for new public TRAIN photos."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_construction_round4_mixed_data import source_polygons, view_geometry, actual_view

DATA = ROOT / 'datasets/construction/round4-pixel-domain/cubit-inseg'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(image, width=600, height=420):
    scale = min(width / image.shape[1], height / image.shape[0])
    small = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    panel = np.full((height, width, 3), 24, np.uint8)
    x, y = (width - small.shape[1]) // 2, (height - small.shape[0]) // 2
    panel[y:y + small.shape[0], x:x + small.shape[1]] = small
    return panel


def main():
    cv2.setNumThreads(1)
    manifest_path = DATA / 'image-assets.json'
    manifest = json.loads(manifest_path.read_text())
    selection_path = DATA / 'image-selection-before-pixels.json'
    if manifest['selection_sha256'] != sha(selection_path) or len(manifest['records']) != 64:
        raise ValueError('Exact frozen64 public TRAIN photos required')
    out = DATA / 'qa'
    out.mkdir(exist_ok=False)
    rows, seen = [], set()
    for record in manifest['records']:
        image = ROOT / record['image_path']
        if sha(image) != record['image_sha256'] or record['image_sha256'] in seen:
            raise ValueError('Original source changed or exact duplicate acquired photo; explicit quarantine needed')
        seen.add(record['image_sha256'])
        with Image.open(image) as header:
            size, orientation = list(header.size), header.getexif().get(274, 1)
        if orientation != 1 or size[0] * size[1] > 40_000_000:
            raise ValueError('Publisher polygon/image orientation or bounded geometry needs review')
        polygons = source_polygons(record, 'inseg')
        if sorted({row['class_id'] for row in polygons}) != record['class_ids_present']:
            raise ValueError('Frozen annotation classes changed')
        rows.append({'image': record['image_path'], 'sha256': record['image_sha256'], 'size_wh': size,
            'exif_orientation': orientation, 'classes': record['class_ids_present'], 'instances': len(polygons),
            'source_group_prefix': image.stem.split('_')[0], 'annotation_sha256': record['label_sha256']})
    # Fixed first/middle/last from frozen selection order for each class. These
    # are bounded annotation alignment review sheets, never model output.
    sheet_records = []
    for category, name in enumerate(['crack', 'spalling']):
        cohort = [record for record in manifest['records'] if category in record['class_ids_present']]
        for ordinal in [0, len(cohort) // 2, len(cohort) - 1]:
            record = cohort[ordinal]
            image = ROOT / record['image_path']
            raw = cv2.imread(str(image))
            polygons = source_polygons(record, 'inseg')
            views = list(view_geometry(raw, polygons))
            _, crop = views[-1]
            x, y, w, h = crop
            closeup = raw[y:y + h, x:x + w].copy()
            traced = closeup.copy()
            for polygon in polygons:
                for part in polygon['parts']:
                    points = np.rint(part * [raw.shape[1], raw.shape[0]] - [x, y]).astype(np.int32)
                    cv2.polylines(traced, [points], True, (0, 220, 255) if polygon['class_id'] == 0 else (255, 190, 50), 1)
            _, classes, boxes, masks, gain, details = actual_view(raw, polygons, crop)
            canvas = np.full((940, 1200, 3), 24, np.uint8)
            cv2.putText(canvas, f'PUBLIC TRAIN / {image.name} / fixed {ordinal + 1} of {len(cohort)} {name} / NOT MODEL OUTPUT', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .49, (240, 240, 240), 1)
            canvas[50:470, :600] = fit(raw)
            canvas[50:470, 600:] = fit(closeup)
            canvas[500:920, :600] = fit(traced)
            # Actual source masks, not predicted masks. Channels overlap.
            label_panel = np.zeros((640, 640, 3), np.uint8)
            for category_id, mask in zip(classes, masks):
                color = np.asarray((0, 220, 255) if category_id == 0 else (255, 190, 50), np.float32)
                label_panel = np.maximum(label_panel, np.clip(mask[:, :, None] * color, 0, 255).astype(np.uint8))
            canvas[500:920, 600:] = fit(label_panel)
            for label, position in [('RAW full source', (8, 46)), ('RAW native closeup (same photo)', (608, 46)),
                                    ('External source polygon contours', (8, 493)), ('Actual640 source coverage; NOT prediction', (608, 493))]:
                cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, .44, (240, 240, 240), 1)
            target = out / f'{name}-{ordinal:02d}-{image.stem}-raw-source-mask.jpg'
            if not cv2.imwrite(str(target), canvas):
                raise ValueError('QA sheet export failed')
            sheet_records.append({'source': record['image_path'], 'selection_ordinal_in_class': ordinal,
                'class': name, 'sheet': target.relative_to(ROOT).as_posix(), 'sha256': sha(target),
                'native_crop_xywh': crop, 'native_gain': gain, 'source_mask_details': details})
    result = {'data_manifest_sha256': sha(manifest_path), 'image_selection_sha256': sha(selection_path),
        'photos': len(rows), 'class_positive_photos': {name: sum(c in row['classes'] for row in rows) for c, name in enumerate(['crack', 'spalling'])},
        'source_group_prefix_counts': dict(Counter(row['source_group_prefix'] for row in rows)),
        'actual_instances': sum(row['instances'] for row in rows), 'exact_duplicate_file_hashes': 0,
        'all_annotation_coordinates_normalized_finite': True, 'all_source_orientation1': True,
        'rows': rows, 'fixed_review_sheets': sheet_records,
        'visual_review_status': 'Pending independent/root bounded review; integrity alone does not establish visible defect/annotation quality.',
        'scope': 'Official TRAIN only. No model/NN/private/CAL/heldout image access. File uniqueness is not physical-site independence.'}
    (out / 'integrity-and-fixed-sheet-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print('QA', len(rows), 'photos', result['actual_instances'], 'actualinstances', flush=True)


if __name__ == '__main__':
    main()
