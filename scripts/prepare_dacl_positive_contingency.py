"""Acquire separate real DACL training positives; never edits an active experiment."""
from pathlib import Path
import hashlib
import json
import math
import sys
from PIL import Image, ImageDraw, ImageChops

from fetch_dacl_round3 import BASE, COMMIT, DATA, ROOT, classes, digest, download

OUT = DATA / 'round3-positive-contingency'
CLASS_MAP = {'Spalling': 0, 'Crack': 1, 'ACrack': 1}
COHORTS = {'crack_family': {'Crack', 'ACrack'}, 'spalling': {'Spalling'}}
SEED = 'dacl-round3-positive-contingency-20261020:'


def polygon_id(poly, index):
    return poly.get('_id', {}).get('$oid', f'polygon-index-{index}')


def clipped_mask(poly, width, height, crop):
    x, y, cw, ch = crop
    mask = Image.new('1', (cw, ch))
    draw = ImageDraw.Draw(mask)
    for contour in poly['points']:
        points = [(float(px) * width - x, float(py) * height - y) for px, py in contour]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return mask


def derive(row, cohort, image_path, *, crop_override=None, output_stem=None, geometry_description=None):
    name = row['name']
    original = Image.open(image_path).convert('RGB')
    width, height = original.size
    if (width, height) != (row['metadata']['width'], row['metadata']['height']):
        raise RuntimeError('Decoded image geometry differs from pinned source metadata')
    polygons = row['semseg_ground_truth_filled']['polylines']
    targets = [(polygon_id(poly, index), index, poly) for index, poly in enumerate(polygons)
               if poly['label'] in COHORTS[cohort]]
    _, anchor_index, anchor = min(targets)
    # TRAIN ONLY: center the largest native4:3 crop around the first stable-ID
    # target polygon bbox. No model scores or validation photos influence it.
    points = [point for contour in anchor['points'] for point in contour]
    cx = (min(p[0] for p in points) + max(p[0] for p in points)) * width / 2
    cy = (min(p[1] for p in points) + max(p[1] for p in points)) * height / 2
    units = min(width // 4, height // 3)
    cw, ch = units * 4, units * 3
    left = min(width - cw, max(0, math.floor(cx - cw / 2)))
    top = min(height - ch, max(0, math.floor(cy - ch / 2)))
    crop = (left, top, cw, ch)
    if crop_override is not None:
        left, top, cw, ch = crop = crop_override
        if min(left, top) < 0 or cw < 1 or ch < 1 or left+cw > width or top+ch > height or cw*3 != ch*4:
            raise RuntimeError('Override crop must be exact4:3 and within original image')
    if not clipped_mask(anchor, width, height, crop).getbbox():
        raise RuntimeError('Frozen target-centered crop contains no rasterized target pixels')
    derived = original.crop((left, top, left + cw, top + ch)).convert('L').resize((640, 480), Image.Resampling.BILINEAR).convert('RGB')
    stem = output_stem or name
    image_out = OUT / 'derived/images/train' / (stem + '.png')
    label_out = OUT / 'derived/labels/train' / (stem + '.txt')
    detail_out = OUT / 'derived/annotations/train' / (stem + '.json')
    for path in (image_out, label_out, detail_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    derived.save(image_out, compress_level=1)
    union = {key: Image.new('1', (640, 480)) for key in [0, 1]}
    instances, yolo = [], []
    for index, poly in enumerate(polygons):
        if poly['label'] not in CLASS_MAP:
            continue
        source_mask = clipped_mask(poly, width, height, crop)
        source_box = source_mask.getbbox()
        if source_box is None:
            continue
        mask = source_mask.resize((640, 480), Image.Resampling.NEAREST)
        mask_box = mask.getbbox()
        # Preserve every genuinely visible polygon, including sub-output-pixel
        # cracks that a nearest-neighbour semantic mask can erase. The box is
        # the continuous projection of the actually clipped SOURCE mask bbox.
        box = (source_box[0] * 640 / cw, source_box[1] * 480 / ch,
               source_box[2] * 640 / cw, source_box[3] * 480 / ch)
        cls = CLASS_MAP[poly['label']]
        union[cls] = ImageChops.logical_or(union[cls], mask)
        x1, y1, x2, y2 = box
        yolo.append(f'{cls} {(x1+x2)/1280:.10f} {(y1+y2)/960:.10f} {(x2-x1)/640:.10f} {(y2-y1)/480:.10f}')
        instances.append({'source_polygon_index': index, 'source_polygon_id': polygon_id(poly, index),
                          'source_label': poly['label'], 'class_id': cls,
                          'source_crop_clipped_mask_bbox_xyxy': source_box,
                          'output_nearest_mask_bbox_xyxy': mask_box,
                          'output_projected_source_bbox_xyxy': box})
    if not instances or not any(item['source_label'] in COHORTS[cohort] for item in instances):
        raise RuntimeError('No retained cohort annotation after nearest-neighbor projection')
    label_out.write_text('\n'.join(yolo) + '\n', encoding='utf-8')
    mask_paths = {}
    for cls, mask in union.items():
        target = OUT / 'derived/masks/train' / f'{stem}-class{cls}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        mask.save(target)
        mask_paths[str(cls)] = {'path': target.relative_to(OUT).as_posix(), 'sha256': digest(target)}
    details = {'source_name': name, 'source_split': 'train', 'cohort': cohort,
               'source_rgb_sha256': digest(image_path), 'source_size': [width, height],
               'crop_xywh': list(crop), 'output_size': [640, 480], 'anchor_polygon_index': anchor_index,
               'anchor_polygon_id': polygon_id(anchor, anchor_index),
               'geometry': geometry_description or 'Largest exact-native4:3 crop, stable-ID target-bbox centered then image-clamped; gray and bilinear resize640x480; GT nearest-neighbor. TRAIN ONLY.',
               'class_map': CLASS_MAP, 'instances': instances, 'class_masks': mask_paths,
               'annotation_scope': 'All Crack/ACrack/Spalling source polygons with nonzero clipped SOURCE pixels retained, independent of cohort. Boxes continuously project each clipped source-mask bbox, preserving sub-output-pixel cracks even when nearest semantic masks vanish. Other source labels preserved in complete original annotation but not YOLO targets. Boxes derived from semantic polygons, not native instance boxes.',
               'usage': 'Unused contingency training data; not active experiment, calibration, heldout, video or model evidence.'}
    detail_out.write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    return {**details, 'image': image_out.relative_to(OUT).as_posix(), 'image_sha256': digest(image_out),
            'label': label_out.relative_to(OUT).as_posix(), 'label_sha256': digest(label_out),
            'derivation': detail_out.relative_to(OUT).as_posix(), 'derivation_sha256': digest(detail_out)}


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    if digest(DATA / 'samples.json') != '329e842ee83c700c3717b629ea199faa7a9d3451b2d5fff5923b5f44e8abc588':
        raise RuntimeError('Pinned mirror metadata hash changed')
    existing = json.loads((DATA / 'round3-asset-manifest.json').read_text(encoding='utf-8'))
    excluded_names = {r['source_name'] for r in existing['records']}
    excluded_hashes = {r['image_sha256'] for r in existing['records']}
    samples = json.loads((DATA / 'samples.json').read_text(encoding='utf-8'))['samples']
    samples = [r for r in samples if r['tags'] == ['train'] and r['name'] not in excluded_names]
    candidates = {key: sorted([r for r in samples if classes(r) & labels],
                              key=lambda r: hashlib.sha256((SEED + r['name']).encode()).hexdigest())
                  for key, labels in COHORTS.items()}
    protocol = {'source': 'https://github.com/phiyodr/dacl10k-toolkit', 'mirror_commit': COMMIT,
                'license': 'CC BY-NC4.0 author data terms prevail over mirror CC BY4 card',
                'selection': 'For crack_family then spalling: first40 eligible officialTRAIN names by SHA256(seed+name), skipping previously selected names and exact RGB-byte hash duplicates including all95 existing assets. No model outputs used.',
                'seed': SEED, 'excluded_existing_names': sorted(excluded_names),
                'excluded_existing_sha256': sorted(excluded_hashes),
                'ordered_candidate_names': {key: [r['name'] for r in rows] for key, rows in candidates.items()},
                'crop_protocol': 'Largest exact4:3 native crop centered at stable-lowest-polygon-ID target bbox; clamp to image, verify nonzero target pixels; grayscale bilinear640x480 RGBPNG. All visible crack/spall polygons clipped via rasterization and nearest-neighbor labels.',
                'group_scope': 'Source-image/hash disjoint from existing95 assets; physical bridge IDs unavailable, no bridge independence claim.',
                'experiment_status': 'CONTINGENCY ONLY, UNUSED. No current experiment or calibration/heldout changes. GT-centered selection is training-only, never evaluation.'}
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / 'protocol-before-download.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Frozen contingency acquisition protocol changed')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    names, hashes, records, rejected = set(excluded_names), set(excluded_hashes), [], []
    for cohort, rows in candidates.items():
        count = 0
        for row in rows:
            if count == 40:
                break
            if row['name'] in names:
                continue
            relative = Path(row['filepath'])
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unsafe mirror path')
            image = OUT / 'original/images' / relative.name
            download(BASE + row['filepath'], image)
            if image.stat().st_size != row['metadata']['size_bytes']:
                raise RuntimeError('Source byte length changed from pinned manifest')
            sha = digest(image)
            if sha in hashes:
                rejected.append({'source_name': row['name'], 'sha256': sha, 'reason': 'Exact source-byte duplicate; not selected or derived'})
                continue
            names.add(row['name']); hashes.add(sha)
            annotation = {'source_image': row['name'], 'source_file': row['filepath'], 'source_split': 'train',
                          'image_width': row['metadata']['width'], 'image_height': row['metadata']['height'],
                          'labels': sorted(classes(row)), 'polygon_format': 'Normalized xy coordinates; labels can overlap',
                          'polygons': row['semseg_ground_truth_filled']['polylines']}
            ann = OUT / 'original/annotations' / (row['name'] + '.json')
            ann.parent.mkdir(parents=True, exist_ok=True)
            ann.write_text(json.dumps(annotation, indent=2) + '\n', encoding='utf-8')
            derived = derive(row, cohort, image)
            records.append({'source_name': row['name'], 'cohort': cohort, 'source_split': 'train',
                            'original_image': image.relative_to(OUT).as_posix(), 'original_image_sha256': sha,
                            'original_annotation': ann.relative_to(OUT).as_posix(), 'original_annotation_sha256': digest(ann),
                            'source_url': BASE + row['filepath'], 'source_labels': sorted(classes(row)), 'derived': derived})
            count += 1
            print(cohort, count, row['name'], len(derived['instances']), 'boxes', flush=True)
    manifest = {'protocol_sha256': digest(protocol_path), 'usage': protocol['experiment_status'],
                'class_names': ['spalling', 'crack'], 'class_map': CLASS_MAP,
                'license': protocol['license'], 'records': records, 'rejected_duplicate_hashes': rejected,
                'counts': {key: sum(r['cohort'] == key for r in records) for key in COHORTS}}
    path = OUT / 'asset-manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(path), 'sha256': digest(path), 'counts': manifest['counts']}, indent=2))


if __name__ == '__main__':
    main()
