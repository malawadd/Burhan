"""Evaluate actual DACL local masks on a frozen real-photo diagnostic subset.

This launches inference: run only during an exclusive compute window.
Ground-truth masks never enter inference; figures keep raw/GT/prediction separate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_dacl import DaclConstructionModel, AUTHOR_CLASSES, CLASSES, DAMAGE_CLASSES, CAPABILITIES
from factory_localization_audit import mask_metrics


def truth_mask(annotation, class_name):
    """Pillow polygon rasterization, preserving author's multilabel union rule."""
    width, height = annotation['image_width'], annotation['image_height']
    image = Image.new('L', (width, height), 0)
    drawing = ImageDraw.Draw(image)
    author_name = AUTHOR_CLASSES[CLASSES.index(class_name)]
    for polygon in annotation['polygons']:
        if polygon['label'] != author_name:
            continue
        for contour in polygon['points']:
            drawing.polygon([(float(x) * width, float(y) * height) for x, y in contour], outline=1, fill=1)
    return np.asarray(image, dtype=bool)


def render_case(frame, expected, predicted, name, target):
    h, w = frame.shape[:2]
    scale = min(1, 480 / max(h, w))
    raw = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    panels = [raw.copy(), raw.copy(), raw.copy()]
    for index, mask in [(1, expected), (2, predicted)]:
        selected = cv2.resize(mask.astype(np.uint8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        panels[index][selected] = (.45 * panels[index][selected] + .55 * np.array((30, 100, 240))).astype(np.uint8)
    canvas = np.full((raw.shape[0] + 70, raw.shape[1] * 3, 3), 22, np.uint8)
    for index, (panel, label) in enumerate(zip(panels, ['RAW CAMERA PHOTO', 'PUBLISHER ANNOTATION', 'ACTUAL LOCAL MODEL MASK'])):
        x = index * raw.shape[1]
        canvas[45:45 + raw.shape[0], x:x + raw.shape[1]] = panel
        cv2.putText(canvas, label, (x + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)
    cv2.putText(canvas, f'{name} | original photo, not a video | no engineering/code acceptance', (5, canvas.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)
    cv2.imwrite(str(target), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=[320, 512], default=512)
    parser.add_argument('--engine', choices=['openvino', 'onnxruntime'], default='openvino')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    data = ROOT / 'datasets/construction/dacl10k'
    manifest_path = data / 'round3-asset-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = [row for row in manifest['records'] if row['partition'] == 'damage_diagnostic']
    config = json.loads((ROOT / 'models/construction/dacl19' / f'export-{args.size}.json').read_text(encoding='utf-8'))
    spec = {'file': f'dacl19/dacl-segformer-b1-{args.size}.onnx', 'sha256': config['onnx_sha256'], 'classes': CLASSES,
            'capabilities': CAPABILITIES, 'input_size': args.size, 'source': 'https://huggingface.co/spaces/phiyodr/dacl-challenge/tree/f5bee34782e7e02393ba4f206a95ba5ed67eb8a5'}
    protocol = {'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'images': [row['source_name'] for row in records], 'image_count': len(records),
                'threshold': .5, 'minimum_component_grid_pixels': 4, 'largest_components_per_class': 20,
                'largest_detections_total': 100, 'mask_projection': 'Nearest-neighbor independent class maps',
                'gt_rasterization': 'Author Pillow polygon fill convention restored from mirror normalized coordinates; conversion precision may differ at individual boundary pixels.',
                'selection': 'Frozen before all inference; source-metadata class-stratified photos, no choosing images by model output',
                'scope': 'Small public author-validation photo diagnostic; unknown model-training/selection overlap, no independent bridge or Saudi validation. Photos are not video.',
                'score_scope': 'Foreground pixels only; no background true-negative accuracy. Connected semantic components are derived region diagnostics, not native instance labels.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'protocol-before-inference.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    model = DaclConstructionModel({'engine': args.engine, 'openvino_device': args.device, 'confidence': .5,
                                   'minimum_component_pixels': 4, 'mask_rendering': 'masks', 'damage_classes': DAMAGE_CLASSES}, spec)
    rows, durations, visualized = [], [], set()
    for asset in records:
        image = data / asset['image']
        if hashlib.sha256(image.read_bytes()).hexdigest() != asset['image_sha256']:
            raise RuntimeError('Frozen image hash mismatch')
        frame = cv2.imread(str(image))
        annotation = json.loads((data / asset['annotation']).read_text(encoding='utf-8'))
        if frame.shape[:2] != (annotation['image_height'], annotation['image_width']):
            raise RuntimeError('Source image and polygon geometry mismatch')
        began = time.perf_counter()
        predictions = model.infer(frame)
        durations.append((time.perf_counter() - began) * 1000)
        result = {'source_name': asset['source_name'], 'source_sha256': asset['image_sha256'], 'classes': {}}
        for name in DAMAGE_CLASSES:
            expected = truth_mask(annotation, name)
            predicted = np.zeros(expected.shape, dtype=bool)
            for finding in predictions:
                if finding['label'] == name:
                    predicted |= finding['mask']
            values = mask_metrics(predicted, expected)
            result['classes'][name] = values
            if expected.any() and name not in visualized:
                render_case(frame, expected, predicted, name, args.output / f'{name}-raw-annotation-model.jpg')
                visualized.add(name)
        rows.append(result)
        print(asset['source_name'], len(predictions), 'candidates', flush=True)
    per_class = {}
    for name in DAMAGE_CLASSES:
        observations = [r['classes'][name] for r in rows]
        pixel_tp = sum(r['pixel_tp'] for r in observations)
        pixel_pred = sum(r['pixel_predicted'] for r in observations)
        pixel_target = sum(r['pixel_target'] for r in observations)
        region_tp = sum(r['region_matches_at_iou']['0.5'] for r in observations)
        gt_regions = sum(r['gt_regions'] for r in observations)
        pred_regions = sum(r['predicted_regions'] for r in observations)
        positives = [r for r in observations if r['pixel_target'] > 0]
        negative = [r for r in observations if not r['pixel_target']]
        per_class[name] = {'positive_images': len(positives), 'negative_images': len(negative),
                           'pixel_tp': pixel_tp, 'pixel_fp': pixel_pred - pixel_tp, 'pixel_fn': pixel_target - pixel_tp,
                           'pixel_precision': pixel_tp / max(1, pixel_pred), 'pixel_recall': pixel_tp / max(1, pixel_target),
                           'pixel_iou': pixel_tp / max(1, pixel_pred + pixel_target - pixel_tp),
                           'derived_region_tp_iou_0_5': region_tp, 'derived_region_fp_iou_0_5': pred_regions - region_tp,
                           'derived_region_fn_iou_0_5': gt_regions - region_tp,
                           'localized_positive_images_any_pixel': sum(r['localized_any_pixel'] for r in positives),
                           'positive_images_10pct_coverage': sum(r['localized_10pct_coverage'] for r in positives),
                           'positive_images_50pct_coverage': sum(r['localized_50pct_coverage'] for r in positives),
                           'negative_images_flagged': sum(r['flagged'] for r in negative)}
    report = {'protocol': protocol, 'model': model.describe(), 'per_class': per_class, 'records': rows,
              'timing': {'observations': len(durations), 'full_mask_adapter_ms_mean': float(np.mean(durations)),
                         'full_mask_adapter_ms_p95': float(np.percentile(durations, 95)),
                         'scope': 'Includes preprocessing, inference, per-class component decode and full-original masks; excludes JPEG decode, GT/matching/figure export. Accuracy run, not isolated sustained shared-runtime benchmark.'}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'per_class': per_class, 'timing': report['timing']}, indent=2))


if __name__ == '__main__':
    main()
