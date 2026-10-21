"""Bounded local adapter benchmark on varied original public photos, no zero-frame timing."""
from pathlib import Path
import argparse
import json
import sys
import time
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_dacl import DaclConstructionModel, CLASSES, CAPABILITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=[320, 512], required=True)
    parser.add_argument('--engine', choices=['openvino', 'onnxruntime'], default='openvino')
    parser.add_argument('--device', choices=['GPU', 'CPU'], default='GPU')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    data = ROOT / 'datasets/construction/dacl10k'
    manifest = json.loads((data / 'round3-asset-manifest.json').read_text(encoding='utf-8'))
    selected = [row for row in manifest['records'] if row['partition'] == 'damage_diagnostic'][:6]
    frames = []
    for row in selected:
        image = cv2.imread(str(data / row['image']))
        scale = min(1, 1920 / max(image.shape[:2]))
        frames.append(cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale))))
    exported = json.loads((ROOT / f'models/construction/dacl19/export-{args.size}.json').read_text(encoding='utf-8'))
    spec = {'file': f'dacl19/dacl-segformer-b1-{args.size}.onnx', 'sha256': exported['onnx_sha256'],
            'input_size': args.size, 'classes': CLASSES, 'capabilities': CAPABILITIES,
            'source': 'https://huggingface.co/spaces/phiyodr/dacl-challenge/tree/f5bee34782e7e02393ba4f206a95ba5ed67eb8a5'}
    model = DaclConstructionModel({'engine': args.engine, 'openvino_device': args.device, 'mask_rendering': 'boxes'}, spec)
    model.infer(frames[0])
    result = {'scope': 'Varied six real public photos, two passes each; preload/JPEG decode and model initialization excluded from warm timings. Not moving video, shared-runtime or accuracy evidence.',
              'input_assets': [r['source_name'] for r in selected], 'analysis_max_edge': 1920, 'model': model.describe(), 'modes': {}}
    for rendering in ['boxes', 'masks']:
        model.mask_rendering = rendering
        records = []
        for index in range(12):
            model.infer(frames[index % len(frames)])
            records.append(dict(model.last_timing_ms))
        result['modes'][rendering] = {key: {'mean': float(np.mean([r[key] for r in records])),
                                          'p50': float(np.median([r[key] for r in records])),
                                          'p95': float(np.percentile([r[key] for r in records], 95))} for key in records[0]}
        result['modes'][rendering]['serial_adapter_fps'] = 1000 / result['modes'][rendering]['adapter']['mean']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['modes'], indent=2))


if __name__ == '__main__':
    main()
