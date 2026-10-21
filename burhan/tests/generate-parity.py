"""Regenerate compact decoder fixtures using the existing Python/OpenCV reference.

Run from burhan: ..\\.venv\\Scripts\\python.exe tests/generate-parity.py
The input is generated here (no external image or image licence). These are
numerical regression fixtures, not an accuracy evaluation dataset.
"""
from pathlib import Path
import gzip
import hashlib
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP.parent))
from edge_runtime.profiles.construction_model import prepare, decode
from edge_runtime.profiles.construction_dacl import decode_masks, DAMAGE_CLASSES

OUT = APP / 'tests/fixtures'
OUT.mkdir(parents=True, exist_ok=True)
width, height = 193, 127
y, x = np.indices((height, width))
rgb = np.stack([(x * 3 + y * 2) % 256, (x + y * 4) % 256, (x * 2 + y) % 256], axis=2).astype(np.uint8)
# Seeded texture and irregular dark paths exercise spatial interpolation.
noise = np.random.default_rng(123).integers(0, 25, (height, width, 1))
rgb = np.clip(rgb.astype(int) + noise, 0, 255).astype(np.uint8)
for path in [np.array([[20, 5], [34, 28], [29, 47], [45, 66], [44, 100], [58, 124]]), np.array([[1, 82], [46, 78], [88, 89], [145, 85], [192, 101]])]:
    cv2.polylines(rgb, [path], False, (5, 6, 7), 2)
rgba = np.concatenate([rgb, np.full((height, width, 1), 255, np.uint8)], axis=2)
(OUT / 'procedural.rgba').write_bytes(rgba.tobytes())
cv2.imwrite(str(OUT / 'procedural.png'), rgb[:, :, ::-1])

def packed(name, array):
    raw = np.ascontiguousarray(array).tobytes()
    file = name + '.gz'
    (OUT / file).write_bytes(gzip.compress(raw, mtime=0))
    return {'file': file, 'sha256': hashlib.sha256(raw).hexdigest(), 'dtype': str(array.dtype), 'dims': list(array.shape)}

def encode_mask(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.r_[0, change, len(flat)]).tolist()
    if flat[0]: runs.insert(0, 0)
    return {'width': mask.shape[1], 'height': mask.shape[0], 'runs': runs}

manifest = json.loads((APP / 'assets/models/manifest.json').read_text())
reference = {'description': 'Procedural numerical parity only; no accuracy claim. Full ONNX outputs, no anchor pruning or quantization.', 'input': {'file': 'procedural.rgba', 'png': 'procedural.png', 'sha256': hashlib.sha256(rgba.tobytes()).hexdigest(), 'width': width, 'height': height}, 'python': {'opencv': cv2.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__}, 'models': []}
for model in manifest:
    size = model['inputSize']
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if model['kind'] == 'dacl':
        tensor = np.ascontiguousarray(cv2.resize(bgr, (size, size)).transpose(2, 0, 1)[None])
        transform = None
    else:
        tensor, transform = prepare(bgr, model['mode'], 'letterbox', size)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(APP / 'assets/models' / model['file']), options, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    if model['kind'] == 'dacl':
        probabilities = 1 / (1 + np.exp(-np.clip(outputs[0][0], -40, 40)))
        observations = decode_masks(probabilities, (height, width), DAMAGE_CLASSES, model['confidence'], 4, 'masks')
    else:
        observations = decode(outputs, model['mode'], transform, (height, width), model['confidence'], model['nms'], 'masks', model['classes'])
    gold = []
    for row in observations:
        item = {'classId': int(row['class_id']), 'label': row['label'], 'score': float(row['score']), 'bbox': [int(v) for v in row['bbox']]}
        if 'mask' in row: item['mask'] = encode_mask(row['mask'])
        gold.append(item)
    reference['models'].append({'id': model['id'], 'modelSha256': model['sha256'], 'input': packed(model['id'] + '-input', tensor), 'outputs': [packed(model['id'] + '-output' + str(i), output) for i, output in enumerate(outputs)], 'observations': gold})
    print(model['id'], len(gold), 'observations', flush=True)
(OUT / 'reference.json').write_text(json.dumps(reference, indent=2), encoding='utf-8')
