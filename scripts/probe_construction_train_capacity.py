"""Bounded TRAIN-only batch4/8/16 fit-capacity probe; never save trained weights."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_model import prepare

DATA = ROOT / 'datasets/construction/evcivil-round4-gain'
SOURCE = ROOT / 'output/training/construction-round3/evcivil-yolov8n-r3/weights/best.pt'
OUT = ROOT / 'output/training/construction-round4-capacity'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_batch(rows):
    images, classes, boxes, batch_indices = [], [], [], []
    for index, row in enumerate(rows):
        if digest(ROOT/row['image']) != row['image_sha256'] or digest(ROOT/row['label']) != row['label_sha256']:
            raise RuntimeError('Frozen training probe image or label changed')
        frame = cv2.imread(str(ROOT / row['image']))
        tensor, transform = prepare(frame, 'damage', size=640)
        images.append(tensor[0])
        sx, sy, px, py = transform
        height, width = frame.shape[:2]
        for line in (ROOT / row['label']).read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = map(float, line.split())
            classes.append([cls])
            boxes.append([(cx*width*sx+px)/640, (cy*height*sy+py)/640,
                          bw*width*sx/640, bh*height*sy/640])
            batch_indices.append(index)
    return {'img': torch.from_numpy(np.stack(images)),
            'cls': torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
            'bboxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'batch_idx': torch.tensor(batch_indices, dtype=torch.float32)}


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((DATA/'native-input-manifest.json').read_text(encoding='utf8'))
    selected = sorted([r for r in inputs if r['split']=='train'],
                      key=lambda r: r['raw_source_file_sha256'])[:16]
    protocol = {'schema_version': 1, 'frozen_before_probe': True,
                'source_checkpoint_sha256': digest(SOURCE), 'image_size': 640,
                'freeze_first_blocks': 10, 'cpu_threads': 4,
                'batch_trials': [4, 8, 16], 'maximum_steps_per_trial': 4,
                'timing': 'First optimizer step warmup; next3 measured; all steps TRAIN only',
                'source_reset': 'Identical retained local checkpoint, fresh optimizer and seed for each trial; no checkpoint saved',
                'maximum_rss_bytes': 6*1024**3,
                'selection': 'Highest measured training images/s among successful trials within6GiB RSS; no calibration or heldout performance used',
                'native_input_manifest_sha256': digest(DATA/'native-input-manifest.json'),
                'training_contention':'A fixed one-thread factory fit may overlap; training capacity is contended, never an inference qualification',
                'ordered_training_images': [r['image'] for r in selected],
                'scope': 'Training capacity only; not inference FPS or inspection accuracy'}
    path = OUT/'preregistered-protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != protocol:
        raise RuntimeError('Frozen capacity protocol changed')
    path.write_text(json.dumps(protocol, indent=2), encoding='utf8')
    results = []
    process = psutil.Process()
    for batch_size in protocol['batch_trials']:
        torch.manual_seed(32026)
        network = YOLO(str(SOURCE), task='detect').model.float().train()
        # Loading a checkpoint through YOLO retains only inference arguments;
        # restore official training defaults before asking for actual loss.
        network.args = get_cfg(overrides=dict(network.args))
        network.args.cls_pw = .5
        if network.names != {0:'spalling', 1:'crack'}:
            raise RuntimeError('Actual retained head semantics changed')
        for name, parameter in network.named_parameters():
            frozen = '.dfl.' in name or any(name.startswith(f'model.{index}.') for index in range(10))
            parameter.requires_grad_(not frozen)
        optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=.0005)
        batch = build_batch(selected[:batch_size])
        initial_memory = process.memory_info()
        times, peak_rss = [], max(initial_memory.rss, getattr(initial_memory, 'peak_wset', initial_memory.rss))
        error = None
        try:
            for step in range(4):
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, _ = network.loss(batch)
                loss.sum().backward()
                optimizer.step()
                times.append(time.perf_counter()-started)
                memory = process.memory_info()
                peak_rss = max(peak_rss, memory.rss, getattr(memory, 'peak_wset', memory.rss))
                if peak_rss > protocol['maximum_rss_bytes']:
                    error = 'Exceeded preregistered6GiB process RSS guard'
                    break
        except RuntimeError as failure:
            error = str(failure)
        result = {'batch_size':batch_size, 'steps_completed':len(times),
                  'step_seconds':times, 'peak_process_rss_bytes':peak_rss,
                  'memory_scope':'Windows process peak working set when available; otherwise maximum sampled post-step RSS; lifetime process peak includes earlier smaller trials',
                  'training_images_per_second':batch_size/float(np.mean(times[1:])) if len(times)>1 else None,
                  'status':'passed' if error is None and len(times)==4 else 'rejected', 'error':error}
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        del network, optimizer, batch
        import gc
        gc.collect()
    passed = [r for r in results if r['status']=='passed']
    if not passed:
        raise RuntimeError('No safe measured training batch candidate')
    choice = max(passed, key=lambda r:r['training_images_per_second'])
    result = {'protocol_sha256':digest(path), 'results':results,
              'selected_batch_size':choice['batch_size'],
              'scope':'TRAIN capacity only; identical checkpoint reset per trial; no training output saved, no calibration/heldout input'}
    (OUT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__':
    main()
