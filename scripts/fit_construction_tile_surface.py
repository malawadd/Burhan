"""CPU1 normal-only real tile fit/cal/test; no GPU and no installation claims."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import time
import cv2
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_tile import TileSurfaceModel
from factory_localization_audit import mask_metrics
from evaluate_dacl_round3 import render_case
DATA = ROOT / 'datasets/construction/finishes-expansion/mvtec-tile'
OUT = ROOT / 'output/evaluation/construction-finishes-tile'
MODEL = ROOT / 'models/construction/finishes-tile'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def original(record, key='source_path'):
    path = DATA / 'original' / str(PurePosixPath(record[key]).relative_to('MVTec-AD/tile'))
    expected = record['sha256' if key == 'source_path' else 'mask_sha256']
    if sha(path) != expected:
        raise RuntimeError('Original tile data changed')
    return path


class Encoder(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features[:4]
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).reshape(1,3,1,1))

    def forward(self, bgr):
        x = (bgr[:, [2,1,0]].float() / 255 - self.mean) / self.std
        first = None
        for index, block in enumerate(self.features):
            x = block(x)
            if index == 0:
                first = x
        return torch.cat([torch.nn.functional.adaptive_avg_pool2d(first, (32,32)), x], 1)


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    if (OUT / 'report.json').exists() or (MODEL / 'model.json').exists():
        raise RuntimeError('Preserve completed tile candidate; use a distinct protocol for any new experiment')
    manifest_path = DATA / 'asset-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    weights = ROOT / 'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
    if sha(weights) != '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f':
        raise RuntimeError('Official local ImageNet encoder differs')
    protocol = {'source_manifest_sha256': sha(manifest_path), 'source_protocol_sha256': sha(DATA / 'protocol-before-acquisition.json'),
                'trainer_sha256': sha(Path(__file__)), 'adapter_sha256': sha(ROOT / 'edge_runtime/profiles/construction_tile.py'),
                'weights_sha256': sha(weights), 'source': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
                'torch_load': 'weights_only=True; original local official weights; no unrestricted pickle',
                'preprocessing': 'Original BGR -> INTER_AREA256; RGB/ImageNet inside encoder; concatenate pooled layer0 and layer3 at32x32',
                'method': 'Standardize40 channels by FIT-only patch mean/std (floor.05); seed20261020 sample1024 FIT patches; exact nearest Euclidean normal distance; Gaussian5x5 grid smoothing then native bilinear projection',
                'threshold_rule': 'np.quantile(.95) of10 normal CAL image maxima; strict greater-than; no defect calibration or test selection',
                'scope': 'Manufacturing still-photo diagnostic; all117 original test photos. Not installed flooring/video/SBC. Host CPU work overlaps other accuracy jobs; incidental times are NOT capacity.',
                'test_opening': 'No test image decode/inference until model+threshold JSON frozen below; original testGT never enters fitting/inference',
                'operating_point_selection': 'ONE256/1024bank candidate, no data/model/threshold search after test'}
    write(OUT / 'protocol-before-any-fit.json', protocol)
    backbone = torchvision.models.mobilenet_v3_small(weights=None)
    backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    encoder = Encoder(backbone).eval()
    encoder_path = MODEL / 'mobilenet-early-tile-256.onnx'
    with torch.no_grad():
        torch.onnx.export(encoder, torch.zeros((1,3,256,256), dtype=torch.uint8), str(encoder_path),
                          input_names=['bgr'], output_names=['features'], opset_version=17, dynamo=False)
    # Temporary in-memory Torch extraction is TRAIN-only and uses identical encoder/preprocessing.
    fit = [r for r in manifest['records'] if r['partition'] == 'normal_fit']
    descriptors = []
    for row in fit:
        frame = cv2.imread(str(original(row)))
        resized = cv2.resize(frame, (256,256), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            features = encoder(torch.from_numpy(np.ascontiguousarray(resized.transpose(2,0,1)[None]))).numpy()[0]
        descriptors.append(features.reshape(40,-1).T)
    points = np.concatenate(descriptors)
    mean, scale = points.mean(0), np.maximum(points.std(0), .05)
    chosen = np.random.default_rng(20261020).choice(len(points), 1024, replace=False)
    bank = ((points[chosen]-mean)/scale).astype(np.float32)
    bank_path = MODEL / 'normal-patches.npz'
    np.savez_compressed(bank_path, bank=bank, mean=mean, scale=scale)
    spec = {'encoder': {'file': encoder_path.name, 'sha256': sha(encoder_path)},
            'bank': {'file': bank_path.name, 'sha256': sha(bank_path)}, 'threshold': 1.0,
            'license': 'MVTec AD CC BY-NC-SA4.0; torchvision code BSD3, official ImageNet weight/source terms apply',
            'fit_protocol_sha256': sha(OUT / 'protocol-before-any-fit.json'), 'fit_photos': len(fit), 'normal_calibration_photos': 10}
    temporary = MODEL / 'calibration-only-model.json'
    write(temporary, spec)
    model = TileSurfaceModel(temporary)
    # Verify actual exported neural features on one TRAIN photo, never a test view.
    first = cv2.imread(str(original(fit[0])))
    tensor = torch.from_numpy(np.ascontiguousarray(cv2.resize(first,(256,256),interpolation=cv2.INTER_AREA).transpose(2,0,1)[None]))
    with torch.no_grad():
        expected = encoder(tensor).numpy()[0]
    difference = float(np.max(np.abs(model.features(first)-expected)))
    if difference > .001:
        raise RuntimeError('Torch/actualCPU ONNX encoder parity failed')
    calibration = [{'source_path': r['source_path'], 'maximum_anomaly': float(model.predict_maps(cv2.imread(str(original(r)))).max())}
                   for r in manifest['records'] if r['partition'] == 'normal_calibration']
    spec['threshold'] = float(np.quantile([r['maximum_anomaly'] for r in calibration], .95))
    spec['calibration'] = calibration
    spec['torch_onnx_max_abs_error_train_photo'] = difference
    write(MODEL / 'model.json', spec)
    model = TileSurfaceModel(MODEL / 'model.json')
    print('FROZEN model', sha(MODEL / 'model.json'), 'threshold', model.threshold, flush=True)
    # Test access begins only after frozen bank/operating point, never tune from it.
    rows, visualized = [], set()
    for row in manifest['records']:
        if row['partition'] != 'test':
            continue
        frame = cv2.imread(str(original(row)))
        expected = cv2.imread(str(original(row,'mask_source_path')), cv2.IMREAD_GRAYSCALE)>0 if row['mask_source_path'] else np.zeros(frame.shape[:2],bool)
        began = time.perf_counter()
        anomaly = model.predict_maps(frame)
        elapsed = time.perf_counter()-began
        predicted = anomaly > model.threshold
        values = mask_metrics(predicted, expected)
        rows.append({'source_path': row['source_path'], 'source_sha256': row['sha256'], 'defect_type': row['defect_type'],
                     'maximum_anomaly': float(anomaly.max()), 'flagged': bool(predicted.any()), 'metrics': values,
                     'incidental_adapter_seconds': elapsed})
        if row['defect_type'] not in visualized:
            render_case(frame, expected, predicted, 'TILE '+row['defect_type'], OUT / (row['defect_type']+'-raw-gt-model.jpg'))
            visualized.add(row['defect_type'])
    by_class = {}
    for name in sorted({r['defect_type'] for r in rows}):
        group = [r for r in rows if r['defect_type'] == name]
        tp=sum(r['metrics']['pixel_tp'] for r in group)
        predicted=sum(r['metrics']['pixel_predicted'] for r in group)
        target=sum(r['metrics']['pixel_target'] for r in group)
        by_class[name]={'photos':len(group),'flagged_photos':sum(r['flagged'] for r in group),'pixel_tp':tp,'pixel_fp':predicted-tp,'pixel_fn':target-tp,
                        'pixel_precision':tp/predicted if predicted else None,'pixel_recall':tp/target if target else None,
                        'pixel_iou':tp/(predicted+target-tp) if predicted+target-tp else None,
                        'derived_region_tp_iou_0_5':sum(r['metrics']['region_matches_at_iou']['0.5'] for r in group),
                        'derived_gt_regions':sum(r['metrics']['gt_regions'] for r in group),
                        'derived_pred_regions':sum(r['metrics']['predicted_regions'] for r in group)}
    pos=[r['maximum_anomaly'] for r in rows if r['defect_type']!='good']; neg=[r['maximum_anomaly'] for r in rows if r['defect_type']=='good']
    auc=float(np.mean([float(p>n)+.5*float(p==n) for p in pos for n in neg]))
    write(OUT/'report.json',{'protocol_sha256':sha(OUT/'protocol-before-any-fit.json'),'model':model.describe(),'model_metadata_sha256':sha(MODEL/'model.json'),
                           'scope':protocol['scope'],'image_auroc':auc,'by_defect_type':by_class,'records':rows})
    print(json.dumps({'image_auroc':auc,'by_defect_type':by_class,'scope':protocol['scope']},indent=2),flush=True)


if __name__ == '__main__':
    main()
