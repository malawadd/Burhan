"""Fit a NEW C-specific full-context bank after fresh C preflight/source parity."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import openvino as ov
from factory_c_density_math import whitening_from_moments, normalized_log_distance, precision_corrected_moments, quantized_quantiles
from factory_c_density_common import verify_protocol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3k_c200_density_f16'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenNormalFeatures:
    def __init__(self, path):
        core = ov.Core()
        graph = core.read_model(str(path))
        feature_outputs=[out for out in graph.outputs if list(out.shape)==[1,16,640,256]]
        if list(graph.input().shape) != [1, 3, 1280, 512] or len(feature_outputs)!=1:
            raise ValueError('Unexpected frozen normal-feature geometry')
        if not any(device.startswith('GPU') for device in core.available_devices):
            raise ValueError('Explicit Intel GPU is required')
        self.compiled = core.compile_model(graph, 'GPU', {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', 'INFERENCE_PRECISION_HINT': 'f16'})
        self.request = self.compiled.create_infer_request()
        self.input=self.compiled.input()
        self.output=next(out for out in self.compiled.outputs if list(out.shape)==[1,16,640,256])

    def extract(self, image):
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError('Normal TRAIN image cannot decode')
        resized = cv2.resize(frame, (512, 1280), interpolation=cv2.INTER_AREA)
        value = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
        result = np.array(self.request.infer({self.input: value})[self.output][0], dtype=np.float32, copy=True)
        if result.shape != (16, 640, 256) or not np.isfinite(result).all():
            raise ValueError('Invalid actual normal features')
        return result


def main():
    global NAME
    parser=argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(precision_corrected=True);args=parser.parse_args()
    if args.precision_corrected:NAME='spatial_round3k_c200_density_f16'
    cv2.setNumThreads(1)
    start = time.perf_counter()
    protocol_path,protocol=verify_protocol()
    source = ROOT / protocol['warmstart']
    source_onnx = ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.onnx'
    if (sha(source) != protocol['warmstart_sha256'] or sha(source_onnx) != protocol['source_onnx_sha256']
            or sha(source_onnx.with_suffix('.onnx.json')) != protocol['source_metadata_sha256']):
        raise ValueError('Current source model/metadata differs from protected identity')
    manifest_path = OUT / (NAME + '-export-manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['protocol_sha256'] != sha(protocol_path) or manifest['source_pt_sha256'] != sha(source)
            or manifest['source_onnx_sha256'] != sha(source_onnx)
            or manifest['source_metadata_sha256'] != sha(source_onnx.with_suffix('.onnx.json'))):
        raise ValueError('Export manifest source/protocol identity mismatch')
    preflight_path=OUT/(NAME+'-fullprofile-preflight.json')
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    if (preflight['protocol_sha256'] != sha(protocol_path)
            or preflight['profile_ms']['serial_calls_per_second'] < 15):
        raise ValueError('Required protected pre-fit capacity gate has not passed')
    parity_path=OUT/(NAME+'-pre-fit-gpu-parity.json')
    parity = json.loads(parity_path.read_text(encoding='utf-8'))
    if not parity['all_checks_passed'] or parity['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Required source/control GPU serving parity has not passed')
    probe = ROOT / 'models/factory' / (NAME + '-capacity-probe.onnx')
    if sha(probe) != preflight['model_sha256'] or sha(probe) != manifest['probe_sha256']:
        raise ValueError('Preflight graph changed')
    if (OUT / (NAME + '-normal-bank.npz')).exists():
        raise ValueError('Normal bank already exists; do not refit after calibration')
    split = protocol['split_items']
    train_items = set(split['train'])
    if len(train_items) != 30 or train_items & (set(split['calibration']) | set(split['test'])):
        raise ValueError('Physical-item TRAIN isolation changed')
    root = ROOT / 'datasets/factory/KolektorSDD'
    rows, all_count, positive_count = [], 0, 0
    for item in split['train']:
        for image in sorted((root / item).glob('*.jpg')):
            mask = image.with_name(image.stem + '_label.bmp')
            truth = cv2.imread(str(mask), 0)
            frame = cv2.imread(str(image))
            if truth is None or frame is None or truth.shape != frame.shape[:2]:
                raise ValueError('Invalid original TRAIN pair')
            all_count += 1
            if np.any(truth):
                positive_count += 1
                continue
            rows.append({'image': str(image.relative_to(root)), 'image_sha256': sha(image),
                         'mask_sha256': sha(mask), 'native_shape': list(frame.shape),
                         'defective': False})
    if (all_count, positive_count, len(rows)) != (239, 31, 208):
        raise ValueError('Protected original TRAIN normal counts changed')
    if rows!=protocol['normal_sources']:
        raise ValueError('Actual208 normal TRAIN file identities differ from protected original sources')
    extractor_path = ROOT / 'models/factory' / (NAME + '-normal-feature-extractor.onnx')
    if sha(extractor_path) != manifest['feature_extractor_sha256']:
        raise ValueError('Unverified normal-feature extractor')
    if sha(extractor_path)!=manifest['pre_fit_diagnostic_sha256']:
        raise ValueError('C full compiler-context feature graph must retain byte-identical complete diagnostic outputs')
    extractor = FrozenNormalFeatures(extractor_path)
    column_sum = np.zeros((16, 256), np.float64)
    outer_sum = np.zeros((16, 16), np.float64)
    for index, row in enumerate(rows):
        feature = extractor.extract(root / row['image']).astype(np.float64)
        column_sum += feature.sum(axis=1)
        flat = feature.reshape(16, -1)
        outer_sum += flat @ flat.T
        if (index + 1) % 25 == 0:
            print(json.dumps({'phase': 'normal_moments', 'views': index + 1, 'elapsed_seconds': time.perf_counter() - start}), flush=True)
    mean64 = column_sum / (len(rows) * 640)
    second = outer_sum / (len(rows) * 640 * 256)
    if args.precision_corrected:
        mean,whitening,covariance_report=precision_corrected_moments(mean64,second)
    else:
        mean=mean64
        whitening, covariance_report = whitening_from_moments(mean, second)
    rng = np.random.default_rng(protocol['seed'])
    quantiles = []
    clipping = {'centered_values_clipped': 0, 'whitened_values_clipped': 0, 'channel_values': 0}
    for index, row in enumerate(rows):
        log_distance, counts = normalized_log_distance(extractor.extract(root / row['image']), mean, whitening, diagnostics=True)
        for key in clipping:
            clipping[key] += counts[key]
        selection = rng.choice(log_distance.size, 2048, replace=False)
        quantiles.append(log_distance.ravel()[selection])
        if (index + 1) % 25 == 0:
            print(json.dumps({'phase': 'normal_quantiles', 'views': index + 1, 'elapsed_seconds': time.perf_counter() - start}), flush=True)
    values = np.concatenate(quantiles)
    rawq50, rawq95 = map(float, np.percentile(values, [50, 95]))
    if not np.isfinite([rawq50, rawq95]).all() or rawq95 < rawq50:
        raise ValueError('Invalid normal-only log-distance quantiles')
    q50,q95,qspan=quantized_quantiles(rawq50,rawq95)
    clipping['centered_fraction_clipped'] = clipping['centered_values_clipped'] / clipping['channel_values']
    clipping['whitened_fraction_clipped'] = clipping['whitened_values_clipped'] / clipping['channel_values']
    for row in rows:
        image = root / row['image']
        if sha(image) != row['image_sha256'] or sha(image.with_name(image.stem + '_label.bmp')) != row['mask_sha256']:
            raise ValueError('Normal source changed during moment/quantile passes')
    target = OUT / (NAME + '-normal-bank.npz')
    temporary = target.with_name(target.stem + '.writing.npz')
    np.savez_compressed(temporary, mean=mean.astype(np.float32), whitening=whitening,
                        q50=np.asarray(q50), q95=np.asarray(q95),
                        **({'qspan':np.asarray(qspan)} if args.precision_corrected else {}))
    temporary.replace(target)
    report = {'protocol_sha256': sha(protocol_path), 'bank_sha256': sha(target),
              'feature_extractor_sha256': sha(extractor_path), 'source_pt_sha256': sha(source),
              'export_manifest_sha256': sha(manifest_path),
              'normal_views': len(rows), 'positive_train_views_excluded': positive_count,
              'normal_feature_pixels': len(rows) * 640 * 256, 'quantile_sample_pixels': int(values.size),
              'normal_sources': rows, 'normal_source_identity_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              'covariance': covariance_report, 'q50': q50, 'q95': q95, 'qspan_with_floor': qspan,
              'raw_normal_q50':rawq50,'raw_normal_q95':rawq95,
              'parameter_quantization':'Explicit representable half mu/W/q95/qspan, qspan stored independently' if args.precision_corrected else 'Original float32 bank, preserved failed F',
              'normal_clipping_diagnostics': clipping, 'elapsed_seconds': time.perf_counter() - start,
              'scope': 'Normal-only fitted statistical head; no optimizer/BN updates; no calibration or original comparison pixels opened; offline feature graph never runs in serving',
              'calibration_views_opened': 0, 'comparison_images_opened': 0, 'official_sdd2_test_still_sealed': True}
    (OUT / (NAME + '-normal-fit.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'normal_sources'}, indent=2))


if __name__ == '__main__':
    main()
