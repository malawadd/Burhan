"""NEW C-density nonconstant full-profile preflight;16TRAINviews,no labels."""
from pathlib import Path
import hashlib
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from edge_runtime.profiles.factory import FactoryProfile
from edge_runtime.contracts import FrameContext
from factory_c_density_common import verify_protocol


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    out = ROOT / 'output/optimization/factory-round3'
    name = 'spatial_round3k_c200_density_f16'
    protocol_path,protocol=verify_protocol()
    report_path=out/(name+'-fullprofile-preflight.json')
    if report_path.exists():raise ValueError('Preserve initial C-density preflight; version any retry separately')
    path = ROOT / 'models/factory' / (name + '-capacity-probe.onnx')
    manifest = json.loads((out / (name + '-export-manifest.json')).read_text(encoding='utf-8'))
    if manifest['protocol_sha256'] != sha(protocol_path) or manifest['probe_sha256'] != sha(path):
        raise ValueError('Nonconstant preflight export identity changed')
    source = ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.onnx.json'
    metadata = json.loads(source.read_text(encoding='utf-8'))
    metadata.update(model_sha256=sha(path), profile_version=name + '_capacity_probe',
                    model_name='UNFITTED nonconstant normal-density capacity probe',
                    architecture=protocol['architecture'], training_protocol=protocol,
                    pixel_threshold=.999, minimum_component_pixels=16, hysteresis_growth_threshold=.95,
                    capacity_probe_only=True, automatic_acceptance=False, automatic_rejection=False)
    path.with_suffix('.onnx.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    selected = []
    for item in protocol['split_items']['train']:
        selected.extend(sorted((ROOT / 'datasets/factory/KolektorSDD' / item).glob('*.jpg'))[:2])
        if len(selected) >= 16:
            break
    decoded = [(p, cv2.imread(str(p))) for p in selected]
    if len(decoded) != 16 or any(frame is None for _, frame in decoded):
        raise ValueError('TRAIN preflight inputs changed')
    started = time.perf_counter()
    profile = FactoryProfile({'defect_model_path': str(path), 'defect_backend': 'auto', 'confirmation_frames': 1})
    profile.prepare_worker()
    startup = time.perf_counter() - started
    for index, (_, frame) in enumerate(decoded[:4]):
        profile.process(frame, FrameContext(index, time.perf_counter(), unit_id=f'preflight-warm-{index}'))
    profile.reset()
    records = []
    for index in range(64):
        image, frame = decoded[index % len(decoded)]
        tick = time.perf_counter()
        result = profile.process(frame, FrameContext(index, tick, unit_id=f'preflight-{index}'))
        records.append({'image': str(image.relative_to(ROOT / 'datasets/factory/KolektorSDD')),
                        'profile_ms': (time.perf_counter() - tick) * 1000, 'status': result.status,
                        'stage_ms': result.metrics.get('factory_neural_stage_ms')})
    values = [row['profile_ms'] for row in records]
    if any(row['stage_ms'] is None for row in records):
        raise ValueError('An input abstained/skipped the actual neural graph')
    report = {'scope': 'Unfitted nonconstant dense16x16-whitening head;16TRAIN views/64full-profile calls; no GT read or accuracy claim',
              'project_quiescent_slot_authorized_by_root': True, 'host_exclusive': False,
              'model_sha256': sha(path), 'protocol_sha256': sha(protocol_path),
              'startup_seconds': startup, 'initialization': profile.describe().get('model_initialization'),
              'profile_ms': {'mean': float(np.mean(values)), 'p50': float(np.median(values)),
                             'p95': float(np.percentile(values, 95)), 'p99': float(np.percentile(values, 99)),
                             'max': max(values), 'serial_calls_per_second': 1000 / float(np.mean(values))},
              'records': records, 'comparison_images_opened': 0, 'official_sdd2_test_still_sealed': True}
    report['capacity_gate_passed']=report['profile_ms']['serial_calls_per_second']>=15
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))
    if not report['capacity_gate_passed']:raise ValueError('C-density failed NEW15calls/s gate; no bankfit permitted')


if __name__ == '__main__':
    main()
