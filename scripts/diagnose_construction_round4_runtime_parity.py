"""One failed-source technical diagnostic. No GT/accuracy scoring or policy tuning."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/evaluation/construction-round4-mixed-v3/parity-technical-diagnostic-v1'
FAILED = ROOT / 'output/evaluation/construction-round4-mixed-v3/step0300'
MODEL = ROOT / 'models/construction/round4-mixed-v3/crack-spalling-step0300-640.onnx'
PROPOSAL = OUT / 'proposal-before-network.json'
APPROVAL = OUT / 'approval-before-network.json'
CLASSES = ['crack', 'spalling']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def proposal():
    failed = read(FAILED / 'full-profile-parity-failure.json')
    if failed['index'] != 21 or sha(MODEL) != '23f3f5c26cb3148a9045439dea58356129557060ef64119da814e9230244d25c':
        raise ValueError('Only original failed step300/source21 technical diagnosis allowed')
    source = failed['input']
    if sha(ROOT / source['image']) != source['image_sha256']:
        raise ValueError('Original failure source image changed')
    paths = [Path(__file__), MODEL, MODEL.with_suffix('.provenance.json'), ROOT / source['image'],
        FAILED / 'candidate-protocol-before-network.json', FAILED / 'calibration-execution.json',
        FAILED / 'full-profile-parity-failure.json', FAILED / 'full-profile-parity-failure-raw-cache.npz',
        ROOT / 'edge_runtime/profiles/construction_model.py', ROOT / 'edge_runtime/profiles/construction_cubit.py',
        ROOT / 'edge_runtime/profiles/construction.py', ROOT / 'edge_runtime/profiles/construction_preprocessing.py']
    value = {'scope': 'Single already failed public CAL source; numerical/input-route diagnosis ONLY, no GT or accuracy scoring.',
        'source': {key: source[key] for key in ['source', 'frame_index', 'image', 'image_sha256']},
        'model_sha256': sha(MODEL), 'classes': CLASSES,
        'policy': {'confidence': .25, 'nms_iou': .45, 'exposure_policy': 'low_light_gain', 'mask_rendering': 'masks'},
        'calls': ['direct_A', 'profile_B', 'direct_A', 'profile_B', 'direct_A', 'profile_B'],
        'initialization': 'Two independently compiled adapters of identical model/options; exactly3 stateless127 warmup calls each =6, discarded, not evidence.',
        'capture': 'Copy each actual NN tensor/raw output/native mask bits BEFORE next call; RAW/gained before/after hashes and queried config including Winograd. No GT boxes loaded into scorer.',
        'cpu_reference': 'Exactly1 ORTCPU numerical call per unique actual captured NN input hash, maximum6. No quality/model/threshold selection.',
        'failure_rule': 'First started/partial/failure receipts retained; no retry, tolerance relaxation or CAL rerun. Diagnostic reports differences, never declares quality guards passed.',
        'pins': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    OUT.mkdir(parents=True, exist_ok=True)
    write(PROPOSAL, value, exclusive=True)
    print('DIAGNOSTIC_PROPOSAL', sha(PROPOSAL), flush=True)


def run():
    import cv2
    import numpy as np
    import onnxruntime as ort
    import edge_runtime.profiles.construction_cubit as module
    from edge_runtime.profiles.construction import ConstructionProfile
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS, decode
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from edge_runtime.contracts import FrameContext
    frozen, approval = read(PROPOSAL), read(APPROVAL)
    if approval.get('parent_approved_network') is not True or approval['approved_proposal_sha256'] != sha(PROPOSAL):
        raise ValueError('Exact separate technical network approval required')
    for relative, digest in frozen['pins'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Frozen technical source changed: ' + relative)
    marker = OUT / 'execution.json'
    receipt = {'state': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(), 'proposal_sha256': sha(PROPOSAL),
               'initialization_calls': 0, 'source_calls': 0, 'cpu_reference_calls': 0}
    write(marker, receipt, exclusive=True)
    cv2.setNumThreads(1)
    original_prepare = module.prepare
    try:
        meta = read(MODEL.with_suffix('.provenance.json'))
        spec = {'file': MODEL.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(MODEL),
            'wrapper_sha256': sha(ROOT / 'edge_runtime/profiles/construction_cubit.py'), 'classes': CLASSES,
            'capabilities': ['surface_crack', 'surface_spalling'], 'decoder': 'cubit_yolo_seg', 'input_size': 640,
            'source': meta['source'], 'license': meta['license'], 'diagnostic_only': True}
        options = {'mode': 'damage', 'damage_variant': 'round4_parity_technical_only', 'engine': 'openvino',
            'openvino_device': 'GPU', 'cpu_threads': 1, 'confidence': .25, 'nms_iou': .45,
            'mask_rendering': 'masks', 'exposure_policy': 'low_light_gain', 'stage': 'structure', 'confirm_frames': 1}
        direct = module.CubitConstructionModel(options, spec)
        DAMAGE_VARIANTS['round4_parity_technical_only'] = spec
        profile = ConstructionProfile(options)
        adapters = {'direct_A': direct, 'profile_B': profile.model.delegate}
        configs = {}
        for role, adapter in adapters.items():
            actual = dict(adapter.actual_execution_config)
            devices = actual['EXECUTION_DEVICES']
            if not isinstance(devices, list) or not devices or any(not str(x).startswith('GPU') for x in devices) or not any(x in str(actual['INFERENCE_PRECISION_HINT']).lower() for x in ['float32', 'f32']) or str(actual['NUM_STREAMS']) != '1':
                raise ValueError('Actual GPU/f32/one-stream differs')
            for property_name in ['GPU_DISABLE_WINOGRAD_CONVOLUTION', 'GPU_ENABLE_LOOP_UNROLLING']:
                try:
                    actual[property_name] = str(adapter.compiled.get_property(property_name))
                except Exception as error:
                    actual[property_name] = {'unavailable': str(error)}
            configs[role] = {'actual': actual, 'requested': adapter.requested_execution_config}
            for _ in range(3):
                adapter.infer(np.full((640, 640, 3), 127, np.uint8))
                receipt['initialization_calls'] += 1
                write(marker, receipt)
        write(OUT / 'queried-configs.json', configs, exclusive=True)
        raw = cv2.imread(str(ROOT / frozen['source']['image']), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError('Failure source failed to decode')
        raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
        call_rows, arrays_by_call, unique_inputs = [], [], {}
        for index, role in enumerate(frozen['calls']):
            adapter = adapters[role]
            captured = {}
            def capture_prepare(frame, *args, **kwargs):
                before = hashlib.sha256(frame.tobytes()).hexdigest()
                tensor, transform = original_prepare(frame, *args, **kwargs)
                if captured:
                    raise ValueError('More than one NN input preparation per actual source call')
                captured.update(tensor=np.array(tensor, copy=True), transform=transform,
                    gained_before_hash=before, gained_after_prepare_hash=hashlib.sha256(frame.tobytes()).hexdigest())
                return tensor, transform
            original_raw = adapter.predict_raw
            def capture_raw(frame):
                output, transform = original_raw(frame)
                captured.update(predictions=np.array(output[0], copy=True), prototypes=np.array(output[1], copy=True))
                return output, transform
            module.prepare, adapter.predict_raw = capture_prepare, capture_raw
            rows = []
            try:
                if role == 'direct_A':
                    rows = direct.infer(raw)
                else:
                    original_infer = profile.model.infer
                    def capture_infer(frame):
                        output = original_infer(frame)
                        rows.extend(output)
                        return output
                    profile.model.infer = capture_infer
                    try:
                        profile.process(raw, FrameContext(sequence=index, captured_monotonic=float(index),
                            source_id='single-failed-public-source', session_id='technical-six-calls'))
                    finally:
                        profile.model.infer = original_infer
            finally:
                module.prepare, adapter.predict_raw = original_prepare, original_raw
            receipt['source_calls'] += 1
            gained, gain = camera_exposure(raw, 'low_light_gain')
            gained_hash = hashlib.sha256(gained.tobytes()).hexdigest()
            if hashlib.sha256(raw.tobytes()).hexdigest() != raw_hash or captured['gained_before_hash'] != gained_hash or captured['gained_after_prepare_hash'] != gained_hash or adapter.last_input_exposure_gain != gain:
                raise ValueError('RAW/input/gain changed in actual route')
            captured['native_masks'] = [np.array(row['mask'], copy=True) for row in rows]
            bits = {f'mask_{i}': np.packbits(row['mask']).copy() for i, row in enumerate(rows)}
            serialized = [{key: value for key, value in row.items() if key != 'mask'} | {
                'mask_positive_pixels': int(row['mask'].sum()), 'mask_sha256': hashlib.sha256(bits[f'mask_{i}'].tobytes()).hexdigest()} for i, row in enumerate(rows)]
            target = OUT / f'call_{index:02d}-{role}-actual-input-output-masks.npz'
            np.savez_compressed(target, actual_nn_input=captured['tensor'], raw_predictions=captured['predictions'],
                raw_prototypes=captured['prototypes'], raw_native_bgr=raw, gained_native_bgr=gained, **bits)
            input_hash = hashlib.sha256(captured['tensor'].tobytes()).hexdigest()
            unique_inputs.setdefault(input_hash, captured['tensor'])
            item = {'call': index, 'route': role, 'actual_gain': gain, 'actual_input_sha256': input_hash,
                'actual_letterbox_transform': list(captured['transform']),
                'raw_before_sha256': raw_hash, 'raw_after_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                'gained_before_sha256': captured['gained_before_hash'], 'gained_after_sha256': gained_hash,
                'outputs': target.relative_to(ROOT).as_posix(), 'outputs_sha256': sha(target), 'predictions': serialized}
            write(OUT / f'call_{index:02d}.json', item, exclusive=True)
            call_rows.append(item)
            arrays_by_call.append(captured)
            write(marker, receipt)
            print('TECHNICAL_CALL', index, role, len(rows), flush=True)
        if receipt['source_calls'] != 6 or receipt['initialization_calls'] != 6 or len(unique_inputs) > 6:
            raise ValueError('Declared bounded calls differ')
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = settings.inter_op_num_threads = 1
        settings.add_session_config_entry('session.intra_op.allow_spinning', '0')
        cpu = ort.InferenceSession(str(MODEL), sess_options=settings, providers=['CPUExecutionProvider'])
        references = {}
        for number, (input_hash, tensor) in enumerate(unique_inputs.items()):
            outputs = cpu.run(None, {cpu.get_inputs()[0].name: tensor})
            if any(not np.isfinite(output).all() for output in outputs):
                raise ValueError('Nonfinite CPU numerical reference')
            target = OUT / f'unique-input-{number:02d}-ortcpu-reference.npz'
            np.savez_compressed(target, input=tensor, predictions=outputs[0], prototypes=outputs[1])
            references[input_hash] = outputs
            receipt['cpu_reference_calls'] += 1
            write(marker, receipt)
        comparisons = []
        for left in range(6):
            for right in range(left + 1, 6):
                a, b = arrays_by_call[left], arrays_by_call[right]
                comparisons.append({'left': left, 'right': right,
                    'same_route': call_rows[left]['route'] == call_rows[right]['route'],
                    'same_input': call_rows[left]['actual_input_sha256'] == call_rows[right]['actual_input_sha256'],
                    'coordinate_max_abs': float(np.max(np.abs(a['predictions'][:, :4] - b['predictions'][:, :4]))),
                    'class_score_max_abs': float(np.max(np.abs(a['predictions'][:, 4:6] - b['predictions'][:, 4:6]))),
                    'mask_coefficient_max_abs': float(np.max(np.abs(a['predictions'][:, 6:] - b['predictions'][:, 6:]))),
                    'prototype_max_abs': float(np.max(np.abs(a['prototypes'] - b['prototypes']))),
                    'candidate_count': [len(call_rows[left]['predictions']), len(call_rows[right]['predictions'])],
                    'native_pixel_xor_by_same_rank': ([int(np.count_nonzero(x != y)) for x, y in zip(a['native_masks'], b['native_masks'])]
                        if len(a['native_masks']) == len(b['native_masks']) and [r['label'] for r in call_rows[left]['predictions']] == [r['label'] for r in call_rows[right]['predictions']] else None),
                    'xor_scope': 'Same score-rank/class only; no GT matching or accuracy inference.',
                    'same_decoded_predictions': call_rows[left]['predictions'] == call_rows[right]['predictions']})
        numerical = []
        for call, arrays in zip(call_rows, arrays_by_call):
            reference = references[call['actual_input_sha256']]
            numerical.append({'call': call['call'],
                'coordinate_max_abs': float(np.max(np.abs(arrays['predictions'][:, :4] - reference[0][:, :4]))),
                'class_score_max_abs': float(np.max(np.abs(arrays['predictions'][:, 4:6] - reference[0][:, 4:6]))),
                'mask_coefficient_max_abs': float(np.max(np.abs(arrays['predictions'][:, 6:] - reference[0][:, 6:]))),
                'prototype_max_abs': float(np.max(np.abs(arrays['prototypes'] - reference[1])))})
        report = {'scope': frozen['scope'], 'proposal_sha256': sha(PROPOSAL), 'calls': call_rows,
            'pairwise_raw_comparisons': comparisons, 'ortcpu_raw_comparisons': numerical,
            'unique_actual_inputs': len(unique_inputs), 'no_accuracy_or_guard_relaxation': True}
        write(OUT / 'technical-report.json', report, exclusive=True)
        receipt.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(OUT / 'technical-report.json'))
        write(marker, receipt)
    except BaseException:
        module.prepare = original_prepare
        receipt.update(state='failed', traceback=traceback.format_exc())
        write(marker, receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['proposal', 'run'], required=True)
    args = parser.parse_args()
    (proposal if args.stage == 'proposal' else run)()
