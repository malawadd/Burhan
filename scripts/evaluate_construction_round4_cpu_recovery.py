"""Versioned CPUORT numerical recovery, identical assets/data/policy; no fitting.

Inert on import. Metadata preparation is permitted separately from NN execution.
The failed GPU attempt and all model/protocol bytes remain untouched.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'output/evaluation/construction-round4-cpu-recovery-v1'
TRAIN = ROOT / 'output/training/construction-round4-mixed-v3'
GPU = ROOT / 'output/evaluation/construction-round4-mixed-v3'
ORIGINAL = ROOT / 'output/evaluation/construction-cubit-author-v2'
PROPOSAL = OUT / 'recovery-proposal-before-cpu-network.json'
APPROVAL = OUT / 'approval-before-cpu-network.json'
CLASSES = ['crack', 'spalling']
POLICY = {'confidence': .25, 'nms_iou': .45, 'exposure_policy': 'low_light_gain', 'mask_rendering': 'masks'}
LABELS = ['author-baseline', 'step0300', 'step0600', 'step0900']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def assets(label):
    if label == 'author-baseline':
        return ROOT / 'models/construction/cubit-author-release-v1/author-yolov8n-seg-640-v2.onnx', ORIGINAL
    if label not in LABELS:
        raise ValueError('Only fixed baseline plus three exact snapshots allowed')
    return ROOT / f'models/construction/round4-mixed-v3/crack-spalling-{label}-640.onnx', GPU / label


def proposal():
    from train_construction_round4_mixed_v3 import approved
    plan, _, plan_hash = approved('fit')
    fit = read(TRAIN / 'fit-execution.json')
    if fit['state'] != 'completed' or fit['completed_steps'] != 900 or fit['plan_sha256'] != plan_hash:
        raise ValueError('Completed unchanged900 fit required')
    original = read(ORIGINAL / 'candidate-protocol-before-network.json')
    rows = original['calibration_inputs']
    if len(rows) != 122 or sum(row['negative'] for row in rows) != 10:
        raise ValueError('Unchanged complete122 calibration scope required')
    files = [Path(__file__), TRAIN / 'approved-plan-before-probe-and-fit.json', TRAIN / 'fit-execution.json',
        ORIGINAL / 'candidate-protocol-before-network.json', GPU / 'step0300/calibration-execution.json',
        GPU / 'step0300/full-profile-parity-failure.json', GPU / 'parity-technical-diagnostic-v1/technical-report.json',
        ROOT / 'edge_runtime/profiles/construction_cubit.py', ROOT / 'edge_runtime/profiles/construction_model.py',
        ROOT / 'edge_runtime/profiles/construction.py', ROOT / 'edge_runtime/profiles/construction_preprocessing.py',
        ROOT / 'scripts/evaluate_construction_round3.py', ROOT / 'scripts/construction_annotations.py']
    models = {}
    for label in LABELS:
        model, inherited = assets(label)
        export_path = inherited / 'export-execution.json'
        export = read(export_path)
        if export['state'] != 'completed' or export['model_sha256'] != sha(model) or export['metadata_sha256'] != sha(model.with_suffix('.provenance.json')):
            raise ValueError('Exact existing export/model/metadata identity differs')
        fixture = inherited / 'export-fixture-torch-outputs.npz'
        if export['torch_fixture_sha256'] != sha(fixture):
            raise ValueError('Original pretrace Torch fixture changed')
        models[label] = {'file': model.relative_to(ROOT).as_posix(), 'sha256': sha(model),
            'metadata': model.with_suffix('.provenance.json').relative_to(ROOT).as_posix(),
            'torch_fixture': fixture.relative_to(ROOT).as_posix(), 'inherited_export': inherited.relative_to(ROOT).as_posix()}
        files.extend([model, model.with_suffix('.provenance.json'), export_path, fixture])
    for row in rows:
        image = ROOT / row['image']
        if sha(image) != row['image_sha256']:
            raise ValueError('Original frozen CAL image changed')
        files.append(image)
    value = {'technical_recovery_only': True, 'approved_fit_plan_sha256': plan_hash,
        'cause_evidence': 'Six identical-input calls reproduced within-instance GPU detection-output variation before CPU decode. No input/gain change; low-level driver/kernel cause unproven.',
        'preservation': 'All original GPU sources/models/failed results remain byte-for-byte intact. No new training or threshold search.',
        'models': models, 'labels': LABELS, 'classes': CLASSES, 'policy': POLICY, 'calibration_inputs': rows,
        'backend': {'engine': 'onnxruntime', 'device': 'CPU', 'provider': 'CPUExecutionProvider', 'intra_op_threads': 1, 'inter_op_threads': 1, 'input_type': 'tensor(float)'},
        'order': LABELS, 'baseline': 'Exactly one original-author CPU baseline on identical122, no model selection from baseline.',
        'numeric_guards': {'coordinate_max_abs': 1., 'class_score_max_abs': .01, 'mask_coefficient_max_abs': .1, 'prototype_max_abs': .1, 'all_finite': True},
        'serving_parity': 'Identical actual prepared inputs; strict original rounded6 score/box/packed-native-mask signature equality between direct and actual full profile. Two CPU forwards/frame; actual paired raw tensors saved on first failure, no retry.',
        'selection': plan['selection'], 'minimum_guards': plan['minimum_calibration_guards'],
        'downstream': 'One CPU global choice, or fixed900 FAILED, for all5public+4private videos and55photos. All55 BEFORE/AFTER CPU. Cached video BEFORE remains actual historicalGPU and explicitly confounds backend/training changes.',
        'speed': 'CPU may be slower; no construction15FPS qualification. Factory15FPS target separate. Measured speed only after project-quiescent timing release.',
        'license': plan['license'], 'pins': {path.relative_to(ROOT).as_posix(): sha(path) for path in files}}
    OUT.mkdir(parents=True, exist_ok=True)
    write(PROPOSAL, value, exclusive=True)
    print('CPU_RECOVERY_PROPOSAL', sha(PROPOSAL), flush=True)


def checked():
    frozen, approval = read(PROPOSAL), read(APPROVAL)
    if approval.get('parent_approved_network') is not True or approval['approved_proposal_sha256'] != sha(PROPOSAL):
        raise ValueError('Separate exact CPU recovery approval required')
    for relative, digest in frozen['pins'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Frozen CPU recovery source differs: ' + relative)
    if frozen['labels'] != LABELS or frozen['classes'] != CLASSES or frozen['policy'] != POLICY:
        raise ValueError('No asset/data/policy changes allowed')
    return frozen


def numeric(adapter, fixture_path, folder, frozen):
    import numpy as np
    actual = adapter.actual_execution_config
    if actual['providers'] != ['CPUExecutionProvider'] or actual['intra_op_num_threads'] != 1 or actual['inter_op_num_threads'] != 1 or adapter.session.get_inputs()[0].type != 'tensor(float)':
        raise ValueError('Actual CPUORT one-thread float32 recipe differs')
    with np.load(fixture_path, allow_pickle=False) as source:
        output = adapter.session.run(None, {adapter.input_name: source['images']})
        np.savez_compressed(folder / 'first-cpu-numeric-outputs.npz', predictions=output[0], prototypes=output[1])
        stats = {'coordinate_max_abs': float(np.max(np.abs(output[0][:, :4] - source['predictions'][:, :4]))),
            'class_score_max_abs': float(np.max(np.abs(output[0][:, 4:6] - source['predictions'][:, 4:6]))),
            'mask_coefficient_max_abs': float(np.max(np.abs(output[0][:, 6:] - source['predictions'][:, 6:]))),
            'prototype_max_abs': float(np.max(np.abs(output[1] - source['prototypes']))),
            'all_finite': all(bool(np.isfinite(value).all()) for value in output)}
    passed = stats['all_finite'] and all(stats[name] <= limit for name, limit in frozen['numeric_guards'].items() if name != 'all_finite')
    write(folder / 'first-cpu-numeric-report.json', {'stats': stats, 'pass': passed, 'actual_execution_config': actual}, exclusive=True)
    if not passed:
        raise ValueError('First CPU numeric guard failed; CAL forbidden, no retry')


def calibration(label):
    import cv2
    import numpy as np
    import edge_runtime.profiles.construction_cubit as module
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS, decode
    from edge_runtime.profiles.construction import ConstructionProfile
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from edge_runtime.contracts import FrameContext
    from evaluate_construction_round3 import score_frame, summarize
    frozen = checked()
    if label not in LABELS:
        raise ValueError('Only fixed labels allowed')
    if label != LABELS[0] and read(OUT / LABELS[LABELS.index(label) - 1] / 'calibration-execution.json')['state'] != 'completed':
        raise ValueError('Mandatory fixed CPU evaluation order required')
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    marker = folder / 'calibration-execution.json'
    asset = frozen['models'][label]
    execution = {'state': 'started', 'label': label, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(PROPOSAL), 'model_sha256': asset['sha256'], 'completed_frames': 0}
    write(marker, execution, exclusive=True)
    cv2.setNumThreads(1)
    original_prepare = module.prepare
    try:
        model_path = ROOT / asset['file']
        meta = read(ROOT / asset['metadata'])
        spec = {'file': model_path.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': asset['sha256'],
            'wrapper_sha256': sha(ROOT / 'edge_runtime/profiles/construction_cubit.py'), 'classes': CLASSES,
            'capabilities': ['surface_crack', 'surface_spalling'], 'decoder': 'cubit_yolo_seg', 'input_size': 640,
            'source': meta['source'], 'license': meta['license'], 'diagnostic_only': True,
            'validation_scope': 'Same exposed122 CAL; actual results recorded separately, source-checkpoint overlap/project independence unresolved; no production qualification.'}
        DAMAGE_VARIANTS['round4_cpu_recovery_only'] = spec
        options = {'mode': 'damage', 'damage_variant': 'round4_cpu_recovery_only', 'engine': 'onnxruntime',
            'openvino_device': 'CPU', 'cpu_threads': 1, 'confidence': .25, 'nms_iou': .45, 'mask_rendering': 'masks',
            'exposure_policy': 'low_light_gain', 'stage': 'structure', 'confirm_frames': 1}
        profile = ConstructionProfile(options)
        adapter = profile.model.delegate
        numeric(adapter, ROOT / asset['torch_fixture'], folder, frozen)
        execution['numeric_report_sha256'] = sha(folder / 'first-cpu-numeric-report.json')
        write(marker, execution)
        profile.prepare_worker()
        records = []
        progress = folder / 'actual-frames'
        progress.mkdir()
        def signature(rows):
            return [(row['label'], tuple(row['bbox']), round(float(row['score']), 6), hashlib.sha256(np.packbits(row['mask']).tobytes()).hexdigest()) for row in rows]
        for index, source in enumerate(frozen['calibration_inputs']):
            raw = cv2.imread(str(ROOT / source['image']))
            if raw is None:
                raise ValueError('Frozen CAL image failed to decode')
            raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
            inputs, outputs = [], []
            def capture_prepare(frame, *args, **kwargs):
                tensor, transform = original_prepare(frame, *args, **kwargs)
                inputs.append(np.array(tensor, copy=True))
                return tensor, transform
            original_raw = adapter.predict_raw
            def capture_raw(frame):
                output, transform = original_raw(frame)
                outputs.append([np.array(value, copy=True) for value in output])
                return output, transform
            module.prepare, adapter.predict_raw = capture_prepare, capture_raw
            captured = []
            original_infer = profile.model.infer
            try:
                direct, transform = adapter.predict_raw(raw)
                cached = decode(direct, 'crack', transform, raw.shape[:2], .25, .45, 'masks', classes=CLASSES)
                def capture_infer(frame):
                    rows = original_infer(frame)
                    captured.extend(rows)
                    return rows
                profile.model.infer = capture_infer
                result = profile.process(raw, FrameContext(sequence=index, captured_monotonic=float(index),
                    source_id='public-fixed122-calibration', session_id='round4-cpu-recovery-' + label))
            finally:
                module.prepare, adapter.predict_raw, profile.model.infer = original_prepare, original_raw, original_infer
            gained, gain = camera_exposure(raw, 'low_light_gain')
            expected_input, _ = original_prepare(gained, 'crack', size=640)
            if len(inputs) != 2 or len(outputs) != 2 or not np.array_equal(inputs[0], inputs[1]) or not np.array_equal(inputs[1], expected_input) or hashlib.sha256(raw.tobytes()).hexdigest() != raw_hash or result.metrics['input_exposure_gain'] != gain or signature(cached) != signature(captured):
                np.savez_compressed(folder / 'first-parity-failure-actual-inputs-outputs.npz',
                    **{f'input_{i}': value for i, value in enumerate(inputs)},
                    **{f'raw_{i}_{j}': value for i, pair in enumerate(outputs) for j, value in enumerate(pair)},
                    **{f'direct_mask_{i}': np.packbits(row['mask']) for i, row in enumerate(cached)},
                    **{f'profile_mask_{i}': np.packbits(row['mask']) for i, row in enumerate(captured)})
                write(folder / 'first-parity-failure.json', {'index': index, 'source': source,
                    'direct_signature': signature(cached), 'profile_signature': signature(captured)}, exclusive=True)
                raise ValueError('Actual same-input strict CPU full-profile parity failed; no retry')
            observations = result.domain_state['observations']
            if [(row['label'], tuple(row['bbox']), round(float(row['score']), 6)) for row in observations] != [(row['label'], tuple(row['bbox']), round(float(row['score']), 6)) for row in captured]:
                raise ValueError('Actual full profile changed candidates')
            scored = score_frame(captured, source['expected'], set(CLASSES), negative=source['negative'])
            if len(scored['predictions']) != len(captured):
                raise ValueError('Scored/captured candidate count differs')
            masks = {}
            for number, (prediction, actual) in enumerate(zip(scored['predictions'], captured)):
                mask = actual['mask']
                if mask.shape != raw.shape[:2] or mask.dtype != np.bool_:
                    raise ValueError('Actual native mask geometry differs')
                key = f'prediction_{number:03d}'
                masks[key] = np.packbits(mask)
                prediction.update(native_mask_positive_pixels=int(mask.sum()), native_mask_empty=not bool(mask.any()),
                    mask_reference={'npz_key': key, 'shape_hw': list(mask.shape), 'encoding': 'numpy.packbits row-major',
                                    'packed_mask_sha256': hashlib.sha256(masks[key].tobytes()).hexdigest()})
            mask_path = progress / f'frame_{index:03d}-actual-masks.npz'
            np.savez_compressed(mask_path, **masks)
            record = scored | {key: value for key, value in source.items() if key != 'expected'} | {
                'input_exposure_gain': gain, 'raw_native_bgr_sha256': raw_hash,
                'model_input_native_bgr_sha256': hashlib.sha256(gained.tobytes()).hexdigest(),
                'actual_nn_input_sha256': hashlib.sha256(inputs[1].tobytes()).hexdigest(), 'direct_and_profile_actual_inputs_identical': True,
                'first_pass_masks': {'path': mask_path.relative_to(ROOT).as_posix(), 'sha256': sha(mask_path)},
                'profile_observations': observations}
            records.append(record)
            write(progress / f'frame_{index:03d}.json', record, exclusive=True)
            execution['completed_frames'] += 1
            write(marker, execution)
            print('CPU_CAL', label, index + 1, '/122', flush=True)
        summary = summarize(records)
        thin = {item['id'] for row in records for item in row['expected'] if item['label'] == 'crack' and min(item['bbox'][2:]) <= 30 and max(item['bbox'][2:]) >= 2 * min(item['bbox'][2:])}
        hits = sum(prediction.get('matched_annotation_id') in thin for row in records for prediction in row['predictions'])
        negatives = [row for row in records if row['negative']]
        negative_rate = sum(bool(row['predictions']) for row in negatives) / len(negatives)
        by = summary['by_class']
        checks = {name + '_precision': (by[name]['precision'] or 0) >= .8 for name in CLASSES}
        checks.update({name + '_recall': (by[name]['recall'] or 0) >= .5 for name in CLASSES})
        checks.update(geometric_thin_recall=hits / len(thin) >= .5, combined_negative_alarm_rate=negative_rate <= .1,
            full_profile_parity=True, numeric_parity=True)
        record_path = folder / 'calibration-runtime-records.json'
        write(record_path, records, exclusive=True)
        report = {'label': label, 'protocol_sha256': sha(PROPOSAL), 'model': profile.describe(), 'record_count': len(records),
            'summary': summary, 'geometric_thin': {'localized_matches': hits, 'annotations': len(thin), 'recall': hits / len(thin)},
            'combined_negative_image_alarm_rate': negative_rate, 'minimum_checks': checks, 'all_minimum_checks_pass': all(checks.values()),
            'full_profile_prediction_parity': True, 'numeric_parity': True, 'records_sha256': sha(record_path),
            'native_mask_audit': {'predictions': sum(len(row['predictions']) for row in records),
                'empty_masks': sum(item['native_mask_empty'] for row in records for item in row['predictions']),
                'bbox_matches_with_empty_masks': sum(item['native_mask_empty'] and bool(item.get('matched_annotation_id')) for row in records for item in row['predictions'])},
            'scope': 'Same exposed122 CAL; boxIoU.50 only, no video pixelGT/private fitting/production or15FPS qualification.',
            'speed': frozen['speed']}
        report_path = folder / 'fixed-policy-calibration-report.json'
        write(report_path, report, exclusive=True)
        execution.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(report_path))
        write(marker, execution)
    except BaseException:
        module.prepare = original_prepare
        execution.update(state='failed', traceback=traceback.format_exc())
        write(marker, execution)
        raise


def select():
    from evaluate_construction_round3 import score_frame, summarize
    frozen = checked()
    outcomes = {}
    for label in LABELS:
        folder = OUT / label
        execution = read(folder / 'calibration-execution.json')
        report_path = folder / 'fixed-policy-calibration-report.json'
        record_path = folder / 'calibration-runtime-records.json'
        report, records = read(report_path), read(record_path)
        numeric_path = folder / 'first-cpu-numeric-report.json'
        if execution['state'] != 'completed' or execution['completed_frames'] != 122 or execution['protocol_sha256'] != sha(PROPOSAL) or execution['report_sha256'] != sha(report_path) or execution['model_sha256'] != frozen['models'][label]['sha256']:
            raise ValueError('Every first CPU baseline/checkpoint execution must complete')
        if report['model']['sha256'] != execution['model_sha256'] or report['protocol_sha256'] != sha(PROPOSAL) or report['records_sha256'] != sha(record_path) or execution['numeric_report_sha256'] != sha(numeric_path) or read(numeric_path)['pass'] is not True:
            raise ValueError('Actual report/record/model/numeric/protocol identities differ')
        if len(records) != 122:
            raise ValueError('Complete122 records required')
        rescored = []
        for actual, expected in zip(records, frozen['calibration_inputs']):
            if any(actual[key] != expected[key] for key in ['source', 'frame_index', 'image', 'expected', 'negative']):
                raise ValueError('Actual CPU source/GT/frame identity differs')
            rescored.append(score_frame(actual['predictions'], actual['expected'], set(CLASSES), negative=actual['negative']) | {'negative': actual['negative']})
        summary = summarize(rescored)
        if summary != report['summary']:
            raise ValueError('Recomputed actual matching differs')
        thin = {item['id'] for row in records for item in row['expected'] if item['label'] == 'crack' and min(item['bbox'][2:]) <= 30 and max(item['bbox'][2:]) >= 2 * min(item['bbox'][2:])}
        hits = sum(item.get('matched_annotation_id') in thin for row in rescored for item in row['predictions'])
        negatives = [row for row in rescored if row['negative']]
        negative_rate = sum(bool(row['predictions']) for row in negatives) / len(negatives)
        by = summary['by_class']
        checks = {name + '_precision': (by[name]['precision'] or 0) >= .8 for name in CLASSES}
        checks.update({name + '_recall': (by[name]['recall'] or 0) >= .5 for name in CLASSES})
        checks.update(geometric_thin_recall=hits / len(thin) >= .5, combined_negative_alarm_rate=negative_rate <= .1,
            full_profile_parity=report['full_profile_prediction_parity'] is True, numeric_parity=report['numeric_parity'] is True)
        if checks != report['minimum_checks'] or all(checks.values()) != report['all_minimum_checks_pass'] or hits != report['geometric_thin']['localized_matches'] or negative_rate != report['combined_negative_image_alarm_rate']:
            raise ValueError('Actual recomputed minimum guards differ')
        outcomes[label] = {'step': None if label == 'author-baseline' else int(label[4:]),
            'model_file': frozen['models'][label]['file'], 'model_sha256': frozen['models'][label]['sha256'],
            'report_sha256': sha(report_path), 'all_minimum_checks_pass': all(checks.values()),
            'rank': None if label == 'author-baseline' else [min(by['crack']['recall'] or 0, by['spalling']['recall'] or 0, hits / len(thin)),
                min(by['crack']['precision'] or 0, by['spalling']['precision'] or 0), -sum(by[name]['fp'] for name in CLASSES), -int(label[4:])]}
    checkpoints = [outcomes[label] for label in LABELS[1:]]
    eligible = [row for row in checkpoints if row['all_minimum_checks_pass']]
    winner = max(eligible, key=lambda row: row['rank']) if eligible else None
    value = {'approved_fit_plan_sha256': frozen['approved_fit_plan_sha256'], 'recovery_protocol_sha256': sha(PROPOSAL),
        'backend': frozen['backend'], 'rule': frozen['selection'], 'checkpoints': checkpoints, 'winner': winner,
        'before_same_cpu': outcomes['author-baseline'],
        'if_no_winner': 'Exact900 remains FAILED diagnostic, no promotion/threshold rescue/heldout opening.',
        'production_qualification': False, 'private_video_training_performed': False}
    write(OUT / 'global-three-checkpoint-freeze.json', value, exclusive=True)
    print('CPU_GLOBAL_WINNER', winner, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['proposal', 'calibration', 'select'], required=True)
    parser.add_argument('--label', choices=LABELS)
    args = parser.parse_args()
    if args.stage == 'proposal':
        proposal()
    elif args.stage == 'select':
        select()
    elif args.label is None:
        parser.error('--label required')
    else:
        calibration(args.label)
