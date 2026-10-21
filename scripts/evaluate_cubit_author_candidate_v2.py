"""One fixed-policy author segmentation candidate; no fitting or threshold grid.

Only explicit 122 CAL inputs are loaded. Video identities are pinned before NN,
but their separate full-recording exporter must be reviewed before execution.
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
OUT = ROOT / 'output/evaluation/construction-cubit-author-v2'
DIRECTORY = ROOT / 'models/construction/cubit-author-release-v1'
MODEL = DIRECTORY / 'author-yolov8n-seg-640-v2.onnx'
TARGET = OUT / 'candidate-protocol-before-network.json'
PROPOSAL = OUT / 'candidate-proposal-revision3-before-network.json'
CLASSES = ['crack', 'spalling']
BEFORE = ROOT / 'models/construction/evcivil-yolov8n-r4-real-lowlight-640-gain.onnx'
BEFORE_RECORDS = ROOT / 'output/evaluation/construction-round3/evcivil-yolov8n-r4-real-lowlight-640-gain/calibration-runtime-records.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def proposal():
    import ultralytics
    from evaluate_construction_round3 import sampled
    from compare_construction_development_videos import selected_sources
    paths = [Path(__file__), ROOT / 'scripts/export_cubit_author_candidate_v2.py',
             ROOT / 'edge_runtime/profiles/construction_cubit.py',
             ROOT / 'edge_runtime/profiles/construction_model.py',
             ROOT / 'edge_runtime/profiles/construction.py', ROOT / 'edge_runtime/contracts.py',
             ROOT / 'edge_runtime/profiles/construction_preprocessing.py',
             ROOT / 'scripts/evaluate_construction_round3.py',
             ROOT / 'scripts/evaluate_construction_defect_video.py',
             ROOT / 'scripts/construction_annotations.py',
             ROOT / 'scripts/compare_construction_development_videos.py',
             DIRECTORY / 'author-best.pt', DIRECTORY / 'author-state-dict.pt',
             DIRECTORY / 'author-model-yaml.json', DIRECTORY / 'safe-extraction.json',
             ROOT / 'datasets/construction/evcivil-round3/protocol.json',
             ROOT / 'datasets/construction/evcivil-round3/exposures.json',
             ROOT / 'datasets/construction/evcivil-round3/background-import.json',
             ROOT / 'output/evaluation/construction-cubit-author-v1/export-execution.json',
             ROOT / 'output/evaluation/construction-cubit-author-v1/candidate-protocol-before-network.json',
             ROOT / 'scripts/export_cubit_author_candidate.py',
             ROOT / 'scripts/evaluate_cubit_author_candidate.py',
             BEFORE, BEFORE.with_suffix('.provenance.json'), BEFORE.with_suffix('.operating-point.json'), BEFORE_RECORDS]
    for relative in ['datasets/construction/evcivil-round3/night-holdout-protocol.json',
                     'models/construction/cubit-author-release-v1/extraction-protocol-before-weights-only-load.json',
                     'models/construction/cubit-author-release-v1/acquisition-protocol-before-download.json']:
        path = ROOT / relative
        if path.exists():
            paths.append(path)
    rows = sampled('calibration')  # Explicit partition only, not compare/video_cases.
    if len(rows) != 122 or sum(r['negative'] for r in rows) != 10:
        raise RuntimeError('Unchanged complete122 CAL and10 negative images required')
    inputs = []
    for row in rows:
        inputs.append({k: v for k, v in row.items() if k != 'image'} |
                      {'image': row['image'].relative_to(ROOT).as_posix(), 'image_sha256': sha(row['image'])})
    counts = {name: sum(e['label'] == name for r in inputs for e in r['expected']) for name in CLASSES}
    if counts != {'crack': 125, 'spalling': 38}:
        raise RuntimeError('Protected CAL labels differ')
    sources = selected_sources()  # Whitelist/metadata hashes only; no H5 decode.
    extra_folder = ROOT / 'datasets/construction/evcivil-round3/raw/dset_6/spalling_76'
    extra = [extra_folder / 'spalling_76_frames.h5', extra_folder / 'spalling_76_frame_label.npy']
    if not extra[1].exists():
        extra[1] = extra_folder / 'spalling_76_label.npy'
    schema = Path(ultralytics.__file__).parent / 'cfg/models/v8/yolov8-seg.yaml'
    previous = ROOT / 'output/evaluation/construction-cubit-author-v1'
    prior = previous / 'candidate-proposal-before-network.json'
    return {'schema_version': 1, 'parent_approved_network': False,
            'prior_unexecuted_proposal_sha256': sha(prior),
            'prior_revision2_sha256': sha(previous / 'candidate-proposal-revision2-before-network.json'),
            'revision_reason': 'Separate execution-only recovery after preserved v1 trace-cache failure: initial fixture uses no_grad instead of inference_mode. Exact original tensors/fixture/policy/data/numeric guards retained; no numerical or CAL result existed in v1.',
            'failed_export_receipt_sha256': sha(previous / 'export-execution.json'),
            'failed_protocol_sha256': sha(previous / 'candidate-protocol-before-network.json'),
            'candidate': 'Author YOLOv8n crack/spall instance masks; one fixed policy, no fit/grid rescue',
            'source_checkpoint_sha256': sha(DIRECTORY / 'author-best.pt'),
            'installed_trusted_schema_sha256': sha(schema),
            'pinned_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            'classes': CLASSES, 'class_confidence': {'crack': .25, 'spalling': .25},
            'input_size': 640, 'nms_iou': .45, 'mask_threshold': 'prototype logit >0, equivalent sigmoid>.50',
            'preprocessing': 'Shared low_light_gain once on full native RAW BGR, then aspect-preserving640 letterbox, RGB/255; no GT ROI',
            'class_mapping_warning': 'Author0crack/1spalling is opposite prior camera models; labels remain author order.',
            'calibration_inputs': inputs, 'calibration_annotations': counts,
            'before_baseline': {'model_path': BEFORE.relative_to(ROOT).as_posix(), 'model_sha256': sha(BEFORE),
                 'classes': ['spalling', 'crack'], 'class_confidence': {'crack': .10, 'spalling': .25},
                 'records_path': BEFORE_RECORDS.relative_to(ROOT).as_posix(), 'records_sha256': sha(BEFORE_RECORDS),
                 'reuse': 'Complete original actual122 serving outputs, no new BEFORE NN; require identical source/index/raw-image/GT identities and recompute scores/common scopes. BEFORE supplies boxes only.'},
            'fixed_development_videos': sources,
            'supplementary_posthoc_video': {'sequence': 'spalling_76', 'group': 'dset_6',
                 'reason': 'Known-hit illustration already selected after old model CAL, zero independent accuracy credit.',
                 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in extra}},
            'scoring': 'All122 fixed-policy class-aware publisherboxIoU>=.50. Retain all wrong-class alerts. Full actual ConstructionProfile.process parity against cached decoder boxes/scores/native masks. No pixel accuracy from box GT.',
            'minimum_review': {'each_class_precision': .8, 'each_class_recall': .5, 'geometric_thin_recall': .5,
                               'combined_negative_image_alarm_rate_max': .1, 'full_profile_parity': True},
            'before_any_cal_numeric': {'fixture': 'One saved deterministic nonconstant Torch export tensor, not inspection data',
                'coordinate_max_abs_input_pixels': 1, 'class_scores_max_abs': .01,
                'mask_coefficients_max_abs': .1, 'prototype_logits_max_abs': .1,
                'all_finite': True, 'queried_device': 'GPU', 'queried_precision': 'float32', 'queried_streams': 1,
                'failure_rule': 'Save first raw outputs and failed receipt; forbid CAL and no retry or relaxed guards.'},
            'not_release_criteria': 'Passing these prototype minima would not establish a solved, production-safe or15FPS system.',
            'forbidden': 'No protected heldout loader/inference, no fitting, no threshold search, no replacing fixed sources, no photo slideshows as video, no GT-derived predicted masks, no automatic app registration/default.',
            'rights_scope': 'Standalone author checkpoint rights and exact training overlap unknown. Research-only; CUBIT photo dataset terms do not automatically clear model/video rights.',
            'resource_scope': 'No constructor/dummy stride forward/export/NN until root approval and GPU release; contended accuracy is not capacity.'}


def checked_protocol():
    frozen = read(TARGET)
    for path, digest in frozen['pinned_files'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError('Frozen author candidate dependency differs: ' + path)
    if not frozen['parent_approved_network']:
        raise RuntimeError('Explicit reviewed candidate approval required')
    for row in frozen['calibration_inputs']:
        if sha(ROOT / row['image']) != row['image_sha256']:
            raise RuntimeError('Frozen raw CAL pixels differ')
    baseline = read(BEFORE_RECORDS)
    before_point = read(BEFORE.with_suffix('.operating-point.json'))
    before_meta = read(BEFORE.with_suffix('.provenance.json'))
    if len(baseline) != 122 or before_point['model_sha256'] != sha(BEFORE) or before_point['class_confidence'] != {'crack': .10, 'spalling': .25} or before_meta['classes'] != ['spalling', 'crack']:
        raise RuntimeError('Original cached before122 point/class identity differs before any candidate NN')
    for old, new in zip(baseline, frozen['calibration_inputs']):
        if old['source'] != new['source'] or old['original_frame_index'] != new['frame_index'] or Path(old['image']).resolve() != (ROOT / new['image']).resolve() or old['expected'] != new['expected'] or old['negative'] != new['negative']:
            raise RuntimeError('Before/after source/image/index/GT scope differs before any candidate NN')
    return frozen


def gpu_numeric(adapter, export):
    """First one-fixture GPU result, saved even if guards fail; no CAL inputs."""
    import numpy as np
    receipt_path = OUT / 'gpu-numeric-execution.json'
    if receipt_path.exists():
        raise RuntimeError('Preserve first candidate GPU numeric attempt; no retry')
    fixture_path = OUT / 'export-fixture-torch-outputs.npz'
    if sha(fixture_path) != export['torch_fixture_sha256']:
        raise RuntimeError('Original Torch numeric fixture differs')
    receipt = {'state': 'started', 'protocol_sha256': sha(TARGET),
               'started_utc': datetime.now(timezone.utc).isoformat(), 'fixture_sha256': sha(fixture_path)}
    write(receipt_path, receipt)
    try:
        actual = adapter.actual_execution_config
        devices = actual['EXECUTION_DEVICES']
        if not isinstance(devices, list) or not devices or any(not str(x).startswith('GPU') for x in devices) or not any(x in str(actual['INFERENCE_PRECISION_HINT']).lower() for x in ['float32', 'f32']) or str(actual['NUM_STREAMS']) != '1':
            raise RuntimeError('Queried actual GPU/float32/one-stream execution differs')
        with np.load(fixture_path, allow_pickle=False) as fixture:
            result = adapter.request.infer({0: fixture['images']})
            outputs = [np.array(result[adapter.compiled.output(i)], copy=True) for i in range(2)]
            np.savez_compressed(OUT / 'first-gpu-numeric-raw-outputs.npz', predictions=outputs[0], prototypes=outputs[1])
            stats = {'coordinate_max_abs': float(np.max(np.abs(outputs[0][:, :4] - fixture['predictions'][:, :4]))),
                     'class_score_max_abs': float(np.max(np.abs(outputs[0][:, 4:6] - fixture['predictions'][:, 4:6]))),
                     'mask_coefficient_max_abs': float(np.max(np.abs(outputs[0][:, 6:] - fixture['predictions'][:, 6:]))),
                     'prototype_max_abs': float(np.max(np.abs(outputs[1] - fixture['prototypes']))),
                     'all_finite': all(bool(np.isfinite(x).all()) for x in outputs)}
        passed = stats['all_finite'] and stats['coordinate_max_abs'] <= 1 and stats['class_score_max_abs'] <= .01 and stats['mask_coefficient_max_abs'] <= .1 and stats['prototype_max_abs'] <= .1
        write(OUT / 'first-gpu-numeric-report.json', {'stats': stats, 'pass': passed,
              'actual_execution_config': actual, 'requested_execution_config': adapter.requested_execution_config,
              'scope': 'Exactly one original deterministic export fixture; no CAL pixels, defect accuracy, timing or source footage claim.'})
        if not passed:
            raise RuntimeError('First actual GPU numeric fixture failed; CAL forbidden, no retry')
        receipt.update(state='completed', report_sha256=sha(OUT / 'first-gpu-numeric-report.json'),
                       raw_outputs_sha256=sha(OUT / 'first-gpu-numeric-raw-outputs.npz'))
        write(receipt_path, receipt)
    except Exception:
        receipt.update(state='failed', traceback=traceback.format_exc())
        write(receipt_path, receipt)
        raise


def calibration():
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction_cubit import CubitConstructionModel
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS, decode
    from edge_runtime.profiles.construction import ConstructionProfile
    from edge_runtime.contracts import FrameContext
    from evaluate_construction_round3 import score_frame, summarize, class_annotation_scope
    frozen = checked_protocol()
    export = read(OUT / 'export-execution.json')
    metadata = read(MODEL.with_suffix('.provenance.json'))
    if export['state'] != 'completed' or export['model_sha256'] != sha(MODEL) or export['metadata_sha256'] != sha(MODEL.with_suffix('.provenance.json')) or metadata['candidate_protocol_sha256'] != sha(TARGET):
        raise RuntimeError('Only complete bound author export can enter calibration')
    for name, key in [('export-fixture-ort-cpu-outputs.npz', 'ort_cpu_fixture_sha256'),
                      ('export-fixture-ort-cpu-numeric.json', 'ort_cpu_numeric_report_sha256')]:
        if sha(OUT / name) != export[key]:
            raise RuntimeError('First CPU numeric/export artifact differs')
    receipt_path = OUT / 'calibration-execution.json'
    if receipt_path.exists():
        raise RuntimeError('Preserve previous complete/failed/partial CAL attempt')
    receipt = {'state': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'protocol_sha256': sha(TARGET), 'model_sha256': sha(MODEL)}
    write(receipt_path, receipt)
    try:
        spec = {'file': MODEL.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(MODEL),
                'classes': CLASSES, 'capabilities': ['surface_crack', 'surface_spalling'],
                'decoder': 'cubit_yolo_seg', 'input_size': 640, 'confidence': .25,
                'wrapper_sha256': sha(ROOT / 'edge_runtime/profiles/construction_cubit.py'),
                'class_confidence': dict.fromkeys(CLASSES, .25), 'exposure_policy': 'low_light_gain',
                'source': metadata['source'], 'license': metadata['license'], 'diagnostic_only': True}
        options = {'mode': 'damage', 'damage_variant': 'cubit_author_evaluation_only', 'engine': 'openvino',
                   'openvino_device': 'GPU', 'cpu_threads': 1, 'confidence': .25, 'mask_rendering': 'masks',
                   'exposure_policy': 'low_light_gain', 'stage': 'structure', 'confirm_frames': 1}
        DAMAGE_VARIANTS['cubit_author_evaluation_only'] = spec
        profile = ConstructionProfile(options)
        if not isinstance(profile.model.delegate, CubitConstructionModel):
            raise RuntimeError('Full serving dispatcher did not select actual author adapter')
        adapter = profile.model.delegate
        gpu_numeric(adapter, export)  # First numeric result must pass BEFORE any CAL source forward.
        profile.prepare_worker()  # Declared initialization only, no evidence/tracker observations.
        records = []
        native_masks = {}
        profile_observations = []
        progress = OUT / 'first-pass-records'
        progress.mkdir()

        def signature(rows):
            return [(p['label'], tuple(p['bbox']), round(float(p['score']), 6),
                     hashlib.sha256(np.packbits(p['mask']).tobytes()).hexdigest()) for p in rows]

        for index, row in enumerate(frozen['calibration_inputs']):
            frame = cv2.imread(str(ROOT / row['image']))
            if frame is None:
                raise RuntimeError('Could not decode frozen native CAL source')
            outputs, transform = adapter.predict_raw(frame)
            cached = decode(outputs, 'crack', transform, frame.shape[:2], .25, .45, 'masks', classes=CLASSES)
            captured = []
            original = profile.model.infer

            def capture(raw_frame):
                actual = original(raw_frame)
                captured.extend(actual)
                return actual

            profile.model.infer = capture
            try:
                result = profile.process(frame, FrameContext(sequence=index, captured_monotonic=float(index),
                         source_id='public-calibration-appearance', session_id='cubit-fixed122'))
            finally:
                profile.model.infer = original
            if signature(cached) != signature(captured):
                write(OUT / 'full-profile-parity-failure.json', {'index': index, 'input': row,
                        'cached_signature': signature(cached), 'actual_signature': signature(captured)})
                np.savez_compressed(OUT / 'full-profile-parity-failure-raw-cache.npz', predictions=outputs[0], prototypes=outputs[1])
                raise RuntimeError('Cached decoder/full actual profile masks/boxes/scores differ at ' + str(index))
            observations = result.domain_state['observations']
            if result.metrics.get('input_exposure_gain') != adapter.last_input_exposure_gain:
                raise RuntimeError('Full serving profile does not preserve actual native input gain')
            if [(x['label'], tuple(x['bbox']), round(float(x['score']), 6)) for x in observations] != [(x['label'], tuple(x['bbox']), round(float(x['score']), 6)) for x in captured]:
                raise RuntimeError('Full profile lost or changed actual candidates')
            scored = score_frame(captured, row['expected'], set(CLASSES), negative=row['negative'])
            frame_masks = {}
            for pindex, (prediction, raw) in enumerate(zip(scored['predictions'], captured)):
                key = f'frame_{index:03d}_prediction_{pindex:03d}'
                mask = np.asarray(raw['mask'], bool)
                native_masks[key] = np.packbits(mask)
                frame_masks[key] = native_masks[key]
                prediction['native_mask_positive_pixels'] = int(np.count_nonzero(mask))
                prediction['native_mask_empty'] = not bool(np.any(mask))
                prediction['mask_reference'] = {'npz_key': key, 'shape_hw': list(mask.shape), 'encoding': 'numpy.packbits row-major',
                                                'packed_mask_sha256': hashlib.sha256(native_masks[key].tobytes()).hexdigest()}
            frame_mask_path = progress / f'frame_{index:03d}-actual-masks.npz'
            np.savez_compressed(frame_mask_path, **frame_masks)
            records.append(scored | {k: v for k, v in row.items() if k != 'expected'} |
                           {'input_exposure_gain': adapter.last_input_exposure_gain,
                            'first_pass_masks': {'path': frame_mask_path.relative_to(ROOT).as_posix(), 'sha256': sha(frame_mask_path)}})
            profile_observations.append({'image': row['image'], 'observations': observations,
                                         'review_candidates': result.metrics['review_candidates']})
            write(progress / f'frame_{index:03d}.json', records[-1] | {'profile_observations': observations})
            print('CAL', index + 1, '/122', flush=True)
        summary = summarize(records)
        baseline = read(BEFORE_RECORDS)
        before_point = read(BEFORE.with_suffix('.operating-point.json'))
        if len(baseline) != len(records) or before_point['model_sha256'] != sha(BEFORE) or before_point['class_confidence'] != {'crack': .10, 'spalling': .25}:
            raise RuntimeError('Original cached before122 point identity differs')
        rescored_before = []
        for old, new in zip(baseline, records):
            if old['source'] != new['source'] or old['original_frame_index'] != new['frame_index'] or Path(old['image']).resolve() != (ROOT / new['image']).resolve() or old['expected'] != new['expected'] or old['negative'] != new['negative']:
                raise RuntimeError('Before/after source/image/index/GT scope differs')
            rescored_before.append(score_frame(old['predictions'], old['expected'], set(CLASSES), negative=old['negative']) |
                                   {'negative': old['negative']})
        before_summary = summarize(rescored_before)
        thin = {e['id'] for r in records for e in r['expected'] if e['label'] == 'crack' and min(e['bbox'][2:]) <= 30 and max(e['bbox'][2:]) >= 2 * min(e['bbox'][2:])}
        matched_thin = sum(p.get('matched_annotation_id') in thin for r in records for p in r['predictions'])
        negatives = [r for r in records if r['negative']]
        alarm_rate = sum(bool(r['predictions']) for r in negatives) / len(negatives)
        checks = {name + '_precision': (summary['by_class'][name]['precision'] or 0) >= .8 for name in CLASSES}
        checks.update({name + '_recall': (summary['by_class'][name]['recall'] or 0) >= .5 for name in CLASSES})
        checks.update(geometric_thin_recall=matched_thin / len(thin) >= .5, combined_negative_alarm_rate=alarm_rate <= .1,
                      full_profile_parity=True)
        write(OUT / 'calibration-runtime-records.json', records)
        write(OUT / 'full-profile-observations.json', profile_observations)
        np.savez_compressed(OUT / 'actual-native-prediction-masks.npz', **native_masks)
        report = {'model': profile.describe(), 'protocol_sha256': sha(TARGET), 'class_confidence': dict.fromkeys(CLASSES, .25),
                  'matching_iou_threshold': .5, 'threshold_search_performed': False, 'summary': summary,
                  'geometric_thin': {'annotations': len(thin), 'localized_matches': matched_thin, 'recall': matched_thin / len(thin),
                                     'scope': 'Publisher-box shortside<=30px/longside>=2xshort; not physical crack width or pixel GT'},
                  'combined_negative_image_alarm_rate': alarm_rate, 'minimum_checks': checks,
                  'all_minimum_checks_pass': all(checks.values()), 'full_profile_prediction_parity': True,
                  'common_class_annotation_scopes': {label: summarize(class_annotation_scope(records, label, set(CLASSES))) for label in CLASSES},
                  'before_baseline': {'model_sha256': sha(BEFORE), 'class_confidence': before_point['class_confidence'],
                       'cached_records_sha256': sha(BEFORE_RECORDS), 'new_before_nn_performed': False,
                       'summary': before_summary,
                       'common_class_annotation_scopes': {label: summarize(class_annotation_scope(rescored_before, label, set(CLASSES))) for label in CLASSES},
                       'scope': 'Exact same122 source/frame/image/GT; original frozen operating policy, box-only masks unavailable. Both reused development CAL and source overlap unknown; no independent credit.'},
                  'record_count': len(records), 'records_sha256': sha(OUT / 'calibration-runtime-records.json'),
                  'actual_masks_sha256': sha(OUT / 'actual-native-prediction-masks.npz'),
                  'native_mask_audit': {'predictions': sum(len(r['predictions']) for r in records),
                       'empty_masks': sum(p['native_mask_empty'] for r in records for p in r['predictions']),
                       'bbox_matches_with_empty_masks': sum(bool(p.get('matched_annotation_id')) and p['native_mask_empty'] for r in records for p in r['predictions']),
                       'by_class': {name: {'empty_masks': sum(p['native_mask_empty'] for r in records for p in r['predictions'] if p['label'] == name),
                                          'bbox_matches_with_empty_masks': sum(bool(p.get('matched_annotation_id')) and p['native_mask_empty'] for r in records for p in r['predictions'] if p['label'] == name)} for name in CLASSES},
                       'scope': 'Actual native predicted foreground pixels, not GT pixel accuracy. Empty masks retained; a bbox match alone does not prove a crack trace or damage contour.'},
                  'scope': 'Reused development CAL. Unknown source checkpoint training overlap. Actual masks preserved, publisher GT boxes only; no pixel metrics, independent accuracy credit, 15FPS or automatic qualification.'}
        write(OUT / 'fixed-policy-calibration-report.json', report)
        receipt.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                       report_sha256=sha(OUT / 'fixed-policy-calibration-report.json'))
        write(receipt_path, receipt)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        receipt.update(state='failed', failed_utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc())
        write(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['proposal', 'freeze', 'calibration'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'proposal':
        target = PROPOSAL
        if target.exists():
            raise RuntimeError('Preserve reviewed candidate proposal')
        write(target, proposal())
        print('PROPOSED', sha(target), flush=True)
    elif args.stage == 'freeze':
        if TARGET.exists():
            raise RuntimeError('Preserve approved candidate network protocol')
        original = read(PROPOSAL)
        if original != proposal():
            raise RuntimeError('Candidate proposal sources or policy changed before approval')
        original['parent_approved_network'] = True
        write(TARGET, original)
        print('FROZEN', sha(TARGET), flush=True)
    else:
        calibration()


if __name__ == '__main__':
    main()
