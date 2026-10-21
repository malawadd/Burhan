"""Full private-video AFTER diagnostic, frozen only after public-CAL global choice.

Inert on import. No old-model inference, training, upload, invented ground truth,
frame selection or policy search. Run with the existing export environment.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
BASE = ROOT / 'output/evaluation/user-construction-videos-round4'
OUT = BASE / 'after-public-cal-choice-mixed-v3'
TRAIN = ROOT / 'output/training/construction-round4-mixed-v3'
EVAL = ROOT / 'output/evaluation/construction-round4-mixed-v3'
CHOICE = EVAL / 'global-three-checkpoint-freeze.json'
PLAN = OUT / 'plan-before-any-private-after-network.json'
CLASSES = ['crack', 'spalling']
OPTIONS = {'engine': 'openvino', 'openvino_device': 'GPU', 'cpu_threads': 1,
           'confidence': .25, 'nms_iou': .45, 'mask_rendering': 'masks',
           'exposure_policy': 'low_light_gain', 'damage_classes': CLASSES}
POLICY = {'confidence': .25, 'nms_iou': .45, 'mask_rendering': 'masks', 'exposure_policy': 'low_light_gain'}
SCOPE = ('Private unannotated original videos, all878 exposures. Unscored actual model masks; '
         'no precision/recall/FP/FN, moisture measurement, fixing adequacy, structural severity or code pass. '
         'No training, upload, frame picking, test-time crop or threshold adjustment.')
INFERENCE_SCOPE = ('Actual CubitConstructionModel.infer on each native RAW exposure, not ConstructionProfile.process. '
                   'Full-profile parity was checked separately on public CAL. Offline model diagnostics; no FPS qualification.')
BASELINE_PLAN_SHA = 'b5ce58df881c8ec2c84ea1d066884823c8e1c3066a6cabc86ee0635656b2c4ec'
BASELINE_AUDIT_SHA = '0dff579e8c01e228426cce83d435f9f727554c2ba40aa02dbb1f32e414ab98cc'
INVENTORY_SHA = 'fffc0847cd48b3dc5faddf91df75cf07ae2aa80a3ca3556cafe01a6450367349'


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256')
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def selected_checkpoint(choice):
    rows = choice['checkpoints']
    if sorted(r['step'] for r in rows) != [300, 600, 900]:
        raise ValueError('Exactly three unique fixed checkpoint outcomes required')
    if any(type(r['all_minimum_checks_pass']) is not bool for r in rows):
        raise ValueError('Explicit boolean guard outcomes required')
    eligible = [r for r in rows if r['all_minimum_checks_pass']]
    expected = max(eligible, key=lambda r: r['rank']) if eligible else None
    if choice['winner'] != expected:
        raise ValueError('Global winner differs from fixed public-CAL rank')
    return expected or next(r for r in rows if r['step'] == 900), bool(eligible)


def report_source(source, frame_count):
    """UI exact-PTS eligibility without changing the saved original inventory."""
    if type(frame_count) is not int or frame_count != source['decoded_frames']:
        raise ValueError('Report must retain every original camera exposure')
    return dict(source, original_camera_frames=frame_count, original_path=source['file'])


def metadata_plan():
    """No network imports or source-pixel decode; binds completed public experiment."""
    for path, digest in [(BASE / 'baseline-plan-v2-before-inference.json', BASELINE_PLAN_SHA),
                         (BASE / 'independent-private-baseline-audit-v1.json', BASELINE_AUDIT_SHA),
                         (BASE / 'original-video-inventory.json', INVENTORY_SHA)]:
        if sha(path) != digest:
            raise ValueError('Original independently audited baseline identity changed')
    fit_path = TRAIN / 'fit-execution.json'
    approved_path = TRAIN / 'approved-plan-before-probe-and-fit.json'
    approved, fit, choice = read(approved_path), read(fit_path), read(CHOICE)
    if approved.get('parent_approved_export_and_calibration') is not True or approved['policy'] != POLICY:
        raise ValueError('Explicit downstream approval and unchanged .25/.45 policy required')
    if fit['state'] != 'completed' or fit['completed_steps'] != 900 or fit['plan_sha256'] != sha(approved_path) or choice['approved_fit_plan_sha256'] != sha(approved_path):
        raise ValueError('All900 fixed updates and public-CAL choice must complete first')
    selected, passed = selected_checkpoint(choice)
    paths = [Path(__file__), approved_path, fit_path, CHOICE,
        ROOT / 'scripts/evaluate_user_construction_videos_round4.py',
        ROOT / 'scripts/render_saved_user_video_frame_round4.py',
        ROOT / 'scripts/train_construction_round4_mixed_v3.py',
        ROOT / 'scripts/export_construction_round4_mixed_v3.py',
        ROOT / 'scripts/evaluate_construction_round4_mixed_v3.py',
        ROOT / 'edge_runtime/profiles/construction_cubit.py',
        ROOT / 'edge_runtime/profiles/construction_model.py',
        ROOT / 'edge_runtime/profiles/construction_preprocessing.py',
        BASE / 'original-video-inventory.json', BASE / 'baseline-plan-v2-before-inference.json',
        BASE / 'all-four-before-v2-completed.json',
        BASE / 'baseline-v2-executed-source-archive/manifest.json',
        BASE / 'independent-private-baseline-audit-v1.json']
    for row in choice['checkpoints']:
        folder = EVAL / f'step{row["step"]:04d}'
        model = ROOT / f'models/construction/round4-mixed-v3/crack-spalling-step{row["step"]:04d}-640.onnx'
        report_path, export_path = folder / 'fixed-policy-calibration-report.json', folder / 'export-execution.json'
        protocol_path, execution_path = folder / 'candidate-protocol-before-network.json', folder / 'calibration-execution.json'
        report, export, protocol, execution = map(read, [report_path, export_path, protocol_path, execution_path])
        numeric_path = folder / 'export-fixture-ort-cpu-numeric.json'
        records_path = folder / 'calibration-runtime-records.json'
        metadata = model.with_suffix('.provenance.json')
        digest = sha(model)
        if export['state'] != 'completed' or execution['state'] != 'completed' or read(numeric_path)['pass'] is not True:
            raise ValueError('All three actual export/numeric/CAL outcomes must complete')
        if any(value != digest for value in [row['model_sha256'], export['model_sha256'], execution['model_sha256'], report['model']['sha256']]):
            raise ValueError('Checkpoint actual ONNX identity mismatch')
        if sha(report_path) != row['report_sha256'] or execution['report_sha256'] != row['report_sha256'] or report['records_sha256'] != sha(records_path):
            raise ValueError('Complete public-CAL evidence changed')
        if execution['protocol_sha256'] != sha(protocol_path) or report['protocol_sha256'] != sha(protocol_path) or protocol['approved_fit_plan_sha256'] != sha(approved_path):
            raise ValueError('Actual public-CAL protocol lineage mismatch')
        if report['all_minimum_checks_pass'] is not row['all_minimum_checks_pass'] or report['full_profile_prediction_parity'] is not True:
            raise ValueError('Actual public-CAL guards/serving parity differ')
        if export['metadata_sha256'] != sha(metadata) or export['ort_cpu_numeric_report_sha256'] != sha(numeric_path):
            raise ValueError('Export metadata/numeric identity mismatch')
        if len(read(records_path)) != 122 or protocol['classes'] != CLASSES or protocol['policy'] != POLICY:
            raise ValueError('Unchanged complete122 public calibration/classes/policy required')
        paths.extend([model, metadata, report_path, export_path, protocol_path, execution_path, numeric_path, records_path])
    inventory = read(BASE / 'original-video-inventory.json')
    if len(inventory) != 4 or sum(r['decoded_frames'] for r in inventory) != 878:
        raise ValueError('Exact original four/878 source inventory required')
    audit = read(BASE / 'independent-private-baseline-audit-v1.json')
    if audit['all_checks_passed'] is not True or audit['failures'] or read(BASE / 'all-four-before-v2-completed.json')['original_exposures'] != 878:
        raise ValueError('Completed independently audited original baseline required')
    baseline_plan = read(BASE / 'baseline-plan-v2-before-inference.json')
    archive = read(BASE / 'baseline-v2-executed-source-archive/manifest.json')
    if archive['baseline_plan_sha256'] != sha(BASE / 'baseline-plan-v2-before-inference.json') or audit['source_pins']['inventory_sha256'] != sha(BASE / 'original-video-inventory.json') or audit['source_pins']['source_archive_manifest_sha256'] != sha(BASE / 'baseline-v2-executed-source-archive/manifest.json'):
        raise ValueError('Original baseline inventory/source archive lineage changed')
    gain_source = 'edge_runtime/profiles/construction_preprocessing.py'
    if sha(ROOT / gain_source) != archive['source_pins'][gain_source]:
        raise ValueError('Shared native gain math differs from cached original-author input recipe')
    for name, digest in archive['source_pins'].items():
        archived = BASE / 'baseline-v2-executed-source-archive' / name
        if sha(archived) != digest:
            raise ValueError('Original executed baseline source archive bytes changed')
        paths.append(archived)
    sources = []
    for source in inventory:
        stem = Path(source['file']).stem
        report_path = BASE / stem / 'before-three-models-v2.json'
        report = read(report_path)
        prior_audit = next(r for r in audit['videos'] if r['source_name'] == stem)
        if sha(report_path) != prior_audit['report_sha256'] or sha(source['file']) != source['sha256'] or report['selection_plan_sha256'] != sha(BASE / 'baseline-plan-v2-before-inference.json'):
            raise ValueError('Original file/baseline report changed')
        if len(report['frames']) != source['decoded_frames'] or len(source['all_frame_times']) != source['decoded_frames'] or report['source']['sha256'] != source['sha256']:
            raise ValueError('Baseline source/frame inventory differs')
        for index, (frame, timestamp) in enumerate(zip(report['frames'], source['all_frame_times'])):
            if frame['source_frame_index'] != index or frame['source_pts'] != timestamp['pts'] or frame['source_time_base'] != timestamp['time_base'] or frame['evaluated'] is not False or frame['expected'] != []:
                raise ValueError('Original baseline frame identity/unscored scope differs')
            if any(p.get('evaluation_status') != 'unscored' for p in frame['models']['author_masks']['predictions']):
                raise ValueError('Cached BEFORE predictions must remain unscored')
        paths.append(report_path)
        sources.append({'inventory': source, 'baseline_report': rel(report_path), 'baseline_report_sha256': sha(report_path)})
    model = ROOT / f'models/construction/round4-mixed-v3/crack-spalling-step{selected["step"]:04d}-640.onnx'
    return {'schema_version': 1, 'selected_step': selected['step'], 'global_choice_sha256': sha(CHOICE),
        'candidate_passed_public_cal_guards': passed,
        'candidate_role': 'Public-CAL globally chosen research model' if passed else 'EXACT900 FAILED research diagnostic; no eligible model',
        'model': {'file': model.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(model),
            'input_size': 640, 'classes': CLASSES, 'source': 'Public TRAIN mixed supervision, fixed public-CAL global choice',
            'license': approved['license'], 'validation_scope': SCOPE},
        'options': OPTIONS, 'sources': sources, 'scope': SCOPE, 'inference_scope': INFERENCE_SCOPE,
        'before': {'role': 'Cached actual original-author masks, zero old-model calls', 'options': baseline_plan['models']['author_masks']['options']},
        'network_calls': {'initialization': 3, 'private_source': 878, 'old_model': 0},
        'software': {'python': sys.version, 'executable': sys.executable,
            'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'opencv-python', 'av', 'openvino']}},
        'timing_scope': 'Offline sequential diagnostic under host/project contention; not real-time capacity.',
        'rendering': '1400x1240 RAW / cached original-author BEFORE / actual new AFTER native masks / information. Gained display separate from raw; no GT.',
        'pins': {rel(path): sha(path) for path in paths}}


def freeze():
    if PLAN.exists():
        raise ValueError('Preserve first frozen plan')
    value = metadata_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / 'executed-source-before-private-after'
    archive.mkdir(exist_ok=False)
    archived = {}
    for name, digest in value['pins'].items():
        if name.endswith('.py'):
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
            if sha(target) != digest:
                raise ValueError('Archived source differs')
            archived[name] = digest
    archive_manifest = archive / 'manifest.json'
    write(archive_manifest, {'source_pins': archived, 'scope': SCOPE}, exclusive=True)
    value['source_archive'] = {'path': rel(archive_manifest), 'sha256': sha(archive_manifest)}
    write(PLAN, value, exclusive=True)
    print('PRIVATE_AFTER_PLAN', sha(PLAN), flush=True)


def verify_plan():
    value = read(PLAN)
    archive = value['source_archive']
    if sha(ROOT / archive['path']) != archive['sha256']:
        raise ValueError('Executed source archive changed')
    for name, digest in read(ROOT / archive['path'])['source_pins'].items():
        if sha((ROOT / archive['path']).parent / name) != digest:
            raise ValueError('Archived source bytes changed')
    body = dict(value)
    del body['source_archive']
    if body != metadata_plan():
        raise ValueError('Frozen choice/options/source/baseline/NN scope changed')
    return value


def cached_author(frame, native_shape):
    import numpy as np
    path = ROOT / frame['actual_masks']['path']
    if sha(path) != frame['actual_masks']['sha256']:
        raise ValueError('Cached BEFORE packed-mask file changed')
    rows = []
    with np.load(path, allow_pickle=False) as arrays:
        for prediction in frame['models']['author_masks']['predictions']:
            item = dict(prediction)
            ref = item['actual_mask']
            packed = arrays[ref['key']]
            count = int(np.prod(native_shape))
            if ref['shape_hw'] != list(native_shape) or packed.dtype != np.uint8 or packed.ndim != 1 or packed.size != (count + 7) // 8 or hashlib.sha256(packed.tobytes()).hexdigest() != ref['packed_sha256']:
                raise ValueError('Cached BEFORE actual mask identity/geometry differs')
            unpacked = np.unpackbits(packed)
            if unpacked[count:].any():
                raise ValueError('Cached BEFORE mask padding is nonzero')
            item['mask'] = unpacked[:count].reshape(native_shape).astype(bool)
            if int(item['mask'].sum()) != ref['positive_pixels']:
                raise ValueError('Cached BEFORE native mask count differs')
            rows.append(item)
    return rows


def exact_timeline(path):
    import av
    with av.open(str(path)) as source:
        return [Fraction(frame.pts) * frame.time_base for frame in source.decode(video=0)]


def config_passes(config):
    devices = config.get('EXECUTION_DEVICES')
    return (isinstance(devices, list) and bool(devices) and all(str(x).startswith('GPU') for x in devices)
            and any(x in str(config.get('INFERENCE_PRECISION_HINT')).lower() for x in ['float32', 'f32'])
            and str(config.get('NUM_STREAMS')) == '1')


def run():
    import av
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction_cubit import CubitConstructionModel
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from evaluate_user_construction_videos_round4 import fit_panel, ISSUES
    value = verify_plan()
    marker = OUT / 'execution.json'
    execution = {'state': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(PLAN), 'initialization_calls': 0, 'private_source_calls': 0, 'old_model_calls': 0,
        'completed_videos': [], 'scope': SCOPE}
    write(marker, execution, exclusive=True)
    cv2.setNumThreads(1)
    try:
        model = CubitConstructionModel(value['options'], value['model'])
        if not config_passes(model.actual_execution_config):
            raise ValueError('Actual GPU/f32/one-stream recipe differs')
        for _ in range(3):
            model.infer(np.zeros((640, 640, 3), np.uint8))
            execution['initialization_calls'] += 1
            write(marker, execution)
        for source_row in value['sources']:
            source = source_row['inventory']
            stem = Path(source['file']).stem
            folder = OUT / stem
            folder.mkdir(exist_ok=False)
            mask_folder = folder / 'actual-after-masks'
            mask_folder.mkdir()
            baseline = read(ROOT / source_row['baseline_report'])
            times = source['all_frame_times']
            origin = Fraction(times[0]['pts']) * Fraction(times[0]['time_base'])
            expected_times = [Fraction(r['pts']) * Fraction(r['time_base']) - origin for r in times]
            if exact_timeline(source['file']) != [t + origin for t in expected_times]:
                raise ValueError('Original full PTS/count changed before private inference')
            timebase = Fraction(times[0]['time_base'])
            path = folder / 'raw-cached-before-actual-after.mp4'
            records, counts, present = [], Counter(), Counter()
            cap = cv2.VideoCapture(source['file'])
            if not cap.isOpened():
                raise ValueError('Original OpenCV serving decoder unavailable')
            try:
                with av.open(str(path), 'w') as video:
                    stream = video.add_stream('libx264', rate=Fraction(source['declared_average_rate']))
                    stream.width, stream.height, stream.pix_fmt = 1400, 1240, 'yuv420p'
                    stream.time_base = timebase
                    stream.codec_context.time_base = timebase
                    stream.options = {'crf': '18', 'preset': 'fast', 'threads': '1', 'bf': '0'}
                    for index, (timestamp, old, relative) in enumerate(zip(times, baseline['frames'], expected_times)):
                        ok, raw = cap.read()
                        if not ok or list(raw.shape[1::-1]) != old['frame_size_wh'] or hashlib.sha256(raw.tobytes()).hexdigest() != old['raw_native_bgr_sha256']:
                            raise ValueError('Original native OpenCV BGR exposure differs from audited baseline')
                        before = cached_author(old, raw.shape[:2])
                        started = time.perf_counter()
                        predictions = model.infer(raw)
                        elapsed = (time.perf_counter() - started) * 1000
                        execution['private_source_calls'] += 1
                        masks, serialized = {}, []
                        for pi, item in enumerate(predictions):
                            mask = item['mask']
                            if mask.shape != raw.shape[:2] or mask.dtype != np.bool_ or item['label'] not in CLASSES or not np.isfinite(float(item['score'])):
                                raise ValueError('Actual native AFTER prediction contract differs')
                            key = 'after_' + str(pi)
                            masks[key] = np.packbits(mask)
                            row = {k: v for k, v in item.items() if k != 'mask'}
                            row.update(model='after_public_cal_choice', evaluation_status='unscored',
                                actual_mask={'key': key, 'shape_hw': list(mask.shape), 'positive_pixels': int(mask.sum()),
                                    'packed_sha256': hashlib.sha256(masks[key].tobytes()).hexdigest()})
                            serialized.append(row)
                        counts.update(r['label'] for r in serialized)
                        present.update(set(r['label'] for r in serialized))
                        present['any'] += int(bool(serialized))
                        mask_path = mask_folder / f'frame_{index:05d}.npz'
                        np.savez_compressed(mask_path, **masks)
                        gained, display_gain = camera_exposure(raw, 'low_light_gain')
                        if display_gain != model.last_input_exposure_gain or display_gain != old['models']['author_masks']['input_exposure_gain']:
                            raise ValueError('Shared RAW-to-gain display differs from actual BEFORE/AFTER input policy')
                        record = {'source_frame_index': index, 'source_pts': timestamp['pts'], 'source_time_base': timestamp['time_base'],
                            'video_time_s': float(relative), 'frame_size_wh': old['frame_size_wh'], 'evaluated': False, 'expected': [],
                            'raw_native_bgr_sha256': old['raw_native_bgr_sha256'],
                            'model_input_native_bgr_sha256': hashlib.sha256(gained.tobytes()).hexdigest(), 'predictions': serialized,
                            'before_cached_predictions': old['models']['author_masks']['predictions'],
                            'before_mask_file': old['actual_masks'], 'actual_masks': {'path': rel(mask_path), 'sha256': sha(mask_path)},
                            'input_exposure_gain': model.last_input_exposure_gain, 'model_ms_offline_not_FPS': elapsed}
                        write(mask_folder / f'frame_{index:05d}.json', record, exclusive=True)
                        records.append(record)
                        canvas = np.full((1240, 1400, 3), 24, np.uint8)
                        cv2.putText(canvas, f'PRIVATE CAMERA / frame {index} / {float(relative):.3f}s / UNSCORED MASKS', (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
                        canvas[50:620, :700] = fit_panel(raw, 'RAW - original private camera exposure', None)
                        canvas[50:620, 700:] = fit_panel(gained, 'BEFORE - cached actual original-author masks', before)
                        canvas[620:1190, :700] = fit_panel(gained, 'AFTER - actual new native prototype masks', predictions)
                        information = np.full((570, 700, 3), 24, np.uint8)
                        lines = [value['candidate_role'], f'Public-CAL selected step: {value["selected_step"]}',
                            'Both overlay panels use the SAME shared gained display.',
                            f'Native input gain: {model.last_input_exposure_gain}',
                            'Masks are model candidates; no frame ground truth exists.',
                            'Empty output is NOT an inspection pass.', 'Supported appearance classes: crack, surface material loss.',
                            'Moisture, fixings, code compliance: NOT assessed.',
                            'Original files private; never uploaded or used for fitting.', ISSUES[stem]]
                        y = 28
                        for line in lines:
                            for offset in range(0, len(line), 84):
                                cv2.putText(information, line[offset:offset + 84], (8, y), cv2.FONT_HERSHEY_SIMPLEX, .40, (230, 230, 230), 1)
                                y += 22
                        canvas[620:1190, 700:] = information
                        cv2.putText(canvas, 'RAW is separate. BEFORE is cached, AFTER is actual inference. No damage verification or accuracy/compliance credit.', (8, 1220), cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 225, 225), 1)
                        frame = av.VideoFrame.from_ndarray(canvas, format='bgr24')
                        pts = relative / timebase
                        if pts.denominator != 1:
                            raise ValueError('Original PTS cannot be represented exactly')
                        frame.pts, frame.time_base = int(pts), timebase
                        for packet in stream.encode(frame):
                            video.mux(packet)
                        write(marker, execution)
                        if index % 30 == 0:
                            print('PRIVATE_AFTER', stem, index, len(predictions), flush=True)
                    for packet in stream.encode():
                        video.mux(packet)
                extra, _ = cap.read()
                if extra or len(records) != source['decoded_frames']:
                    raise ValueError('Original whole-video frame count differs')
            finally:
                cap.release()
            if exact_timeline(path) != expected_times:
                raise ValueError('Derived full video lost original rational PTS/count')
            if sha(source['file']) != source['sha256']:
                raise ValueError('Original file changed during full diagnostic')
            report = {'schema_version': 1, 'purpose': 'unscored_private_after_diagnostic', 'independent_accuracy_credit': False,
                'title': stem + ' / cached author BEFORE / actual mixed-supervision AFTER', 'candidate_role': value['candidate_role'],
                'candidate_passed_public_cal_guards': value['candidate_passed_public_cal_guards'], 'selected_step': value['selected_step'],
                'issue_summary': ISSUES[stem], 'annotation_scope': SCOPE, 'plan_sha256': sha(PLAN),
                'baseline_report_sha256': source_row['baseline_report_sha256'], 'source': report_source(source, len(records)),
                'model': model.describe(), 'summary': {'annotated_frames': None, 'localized_matches': None, 'missed_defects': None,
                    'false_alerts': None, 'unsupported_annotations': None, 'unscored_prediction_counts': dict(counts),
                    'prediction_present_frame_counts': dict(present), 'original_frames': len(records)},
                'frames': records, 'derived_video': {'path': rel(path), 'sha256': sha(path), 'encoded_frames': len(records),
                    'all_original_PTS_exact': True}, 'timing_scope': value['timing_scope'],
                'inference_scope': INFERENCE_SCOPE,
                'privacy': 'All local only; original private videos never training, selection or upload.'}
            report_path = folder / 'after-public-cal-choice.json'
            write(report_path, report, exclusive=True)
            execution['completed_videos'].append({'source': stem, 'frames': len(records), 'report_sha256': sha(report_path), 'video_sha256': sha(path)})
            write(marker, execution)
        if execution['initialization_calls'] != 3 or execution['private_source_calls'] != 878 or len(execution['completed_videos']) != 4:
            raise ValueError('Declared one-candidate full878 +3 initialization accounting differs')
        verify_plan()
        execution.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(), model=model.describe())
        write(marker, execution)
    except BaseException as error:
        execution.update(state='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        write(marker, execution)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'run'], required=True)
    parser.add_argument('--root-network-release', action='store_true')
    args = parser.parse_args()
    if args.stage == 'run' and not args.root_network_release:
        parser.error('Explicit root GPU/private-diagnostic release required; no automatic execution')
    (freeze if args.stage == 'freeze' else run)()
