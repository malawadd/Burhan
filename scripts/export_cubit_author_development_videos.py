"""Fixed genuine video diagnostics, including misses and actual prototype masks.

No new source selection: four pre-NN fixed recordings plus previously declared
post-hoc spall76. Never calls generic compare()/video_cases() or heldout loaders.
"""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'output/evaluation/construction-cubit-author-v2/videos'
CANDIDATE = ROOT / 'output/evaluation/construction-cubit-author-v2'
TARGET = OUT / 'fixed-five-video-protocol-before-forward.json'
NAMES = ['crack_89', 'crack_129', 'spalling_75', 'ros_crack_21', 'spalling_76']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def baseline_path(name):
    if name == 'spalling_76':
        return ROOT / 'output/construction-posthoc-before-spalling_76.json'
    return ROOT / f'output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-{name}.json'


def freeze():
    from evaluate_cubit_author_candidate_v2 import checked_protocol
    from compare_construction_development_videos import selected_sources
    candidate = checked_protocol()
    point_path = CANDIDATE / 'fixed-policy-calibration-report.json'
    point = read(point_path)
    receipt = read(CANDIDATE / 'calibration-execution.json')
    if receipt['state'] != 'completed' or receipt['report_sha256'] != sha(point_path) or point['record_count'] != 122 or not point['full_profile_prediction_parity']:
        raise RuntimeError('Complete bound fixed122 serving evaluation required')
    if selected_sources() != candidate['fixed_development_videos']:
        raise RuntimeError('Pre-NN four-source whitelist changed')
    sources = candidate['fixed_development_videos'] + [candidate['supplementary_posthoc_video'] | {'role': 'POST-HOC known-hit illustration; no independent accuracy credit'}]
    if [s['sequence'] for s in sources] != NAMES:
        raise RuntimeError('Exactly five predeclared sources required')
    for source in sources:
        for relative, digest in source['files'].items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError('Pre-NN camera/annotation identity differs')
    paths = [Path(__file__), ROOT / 'scripts/evaluate_cubit_author_candidate_v2.py',
             ROOT / 'scripts/compare_construction_development_videos.py',
             ROOT / 'scripts/compare_construction_posthoc_known_hit_videos.py',
             ROOT / 'scripts/construction_annotations.py', ROOT / 'scripts/prepare_evcivil_lossless.py',
             ROOT / 'edge_runtime/profiles/construction_model.py', ROOT / 'edge_runtime/profiles/construction.py',
             ROOT / 'edge_runtime/profiles/construction_cubit.py', ROOT / 'edge_runtime/profiles/construction_preprocessing.py',
             CANDIDATE / 'candidate-protocol-before-network.json', point_path, CANDIDATE / 'calibration-execution.json']
    paths += [baseline_path(name) for name in NAMES]
    protocol = {'selection': 'Unchanged four sources fixed before author NN; spall76 explicitly supplementary historical post-hoc known-hit source.',
                'sources': sources, 'candidate_protocol_sha256': sha(CANDIDATE / 'candidate-protocol-before-network.json'),
                'model_sha256': point['model']['sha256'], 'calibration_report_sha256': sha(point_path),
                'class_confidence': {'crack': .25, 'spalling': .25}, 'classes': ['crack', 'spalling'],
                'pinned_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                'before': 'Reuse already complete original bestepoch1 source-matched full-recording predictions, no new BEFORE NN/media. Frozen cuts .10 crack/.25spall, original spall0/crack1 classes.',
                'visuals': 'Unchanged RAW / actual native gain / external publisher boxes / actual native prototype-mask pixels plus detector boxes. Model panes never contain GT-derived pixels.',
                'all_frames': 'One original camera exposure per source record and encoded frame at exact original microsecond PTS; unlabelled exposures explicitly unscored.',
                'matching': 'Class-aware one-to-one boxIoU>=.50 only. Empty-mask boxes retained/counts separate, no pixelGTmetric or physical crack width.',
                'status': 'Failed calibration remains failed. No new thresholds, models, source choice, protected heldout or app defaults.',
                'execution': 'GPU/media only after root release; per-case started/failure/completed receipts prohibit silent partial overwrite/reinference.'}
    OUT.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and read(TARGET) != protocol:
        raise RuntimeError('Preserve fixed video protocol')
    write(TARGET, protocol)
    return protocol


def load_cases(protocol):
    from compare_construction_development_videos import load_cases as four_cases
    from compare_construction_posthoc_known_hit_videos import load_cases as pair_cases
    cases = four_cases({'sources': protocol['sources'][:4]})
    pair = pair_cases({'sources': [protocol['sources'][1], protocol['sources'][4]]})
    cases.append(pair[1])
    if [c['name'] for c in cases] != NAMES:
        raise RuntimeError('Loaded cases differ from predeclared whitelist')
    return cases


def render(raw, native, record, predictions, name, status):
    import cv2
    import numpy as np
    from compare_construction_development_videos import panel_video
    canvas = panel_video(raw, native, record, name, 'AUTHOR FIXED .25 MASKS', status)
    predicted = native.copy()
    for i, (actual, scored) in enumerate(zip(predictions, record['predictions'])):
        color = (60, 205, 90) if scored.get('matched_annotation_id') else (30, 170, 245)
        mask = np.asarray(actual['mask'], bool)
        predicted[mask] = (.60 * predicted[mask] + .40 * np.asarray(color)).astype(np.uint8)
        x, y, w, h = map(round, actual['bbox'])
        cv2.rectangle(predicted, (x, y), (x + w, y + h), color, 1)
        label = f'#{i + 1} {actual["label"]} {actual["score"]:.2f}'
        if not mask.any():
            label += ' EMPTY MASK'
        cv2.putText(predicted, label, (max(0, x), max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1)
    canvas[660:1180, 716:1408] = cv2.resize(predicted, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    canvas[625:655, 710:] = (24, 27, 30)
    cv2.putText(canvas, 'ACTUAL MASK ON GAINED NN INPUT (not GT)', (716, 648), cv2.FONT_HERSHEY_SIMPLEX, .50, (235, 235, 235), 1)
    canvas[1220:1252] = (24, 27, 30)
    empty = sum(p['native_mask_empty'] for p in record['predictions'])
    cv2.putText(canvas, f'Green BOX = IoU hit; colored pixels = actual mask, NO pixel GT. Empty masks retained: {empty}. Amber = unmatched/unverified.',
                (8, 1230), cv2.FONT_HERSHEY_SIMPLEX, .43, (225, 225, 225), 1)
    return canvas


def export():
    import av
    import cv2
    import numpy as np
    from evaluate_cubit_author_candidate_v2 import MODEL
    from edge_runtime.profiles.construction import ConstructionProfile
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    from edge_runtime.contracts import FrameContext
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from evaluate_construction_round3 import score_frame, summarize
    from compare_construction_development_videos import verify_replay
    protocol = freeze()
    point = read(CANDIDATE / 'fixed-policy-calibration-report.json')
    meta = read(MODEL.with_suffix('.provenance.json'))
    if sha(MODEL) != protocol['model_sha256']:
        raise RuntimeError('Fixed author model differs')
    spec = {'file': MODEL.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(MODEL),
            'wrapper_sha256': sha(ROOT / 'edge_runtime/profiles/construction_cubit.py'), 'classes': ['crack', 'spalling'],
            'capabilities': ['surface_crack', 'surface_spalling'], 'decoder': 'cubit_yolo_seg', 'input_size': 640,
            'confidence': .25, 'class_confidence': {'crack': .25, 'spalling': .25}, 'exposure_policy': 'low_light_gain',
            'source': meta['source'], 'license': meta['license'], 'diagnostic_only': True}
    DAMAGE_VARIANTS['cubit_author_video_evaluation_only'] = spec
    profile = ConstructionProfile({'mode': 'damage', 'damage_variant': 'cubit_author_video_evaluation_only', 'engine': 'openvino',
            'openvino_device': 'GPU', 'cpu_threads': 1, 'confidence': .25, 'mask_rendering': 'masks',
            'exposure_policy': 'low_light_gain', 'stage': 'structure', 'confirm_frames': 1})
    profile.prepare_worker()
    cases = load_cases(protocol)
    results = []
    status = 'FAILED CALIBRATION' if not point['all_minimum_checks_pass'] else 'MINIMUM REVIEW ONLY - NOT QUALIFIED'
    for case in cases:
        name = case['name']
        receipt_path = OUT / (name + '-execution.json')
        if receipt_path.exists():
            raise RuntimeError('Preserve previous partial/complete video attempt: ' + name)
        target = ROOT / ('output/construction-cubit-author-v2-' + name)
        if target.with_suffix('.mp4').exists() or target.with_suffix('.json').exists():
            raise RuntimeError('Refuse to overwrite existing candidate media/report')
        receipt = {'state': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(), 'video_protocol_sha256': sha(TARGET)}
        write(receipt_path, receipt)
        try:
            baseline = read(baseline_path(name))
            if baseline['model']['sha256'] != point['before_baseline']['model_sha256'] or baseline['class_confidence'] != {'crack': .10, 'spalling': .25} or len(baseline['frames']) != len(case['frames']):
                raise RuntimeError('Complete before source/model/point differs')
            folder = ROOT / 'datasets/construction/evcivil/crack_89' if name == 'crack_89' else ROOT / 'datasets/construction/evcivil-round3/replay' / name
            proof = read(folder / (name + '-lossless-frames.provenance.json'))
            if sha(ROOT / proof['path']) != proof['sha256']:
                raise RuntimeError('Lossless source identity differs')
            case['source']['lossless_replay_input'] = proof
            case['source']['replay_identity_reverified_against_raw'] = verify_replay(case, proof, folder)
            progress = OUT / name
            progress.mkdir()
            records, before_records = [], []
            fps = max(1, round(1e6 / np.median(np.diff(case['times']))))
            with av.open(str(target.with_suffix('.mp4')), 'w') as container:
                stream = container.add_stream('libx264', rate=fps)
                stream.width = 1420
                stream.height = 1300
                stream.pix_fmt = 'yuv420p'
                stream.time_base = Fraction(1, 1_000_000)
                stream.codec_context.time_base = Fraction(1, 1_000_000)
                stream.options = {'crf': '18', 'preset': 'fast', 'threads': '1', 'bf': '0'}
                first_preview = False
                preview_info = None
                for index, (gray, truth, old) in enumerate(zip(case['frames'], case['truth'], baseline['frames'])):
                    raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    native, gain = camera_exposure(raw, 'low_light_gain')
                    if old['source_frame_index'] != index or old['video_time_s'] != truth['video_time_s'] or old['expected'] != truth['expected'] or old['raw_native_bgr_sha256'] != hashlib.sha256(raw.tobytes()).hexdigest() or old['model_input_native_bgr_sha256'] != hashlib.sha256(native.tobytes()).hexdigest():
                        raise RuntimeError('Exact cached before source/GT/PTS/raw/gain identity differs')
                    captured = []
                    original = profile.model.infer
                    def capture(frame):
                        rows = original(frame)
                        captured.extend(rows)
                        return rows
                    profile.model.infer = capture
                    try:
                        result = profile.process(raw, FrameContext(sequence=index, captured_monotonic=truth['video_time_s'], source_id='original-camera-' + name,
                                      session_id='author-five-' + name, source_time_ms=truth['video_time_s'] * 1000, source_frame_index=index))
                    finally:
                        profile.model.infer = original
                    if result.metrics['input_exposure_gain'] != gain:
                        raise RuntimeError('Displayed/native/full profile gain differs')
                    record = truth | score_frame(captured, truth['expected'], {'crack', 'spalling'}) | {
                        'frame_size_wh': [346, 260], 'input_exposure_policy': 'low_light_gain', 'input_exposure_gain': gain,
                        'raw_native_bgr_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                        'model_input_native_bgr_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
                        'profile_observations': result.domain_state['observations'],
                        'profile_review_findings': result.domain_state['review_findings'],
                        'profile_status': result.status, 'profile_guidance': result.guidance}
                    if len(record['profile_observations']) != len(captured):
                        raise RuntimeError('Full profile candidate observation count differs')
                    if len(record['predictions']) != len(captured):
                        raise RuntimeError('Scored/captured candidate count differs')
                    masks = {}
                    for pi, (prediction, actual) in enumerate(zip(record['predictions'], captured)):
                        mask = np.asarray(actual['mask'], bool)
                        if mask.shape != raw.shape[:2]:
                            raise RuntimeError('Actual predicted mask is not native source geometry')
                        key = 'prediction_' + str(pi + 1)
                        masks[key] = np.packbits(mask)
                        prediction.update(native_mask_positive_pixels=int(mask.sum()), native_mask_empty=not bool(mask.any()),
                             profile_track_id=record['profile_observations'][pi]['track_id'],
                             mask_reference={'npz_key': key, 'shape_hw': list(mask.shape), 'encoding': 'numpy.packbits row-major',
                                             'packed_mask_sha256': hashlib.sha256(masks[key].tobytes()).hexdigest()})
                    mask_path = progress / f'frame_{index:05d}-masks.npz'
                    np.savez_compressed(mask_path, **masks)
                    record['actual_masks'] = {'path': mask_path.relative_to(ROOT).as_posix(), 'sha256': sha(mask_path)}
                    write(progress / f'frame_{index:05d}.json', record)
                    records.append(record)
                    before_records.append(score_frame(old['predictions'], old['expected'], {'crack', 'spalling'}))
                    canvas = render(raw, native, record, captured, name, status)
                    if truth['expected'] and not first_preview:
                        preview = target.with_name(target.name + '-fixed-first-annotated').with_suffix('.jpg')
                        if not cv2.imwrite(str(preview), canvas):
                            raise RuntimeError('Fixed first annotated preview could not be saved')
                        preview_info = {'path': preview.relative_to(ROOT).as_posix(), 'sha256': sha(preview),
                                        'selection': 'First publisher-annotated source exposure; never selected for favorable model outcome',
                                        'source_frame_index': index}
                        first_preview = True
                    frame = av.VideoFrame.from_ndarray(canvas, format='bgr24')
                    frame.pts = int(case['times'][index]) - int(case['times'][0])
                    frame.time_base = stream.time_base
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            count = 0
            with av.open(str(target.with_suffix('.mp4'))) as video:
                for index, frame in enumerate(video.decode(video=0)):
                    if index >= len(case['frames']) or Fraction(frame.pts) * frame.time_base * 1_000_000 != int(case['times'][index]) - int(case['times'][0]):
                        raise RuntimeError('Annotated video original exposure count/PTS differs')
                    count += 1
            if count != len(case['frames']):
                raise RuntimeError('Encoded frame count differs')
            if preview_info is None:
                raise RuntimeError('Fixed first publisher-annotated exposure was not found')
            before_summary = summarize(before_records)
            for name_key in ['crack', 'spalling']:
                if any(before_summary['by_class'][name_key][key] != baseline['summary']['by_class'][name_key][key] for key in ['tp', 'fn', 'fp']):
                    raise RuntimeError('Recomputed complete cached BEFORE counts differ')
            report = {'schema_version': 1, 'purpose': 'failure_diagnostic' if not point['all_minimum_checks_pass'] else 'development_diagnostic',
                      'sequence': name, 'candidate_status': status, 'matching_iou_threshold': .5,
                      'issue_summary': 'Actual publisher-labelled crack or surface material loss; fixed author model outputs and all failures.',
                      'why_it_matters': 'Visible damage needs close inspection; a model box or mask cannot establish substrate, cause or structural severity.',
                      'next_check': 'Verify joint/shadow versus crack and measure with a real scale; check loose finish/substrate directly. No automatic acceptance.',
                      'annotation_scope': case['source']['evaluation_role'] + '; original external publisher boxes, no pixel GT; unlabelled exposures unscored',
                      'frame_size_wh': [346, 260], 'source': case['source'], 'model': profile.describe(),
                      'class_confidence': {'crack': .25, 'spalling': .25}, 'selection_protocol_sha256': sha(TARGET),
                      'summary': summarize(records), 'frames': records,
                      'paired_before': {'report_path': baseline_path(name).relative_to(ROOT).as_posix(), 'report_sha256': sha(baseline_path(name)),
                            'model_sha256': baseline['model']['sha256'], 'class_confidence': baseline['class_confidence'], 'summary': before_summary,
                            'reuse': 'Exact complete source-matched original predictions; no new BEFORE NN or media.'},
                      'native_mask_audit': {'predictions': sum(len(r['predictions']) for r in records),
                            'empty_masks': sum(p['native_mask_empty'] for r in records for p in r['predictions']),
                            'bbox_matches_with_empty_masks': sum(p['native_mask_empty'] and bool(p.get('matched_annotation_id')) for r in records for p in r['predictions']),
                            'scope': 'Actual predicted foreground pixels, not pixel accuracy.'},
                      'fixed_preview': preview_info,
                      'derived_video': {'encoded_frames': count, 'max_timestamp_error_s': 0, 'all_original_microsecond_pts_exact': True,
                                        'sha256': sha(target.with_suffix('.mp4'))}, 'independent_accuracy_credit': False}
            write(target.with_suffix('.json'), report)
            receipt.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(target.with_suffix('.json')),
                           video_sha256=sha(target.with_suffix('.mp4')))
            write(receipt_path, receipt)
            results.append({'sequence': name, 'report': target.with_suffix('.json').relative_to(ROOT).as_posix(),
                            'video': target.with_suffix('.mp4').relative_to(ROOT).as_posix(), 'summary': report['summary'],
                            'paired_before': report['paired_before'], 'native_mask_audit': report['native_mask_audit']})
            print(json.dumps(results[-1], indent=2), flush=True)
        except Exception:
            receipt.update(state='failed', traceback=traceback.format_exc())
            write(receipt_path, receipt)
            raise
    write(OUT / 'five-complete-video-summary.json', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'export'], required=True)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
        print('FROZEN', sha(TARGET), flush=True)
    else:
        export()
