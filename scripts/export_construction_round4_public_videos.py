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
OUT = ROOT / 'output/evaluation/construction-round4-mixed-v3/videos'
CANDIDATE = None  # Derived exclusively from completed global choice during freeze().
CHOICE = ROOT / 'output/evaluation/construction-round4-mixed-v3/global-three-checkpoint-freeze.json'
ORIGINAL = ROOT / 'output/evaluation/construction-cubit-author-v2/candidate-protocol-before-network.json'
TARGET = OUT / 'fixed-five-video-protocol-before-forward.json'
NAMES = ['crack_89', 'crack_129', 'spalling_75', 'ros_crack_21', 'spalling_76']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def baseline_path(name):
    if name not in NAMES:
        raise RuntimeError('Only fixed five public sources allowed')
    return ROOT / f'output/construction-cubit-author-v2-{name}.json'


def chosen():
    from train_construction_round4_mixed_v3 import approved
    plan, _, plan_hash = approved('fit')
    if plan.get('parent_approved_export_and_calibration') is not True:
        raise RuntimeError('Reviewed downstream stage required')
    fit = read(ROOT / 'output/training/construction-round4-mixed-v3/fit-execution.json')
    choice = read(CHOICE)
    if fit['state'] != 'completed' or fit['completed_steps'] != 900 or choice['approved_fit_plan_sha256'] != plan_hash or sorted(r['step'] for r in choice['checkpoints']) != [300, 600, 900]:
        raise RuntimeError('Complete900/all3CAL global source-bound choice required')
    if choice['winner'] is not None and choice['winner'] not in choice['checkpoints']:
        raise RuntimeError('Chosen winner must equal an actual fixed checkpoint record')
    if any(type(row['all_minimum_checks_pass']) is not bool for row in choice['checkpoints']):
        raise RuntimeError('Explicit boolean guard outcomes required')
    eligible = [row for row in choice['checkpoints'] if row['all_minimum_checks_pass']]
    if choice['winner'] != (max(eligible, key=lambda row: row['rank']) if eligible else None):
        raise RuntimeError('Chosen winner differs from fixed public-CAL global rank')
    for entry in choice['checkpoints']:
        run_folder = CHOICE.parent / f'step{entry["step"]:04d}'
        execution = read(run_folder / 'calibration-execution.json')
        if execution['state'] != 'completed' or execution['model_sha256'] != entry['model_sha256'] or execution['report_sha256'] != entry['report_sha256'] or sha(run_folder / 'fixed-policy-calibration-report.json') != entry['report_sha256']:
            raise RuntimeError('All three actual completed CAL results required')
    selected = choice['winner'] or next(r for r in choice['checkpoints'] if r['step'] == 900)
    folder = CHOICE.parent / f'step{selected["step"]:04d}'
    model = ROOT / f'models/construction/round4-mixed-v3/crack-spalling-step{selected["step"]:04d}-640.onnx'
    report = folder / 'fixed-policy-calibration-report.json'
    execution = read(folder / 'calibration-execution.json')
    if sha(model) != selected['model_sha256'] or sha(report) != selected['report_sha256'] or execution['state'] != 'completed' or execution['report_sha256'] != sha(report) or execution['model_sha256'] != sha(model):
        raise RuntimeError('Chosen/fallback actual checkpoint report/model differs')
    if read(report)['all_minimum_checks_pass'] != (choice['winner'] is not None):
        raise RuntimeError('Frozen choice eligibility differs')
    return model, folder, choice


def freeze():
    global CANDIDATE
    from compare_construction_development_videos import selected_sources
    model, CANDIDATE, choice = chosen()
    candidate = read(ORIGINAL)
    if selected_sources() != candidate['fixed_development_videos']:
        raise RuntimeError('Original fixed public source whitelist differs')
    sources = candidate['fixed_development_videos'] + [candidate['supplementary_posthoc_video'] | {'role': 'POST-HOC known-hit illustration; no independent accuracy credit'}]
    if [s['sequence'] for s in sources] != NAMES:
        raise RuntimeError('Exactly the original fixed five sources required')
    point_path = CANDIDATE / 'fixed-policy-calibration-report.json'
    before_sha = 'df13a679081023d9093f731631ad199dbaaa9c467bb1e3c0df7054240c51bb6f'
    paths = [Path(__file__), ORIGINAL, CHOICE, model, model.with_suffix('.provenance.json'), point_path,
        CANDIDATE / 'candidate-protocol-before-network.json', CANDIDATE / 'calibration-execution.json', ROOT / 'scripts/compare_construction_development_videos.py',
        ROOT / 'scripts/compare_construction_posthoc_known_hit_videos.py', ROOT / 'scripts/construction_annotations.py',
        ROOT / 'scripts/prepare_evcivil_lossless.py', ROOT / 'edge_runtime/profiles/construction_model.py',
        ROOT / 'edge_runtime/profiles/construction.py', ROOT / 'edge_runtime/profiles/construction_cubit.py',
        ROOT / 'edge_runtime/profiles/construction_preprocessing.py',
        ROOT / 'output/training/construction-round4-mixed-v3/approved-plan-before-probe-and-fit.json']
    expected_counts = [24, 104, 84, 222, 249]
    for source, name, count in zip(sources, NAMES, expected_counts):
        for relative, digest in source['files'].items():
            path = ROOT / relative
            if sha(path) != digest:
                raise RuntimeError('Original raw camera/annotation identity changed')
            paths.append(path)
        baseline_file = baseline_path(name)
        baseline = read(baseline_file)
        if baseline['model']['sha256'] != before_sha or baseline['class_confidence'] != {'crack': .25, 'spalling': .25} or len(baseline['frames']) != count:
            raise RuntimeError('Complete original author cached BEFORE differs')
        paths.append(baseline_file)
        for index, record in enumerate(baseline['frames']):
            if record['source_frame_index'] != index:
                raise RuntimeError('Cached BEFORE source index differs')
            mask_file = ROOT / record['actual_masks']['path']
            if sha(mask_file) != record['actual_masks']['sha256']:
                raise RuntimeError('Cached BEFORE actual masks changed')
            paths.append(mask_file)
    protocol = {'selection': 'Same four predeclared public cases plus explicitly POST-HOC sp76; all683 original exposures, no outcome selection.',
        'sources': sources, 'model_sha256': sha(model), 'model_path': model.relative_to(ROOT).as_posix(),
        'global_choice_sha256': sha(CHOICE), 'selected_step': read(CANDIDATE / 'candidate-protocol-before-network.json')['step'],
        'candidate_passed_minimum_guards': choice['winner'] is not None,
        'candidate_protocol_sha256': sha(CANDIDATE / 'candidate-protocol-before-network.json'),
        'calibration_report_sha256': sha(point_path), 'before_model_sha256': before_sha,
        'classes': ['crack', 'spalling'], 'class_confidence': {'crack': .25, 'spalling': .25},
        'pinned_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'before': 'Exact original author fixed.25/.45 cached actual683 predictions/native prototype masks, no new BEFORE NN.',
        'visuals': 'Separate unchanged RAW, actual native gain input, external publisher boxes, original-author actual masks and new actual masks. No GT-derived model pixels.',
        'matching': 'Class-aware one-to-one boxIoU>=.50; camera boxGT does not establish pixel trace accuracy. Empty masks retained.',
        'status': 'Exact globally chosen checkpoint, or exact900 FAILED diagnostic if no winner. No automatic qualification/default promotion.',
        'all_frames': 'Exact original microsecond PTS, one real exposure per encoded frame, all unlabelled exposures unscored.',
        'execution': 'Accuracy/media-only after explicit root GPU release; preserve first started/failure/completed attempts.'}
    OUT.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and read(TARGET) != protocol:
        raise RuntimeError('Preserve previously frozen video protocol')
    write(TARGET, protocol)
    return protocol


def before_masks(record, shape):
    import numpy as np
    target = ROOT / record['actual_masks']['path']
    if sha(target) != record['actual_masks']['sha256']:
        raise RuntimeError('Cached BEFORE mask file changed')
    masks = []
    with np.load(target, allow_pickle=False) as packed:
        for row in record['predictions']:
            ref = row['mask_reference']
            bits = packed[ref['npz_key']]
            if ref['shape_hw'] != list(shape) or hashlib.sha256(bits.tobytes()).hexdigest() != ref['packed_mask_sha256']:
                raise RuntimeError('Cached BEFORE mask geometry/identity changed')
            mask = np.unpackbits(bits)[:shape[0] * shape[1]].reshape(shape).astype(bool)
            if int(mask.sum()) != row['native_mask_positive_pixels']:
                raise RuntimeError('Cached BEFORE foreground count differs')
            masks.append(row | {'mask': mask})
    return masks


def load_cases(protocol):
    from compare_construction_development_videos import load_cases as four_cases
    from compare_construction_posthoc_known_hit_videos import load_cases as pair_cases
    cases = four_cases({'sources': protocol['sources'][:4]})
    pair = pair_cases({'sources': [protocol['sources'][1], protocol['sources'][4]]})
    cases.append(pair[1])
    if [c['name'] for c in cases] != NAMES:
        raise RuntimeError('Loaded cases differ from predeclared whitelist')
    return cases


def render(raw, native, record, predictions, name, status, old, old_masks):
    import cv2
    import numpy as np
    canvas = np.full((1300, 2140, 3), (24, 27, 30), np.uint8)
    gt = raw.copy()
    after_matched = {p.get('matched_annotation_id') for p in record['predictions'] if p.get('matched_annotation_id')}
    for item in record['expected']:
        x, y, w, h = map(round, item['bbox'])
        color = (245, 210, 30) if item['id'] in after_matched else (70, 70, 245)
        cv2.rectangle(gt, (x, y), (x + w, y + h), color, 1)
        cv2.putText(gt, 'SOURCE ' + item['label'], (max(0, x), max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1)
    def predicted(actual_rows, scored_rows):
        if len(actual_rows) != len(scored_rows):
            raise RuntimeError('Mask/scored count differs in presentation')
        image = native.copy()
        for index, (actual, scored) in enumerate(zip(actual_rows, scored_rows)):
            color = (60, 205, 90) if scored.get('matched_annotation_id') else (30, 170, 245)
            mask = np.asarray(actual['mask'], bool)
            image[mask] = (.60 * image[mask] + .40 * np.asarray(color)).astype(np.uint8)
            x, y, w, h = map(round, actual['bbox'])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            cv2.putText(image, f'#{index+1} {actual["label"]} {actual["score"]:.2f}', (max(0,x),max(12,y-3)), cv2.FONT_HERSHEY_SIMPLEX,.30,color,1)
        return image
    panels = [(raw, 'RAW CAMERA - unchanged'), (native, f'ACTUAL NN INPUT - gain {record["input_exposure_gain"]:.2f}x'),
        (gt, 'EXTERNAL PUBLISHER BOXES - no pixel GT'), (predicted(old_masks, old['predictions']), 'BEFORE ORIGINAL AUTHOR - actual mask pixels'),
        (predicted(predictions, record['predictions']), 'AFTER MIXED REAL SUPERVISION - actual masks')]
    for index, (pixels, label) in enumerate(panels):
        x, y = 8 + (index % 3) * 708, 100 + (index // 3) * 560
        canvas[y:y+520, x:x+692] = cv2.resize(pixels, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas,label,(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1)
    lines = ['WHAT: visible crack or surface material loss.', 'Verify substrate/coating, joints and shadows.', 'NEXT: close inspection and real-scale measurement.', 'Masks are actual predictions, not publisher truth.', 'GT plane: cyan AFTER box match / red AFTER miss.', 'Green model BOX = publisher boxIoU>=.50.', 'Amber candidate = unmatched or unverified.']
    if record['evaluated']:
        lines += [f'BEFORE matched {old["localized_matches"]}, missed {old["missed_defects"]}, unmatched {old["false_alerts"]}',
                  f'AFTER matched {record["localized_matches"]}, missed {record["missed_defects"]}, unmatched {record["false_alerts"]}']
    else:
        lines += ['UNSCORED: no publisher annotations.', 'Matches/misses/unmatched are not scored.']
    for index, line in enumerate(lines):
        cv2.putText(canvas,line,(1424,690+index*42),cv2.FONT_HERSHEY_SIMPLEX,.46,(230,230,230),1)
    header = [f'REAL PUBLIC DAMAGE / {name} / {status}', f'Original source frame {record["source_frame_index"]} at {record["video_time_s"]:.6f}s / DEVELOPMENT ONLY, no independent credit']
    for index, line in enumerate(header):
        cv2.putText(canvas,line,(8,25+index*25),cv2.FONT_HERSHEY_SIMPLEX,.60,(235,235,235),1)
    cv2.putText(canvas,'Actual native pixels shown2x. Source76 is POST-HOC. All frames/misses/false alerts retained. No pixelGT, physical width, cause or code-compliance claim.',(8,1220),cv2.FONT_HERSHEY_SIMPLEX,.47,(230,230,230),1)
    cv2.putText(canvas,'Gamage et al., ev-CIVIL (DTU), CC BY4.0. No source exposure repeated/generated. Full actual masks and original PTS retained separately.',(8,1250),cv2.FONT_HERSHEY_SIMPLEX,.47,(230,230,230),1)
    return canvas


def export():
    import av
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction import ConstructionProfile
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    from edge_runtime.contracts import FrameContext
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from evaluate_construction_round3 import score_frame, summarize
    from compare_construction_development_videos import verify_replay
    protocol = freeze()
    MODEL, _, _ = chosen()
    for name in NAMES:
        target = ROOT / ('output/construction-round4-mixed-v3-' + name)
        if (OUT / (name + '-execution.json')).exists() or target.with_suffix('.mp4').exists() or target.with_suffix('.json').exists() or (OUT / name).exists():
            raise RuntimeError('Preserve prior partial/complete video before any new initialization: ' + name)
    point = read(CANDIDATE / 'fixed-policy-calibration-report.json')
    meta = read(MODEL.with_suffix('.provenance.json'))
    if sha(MODEL) != protocol['model_sha256']:
        raise RuntimeError('Fixed author model differs')
    spec = {'file': MODEL.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(MODEL),
            'wrapper_sha256': sha(ROOT / 'edge_runtime/profiles/construction_cubit.py'), 'classes': ['crack', 'spalling'],
            'capabilities': ['surface_crack', 'surface_spalling'], 'decoder': 'cubit_yolo_seg', 'input_size': 640,
            'confidence': .25, 'class_confidence': {'crack': .25, 'spalling': .25}, 'exposure_policy': 'low_light_gain',
            'source': meta['source'], 'license': meta['license'], 'diagnostic_only': True}
    DAMAGE_VARIANTS['round4_mixed_video_evaluation_only'] = spec
    profile = ConstructionProfile({'mode': 'damage', 'damage_variant': 'round4_mixed_video_evaluation_only', 'engine': 'openvino',
            'openvino_device': 'GPU', 'cpu_threads': 1, 'confidence': .25, 'mask_rendering': 'masks',
            'exposure_policy': 'low_light_gain', 'stage': 'structure', 'confirm_frames': 1})
    actual_config = profile.model.delegate.actual_execution_config
    devices = actual_config['EXECUTION_DEVICES']
    if not isinstance(devices, list) or not devices or any(not str(x).startswith('GPU') for x in devices) or not any(x in str(actual_config['INFERENCE_PRECISION_HINT']).lower() for x in ['float32', 'f32']) or str(actual_config['NUM_STREAMS']) != '1':
        raise RuntimeError('Actual public-video GPU/float32/one-stream execution differs')
    profile.prepare_worker()
    cases = load_cases(protocol)
    results = []
    status = 'FAILED CALIBRATION' if not point['all_minimum_checks_pass'] else 'MINIMUM REVIEW ONLY - NOT QUALIFIED'
    for case in cases:
        profile.reset()  # Independent recording trackers; stateless NN and policy unchanged.
        name = case['name']
        receipt_path = OUT / (name + '-execution.json')
        if receipt_path.exists():
            raise RuntimeError('Preserve previous partial/complete video attempt: ' + name)
        target = ROOT / ('output/construction-round4-mixed-v3-' + name)
        if target.with_suffix('.mp4').exists() or target.with_suffix('.json').exists():
            raise RuntimeError('Refuse to overwrite existing candidate media/report')
        receipt = {'state': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(), 'video_protocol_sha256': sha(TARGET)}
        write(receipt_path, receipt)
        try:
            baseline = read(baseline_path(name))
            if baseline['model']['sha256'] != protocol['before_model_sha256'] or baseline['class_confidence'] != {'crack': .25, 'spalling': .25} or len(baseline['frames']) != len(case['frames']):
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
                stream.width = 2140
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
                    cached_before_masks = before_masks(old, raw.shape[:2])
                    captured = []
                    original = profile.model.infer
                    def capture(frame):
                        rows = original(frame)
                        captured.extend(rows)
                        return rows
                    profile.model.infer = capture
                    try:
                        result = profile.process(raw, FrameContext(sequence=index, captured_monotonic=truth['video_time_s'], source_id='original-camera-' + name,
                                      session_id='round4-mixed-five-' + name, source_time_ms=truth['video_time_s'] * 1000, source_frame_index=index))
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
                        'profile_status': result.status, 'profile_guidance': result.guidance,
                        'paired_before_source_frame_index': old['source_frame_index'], 'paired_before_actual_masks': old['actual_masks'],
                        'paired_before_counts': {key: old[key] for key in ['evaluated', 'localized_matches', 'missed_defects', 'false_alerts']}}
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
                    canvas = render(raw, native, record, captured, name, status, old, cached_before_masks)
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
                      'issue_summary': 'Actual publisher-labelled crack or surface material loss; mixed real-supervision model outputs and all failures.',
                      'why_it_matters': 'Visible damage needs close inspection; a model box or mask cannot establish substrate, cause or structural severity.',
                      'next_check': 'Verify joint/shadow versus crack and measure with a real scale; check loose finish/substrate directly. No automatic acceptance.',
                      'annotation_scope': case['source']['evaluation_role'] + '; original external publisher boxes, no pixel GT; unlabelled exposures unscored',
                      'frame_size_wh': [346, 260], 'source': case['source'], 'model': profile.describe(),
                      'class_confidence': {'crack': .25, 'spalling': .25}, 'selection_protocol_sha256': sha(TARGET),
                      'summary': summarize(records), 'frames': records,
                      'paired_before': {'report_path': baseline_path(name).relative_to(ROOT).as_posix(), 'report_sha256': sha(baseline_path(name)),
                            'model_sha256': baseline['model']['sha256'], 'class_confidence': baseline['class_confidence'], 'summary': before_summary,
                            'reuse': 'Exact complete original author source-matched predictions and actual native masks; no new BEFORE NN or media.'},
                      'native_mask_audit': {'predictions': sum(len(r['predictions']) for r in records),
                            'empty_masks': sum(p['native_mask_empty'] for r in records for p in r['predictions']),
                            'bbox_matches_with_empty_masks': sum(p['native_mask_empty'] and bool(p.get('matched_annotation_id')) for r in records for p in r['predictions']),
                            'scope': 'Actual predicted foreground pixels, not pixel accuracy.'},
                      'global_choice_sha256': sha(CHOICE), 'selected_step': protocol['selected_step'],
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
