"""Full local-video model diagnostics. No training and no invented ground truth."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/evaluation/user-construction-videos-round4'
PLAN = OUT / 'baseline-plan-v2-before-inference.json'
OPTIONS = {
    'author_masks': {'damage_variant': 'cubit_author_v2_failed', 'mask_rendering': 'masks'},
    'wall_appearance': {'damage_variant': 'mbdd_wall_total16_failed', 'mask_rendering': 'boxes'},
    'broad_masks': {'damage_variant': 'dacl_segformer_b1', 'mask_rendering': 'masks',
                    'confidence': .5, 'damage_classes': ['crack', 'alligator_crack', 'wetspot', 'efflorescence',
                        'rust', 'spalling', 'cavity', 'weathering', 'washout']},
}
ISSUES = {
    'VID-20261020-WA0000': 'Visible long wall crack; whether coating, render or deeper substrate is unconfirmed.',
    'VID-20261020-WA0001': 'Railing-to-wall fixings and gaps: adequacy, movement and sealing are not assessed by these models.',
    'VID-20261020-WA0002': 'Owner reports wall humidity damage. Assess visible affected finish only; no measured moisture or cause claim.',
    'VID_20261020_075653953': 'Owner reports cracks near upper wall/cornice; distinguish predicted cracks from irregular finish appearance.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def plan_value():
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    inventory_path = OUT / 'original-video-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    assert len(inventory) == 4 and sum(v['decoded_frames'] for v in inventory) == 878
    pins = [Path(__file__), ROOT / 'edge_runtime/profiles/construction.py', ROOT / 'edge_runtime/profiles/construction_model.py',
            ROOT / 'edge_runtime/profiles/construction_cubit.py', ROOT / 'edge_runtime/profiles/construction_dacl.py',
            ROOT / 'edge_runtime/profiles/construction_preprocessing.py']
    for spec in OPTIONS.values():
        model = DAMAGE_VARIANTS[spec['damage_variant']]
        assert sha(ROOT / 'models/construction' / model['file']) == model['sha256']
    return {'created_before_NN': True, 'inventory_sha256': sha(inventory_path), 'sources': [
                {k: v[k] for k in ['file', 'sha256', 'decoded_frames', 'rotation_degrees']} for v in inventory],
            'models': {key: {'options': {'mode': 'damage', 'engine': 'openvino', 'openvino_device': 'GPU',
                        'cpu_threads': 1, 'confirm_frames': 1, 'stage': 'all', **options},
                        'specification': DAMAGE_VARIANTS[options['damage_variant']]} for key, options in OPTIONS.items()},
            'source_pins': {p.relative_to(ROOT).as_posix(): sha(p) for p in pins}, 'manual_visual_review': ISSUES,
            'scope': 'All 878 original exposures, original PTS. OpenCV5 serving decoder (auto orientation) supplies native BGR; '
                     'PyAV timing is metadata only. No source skipping, test-time ROI, frame selection, fit, cut tuning or upload. '
                     'No frame ground truth: every prediction unscored, no recall/precision/false-alarm or compliance claim.',
            'rendering': 'Separate RAW/author masks/wall boxes/broad masks. Actual model masks only. Every emitted prediction retained in JSON/mask NPZ and rendered. '
                         'DACL nine relevant named channels selected before inference; no component/context or graffiti channel claim.',
            'timing': 'Offline sequential actual-profile diagnostics under ordinary host/project contention; NOT a capacity or real-time benchmark.',
            'technical_v1_failure_sha256': sha(OUT / 'baseline-v1-technical-failure.json'),
            'v1_model_options_unchanged': OPTIONS == {k: {x: y for x, y in v['options'].items() if x in OPTIONS[k]} for k, v in json.loads((OUT / 'baseline-plan-before-inference.json').read_text())['models'].items()}}


def fit_panel(image, title, rows):
    result = np.full((570, 700, 3), 24, np.uint8)
    drawn = image.copy()
    for prediction in rows or []:
        color = (40, 200, 250) if prediction['label'] in {'crack', 'alligator_crack'} else (230, 180, 60)
        mask = prediction.get('mask')
        if mask is not None:
            if mask.shape != image.shape[:2]:
                raise ValueError('Actual native mask geometry differs')
            drawn[mask] = (.65 * drawn[mask] + .35 * np.asarray(color)).astype(np.uint8)
        else:
            x, y, w, h = map(round, prediction['bbox'])
            cv2.rectangle(drawn, (x, y), (x + w, y + h), color, 1)
    scale = min(690 / drawn.shape[1], 465 / drawn.shape[0])
    thumb = cv2.resize(drawn, (max(1, round(drawn.shape[1] * scale)), max(1, round(drawn.shape[0] * scale))))
    x = (700 - thumb.shape[1]) // 2
    result[36:36 + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    cv2.putText(result, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1)
    counts = Counter(r['label'] for r in rows or [])
    text = ('Original camera frame; no model overlay.' if rows is None else
            ' | '.join(f'{name}:{count}' for name, count in sorted(counts.items())) if rows else
            'No model candidate emitted; this is not an inspection pass.')
    for offset in range(0, len(text), 87):
        cv2.putText(result, text[offset:offset + 87], (8, 525 + (offset // 87) * 17), cv2.FONT_HERSHEY_SIMPLEX, .39, (225, 225, 225), 1)
    return result


def run():
    from edge_runtime.profiles.construction import ConstructionProfile
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from edge_runtime.contracts import FrameContext
    plan = json.loads(PLAN.read_text())
    if plan != plan_value():
        raise ValueError('Pre-inference plan/source/model changed')
    inventory = json.loads((OUT / 'original-video-inventory.json').read_text())
    profiles = {}
    for name, definition in plan['models'].items():
        profiles[name] = ConstructionProfile(definition['options'])
        profiles[name].prepare_worker()
    for source in inventory:
        folder = OUT / Path(source['file']).stem
        attempt = folder / 'baseline-v2-attempt.json'
        write_new(attempt, {'started_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(PLAN)})
        if sha(Path(source['file'])) != source['sha256']:
            raise ValueError('Original user file changed')
        for profile in profiles.values():
            profile.reset()
        cap = cv2.VideoCapture(source['file'])
        if not cap.isOpened():
            raise ValueError('OpenCV serving decoder could not open original')
        times = source['all_frame_times']
        timebase = Fraction(times[0]['time_base'])
        origin = Fraction(times[0]['pts']) * timebase
        video_path = folder / 'before-three-models-v2.mp4'
        if video_path.exists():
            raise ValueError('Preserve previous partial media')
        records = []
        mask_folder = folder / 'baseline-v2-masks'
        mask_folder.mkdir()
        per_model = {name: Counter() for name in profiles}
        with av.open(str(video_path), 'w') as video:
            stream = video.add_stream('libx264', rate=Fraction(source['declared_average_rate']))
            stream.width, stream.height, stream.pix_fmt = 1400, 1240, 'yuv420p'
            stream.time_base = timebase
            stream.codec_context.time_base = timebase
            stream.options = {'crf': '18', 'preset': 'fast', 'threads': '1', 'bf': '0'}
            for index, timestamp in enumerate(times):
                ok, raw = cap.read()
                if not ok:
                    raise ValueError('Serving decoder lost an original exposure')
                if raw.shape[1] != (source['native_height'] if source['rotation_degrees'] % 180 else source['native_width']):
                    raise ValueError('Auto-oriented source geometry differs')
                if raw.shape[0] != (source['native_width'] if source['rotation_degrees'] % 180 else source['native_height']):
                    raise ValueError('Auto-oriented source height differs')
                relative = Fraction(timestamp['pts']) * Fraction(timestamp['time_base']) - origin
                rows_by_model, records_by_model, masks = {}, {}, {}
                for name, profile in profiles.items():
                    captured = []
                    original = profile.model.infer
                    def capture(frame):
                        rows = original(frame)
                        captured.extend(rows)
                        return rows
                    profile.model.infer = capture
                    start = time.perf_counter()
                    try:
                        result = profile.process(raw, FrameContext(index, float(relative), source_id='file:' + source['file'],
                            session_id='round4-user-before-' + Path(source['file']).stem, source_time_ms=float(relative) * 1000,
                            source_frame_index=index, source_dimensions=(raw.shape[1], raw.shape[0])))
                    finally:
                        profile.model.infer = original
                    elapsed = (time.perf_counter() - start) * 1000
                    if len(captured) != len(result.domain_state['observations']):
                        raise ValueError('Actual serving predictions/capture differ')
                    serialized = []
                    for pi, actual in enumerate(captured):
                        prediction = {k: v for k, v in actual.items() if k != 'mask'}
                        prediction.update(model=name, evaluation_status='unscored')
                        if 'mask' in actual:
                            if actual['mask'].shape != raw.shape[:2]:
                                raise ValueError('Captured native mask geometry differs')
                            key = name + '_' + str(pi)
                            masks[key] = np.packbits(actual['mask'])
                            prediction['actual_mask'] = {'key': key, 'shape_hw': list(actual['mask'].shape),
                                'positive_pixels': int(actual['mask'].sum()), 'packed_sha256': hashlib.sha256(masks[key].tobytes()).hexdigest()}
                        serialized.append(prediction)
                    per_model[name].update(r['label'] for r in captured)
                    rows_by_model[name] = captured
                    records_by_model[name] = {'predictions': serialized, 'profile_ms_offline': elapsed,
                        'actual_serving_status': result.status, 'input_exposure_gain': result.metrics['input_exposure_gain'],
                        'review_findings': result.domain_state['review_findings']}
                mask_path = mask_folder / f'frame_{index:05d}.npz'
                np.savez_compressed(mask_path, **masks)
                record = {'source_frame_index': index, 'source_pts': timestamp['pts'], 'source_time_base': timestamp['time_base'],
                    'video_time_s': float(relative), 'frame_size_wh': [raw.shape[1], raw.shape[0]], 'evaluated': False,
                    'expected': [], 'raw_native_bgr_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'models': records_by_model, 'predictions': [p for model in records_by_model.values() for p in model['predictions']],
                    'actual_masks': {'path': mask_path.relative_to(ROOT).as_posix(), 'sha256': sha(mask_path)}}
                write_new(mask_folder / f'frame_{index:05d}.json', record)
                records.append(record)
                canvas = np.full((1240, 1400, 3), 24, np.uint8)
                cv2.putText(canvas, f'USER VIDEO / BEFORE / frame {index} / {float(relative):.3f}s / NO GROUND-TRUTH METRICS', (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (245, 245, 245), 1)
                gained, _ = camera_exposure(raw, 'low_light_gain')
                canvas[50:620, :700] = fit_panel(raw, 'RAW - original local video, orientation respected', None)
                canvas[50:620, 700:] = fit_panel(gained, 'AUTHOR: actual crack/material-loss masks', rows_by_model['author_masks'])
                canvas[620:1190, :700] = fit_panel(raw, 'WALL: appearance BOXES only, failed research', rows_by_model['wall_appearance'])
                canvas[620:1190, 700:] = fit_panel(raw, 'BROAD: actual damage-appearance MASKS', rows_by_model['broad_masks'])
                cv2.putText(canvas, 'Colored pixels are actual predictions, not verified defects. Fixing adequacy, humidity and code acceptance UNASSESSED.', (8, 1220), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1)
                frame = av.VideoFrame.from_ndarray(canvas, format='bgr24')
                exact_pts = relative / timebase
                if exact_pts.denominator != 1:
                    raise ValueError('Source timebase conversion must remain exact')
                frame.pts, frame.time_base = int(exact_pts), timebase
                for packet in stream.encode(frame):
                    video.mux(packet)
                if index % 30 == 0:
                    print(Path(source['file']).name, index, {n: len(v) for n, v in rows_by_model.items()}, flush=True)
            for packet in stream.encode():
                video.mux(packet)
        extra, _ = cap.read()
        cap.release()
        if extra or len(records) != source['decoded_frames']:
            raise ValueError('Original full video exposure count differs')
        decoded_times = []
        with av.open(str(video_path)) as video:
            for frame in video.decode(video=0):
                decoded_times.append(Fraction(frame.pts) * frame.time_base)
        expected_times = [Fraction(r['pts']) * Fraction(r['time_base']) - origin for r in times]
        if decoded_times != expected_times:
            raise ValueError('Encoded complete video PTS differ')
        report = {'schema_version': 1, 'purpose': 'failure_diagnostic', 'independent_accuracy_credit': False,
            'title': Path(source['file']).stem + ' / all three baseline models', 'issue_summary': ISSUES[Path(source['file']).stem],
            'why_it_matters': 'Inspect visible material and installation details directly. Model outputs do not determine cause or acceptability.',
            'next_check': 'Confirm intended fault and locations with the owner/inspector; use relevant material, moisture or fixing tests.',
            'annotation_scope': 'User video without frame ground truth. All model outputs unscored; no TP/FN/FP, recall or precision claim.',
            'source': {'original_path': source['file'], 'sha256': source['sha256'], 'original_camera_frames': len(records),
                       'rotation_degrees': source['rotation_degrees'], 'decoder': 'OpenCV ' + cv2.__version__,
                       'privacy': 'Local processing only; never uploaded or used for training.'},
            'summary': {'annotated_frames': None, 'localized_matches': None, 'missed_defects': None, 'false_alerts': None,
                        'unsupported_annotations': None, 'unscored_prediction_counts': {n: dict(c) for n, c in per_model.items()}},
            'models': {name: profile.describe() for name, profile in profiles.items()}, 'selection_plan_sha256': sha(PLAN),
            'frames': records, 'derived_video': {'path': video_path.relative_to(ROOT).as_posix(), 'sha256': sha(video_path),
                        'encoded_frames': len(records), 'all_original_PTS_exact': True},
            'timing_scope': plan['timing']}
        write_new(folder / 'before-three-models-v2.json', report)
        write_new(folder / 'baseline-v2-completed.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
            'report_sha256': sha(folder / 'before-three-models-v2.json'), 'video_sha256': sha(video_path), 'frames': len(records)})
    write_new(OUT / 'all-four-before-v2-completed.json', {'completed': True, 'original_exposures': 878, 'models_per_exposure': 3,
        'NN_calls_excluding_initialization': 2634, 'ground_truth_metrics': False, 'plan_sha256': sha(PLAN)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'run'], required=True)
    args = parser.parse_args()
    if args.stage == 'freeze':
        write_new(PLAN, plan_value())
        print('Frozen baseline plan:', sha(PLAN))
    else:
        cv2.setNumThreads(1)
        run()
