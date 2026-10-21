"""Independent file/pixel/timeline audit of all private CPU AFTER evidence; no NN."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from audit_user_construction_videos_round4 import Checks, encoded_timeline, fraction, protected_path, sha, read
from edge_runtime.profiles.construction_preprocessing import camera_exposure

BASE = ROOT / 'output/evaluation/user-construction-videos-round4'
OUT = BASE / 'after-public-cal-choice-cpu-recovery-v1'
PLAN = OUT / 'plan-before-any-private-after-network.json'
TARGET = OUT / 'independent-native-evidence-audit.json'


def main():
    if TARGET.exists():
        raise ValueError('Preserve first independent audit')
    cv2.setNumThreads(1)
    plan = read(PLAN)
    inventory = read(BASE / 'original-video-inventory.json')
    execution = read(OUT / 'execution.json')
    checks = Checks()
    checks.check(execution['state'] == 'completed' and execution['private_source_calls'] == 878
                 and execution['initialization_calls'] == 3 and execution['old_model_calls'] == 0
                 and execution['plan_sha256'] == sha(PLAN), 'Completed exact878+3 calls / no old-model calls / plan')
    checks.check(plan['options']['engine'] == 'onnxruntime'
                 and plan['options']['openvino_device'] == 'CPU'
                 and plan['options']['cpu_threads'] == 1, 'Frozen CPUORT/CPU1')
    videos, all_masks, all_frames = [], 0, 0
    for source in inventory:
        name = Path(source['file']).stem
        folder = OUT / name
        report_path = folder / 'after-public-cal-choice.json'
        report = read(report_path)
        before = read(BASE / name / 'before-three-models-v2.json')
        rows = report['frames']
        count = source['decoded_frames']
        checks.check(sha(Path(source['file'])) == source['sha256'] == report['source']['sha256'], name + ': original file SHA')
        checks.check(report['plan_sha256'] == sha(PLAN) and report['baseline_report_sha256'] == sha(BASE / name / 'before-three-models-v2.json'), name + ': plan/baseline SHA')
        checks.check(report['model']['sha256'] == plan['model']['sha256'] and report['selected_step'] == plan['selected_step'], name + ': globally frozen model')
        config = report['model']['actual_execution_config']
        checks.check(report['model']['engine'] == 'onnxruntime' and report['model']['device'] == 'CPU'
                     and config.get('providers') == ['CPUExecutionProvider']
                     and config.get('intra_op_num_threads') == 1 and config.get('inter_op_num_threads') == 1,
                     name + ': actual CPUORT provider/threads')
        checks.check(report['candidate_passed_public_cal_guards'] == plan['candidate_passed_public_cal_guards'], name + ': honest qualification flag')
        checks.check(report['independent_accuracy_credit'] is False and len(rows) == count == report['source']['original_camera_frames'], name + ': unscored complete count')
        for key in ['annotated_frames', 'localized_matches', 'missed_defects', 'false_alerts', 'unsupported_annotations']:
            checks.check(report['summary'][key] is None, name + ': unknown ' + key)
        cap = cv2.VideoCapture(source['file'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if not cap.isOpened():
            raise ValueError('Original private file cannot be decoded')
        counts, present, empty, mask_total = Counter(), Counter(), 0, 0
        try:
            for i, row in enumerate(rows):
                context = f'{name}/{i}'
                ok, raw = cap.read()
                if not ok:
                    checks.check(False, context + ': missing source exposure')
                    break
                old, stamp = before['frames'][i], source['all_frame_times'][i]
                raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                gained, gain = camera_exposure(raw, 'low_light_gain')
                checks.check(raw_hash == row['raw_native_bgr_sha256'] == old['raw_native_bgr_sha256'], context + ': native RAW bytes')
                checks.check(row['frame_size_wh'] == list(raw.shape[1::-1]), context + ': native orientation/shape')
                checks.check(row['model_input_native_bgr_sha256'] == hashlib.sha256(gained.tobytes()).hexdigest()
                             and row['input_exposure_gain'] == gain == old['models']['author_masks']['input_exposure_gain'], context + ': exact gained native input')
                checks.check(row['source_frame_index'] == i and row['source_pts'] == stamp['pts'] and row['source_time_base'] == stamp['time_base'], context + ': original index/PTS')
                checks.check(abs(row['video_time_s'] - float(fraction(stamp) - fraction(source['all_frame_times'][0]))) <= 1e-9, context + ': relative timestamp')
                checks.check(row['evaluated'] is False and row['expected'] == [], context + ': no invented GT')
                checks.check(row['before_cached_predictions'] == old['models']['author_masks']['predictions']
                             and row['before_mask_file'] == old['actual_masks'], context + ': cached BEFORE preserved')
                mask_path = protected_path(row['actual_masks']['path'], folder)
                checks.check(sha(mask_path) == row['actual_masks']['sha256'], context + ': native NPZ SHA')
                checks.check(read(mask_path.with_suffix('.json')) == row, context + ': frame receipt')
                expected_keys = set()
                with np.load(mask_path, allow_pickle=False) as arrays:
                    for pred in row['predictions']:
                        ref = pred['actual_mask']
                        expected_keys.add(ref['key'])
                        packed = arrays[ref['key']]
                        pixels = raw.shape[0] * raw.shape[1]
                        checks.check(packed.dtype == np.uint8 and packed.ndim == 1 and packed.size == (pixels + 7) // 8, context + ': packed shape/type')
                        checks.check(ref['shape_hw'] == list(raw.shape[:2]) and hashlib.sha256(packed.tobytes()).hexdigest() == ref['packed_sha256'], context + ': native shape/mask SHA')
                        bits = np.unpackbits(packed)
                        checks.check(not bits[pixels:].any() and int(bits[:pixels].sum()) == ref['positive_pixels'], context + ': exact foreground/padding')
                        checks.check(pred['label'] in ['crack', 'spalling'] and type(pred['score']) in (float, int) and math.isfinite(pred['score'])
                                     and 0 <= pred['score'] <= 1 and pred['evaluation_status'] == 'unscored'
                                     and not pred.get('matched_annotation_id'), context + ': unscored actual candidate')
                        mask_total += 1
                        empty += int(ref['positive_pixels'] == 0)
                    checks.check(set(arrays.files) == expected_keys, context + ': exact mask keys')
                counts.update(p['label'] for p in row['predictions'])
                present.update(set(p['label'] for p in row['predictions']))
                present['any'] += int(bool(row['predictions']))
            checks.check(not cap.read()[0], name + ': clean original EOF')
        finally:
            cap.release()
        checks.check(dict(counts) == report['summary']['unscored_prediction_counts']
                     and dict(present) == report['summary']['prediction_present_frame_counts'], name + ': recomputed activity counts')
        derived = protected_path(report['derived_video']['path'], folder)
        times, meta = encoded_timeline(derived)
        origin = fraction(source['all_frame_times'][0])
        checks.check(sha(derived) == report['derived_video']['sha256'] and len(times) == count
                     and meta['width'] == 1400 and meta['height'] == 1240, name + ': whole encoded video/hash/geometry')
        checks.check([fraction(t) for t in times] == [fraction(t) - origin for t in source['all_frame_times']], name + ': every rational original PTS retained')
        videos.append({'source': name, 'frames': count, 'masks': mask_total, 'empty_native_masks': empty,
                       'unscored_prediction_counts': dict(counts), 'prediction_present_frames': dict(present),
                       'report_sha256': sha(report_path), 'video_sha256': sha(derived)})
        all_frames += count
        all_masks += mask_total
    checks.check(all_frames == 878 and len(videos) == 4, 'Every four original videos/all878 frames')
    result = {'scope': 'Independent file/native-pixel/timestamp integrity only; no NN, private GT, accuracy claim or upload.',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(PLAN),
              'source_sha256': sha(Path(__file__)), 'checks': checks.total, 'all_checks_passed': not checks.failures,
              'failures': checks.failures, 'original_frames': all_frames, 'native_masks': all_masks, 'videos': videos}
    with TARGET.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    if checks.failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
