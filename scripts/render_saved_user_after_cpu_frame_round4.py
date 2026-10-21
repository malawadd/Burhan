"""Render one saved original exposure and actual cached masks; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from evaluate_user_construction_after_round4 import cached_author
from evaluate_user_construction_videos_round4 import fit_panel
from edge_runtime.profiles.construction_preprocessing import camera_exposure

BASE = ROOT / 'output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--frame', required=True, type=int)
    args = parser.parse_args()
    folder = (BASE / args.stem).resolve()
    if not folder.is_relative_to(BASE.resolve()):
        raise ValueError('Source outside local evidence')
    report = json.loads((folder / 'after-public-cal-choice.json').read_text())
    if not 0 <= args.frame < len(report['frames']):
        raise ValueError('Frame must be an existing nonnegative source index')
    row = report['frames'][args.frame]
    cap = cv2.VideoCapture(report['source']['original_path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        for _ in range(args.frame + 1):
            ok, raw = cap.read()
            if not ok:
                raise ValueError('Original exposure missing')
    finally:
        cap.release()
    if hashlib.sha256(raw.tobytes()).hexdigest() != row['raw_native_bgr_sha256']:
        raise ValueError('Native original pixels changed')
    old = {'models': {'author_masks': {'predictions': row['before_cached_predictions']}},
           'actual_masks': row['before_mask_file']}
    before = cached_author(old, raw.shape[:2])
    mask_path = ROOT / row['actual_masks']['path']
    if not mask_path.resolve().is_relative_to(folder):
        raise ValueError('Saved actual masks escape this video folder')
    if hashlib.sha256(mask_path.read_bytes()).hexdigest() != row['actual_masks']['sha256']:
        raise ValueError('Saved actual AFTER masks changed')
    after = []
    with np.load(mask_path, allow_pickle=False) as data:
        for pred in row['predictions']:
            ref = pred['actual_mask']
            packed = data[ref['key']]
            if hashlib.sha256(packed.tobytes()).hexdigest() != ref['packed_sha256'] or ref['shape_hw'] != list(raw.shape[:2]):
                raise ValueError('Native AFTER mask changed')
            mask = np.unpackbits(packed)[:raw.shape[0]*raw.shape[1]].reshape(raw.shape[:2]).astype(bool)
            if int(mask.sum()) != ref['positive_pixels']:
                raise ValueError('Saved actual foreground differs')
            after.append(dict(pred, mask=mask))
    gained, gain = camera_exposure(raw, 'low_light_gain')
    if gain != row['input_exposure_gain'] or hashlib.sha256(gained.tobytes()).hexdigest() != row['model_input_native_bgr_sha256']:
        raise ValueError('Shared actual model input differs')
    canvas = np.full((1240, 1400, 3), 24, np.uint8)
    canvas[50:620, :700] = fit_panel(raw, 'RAW - unchanged camera exposure', None)
    canvas[50:620, 700:] = fit_panel(gained, 'BEFORE - original author / cached GPU masks', before)
    canvas[620:1190, :700] = fit_panel(gained, 'AFTER - trained model / actual CPU masks', after)
    lines = [f'{args.stem} / source frame {args.frame} / {row["video_time_s"]:.3f}s',
             report['candidate_role'], f'BEFORE {len(before)} candidates; AFTER {len(after)} candidates',
             'UNSCORED: no complete private ground truth.', 'CPU + training changed; not training-only attribution.',
             'No measured wall humidity or fastener adequacy detector.', 'Native masks are real saved model output; no GT overlay.']
    cv2.putText(canvas, 'PRIVATE VIDEO / SAVED NATIVE OUTPUT / NO NEW INFERENCE', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (240,240,240), 1)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line[:78], (712, 665 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, .42, (230,230,230), 1)
    qa = folder / 'qa'
    qa.mkdir(exist_ok=True)
    target = qa / f'after-source-frame-{args.frame:05d}.png'
    if target.exists():
        raise ValueError('Preserve original diagnostic rendering')
    if not cv2.imwrite(str(target), canvas):
        raise IOError('Cannot save actual evidence')
    print(target)


if __name__ == '__main__':
    main()
