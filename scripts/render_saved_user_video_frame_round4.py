"""Render a saved native model output; zero inference and no fabricated mask."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_user_construction_videos_round4 import ROOT, OUT, fit_panel
from edge_runtime.profiles.construction_preprocessing import camera_exposure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--frame', type=int, default=0)
    args = parser.parse_args()
    inventory = json.loads((OUT / 'original-video-inventory.json').read_text())
    source = next(row for row in inventory if Path(row['file']).stem == args.stem)
    folder = OUT / args.stem
    record = json.loads((folder / 'baseline-v2-masks' / f'frame_{args.frame:05d}.json').read_text())
    cap = cv2.VideoCapture(source['file'])
    for index in range(args.frame + 1):
        ok, raw = cap.read()
        if not ok:
            raise ValueError('Original frame unavailable')
    cap.release()
    assert record['raw_native_bgr_sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
    masks = ROOT / record['actual_masks']['path']
    assert hashlib.sha256(masks.read_bytes()).hexdigest() == record['actual_masks']['sha256']
    rows = {}
    with np.load(masks, allow_pickle=False) as data:
        for name, model in record['models'].items():
            rows[name] = []
            for prediction in model['predictions']:
                item = dict(prediction)
                if 'actual_mask' in item:
                    ref = item['actual_mask']
                    packed = data[ref['key']]
                    assert hashlib.sha256(packed.tobytes()).hexdigest() == ref['packed_sha256']
                    item['mask'] = np.unpackbits(packed)[:np.prod(ref['shape_hw'])].reshape(ref['shape_hw']).astype(bool)
                    assert int(item['mask'].sum()) == ref['positive_pixels']
                rows[name].append(item)
    canvas = np.full((1240, 1400, 3), 24, np.uint8)
    cv2.putText(canvas, f'ACTUAL SAVED BEFORE OUTPUT / {args.stem} / frame {args.frame} / {record["video_time_s"]:.3f}s', (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1)
    gained, _ = camera_exposure(raw, 'low_light_gain')
    canvas[50:620, :700] = fit_panel(raw, 'RAW - original local camera frame', None)
    canvas[50:620, 700:] = fit_panel(gained, 'AUTHOR: actual crack/material-loss masks', rows['author_masks'])
    canvas[620:1190, :700] = fit_panel(raw, 'WALL: appearance BOXES only, failed research', rows['wall_appearance'])
    canvas[620:1190, 700:] = fit_panel(raw, 'BROAD: actual damage-appearance MASKS', rows['broad_masks'])
    cv2.putText(canvas, 'Unscored model candidates. No pixel GT, moisture measurement, fixing-capacity or code-acceptance determination.', (8, 1220), cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 225, 225), 1)
    qa = folder / 'qa'
    qa.mkdir(exist_ok=True)
    target = qa / f'before-source-frame-{args.frame:05d}.png'
    if target.exists():
        raise ValueError('Preserve original saved-output rendering')
    assert cv2.imwrite(str(target), canvas)
    print(target)


if __name__ == '__main__':
    main()
