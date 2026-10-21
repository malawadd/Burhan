"""Local, deterministic inventory and camera-frame contact sheets; no inference."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'videos'
OUT = ROOT / 'output/evaluation/user-construction-videos-round4'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'original-video-inventory.json'
    if target.exists():
        raise ValueError('Preserve the original inventory')
    inventory = []
    for path in sorted(SOURCE.glob('*.mp4')):
        folder = OUT / path.stem
        folder.mkdir()
        times, samples, rotation = [], [], None
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            declared_duration = float(stream.duration * stream.time_base) if stream.duration else None
            targets = np.linspace(0, max(0, (declared_duration or 5) - .10), 12)
            wanted = 0
            for index, frame in enumerate(container.decode(video=0)):
                seconds = float(Fraction(frame.pts) * frame.time_base)
                times.append({'index': index, 'pts': frame.pts, 'time_base': str(frame.time_base), 'seconds': seconds})
                if rotation is None:
                    rotation = int(getattr(frame, 'rotation', 0))
                if wanted < len(targets) and seconds >= targets[wanted]:
                    native = frame.to_ndarray(format='bgr24')
                    # Display orientation only; original source remains untouched.
                    display = native
                    if rotation % 360 == 90:
                        display = cv2.rotate(native, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation % 360 == 270:
                        display = cv2.rotate(native, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation % 360 == 180:
                        display = cv2.rotate(native, cv2.ROTATE_180)
                    name = f'source-frame-{index:05d}.jpg'
                    if not cv2.imwrite(str(folder / name), display):
                        raise ValueError('Could not save original frame display')
                    samples.append({'source_frame_index': index, 'source_time_s': seconds, 'path': (folder / name).relative_to(ROOT).as_posix(), 'native_bgr_sha256': hashlib.sha256(native.tobytes()).hexdigest(), 'image': display})
                    wanted += 1
            meta = {'codec': stream.codec_context.name, 'native_width': stream.width, 'native_height': stream.height,
                    'declared_average_rate': str(stream.average_rate), 'declared_duration_s': declared_duration,
                    'rotation_degrees': rotation, 'stream_metadata': dict(stream.metadata)}
        sheet = np.full((4 * 280, 3 * 400, 3), 25, np.uint8)
        for i, sample in enumerate(samples):
            image = sample.pop('image')
            scale = min(396 / image.shape[1], 246 / image.shape[0])
            thumb = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
            y, x = (i // 3) * 280 + 30, (i % 3) * 400
            sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
            cv2.putText(sheet, f'frame {sample["source_frame_index"]} / {sample["source_time_s"]:.3f}s', (x + 4, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
        cv2.imwrite(str(folder / 'original-frame-contact-sheet.jpg'), sheet)
        row = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
               'decoded_frames': len(times), 'first_time_s': times[0]['seconds'], 'last_time_s': times[-1]['seconds'],
               **meta, 'samples': samples, 'all_frame_times': times,
               'role': 'User-provided diagnostic only; no training/calibration, no upload, no ground-truth accuracy claim.'}
        (folder / 'source-inventory.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        inventory.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in {'samples', 'all_frame_times'}}, indent=2), flush=True)
    target.write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
