"""Read-only independent source, timeline and box-count audit; never loads a model."""
import hashlib
import json
from pathlib import Path

import av
import h5py
import numpy as np

from audit_construction_recording_counts import audit_document

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'output/evaluation/construction-posthoc-known-hit-videos'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = FOLDER / 'root-independent-media-audit.json'
    if target.exists():
        raise ValueError('Preserve the first independent audit')
    paired = json.loads((FOLDER / 'complete-posthoc-paired-summary.json').read_text())
    results = []
    common_sources = {}
    for tag in ['BEFORE', 'AFTER']:
        assert [v['sequence'] for v in paired[tag]] == ['crack_129', 'spalling_76']
        for entry in paired[tag]:
            report_path, video = ROOT / entry['report'], ROOT / entry['video']
            assert sha(report_path) == entry['report_sha256']
            assert sha(video) == entry['video_sha256']
            report = json.loads(report_path.read_text())
            assert not report['independent_accuracy_credit']
            assert report['posthoc_known_hit_selection']
            counts = audit_document(report)
            assert counts['passed'], counts['errors']
            sources = report['source']['source_files']
            for name, digest in sources.items():
                assert sha(ROOT / name) == digest
            h5_path = next(ROOT / name for name in sources if name.endswith('.h5'))
            frames = report['frames']
            with h5py.File(h5_path) as h5:
                raw = h5['frames'][:]
                timestamps = h5['frame_timestamp'][:]
            assert len(raw) == len(frames) == len(timestamps)
            native_pts = (timestamps - timestamps[0]).astype(np.int64)
            source_identity = []
            for index, (gray, timestamp, row) in enumerate(zip(raw, timestamps, frames)):
                bgr = np.repeat(gray[:, :, None], 3, axis=2)
                assert row['source_frame_index'] == index
                assert row['source_timestamp_us'] == int(timestamp)
                assert round(row['video_time_s'] * 1_000_000) == int(native_pts[index])
                assert hashlib.sha256(bgr.tobytes()).hexdigest() == row['raw_native_bgr_sha256']
                # Independent transcription of the full-frame scalar exposure policy.
                p99 = float(np.percentile(gray, 99))
                gain = min(8.0, 128.0 / max(p99, 1.0)) if p99 < 64 else 1.0
                actual_input = (np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                                if gain != 1 else bgr)
                assert row['input_exposure_gain'] == gain
                assert hashlib.sha256(actual_input.tobytes()).hexdigest() == row['model_input_native_bgr_sha256']
                source_identity.append({key: row[key] for key in [
                    'source_frame_index', 'source_timestamp_us', 'video_time_s', 'expected',
                    'raw_native_bgr_sha256', 'model_input_native_bgr_sha256']})
            if tag == 'BEFORE':
                common_sources[entry['sequence']] = source_identity
            else:
                assert common_sources[entry['sequence']] == source_identity
            for kind, path in [('original_lossless', ROOT / report['source']['lossless_replay_input']['path']),
                               ('annotated_mp4', video)]:
                times = []
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = 'NONE'
                    stream.codec_context.thread_count = 1
                    for index, decoded in enumerate(container.decode(stream)):
                        times.append(round(float(decoded.pts * decoded.time_base) * 1_000_000))
                        if kind == 'original_lossless':
                            assert np.array_equal(decoded.to_ndarray(format='gray'), raw[index])
                assert times == native_pts.tolist(), kind
            results.append({'tag': tag, 'sequence': entry['sequence'], 'report_sha256': sha(report_path),
                            'video_sha256': sha(video), 'all_frames': len(frames),
                            'counts': counts, 'source_and_gain_hashes_passed': True,
                            'lossless_pixels_and_both_container_timelines_passed': True})
    result = {'all_checks_passed': True, 'actual_NN_calls': 0,
              'all_frame_records': sum(v['all_frames'] for v in results), 'results': results,
              'audit_source_sha256': sha(Path(__file__)),
              'scope': 'Independent raw-H5 and lossless pixels, native-input gain/hashes, full MP4 PTS/counts, '
                       'before/after source/GT equality and independently recomputed report box IoU/counts. '
                       'No rerun of predictions or independent semantic annotation/pixel-GT validation; '
                       'post-hoc CAL illustrations, no independent accuracy credit.'}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}, indent=2))


if __name__ == '__main__':
    main()
