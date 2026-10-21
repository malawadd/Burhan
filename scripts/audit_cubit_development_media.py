"""Independent saved-evidence checks; no model imports or inference."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np

from audit_construction_recording_counts import audit_document

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/construction-cubit-author-v2/videos'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = OUT / 'root-independent-five-video-audit.json'
    if target.exists():
        raise ValueError('Preserve the first audit')
    entries = json.loads((OUT / 'five-complete-video-summary.json').read_text())
    assert [v['sequence'] for v in entries] == [
        'crack_89', 'crack_129', 'spalling_75', 'ros_crack_21', 'spalling_76']
    results = []
    for entry in entries:
        report_path, video_path = ROOT / entry['report'], ROOT / entry['video']
        report = json.loads(report_path.read_text())
        assert not report['independent_accuracy_credit']
        assert report['candidate_status'] == 'FAILED CALIBRATION'
        assert sha(video_path) == report['derived_video']['sha256']
        counts = audit_document(report)
        assert counts['passed'], counts['errors']
        before = report['paired_before']
        assert sha(ROOT / before['report_path']) == before['report_sha256']
        old = json.loads((ROOT / before['report_path']).read_text())
        source_files = report['source']['source_files']
        for name, digest in source_files.items():
            assert sha(ROOT / name) == digest
        h5_sources = [ROOT / name for name in source_files if name.endswith('.h5')]
        if h5_sources:
            assert len(h5_sources) == 1
            with h5py.File(h5_sources[0]) as h5:
                frames = h5['frames'][:]
                timestamps = h5['frame_timestamp'][:]
        else:
            # Original crack89 acquisition retains publisher camera arrays.
            frames = np.load(next(ROOT / name for name in source_files if name.endswith('frames.npy')), allow_pickle=False)
            timestamps = np.load(next(ROOT / name for name in source_files if name.endswith('timestamps-us.npy')), allow_pickle=False)
        rows = report['frames']
        assert len(frames) == len(rows) == len(old['frames'])
        mask_count = empty_count = empty_hits = 0
        for index, (gray, row, baseline) in enumerate(zip(frames, rows, old['frames'])):
            raw = np.repeat(gray[:, :, None], 3, axis=2)
            assert row['source_frame_index'] == index
            assert row['source_timestamp_us'] == int(timestamps[index])
            assert round(row['video_time_s'] * 1e6) == int(timestamps[index]) - int(timestamps[0])
            p99 = float(np.percentile(gray, 99))
            gain = min(8., 128. / max(p99, 1.)) if p99 < 64 else 1.
            native = (np.clip(raw.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                      if gain != 1 else raw)
            assert row['input_exposure_gain'] == gain
            assert row['raw_native_bgr_sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
            assert row['model_input_native_bgr_sha256'] == hashlib.sha256(native.tobytes()).hexdigest()
            for key in ['source_frame_index', 'video_time_s', 'expected',
                        'raw_native_bgr_sha256', 'model_input_native_bgr_sha256']:
                assert row[key] == baseline[key]
            mask_path = ROOT / row['actual_masks']['path']
            assert sha(mask_path) == row['actual_masks']['sha256']
            with np.load(mask_path, allow_pickle=False) as saved:
                assert len(saved.files) == len(row['predictions'])
                for prediction in row['predictions']:
                    ref = prediction['mask_reference']
                    packed = saved[ref['npz_key']]
                    assert list(ref['shape_hw']) == list(gray.shape)
                    assert packed.dtype == np.uint8
                    assert hashlib.sha256(packed.tobytes()).hexdigest() == ref['packed_mask_sha256']
                    mask = np.unpackbits(packed)[:gray.size].reshape(gray.shape)
                    assert prediction['native_mask_positive_pixels'] == int(mask.sum())
                    empty = not bool(mask.any())
                    assert prediction['native_mask_empty'] == empty
                    mask_count += 1
                    empty_count += empty
                    empty_hits += empty and bool(prediction.get('matched_annotation_id'))
        mask_audit = report['native_mask_audit']
        assert mask_count == mask_audit['predictions']
        assert empty_count == mask_audit['empty_masks']
        assert empty_hits == mask_audit['bbox_matches_with_empty_masks']
        replay = ROOT / report['source']['lossless_replay_input']['path']
        for kind, path in [('lossless', replay), ('annotated', video_path)]:
            count = 0
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    assert Fraction(frame.pts) * frame.time_base * 1_000_000 == int(timestamps[index]) - int(timestamps[0])
                    if kind == 'lossless':
                        assert np.array_equal(frame.to_ndarray(format='gray'), frames[index])
                    count += 1
            assert count == len(rows)
        results.append({'sequence': entry['sequence'], 'frames': len(rows),
                        'report_sha256': sha(report_path), 'video_sha256': sha(video_path),
                        'counts': counts, 'actual_masks': mask_count, 'empty_masks': empty_count})
    value = {'all_checks_passed': True, 'actual_NN_calls': 0, 'results': results,
             'all_frame_records': sum(r['frames'] for r in results),
             'audit_source_sha256': sha(Path(__file__)),
             'scope': 'Independent raw/replay pixels, native gain and hashes, cached-before source/GT equality, '
                      'actual saved mask hashes/geometry/counts, report box matching, full video frame counts/PTS. '
                      'No independent rerun of predictions, semantic GT validation or video pixel-GT accuracy.'}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
