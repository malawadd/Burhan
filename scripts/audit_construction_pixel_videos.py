"""Independent saved-count, original-input, native-mask and encoded-PTS audit.

No model or inference framework is loaded. Run only after the coordinator's
project timing hold releases: media decoding/hash checks do consume CPU and IO.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/evaluation/construction-pixel-videos'
NAMES = ['crack_89', 'crack_129', 'ros_crack_21']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_arrays(source):
    import numpy as np
    files = source['files']
    for name, digest in files.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen original source file changed')
    if source['group'] == 'exposed':
        frames_path = next(name for name in files if Path(name).name == 'frames.npy')
        times_path = next(name for name in files if Path(name).name == 'timestamps-us.npy')
        return np.load(ROOT / frames_path, allow_pickle=False), np.load(ROOT / times_path, allow_pickle=False)
    import h5py
    path = next(name for name in files if name.endswith('_frames.h5'))
    with h5py.File(ROOT / path, 'r') as handle:
        return handle['frames'][:], handle['frame_timestamp'][:].astype(np.int64)


def audit_masks(document, source, original, times):
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    evidence = document['native_mask_evidence']
    path = ROOT / evidence['path']
    if sha(path) != evidence['sha256']:
        raise ValueError('Saved native mask asset SHA differs')
    records = document['frames']
    if len(records) != len(original) or len(times) != len(original):
        raise ValueError('Every original source exposure must have one record/mask')
    raw_total = served_total = 0
    with np.load(path, allow_pickle=False) as saved:
        shape = tuple(int(value) for value in saved['native_shape_hw'])
        if shape != original.shape[1:]:
            raise ValueError('Native mask geometry differs from original camera')
        if saved['raw_packed'].shape != saved['served_packed'].shape or saved['raw_packed'].shape[0] != len(records):
            raise ValueError('Raw/served packed mask frame counts differ')
        if saved['source_frame_indices'].tolist() != list(range(len(records))) or not np.array_equal(saved['source_timestamps_us'], times):
            raise ValueError('Mask source indices/PTS differ from original camera')
        for index, (record, gray, timestamp) in enumerate(zip(records, original, times)):
            raw_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            gained, _ = camera_exposure(raw_bgr, 'low_light_gain')
            if record['source_frame_index'] != index or record['source_timestamp_us'] != int(timestamp):
                raise ValueError('Recorded source index/PTS differs from original camera')
            if hashlib.sha256(raw_bgr.tobytes()).hexdigest() != record['raw_native_bgr_sha256'] or hashlib.sha256(gained.tobytes()).hexdigest() != record['model_input_native_bgr_sha256']:
                raise ValueError('Recorded raw/gain input hash differs from original camera')
            masks = []
            for kind in ['raw', 'served']:
                unpacked = np.unpackbits(saved[kind + '_packed'][index], bitorder='little')
                cells = shape[0] * shape[1]
                if unpacked[cells:].any():
                    raise ValueError('Nonzero bit-pack padding outside native image')
                mask = unpacked[:cells].reshape(shape).astype(bool)
                if hashlib.sha256(mask.tobytes()).hexdigest() != record[kind + '_native_mask_sha256'] or int(mask.sum()) != record[kind + '_native_positive_pixels']:
                    raise ValueError('Native mask hash/positive-pixel count differs from saved record')
                masks.append(mask)
            raw, served = masks
            if np.any(served & ~raw):
                raise ValueError('Served mask contains pixels outside raw strict-threshold mask')
            count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
            expected_boxes = []
            for label in range(1, count):
                in_component = labels == label
                kept = int(np.count_nonzero(served & in_component))
                if kept and (kept != int(stats[label, cv2.CC_STAT_AREA]) or kept < 4):
                    raise ValueError('Served output includes a partial raw component or an area<4 component')
                if kept:
                    expected_boxes.append(tuple(int(v) for v in stats[label, :4]))
            predictions = record['predictions']
            if len(expected_boxes) > 100 or any(p['label'] != 'crack' for p in predictions) or sorted(tuple(p['bbox']) for p in predictions) != sorted(expected_boxes):
                raise ValueError('Actual emitted boxes do not equal retained native mask components')
            raw_total += int(raw.sum()); served_total += int(served.sum())
    return {'frames': len(records), 'raw_native_positive_pixels_total': raw_total,
            'served_native_positive_pixels_total': served_total, 'all_input_and_mask_hashes_match': True,
            'all_served_components_are_complete_raw_components_area_at_least4': True,
            'all_emitted_boxes_equal_served_mask_components': True,
            'limitation': 'Raw probability384 arrays are not in the video NPZ; this audit checks their recorded SHA format but cannot recompute thresholding, component mean scores or top100 rank from probabilities. It does not rerun NN or grant pixel-accuracy credit.'}


def main():
    import av
    import cv2
    cv2.setNumThreads(1)
    from audit_construction_recording_counts import audit_document
    protocol_path = OUT / 'fixed-pair-three-video-protocol-before-inference.json'
    protocol = json.loads(protocol_path.read_text())
    paired = json.loads((OUT / 'paired-summary.json').read_text())
    if protocol['ordered_sequences'] != NAMES or list(paired) != ['BEFORE', 'AFTER']:
        raise ValueError('Fixed pair/three-source order differs')
    result = {'scope': 'Independent saved JSON counts, original raw/gain frame hashes, packed native masks and actual encoded PTS; no NN and no video pixel-GT accuracy claim.',
              'protocol_sha256': sha(protocol_path), 'reports': [], 'passed': True}
    for tag in ['BEFORE', 'AFTER']:
        if [row['sequence'] for row in paired[tag]] != NAMES:
            raise ValueError('Every declared source must appear once per fixed model')
        for row in paired[tag]:
            path = ROOT / row['report']; report = json.loads(path.read_text())
            checked = {'tag': tag, 'sequence': row['sequence'], 'report': str(path.relative_to(ROOT)), 'report_sha256': sha(path)}
            try:
                if sha(path) != row['report_sha256'] or report['model']['sha256'] != row['model_sha256'] or report['selection_protocol_sha256'] != sha(protocol_path):
                    raise ValueError('Saved report/model/protocol identity differs')
                if report['fixed_probability_threshold'] != .5 or report['candidate_status'] != 'FIXED .50 DEVELOPMENT DIAGNOSTIC - NOT QUALIFIED':
                    raise ValueError('Fixed threshold/failed-diagnostic scope differs')
                counts = audit_document(report)
                if not counts['passed']:
                    raise ValueError('Independent count/geometry errors: ' + repr(counts['errors']))
                source = next(s for s in protocol['sources'] if s['sequence'] == row['sequence'])
                original, times = original_arrays(source)
                checked['counts'] = counts['recomputed']
                checked['masks'] = audit_masks(report, source, original, times)
                video = ROOT / row['video']
                if sha(video) != row['video_sha256'] or sha(video) != report['derived_video']['sha256']:
                    raise ValueError('Derived video byte SHA differs')
                with av.open(str(video)) as container:
                    container.streams.video[0].codec_context.thread_count = 1
                    actual = [Fraction(frame.pts) * frame.time_base * 1_000_000 for frame in container.decode(video=0)]
                expected = [int(t) - int(times[0]) for t in times]
                if actual != expected or len(actual) != report['derived_video']['encoded_frames']:
                    raise ValueError('Actual encoded frames/PTS differ from complete original source')
                checked.update(passed=True, encoded_frames=len(actual), actual_encoded_pts_exact=True)
            except Exception as error:
                checked.update(passed=False, error=repr(error)); result['passed'] = False
            result['reports'].append(checked)
    target = OUT / 'independent-pixel-video-audit.json'
    if target.exists():
        raise ValueError('Preserve prior audit receipt; no silent overwrite')
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'reports': len(result['reports']), 'receipt_sha256': sha(target)}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
