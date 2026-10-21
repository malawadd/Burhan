"""Fixed TUT versus FINAL300 semantic traces on three complete real videos.

No inference occurs on import or --freeze-only. Whole-video inference requires
all three pixel calibrations, numerical/serving parity and explicit GPU release.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_construction_development_videos import selected_sources, load_cases, verify_replay
from evaluate_construction_round3 import score_frame, summarize
from edge_runtime.profiles.construction_lite import LiteCrackConstructionModel, decode_crack_map
from edge_runtime.profiles.construction_preprocessing import camera_exposure

OUT = ROOT / 'output/evaluation/construction-pixel-videos'
PROTOCOL = OUT / 'fixed-pair-three-video-protocol-before-inference.json'
NAMES = ['crack_89', 'crack_129', 'ros_crack_21']
MODELS = {
    'BEFORE': ('models/construction/thin-contingency/litecrackseg-tut-384.onnx',
               '20936464f5fb6ef32cf5d84a87c84616bf1d5461364fbfb466dd7c4846d04055'),
    'AFTER': ('models/construction/pixel-supervision-v2/crack-pixel-step300.onnx',
              'b7cad346d11cb89be6a76fff38ab26fa3e8fd9778981af47498e6f8a99c6d3d7'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    rows = selected_sources()
    sources = [next(r for r in rows if r['sequence'] == name) for name in NAMES]
    prior = ROOT / 'output/evaluation/construction-round3/four-development-video-comparison-protocol.json'
    if sha(prior) != '0006e730fc4951044d395a7921796b1c226884d574ae10ed4d8a09604d19ee7b':
        raise ValueError('Original development source protocol differs')
    original = json.loads(prior.read_text(encoding='utf-8'))
    original_sources = [next(r for r in original['sources'] if r['sequence'] == name) for name in NAMES]
    if sources != original_sources or sha(ROOT / 'datasets/construction/evcivil-round3/protocol.json') != original['source_protocol_sha256']:
        raise ValueError('Actual source files/roles must match the earlier immutable selection, not a new first-freeze inventory')
    for path, digest in MODELS.values():
        if sha(ROOT / path) != digest:
            raise ValueError('Fixed original/final300 model changed')
    value = {
        'schema_version': 1, 'sources': sources, 'ordered_sequences': NAMES,
        'prior_source_protocol_sha256': sha(prior),
        'fixed_models': {tag: {'path': path, 'sha256': digest} for tag, (path, digest) in MODELS.items()},
        'threshold': .5, 'operator': 'strictly_greater_than', 'minimum_component_area': 4,
        'selection': 'Fixed original TUT versus FINAL300, never CAL-winner substitution. Same three earlier development crack sources, no favorable replacement.',
        'input': 'Original camera frame; identical whole-native low_light_gain once,384 linear resize, RGBfloat0..255 in graph; publisher annotations never enter model input.',
        'primary_video_score': 'Unchanged class-aware one-to-one publisher box IoU>=.50 using installed area4/top100 component decoder; all exposures/PTS, misses and unmatched retained. Unlabelled exposures unscored.',
        'semantic_display': 'All native raw sigmoid>.50 pixels yellow; actual area4/top100 served mask blue. These colors do not assert pixel true positives; no publisher video pixel masks exist.',
        'evidence': 'Full original source timelines, separate RAW/native model input/externalGT/model trace panes. Bit-packed raw and served native masks with every source ID/PTS and per-frame hashes; no morphology.',
        'scope': 'Reused/exposed/calibration development only, no independent accuracy credit, no physical width or code acceptance. Crack-only; spalling and other conditions unsupported.',
        'execution_guard': 'All100/200/300 actual calibration reports, global freeze even if rejected, fitted numeric/serving parity and fixed final300 lineage must pass the binding helper before video NN.',
        'speed': 'Contended offline accuracy/export, never real-time qualification; preserve source presentation timestamps rather than inference wallclock.',
        'export_source_sha256': sha(Path(__file__)),
        'dependency_source_sha256': {path: sha(ROOT / path) for path in [
            'scripts/compare_construction_development_videos.py', 'scripts/construction_annotations.py',
            'scripts/evaluate_construction_round3.py', 'edge_runtime/profiles/construction_lite.py',
            'edge_runtime/profiles/construction_preprocessing.py']},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text(encoding='utf-8')) != value:
            raise ValueError('Refusing to overwrite changed fixed video protocol')
    else:
        PROTOCOL.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return value


def render(raw, native, raw_mask, served_mask, record, name, tag):
    canvas = np.full((1260, 1420, 3), (24, 27, 30), np.uint8)
    gt, trace = raw.copy(), raw.copy()
    matched = {p.get('matched_annotation_id') for p in record['predictions']}
    for item in record['expected']:
        x, y, w, h = map(round, item['bbox'])
        cv2.rectangle(gt, (x, y), (x + w, y + h), (245, 210, 30) if item['id'] in matched else (70, 70, 245), 1)
    trace[raw_mask] = (.3 * trace[raw_mask] + .7 * np.array([0, 210, 245])).astype(np.uint8)
    trace[served_mask] = (.3 * trace[served_mask] + .7 * np.array([245, 160, 30])).astype(np.uint8)
    panes = [(raw, 'RAW CAMERA - unchanged'), (native, f'ACTUAL NATIVE INPUT - gain {record["input_exposure_gain"]:.2f}x'),
             (gt, 'EXTERNAL PUBLISHER BOXES - not pixel GT'), (trace, 'ACTUAL MODEL TRACE - not GT colors')]
    for i, (array, title) in enumerate(panes):
        x, y = 8 + (i % 2) * 708, 95 + (i // 2) * 560
        canvas[y:y + 520, x:x + 692] = cv2.resize(array, (692, 520), interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas, title, (x, y - 13), cv2.FONT_HERSHEY_SIMPLEX, .53, (235, 235, 235), 1)
    lines = [f'REAL CRACK VIDEO / {tag} / {name} / FIXED .50 - NOT QUALIFIED',
             f'Source frame {record["source_frame_index"]} at {record["video_time_s"]:.6f}s; complete development recording']
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (8, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, .56, (235, 235, 235), 1)
    counts = (f'Box matches {record["localized_matches"]} / missed {record["missed_defects"]} / unmatched {record["false_alerts"]}'
              if record['evaluated'] else 'UNSCORED exposure: absence of publisher labels is not verified absence of cracks')
    footer = [counts, 'YELLOW: raw sigmoid>.50. BLUE: installed area4/top100 mask. Trace colors do not assert pixel correctness.',
              'Counts use publisher BOX IoU>=.50, not pixel GT. No spalling support, physical crack width or code acceptance.',
              'Gamage et al., ev-CIVIL (DTU), CC BY4.0. Every actual camera exposure/PTS retained; no generated frames.']
    for i, line in enumerate(footer):
        cv2.putText(canvas, line, (8, 1170 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1)
    return canvas


def export():
    import av
    from audit_construction_reserved_pixels import binding_value
    protocol = freeze()
    binding = binding_value()  # Metadata guard only; does not download reserved photos.
    if binding['fixed_pair']['final300']['sha256'] != MODELS['AFTER'][1]:
        raise ValueError('All-three postfit binding differs from fixed final300')
    cases = load_cases(protocol)
    paired = {}
    for tag, (relative, digest) in MODELS.items():
        model_path = ROOT / relative
        provenance = json.loads(model_path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        spec = {'file': str(model_path.relative_to(ROOT / 'models/construction')), 'sha256': digest,
                'source': provenance['source'], 'classes': ['crack'], 'capabilities': ['surface_crack'],
                'input_size': 384, 'confidence': .5, 'experimental': True, 'diagnostic_only': True,
                'license': provenance.get('license', 'Experimental source/model rights documented separately')}
        model = LiteCrackConstructionModel({'engine': 'openvino', 'openvino_device': 'GPU', 'cpu_threads': 1,
                                            'confidence': .5, 'exposure_policy': 'low_light_gain', 'mask_rendering': 'masks'}, spec)
        paired[tag] = []
        for case in cases:
            stem = OUT / f'{tag.lower()}-{case["name"]}'
            report_path, video_path, masks_path = [stem.with_suffix(s) for s in ['.json', '.mp4', '.npz']]
            if any(p.exists() for p in [report_path, video_path, masks_path]):
                raise FileExistsError('Preserve existing complete/partial video evidence; no silent overwrite')
            exposed = case['source']['source_group'] == 'exposed'
            folder = ROOT / 'datasets/construction/evcivil' / case['name'] if exposed else ROOT / 'datasets/construction/evcivil-round3/replay' / case['name']
            proof = json.loads((folder / f'{case["name"]}-lossless-frames.provenance.json').read_text(encoding='utf-8'))
            if sha(ROOT / proof['path']) != proof['sha256']:
                raise ValueError('Source lossless replay identity changed')
            case['source']['lossless_replay_input'] = proof
            case['source']['replay_identity_reverified_against_raw'] = verify_replay(case, proof, folder)
            records, raw_bits, served_bits = [], [], []
            first_time = int(case['times'][0])
            fps = max(1, round(1e6 / np.median(np.diff(case['times']))))
            with av.open(str(video_path), 'w') as container:
                stream = container.add_stream('libx264', rate=fps)
                stream.width, stream.height, stream.pix_fmt = 1420, 1260, 'yuv420p'
                stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
                stream.options = {'crf': '18', 'preset': 'fast', 'threads': '1', 'bf': '0'}
                preview_saved = False
                for gray, truth, timestamp in zip(case['frames'], case['truth'], case['times']):
                    raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    probability = model.predict_maps(raw)
                    rows = decode_crack_map(probability, raw.shape[:2], .5, 4, masks=True)
                    native_probability = cv2.resize(probability, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
                    raw_mask, served_mask = native_probability > .5, np.zeros(raw.shape[:2], dtype=bool)
                    for row in rows:
                        served_mask |= row['mask']
                    native, gain = camera_exposure(raw, 'low_light_gain')
                    record = {**truth, **score_frame(rows, truth['expected'], {'crack'}), 'frame_size_wh': [346, 260],
                              'input_exposure_policy': 'low_light_gain', 'input_exposure_gain': gain,
                              'raw_native_bgr_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                              'model_input_native_bgr_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
                              'probability384_float32_sha256': hashlib.sha256(probability.tobytes()).hexdigest(),
                              'raw_native_mask_sha256': hashlib.sha256(raw_mask.tobytes()).hexdigest(),
                              'served_native_mask_sha256': hashlib.sha256(served_mask.tobytes()).hexdigest(),
                              'raw_native_positive_pixels': int(raw_mask.sum()), 'served_native_positive_pixels': int(served_mask.sum())}
                    records.append(record)
                    raw_bits.append(np.packbits(raw_mask, bitorder='little'))
                    served_bits.append(np.packbits(served_mask, bitorder='little'))
                    canvas = render(raw, native, raw_mask, served_mask, record, case['name'], tag)
                    if not preview_saved and record['expected']:
                        if not cv2.imwrite(str(stem) + '-fixed-first-annotated.jpg', canvas):
                            raise IOError('Could not save fixed first annotated preview')
                        preview_saved = True
                    frame = av.VideoFrame.from_ndarray(canvas, format='bgr24')
                    frame.pts, frame.time_base = int(timestamp) - first_time, stream.time_base
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(str(video_path)) as check:
                actual_pts = [Fraction(frame.pts) * frame.time_base * 1_000_000 for frame in check.decode(video=0)]
            expected_pts = [int(t) - first_time for t in case['times']]
            if actual_pts != expected_pts:
                raise ValueError('Annotated video must retain every exposure and exact microsecond presentation time')
            np.savez_compressed(masks_path, raw_packed=np.stack(raw_bits), served_packed=np.stack(served_bits),
                                native_shape_hw=np.asarray(case['frames'].shape[1:]),
                                source_frame_indices=np.asarray([r['source_frame_index'] for r in records]),
                                source_timestamps_us=case['times'])
            report = {'schema_version': 1, 'purpose': 'development_diagnostic', 'sequence': case['name'],
                      'candidate_status': 'FIXED .50 DEVELOPMENT DIAGNOSTIC - NOT QUALIFIED',
                      'independent_accuracy_credit': False, 'matching_iou_threshold': .5,
                      'issue_summary': 'Actual crack semantic traces and complete publisher-box comparison; no video pixel GT.',
                      'why_it_matters': 'Visible cracks require material/scale review; component boxes and semantic traces are different outputs.',
                      'next_check': 'Verify joint/shadow and measure the actual crack with a scale; no automatic construction acceptance.',
                      'frame_size_wh': [346, 260], 'source': case['source'], 'model': model.describe(),
                      'fixed_probability_threshold': .5, 'selection_protocol_sha256': sha(PROTOCOL),
                      'postfit_binding': binding, 'summary': summarize(records), 'frames': records,
                      'derived_video': {'encoded_frames': len(actual_pts), 'max_timestamp_error_s': 0, 'sha256': sha(video_path)},
                      'native_mask_evidence': {'path': str(masks_path.relative_to(ROOT)), 'sha256': sha(masks_path),
                                               'encoding': 'np.packbits(flat native bool mask, bitorder=little); original shape/indices/PTS inside NPZ',
                                               'pixel_gt_available': False, 'mask_colors_are_not_true_positive_labels': True}}
            report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            item = {'sequence': case['name'], 'model_sha256': digest, 'summary': report['summary'],
                    'report': str(report_path.relative_to(ROOT)), 'report_sha256': sha(report_path),
                    'video': str(video_path.relative_to(ROOT)), 'video_sha256': sha(video_path)}
            paired[tag].append(item)
            print(json.dumps(item), flush=True)
        del model
    (OUT / 'paired-summary.json').write_text(json.dumps(paired, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--freeze-only', action='store_true')
    choice.add_argument('--export', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.freeze_only:
        freeze()
        print(json.dumps({'protocol': str(PROTOCOL), 'sha256': sha(PROTOCOL)}))
    else:
        export()
