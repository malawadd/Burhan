"""Post-choice exposed55-photo actual semantic-mask diagnostic; never model selection.

Freeze only after all900 updates and three CAL results/global choice. Execution
requires a separate GPU release. This helper never loads a private/video/heldout
source or adjusts the fixed .25/.45 policy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'output/evaluation/construction-round4-mixed-v3/fixed55-pixel-diagnostic'
PREPLAN = ROOT / 'output/evaluation/construction-round4-mixed-v2/fixed55-dacl-pixel-diagnostic-plan-before-choice.json'
CHOICE = ROOT / 'output/evaluation/construction-round4-mixed-v3/global-three-checkpoint-freeze.json'
DACL = ROOT / 'datasets/construction/dacl10k'
CLASSES = ['crack', 'spalling']
MAPPING = {'Crack': 0, 'ACrack': 0, 'Spalling': 1}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf8') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def freeze():
    from train_construction_round4_mixed_v3 import approved
    plan, _, plan_hash = approved('fit')
    if plan.get('parent_approved_export_and_calibration') is not True:
        raise ValueError('Exact downstream approval required')
    choice = read(CHOICE)
    fit_path = ROOT / 'output/training/construction-round4-mixed-v3/fit-execution.json'
    fit = read(fit_path)
    if fit['state'] != 'completed' or fit['completed_steps'] != 900 or choice['approved_fit_plan_sha256'] != plan_hash:
        raise ValueError('All900 updates and frozen three-CAL choice required first')
    if sorted(r['step'] for r in choice['checkpoints']) != [300, 600, 900]:
        raise ValueError('Exactly the three unique fixed checkpoint steps required')
    if choice['winner'] is not None and choice['winner'] not in choice['checkpoints']:
        raise ValueError('Chosen winner must equal an actual fixed checkpoint record')
    if any(type(row['all_minimum_checks_pass']) is not bool for row in choice['checkpoints']):
        raise ValueError('Explicit checkpoint guard booleans required')
    eligible = [row for row in choice['checkpoints'] if row['all_minimum_checks_pass']]
    if choice['winner'] != (max(eligible, key=lambda row: row['rank']) if eligible else None):
        raise ValueError('Chosen winner differs from fixed public-CAL rank')
    selected = choice['winner'] or next(r for r in choice['checkpoints'] if r['step'] == 900)
    step = selected['step']
    model = ROOT / f'models/construction/round4-mixed-v3/crack-spalling-step{step:04d}-640.onnx'
    metadata = model.with_suffix('.provenance.json')
    if sha(model) != selected['model_sha256']:
        raise ValueError('Actual globally chosen/fallback asset differs')
    for candidate in choice['checkpoints']:
        folder = ROOT / f'output/evaluation/construction-round4-mixed-v3/step{candidate["step"]:04d}'
        report_path = folder / 'fixed-policy-calibration-report.json'
        execution = read(folder / 'calibration-execution.json')
        if execution['state'] != 'completed' or sha(report_path) != candidate['report_sha256'] or execution['report_sha256'] != candidate['report_sha256'] or execution['model_sha256'] != candidate['model_sha256']:
            raise ValueError('All three completed source-bound CAL results required')
    preplan = read(PREPLAN)
    if sha(PREPLAN) != '184e5f534e37975fadd853cb4a41d96553c093069e0f619868defd90343f5e59' or len(preplan['rows']) != 55:
        raise ValueError('Pre-choice fixed55 source scope differs')
    paths = [Path(__file__), PREPLAN, CHOICE, fit_path, model, metadata,
             ROOT / preplan['model_before']['file'],
             ROOT / 'edge_runtime/profiles/construction_cubit.py',
             ROOT / 'edge_runtime/profiles/construction_model.py',
             ROOT / 'edge_runtime/profiles/construction_preprocessing.py']
    for row in preplan['rows']:
        for key in ['image', 'annotation']:
            path = DACL / row[key]
            if sha(path) != row[key + '_sha256']:
                raise ValueError('Fixed diagnostic source changed')
            paths.append(path)
    if sha(ROOT / preplan['model_before']['file']) != preplan['model_before']['sha256']:
        raise ValueError('Original author model changed')
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'post_choice_only': True, 'source_plan_sha256': sha(PREPLAN),
        'global_choice_sha256': sha(CHOICE), 'selected_step': step,
        'candidate_passed_minimum_guards': choice['winner'] is not None,
        'candidate_role': 'Eligible globally frozen checkpoint' if choice['winner'] else 'Exact900 FAILED diagnostic; no winner',
        'rows': preplan['rows'], 'classes': CLASSES, 'source_mapping': MAPPING,
        'fixed_policy': preplan['fixed_policy'],
        'models': {'before': preplan['model_before'], 'after': {'file': model.relative_to(ROOT).as_posix(), 'sha256': sha(model)}},
        'ground_truth': 'Native normalized source polygons scaled by [width,height], np.rint then cv2.fillPoly; union parts/class, overlap allowed. No GT crop or box fill.',
        'scope': preplan['source_role'], 'rights': plan['license'],
        'metrics': preplan['metrics'], 'timing_scope': 'Accuracy-only; host/project contention; no inferenceFPS.',
        'pins': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    write(OUT / 'execution-protocol-before-pixel-network.json', protocol, exclusive=True)
    print('PIXEL_PROTOCOL', sha(OUT / 'execution-protocol-before-pixel-network.json'), flush=True)


def ground_truth(annotation, shape):
    import cv2
    import numpy as np
    height, width = shape
    if annotation['image_width'] != width or annotation['image_height'] != height:
        raise ValueError('Native source/annotation geometry differs')
    masks = np.zeros((2, height, width), np.uint8)
    for polygon in annotation['polygons']:
        if polygon['label'] not in MAPPING:
            continue
        for part in polygon['points']:
            points = np.asarray(part, np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
                raise ValueError('Actual target source polygon invalid')
            cv2.fillPoly(masks[MAPPING[polygon['label']]], [np.rint(points * [width, height]).astype(np.int32)], 1)
    return masks.astype(bool)


def ratios(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    return dict(counts, precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                iou=tp / (tp + fp + fn) if tp + fp + fn else None)


def run():
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction_cubit import CubitConstructionModel
    protocol_path = OUT / 'execution-protocol-before-pixel-network.json'
    protocol = read(protocol_path)
    for name, digest in protocol['pins'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen pixel-audit source changed: ' + name)
    preplan, choice = read(PREPLAN), read(CHOICE)
    if sha(PREPLAN) != '184e5f534e37975fadd853cb4a41d96553c093069e0f619868defd90343f5e59' or sorted(r['step'] for r in choice['checkpoints']) != [300, 600, 900]:
        raise ValueError('Fixed pre-choice source scope/global three-checkpoint choice differs')
    if choice['winner'] is not None and choice['winner'] not in choice['checkpoints']:
        raise ValueError('Chosen winner must equal an actual fixed checkpoint record')
    if any(type(row['all_minimum_checks_pass']) is not bool for row in choice['checkpoints']):
        raise ValueError('Explicit checkpoint guard booleans required')
    eligible = [row for row in choice['checkpoints'] if row['all_minimum_checks_pass']]
    if choice['winner'] != (max(eligible, key=lambda row: row['rank']) if eligible else None):
        raise ValueError('Chosen winner differs from fixed public-CAL rank')
    selected = choice['winner'] or next(r for r in choice['checkpoints'] if r['step'] == 900)
    expected_models = {'before': preplan['model_before'], 'after': {
        'file': f'models/construction/round4-mixed-v3/crack-spalling-step{selected["step"]:04d}-640.onnx',
        'sha256': selected['model_sha256']}}
    expected_role = 'Eligible globally frozen checkpoint' if choice['winner'] else 'Exact900 FAILED diagnostic; no winner'
    if protocol['rows'] != preplan['rows'] or protocol['models'] != expected_models or protocol['selected_step'] != selected['step'] or protocol['candidate_role'] != expected_role or protocol['candidate_passed_minimum_guards'] != (choice['winner'] is not None) or protocol['classes'] != CLASSES or protocol['source_mapping'] != MAPPING or protocol['fixed_policy'] != preplan['fixed_policy'] or protocol['scope'] != preplan['source_role'] or protocol['source_plan_sha256'] != sha(PREPLAN) or protocol['global_choice_sha256'] != sha(CHOICE):
        raise ValueError('Execution body differs from source-bound post-choice diagnostic')
    marker = OUT / 'execution.json'
    execution = {'state': 'started', 'protocol_sha256': sha(protocol_path), 'completed_model_photos': 0}
    write(marker, execution, exclusive=True)
    try:
        all_results = {}
        for role, asset in protocol['models'].items():
            model_path = ROOT / asset['file']
            spec = {'file': model_path.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': asset['sha256'],
                'input_size': 640, 'classes': CLASSES, 'source': 'Frozen original/real-supervision adaptation diagnostic',
                'license': protocol['rights'], 'validation_scope': protocol['scope']}
            model = CubitConstructionModel({'engine': 'openvino', 'openvino_device': 'GPU', 'cpu_threads': 1,
                'confidence': .25, 'nms_iou': .45, 'mask_rendering': 'masks', 'exposure_policy': 'low_light_gain'}, spec)
            actual = model.actual_execution_config
            devices = actual['EXECUTION_DEVICES']
            if not isinstance(devices, list) or not devices or any(not str(x).startswith('GPU') for x in devices) or not any(x in str(actual['INFERENCE_PRECISION_HINT']).lower() for x in ['float32', 'f32']) or str(actual['NUM_STREAMS']) != '1':
                raise ValueError('Actual GPU/float32/one-stream execution differs')
            rows, totals = [], {label: {'tp': 0, 'fp': 0, 'fn': 0, 'class_absent_photos': 0, 'class_absent_flags': 0} for label in CLASSES}
            folder = OUT / role
            folder.mkdir(exist_ok=False)
            for index, source in enumerate(protocol['rows']):
                raw = cv2.imread(str(DACL / source['image']), cv2.IMREAD_COLOR)
                if raw is None:
                    raise ValueError('Native diagnostic photo failed to decode')
                gt = ground_truth(read(DACL / source['annotation']), raw.shape[:2])
                predictions = model.infer(raw)
                predicted = np.zeros_like(gt)
                actual_rows = []
                for item in predictions:
                    mask = item['mask']
                    if mask.shape != raw.shape[:2] or mask.dtype != np.bool_:
                        raise ValueError('Actual native predicted mask geometry differs')
                    predicted[CLASSES.index(item['label'])] |= mask
                    actual_rows.append({k: v for k, v in item.items() if k != 'mask'} | {'native_positive_pixels': int(mask.sum()), 'native_mask_empty': not bool(mask.any())})
                metrics = {}
                for class_id, label in enumerate(CLASSES):
                    p, g = predicted[class_id], gt[class_id]
                    counts = {'tp': int((p & g).sum()), 'fp': int((p & ~g).sum()), 'fn': int((~p & g).sum())}
                    metrics[label] = ratios(counts)
                    for key, value in counts.items():
                        totals[label][key] += value
                    totals[label]['class_absent_photos'] += int(not g.any())
                    totals[label]['class_absent_flags'] += int(not g.any() and p.any())
                target = folder / f'{index:03d}-{source["source_name"]}-native-masks.npz'
                np.savez_compressed(target, shape=np.asarray(gt.shape, np.int64),
                    actual_gt_packed=np.packbits(gt), actual_prediction_packed=np.packbits(predicted))
                row = {'source': source, 'shape_hw': list(raw.shape[:2]), 'actual_input_exposure_gain': model.last_input_exposure_gain,
                       'predictions': actual_rows, 'pixel_metrics': metrics, 'native_masks': target.relative_to(ROOT).as_posix(), 'native_masks_sha256': sha(target)}
                rows.append(row)
                write(folder / 'actual-photo-records.json', rows)
                execution['completed_model_photos'] += 1
                write(marker, execution)
                print('PIXEL', role, index + 1, '/55', flush=True)
            all_results[role] = {'model': model.describe(), 'photos': len(rows), 'by_class': {k: ratios(v) for k, v in totals.items()},
                                 'records_sha256': sha(folder / 'actual-photo-records.json')}
            del model
        report = {'protocol_sha256': sha(protocol_path), 'scope': protocol['scope'], 'candidate_role': protocol['candidate_role'],
                  'ground_truth': protocol['ground_truth'],
                  'class_absent_flags_definition': 'Native predicted-mask presence on GT-absent photos; distinct from detector-box alarm rates.',
                  'before_after': all_results, 'model_selection_from_these_photos': False, 'production_qualification': False}
        write(OUT / 'paired55-semantic-pixel-report.json', report, exclusive=True)
        execution.update(state='completed', report_sha256=sha(OUT / 'paired55-semantic-pixel-report.json'))
        write(marker, execution)
    except Exception:
        execution.update(state='failed', traceback=traceback.format_exc())
        write(marker, execution)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'run'], required=True)
    args = parser.parse_args()
    (freeze if args.stage == 'freeze' else run)()
