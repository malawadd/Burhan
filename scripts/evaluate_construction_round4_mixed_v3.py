"""Fixed three-snapshot CAL using unchanged audited mask/profile scoring, no cutoff search."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'output/evaluation/construction-round4-mixed-v3'
TRAIN = ROOT / 'output/training/construction-round4-mixed-v3'
MODELS = ROOT / 'models/construction/round4-mixed-v3'
OLD_PROTOCOL = ROOT / 'output/evaluation/construction-cubit-author-v2/candidate-protocol-before-network.json'
AUTHOR_RECORDS = ROOT / 'output/evaluation/construction-cubit-author-v2/calibration-runtime-records.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def step_paths(step):
    return OUT / f'step{step:04d}', MODELS / f'crack-spalling-step{step:04d}-640.onnx'


def freeze(step):
    from train_construction_round4_mixed_v3 import approved, CONFIG
    plan, _, plan_hash = approved('fit')
    if plan.get('parent_approved_export_and_calibration') is not True:
        raise ValueError('Reviewed fixed-three export/CAL approval required')
    if step not in CONFIG['checkpoints']:
        raise ValueError('Only the three fixed snapshots are allowed')
    checkpoint = TRAIN / f'step{step:04d}-state.pt'
    checkpoint_receipt = read(checkpoint.with_suffix('.json'))
    if checkpoint_receipt['sha256'] != sha(checkpoint) or checkpoint_receipt['plan_sha256'] != plan_hash or checkpoint_receipt['classes'] != ['crack', 'spalling']:
        raise ValueError('Actual fixed checkpoint identity/class/lineage differs')
    old = read(OLD_PROTOCOL)
    paths = [Path(__file__), ROOT / 'scripts/export_construction_round4_mixed_v3.py',
        ROOT / 'scripts/evaluate_cubit_author_candidate_v2.py', ROOT / 'scripts/train_construction_round4_mixed_v3.py',
        ROOT / 'scripts/construction_round4_mixed_loss_v3.py', ROOT / 'edge_runtime/profiles/construction_model.py',
        ROOT / 'edge_runtime/profiles/construction_cubit.py', ROOT / 'edge_runtime/profiles/construction.py',
        ROOT / 'edge_runtime/profiles/construction_preprocessing.py', ROOT / 'edge_runtime/contracts.py',
        ROOT / 'scripts/evaluate_construction_round3.py', ROOT / 'scripts/construction_annotations.py',
        TRAIN / 'approved-plan-before-probe-and-fit.json', checkpoint, checkpoint.with_suffix('.json'),
        OLD_PROTOCOL, AUTHOR_RECORDS,
        ROOT / 'output/evaluation/construction-cubit-author-v2/fixed-policy-calibration-report.json']
    # The audited reused helper also scores historical best1 as a secondary
    # reference; preserve its actual cached identities, never run a new baseline.
    import evaluate_cubit_author_candidate_v2 as base
    for p in [base.BEFORE, base.BEFORE.with_suffix('.operating-point.json'), base.BEFORE.with_suffix('.provenance.json'), base.BEFORE_RECORDS]:
        paths.append(p)
    rows = old['calibration_inputs']
    if len(rows) != 122 or sum(r['negative'] for r in rows) != 10:
        raise ValueError('Original complete122 calibration scope differs')
    for row in rows:
        if sha(ROOT / row['image']) != row['image_sha256']:
            raise ValueError('Original calibration source image changed')
    original = read(AUTHOR_RECORDS)
    if len(original) != len(rows) or any(a['source'] != b['source'] or a['frame_index'] != b['frame_index'] or a['expected'] != b['expected'] or a['negative'] != b['negative'] for a, b in zip(original, rows)):
        raise ValueError('Author baseline and frozen calibration source/GT/index differ')
    folder, _ = step_paths(step)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {'parent_approved_network': True, 'approval_basis': 'Explicit parent_approved_export_and_calibration in pinned fit plan',
        'approved_fit_plan_sha256': plan_hash, 'step': step, 'classes': ['crack', 'spalling'],
        'class_confidence': {'crack': .25, 'spalling': .25}, 'nms_iou': .45, 'input_size': 640,
        'calibration_inputs': rows, 'pinned_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'checkpoint_sha256': sha(checkpoint), 'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(),
        'source_checkpoint_sha256': plan['source_state_sha256'],
        'primary_before_records_sha256': sha(AUTHOR_RECORDS), 'primary_before': 'Author original CUBIT fixed.25/.45 actual122 cached outputs, no new baseline NN.',
        'fixed_scoring_helper': 'evaluate_cubit_author_candidate_v2.calibration with only newOUT/MODEL/TARGET bindings; original historicalbest1 secondary comparison retained.',
        'mask_scope': 'Actual prototypes/native foreground bits retained. Publisher camera boxes do not establish pixel trace accuracy or physical crack width.',
        'policy': plan['policy'], 'minimum_review': plan['minimum_calibration_guards'], 'selection': plan['selection'],
        'no_heldout_private_or_video_loading': True, 'rights_scope': plan['license']}
    target = folder / 'candidate-protocol-before-network.json'
    write_new(target, protocol)
    print('EVALPROTOCOL', step, sha(target), flush=True)


def calibration(step):
    import evaluate_cubit_author_candidate_v2 as base
    from evaluate_construction_round3 import score_frame, summarize
    folder, model = step_paths(step)
    protocol_path = folder / 'candidate-protocol-before-network.json'
    protocol = read(protocol_path)
    if protocol['step'] != step or protocol['classes'] != ['crack', 'spalling'] or protocol['policy'] != {'confidence': .25, 'nms_iou': .45, 'exposure_policy': 'low_light_gain', 'mask_rendering': 'masks'}:
        raise ValueError('Frozen snapshot/policy differs')
    # Source-bound, process-local reuse. No installed/runtime source mutation.
    base.OUT, base.DIRECTORY, base.MODEL, base.TARGET = folder, MODELS, model, protocol_path
    base.calibration()
    records = read(folder / 'calibration-runtime-records.json')
    original = read(AUTHOR_RECORDS)
    before = []
    for old, new in zip(original, records):
        if any(old[key] != new[key] for key in ['source', 'frame_index', 'image', 'expected', 'negative']):
            raise ValueError('Actual original/new calibration identity differs')
        before.append(score_frame(old['predictions'], old['expected'], {'crack', 'spalling'}, negative=old['negative']) | {'negative': old['negative']})
    report = read(folder / 'fixed-policy-calibration-report.json')
    write_new(folder / 'paired-original-author-report.json', {'protocol_sha256': sha(protocol_path),
        'before_model': 'Original author CUBIT .25/.45', 'before_cached_records_sha256': sha(AUTHOR_RECORDS),
        'before': summarize(before), 'after': report['summary'], 'step': step,
        'geometric_thin_after': report['geometric_thin'], 'all_minimum_checks_pass': report['all_minimum_checks_pass'],
        'scope': 'Exact same complete122 source/image/index/GT. Reused calibration, no independent-site/private-video accuracy credit. Historicalbest1 remains secondary in original helper report.'})


def select():
    from train_construction_round4_mixed_v3 import CONFIG, approved
    from evaluate_construction_round3 import score_frame, summarize
    plan, _, plan_hash = approved('fit')
    fit = read(TRAIN / 'fit-execution.json')
    if fit['state'] != 'completed' or fit['completed_steps'] != CONFIG['steps'] or fit['plan_sha256'] != plan_hash:
        raise ValueError('All fixed900 updates must complete before checkpoint selection')
    results = []
    for step in CONFIG['checkpoints']:
        folder, model = step_paths(step)
        execution, report = read(folder / 'calibration-execution.json'), read(folder / 'fixed-policy-calibration-report.json')
        protocol_path = folder / 'candidate-protocol-before-network.json'
        protocol = read(protocol_path)
        export = read(folder / 'export-execution.json')
        if execution['state'] != 'completed' or execution['model_sha256'] != sha(model) or execution['report_sha256'] != sha(folder / 'fixed-policy-calibration-report.json'):
            raise ValueError('Every fixed snapshot must complete its first actual CAL/parity')
        if report['protocol_sha256'] != sha(protocol_path) or execution['protocol_sha256'] != sha(protocol_path) or report['model']['sha256'] != sha(model) or export['model_sha256'] != sha(model) or export['state'] != 'completed':
            raise ValueError('Actual report/model/export/protocol bindings differ')
        records_path = folder / 'calibration-runtime-records.json'
        if report['records_sha256'] != sha(records_path):
            raise ValueError('Actual calibration records changed')
        records = read(records_path)
        if len(records) != 122:
            raise ValueError('Complete122 actual frame records required')
        rescored = []
        for actual, expected in zip(records, protocol['calibration_inputs']):
            if any(actual[key] != expected[key] for key in ['source', 'frame_index', 'image', 'expected', 'negative']):
                raise ValueError('Actual frame identity/ground truth differs from frozen source')
            rescored.append(score_frame(actual['predictions'], actual['expected'], {'crack', 'spalling'}, negative=actual['negative']) | {'negative': actual['negative']})
        summary = summarize(rescored)
        if summary != report['summary']:
            raise ValueError('Recomputed actual box matching/summary differs')
        thin = {e['id'] for r in records for e in r['expected'] if e['label'] == 'crack' and min(e['bbox'][2:]) <= 30 and max(e['bbox'][2:]) >= 2 * min(e['bbox'][2:])}
        thin_hits = sum(p.get('matched_annotation_id') in thin for r in rescored for p in r['predictions'])
        thin_recall = thin_hits / len(thin)
        negatives = [r for r in rescored if r['negative']]
        negative_rate = sum(bool(r['predictions']) for r in negatives) / len(negatives)
        by = summary['by_class']
        checks = {name + '_precision': (by[name]['precision'] or 0) >= .8 for name in ['crack', 'spalling']}
        checks.update({name + '_recall': (by[name]['recall'] or 0) >= .5 for name in ['crack', 'spalling']})
        checks.update(geometric_thin_recall=thin_recall >= .5, combined_negative_alarm_rate=negative_rate <= .1,
                      full_profile_parity=report['full_profile_prediction_parity'] is True)
        if checks != report['minimum_checks'] or all(checks.values()) != report['all_minimum_checks_pass'] or thin_hits != report['geometric_thin']['localized_matches'] or negative_rate != report['combined_negative_image_alarm_rate']:
            raise ValueError('Actual recomputed guard/thin/negative results differ')
        rank = [min(by['crack']['recall'] or 0, by['spalling']['recall'] or 0, thin_recall),
                min(by['crack']['precision'] or 0, by['spalling']['precision'] or 0),
                -sum(by[name]['fp'] for name in ['crack', 'spalling']), -step]
        results.append({'step': step, 'all_minimum_checks_pass': all(checks.values()), 'rank': rank,
                        'report_sha256': sha(folder / 'fixed-policy-calibration-report.json'), 'model_sha256': sha(model)})
    eligible = [r for r in results if r['all_minimum_checks_pass']]
    winner = max(eligible, key=lambda r: r['rank']) if eligible else None
    write_new(OUT / 'global-three-checkpoint-freeze.json', {'approved_fit_plan_sha256': plan_hash,
        'rule': plan['selection'], 'checkpoints': results, 'winner': winner,
        'if_no_winner': 'Exact900 final remains failure diagnostic; no threshold rescue, promotion or heldout opening.',
        'production_qualification': False, 'private_video_training_performed': False})
    print('FROZENWINNER', winner, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'calibration', 'select'], required=True)
    parser.add_argument('--step', type=int, choices=[300, 600, 900])
    args = parser.parse_args()
    if args.stage == 'select':
        select()
    else:
        if args.step is None:
            parser.error('--step is required')
        (freeze if args.stage == 'freeze' else calibration)(args.step)
