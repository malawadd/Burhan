"""Post-CAL lower-alarm exploratory comparison, before any original/capacity NN.

Uses only the three PRIMARY points already selected and frozen in Round4.
No new threshold search, source scoring, fit or model/metadata write occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from factory_stfpm_common import rank

R4 = ROOT / 'output/optimization/factory-round4-supervised'
OUT = ROOT / 'output/optimization/factory-round4-lower-alarm'
PROPOSAL = OUT / 'post-CAL-exploratory-proposal-before-any-benchmark.json'
DECISION = OUT / 'parent-approved-post-CAL-exploratory-decision.json'
FREEZE = OUT / 'post-CAL-lower-alarm-candidate-freeze.json'
BASE = R4 / 'winner-frozen-before-root-capacity.json'
BASE_SHA = '822c795aa83350841998b8243a3d072a82192ee4f026c7d95553c8368ba40106'
MODEL_NAME = 'lower_alarm_exploratory'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def choose(points, global_primary):
    """Only previously selected primary points; retain the stated full ordering."""
    usable = [row for row in points if row['point'] is not None and row['point']['foreground_pixel_precision'] >= .65]
    ordered = sorted(usable, key=lambda row: (row['point']['meaningful_iou_0_1_recall'], -row['point']['normal_fp'],
                                              *rank(row['point']), -row['step']), reverse=True)
    if not ordered or global_primary is None:
        return None, ordered
    candidate = ordered[0]
    if (candidate['point']['meaningful_iou_0_1_recall'] != global_primary['point']['meaningful_iou_0_1_recall']
        or candidate['point']['normal_fp'] >= global_primary['point']['normal_fp']):
        return None, ordered
    return candidate, ordered


def binding():
    if sha(BASE) != BASE_SHA:
        raise ValueError('Exact unchanged prospective both-arm freeze required')
    frozen = read(BASE)
    if frozen['all_three_required_and_complete'] is not True or frozen['both_arms_frozen_before_any_original'] is not True:
        raise ValueError('All3 prospective outcomes must already be frozen')
    points = []
    paths = [BASE, R4 / 'protocol-before-fit.json', R4 / 'training-complete.json',
             R4 / 'CAL-identity-v2-correction-before-CAL.json']
    original = {row['step']: row for row in frozen['arms']['primary']['ranked_candidates']}
    for step in [300, 600, 900]:
        report_path = R4 / f'checkpoint{step}-calibration.json'
        numeric_path = R4 / f'checkpoint{step}-numeric.json'
        terminal_path = R4 / f'checkpoint{step}-terminal.json'
        report, numeric, terminal = read(report_path), read(numeric_path), read(terminal_path)
        if (terminal['status'] != 'calibration_complete' or terminal['CAL_inputs_opened'] != 80
            or terminal['calibration_sha256'] != sha(report_path) or terminal['numeric_sha256'] != sha(numeric_path)
            or frozen['all_three_terminal_outcomes'][str(step)]['terminal_sha256'] != sha(terminal_path)
            or numeric['all_checks_passed'] is not True or report['numeric_sha256'] != sha(numeric_path)
            or report['protocol_sha256'] != frozen['protocol_sha256']
            or report['CAL_identity_v2_correction_sha256'] != frozen['CAL_identity_v2_correction_sha256']):
            raise ValueError('Previously frozen actual CAL/numeric/terminal lineage differs')
        result = report['arms']['primary']
        point = result['selected']
        row = {'step': step, 'point': point}
        if point is not None:
            row.update(result['deployment'])
            if row != original.get(step):
                raise ValueError('Only exact already-frozen PRIMARY points are permitted')
            model = Path(row['model_path'])
            meta = read(model.with_suffix('.onnx.json'))
            if (sha(model) != row['model_sha256'] or sha(model.with_suffix('.onnx.json')) != row['metadata_sha256']
                or meta['calibration_selected_point'] != point or meta['round4_policy_arm'] != 'primary'
                or meta['round4_normal_FP_budget'] != 1 or point['normal_fp'] > 1
                or (meta['pixel_threshold'], meta['minimum_component_pixels'], meta['hysteresis_growth_threshold']) != (point['high'], point['area'], point['growth'])):
                raise ValueError('Exact existing PRIMARY artifact/point changed')
            paths.extend([model, model.with_suffix('.onnx.json')])
        points.append(row)
        paths.extend([report_path, numeric_path, terminal_path])
    winner, ordered = choose(points, frozen['arms']['primary']['winner'])
    return {'original_global_freeze_sha256': BASE_SHA, 'original_prospective_primary_winner': frozen['arms']['primary']['winner'],
            'all_three_existing_primary_points': points, 'post_CAL_ranked_primary_points': ordered,
            'candidate': winner, 'artifact_pins': {str(path.relative_to(ROOT)): sha(path) for path in paths}}


def propose():
    # This comparison is declared before original inference and speed results.
    benchmark = ROOT / 'output/optimization/factory-round4-root-benchmark'
    if any(benchmark.glob('*-attempt.json')):
        raise ValueError('Exploratory comparison must be declared before any root capacity/original/runtime attempt')
    data = binding()
    value = {'model_name': MODEL_NAME, 'scope': 'POST-CAL exploratory comparison; not a prospective confirmatory result',
             'selection': 'Only three existing selected PRIMARY points: maximize meaningful IoU>=.1 view recall, then minimize normal-view FP, then existing rank, then earlier step',
             'eligibility': 'Must match frozen global PRIMARY meaningful recall, have pixel precision>=.65 and strictly fewer CAL normal-view alarms; otherwise no candidate',
             'new_threshold_searches': 0, 'new_fits': 0, 'new_source_scoring': 0,
             'original_prospective_two_arms_unchanged': True, 'all_three_candidate_outcomes_mandatory': True,
             'before_any_original_or_capacity': True,
             'root_gates': 'Same first64 complete-profile>=15 before one exposed-original audit; same paired CAL80 and16FPS60s/fullcounters; no original-informed choosing or fallback',
             'candidate_is_qualified_or_promoted': False, 'binding': data,
             'source_pins': {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), ROOT / 'tests/test_factory_round4_lower_alarm.py', ROOT / 'scripts/factory_stfpm_common.py']}}
    write(PROPOSAL, value)
    print(json.dumps({'proposal_sha256': sha(PROPOSAL), 'candidate': data['candidate']}, indent=2))


def freeze():
    proposal, decision = read(PROPOSAL), read(DECISION)
    if decision.get('approved_post_CAL_lower_alarm_exploratory_comparison') is not True or decision.get('proposal_sha256') != sha(PROPOSAL):
        raise ValueError('Root exact exploratory proposal approval required')
    if proposal['binding'] != binding():
        raise ValueError('Already-frozen candidate/source artifacts changed')
    for relative, digest in proposal['source_pins'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Exploratory source changed after proposal')
    if any((ROOT / 'output/optimization/factory-round4-root-benchmark').glob('*-attempt.json')):
        raise ValueError('Exploratory candidate must freeze before any root benchmark attempt')
    archive = OUT / 'executed-sources-before-any-benchmark'
    for relative, digest in proposal['source_pins'].items():
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        if sha(target) != digest:
            raise ValueError('Exploratory source archive mismatch')
    manifest = archive / 'manifest.json'
    write(manifest, {'proposal_sha256': sha(PROPOSAL), 'source_pins': proposal['source_pins']})
    value = dict(proposal, proposal_sha256=sha(PROPOSAL), parent_decision_sha256=sha(DECISION),
                 executed_source_archive=str(archive.relative_to(ROOT)), executed_source_manifest_sha256=sha(manifest),
                 frozen_before_any_NN=True, accepted_for_root_first64_capacity=proposal['binding']['candidate'] is not None,
                 accepted_for_original_audit=False, production_qualified=False)
    write(FREEZE, value)
    print(json.dumps({'freeze_sha256': sha(FREEZE), 'candidate': value['binding']['candidate']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['propose', 'freeze'], required=True)
    args = parser.parse_args()
    propose() if args.stage == 'propose' else freeze()
