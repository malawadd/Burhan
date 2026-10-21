"""Versioned CPUORT private AFTER diagnostic; original GPU preparation untouched.

Reuses the immutable full-video renderer through explicit process-local bindings.
No runtime/default/file mutation. Freeze and inference require completed CPU
public calibration/global choice and a separate root release respectively.
"""
from pathlib import Path
import argparse
import importlib.metadata
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_user_construction_after_round4 as saved

BASE = saved.BASE
OUT = BASE / 'after-public-cal-choice-cpu-recovery-v1'
PLAN = OUT / 'plan-before-any-private-after-network.json'
RECOVERY = ROOT / 'output/evaluation/construction-round4-cpu-recovery-v1'
PROPOSAL = RECOVERY / 'recovery-proposal-before-cpu-network.json'
APPROVAL = RECOVERY / 'approval-before-cpu-network.json'
CHOICE = RECOVERY / 'global-three-checkpoint-freeze.json'
TRAIN = saved.TRAIN
ORIGINAL = ROOT / 'output/evaluation/construction-cubit-author-v2/candidate-protocol-before-network.json'
ORIGINAL_HELPER = ROOT / 'scripts/evaluate_user_construction_after_round4.py'
ORIGINAL_HELPER_SHA = '9859a61412192fe2252fc3bad7b83aefe98d054fee207b2f22588eba249cbca4'
BACKEND = {'engine': 'onnxruntime', 'device': 'CPU', 'provider': 'CPUExecutionProvider',
           'intra_op_threads': 1, 'inter_op_threads': 1, 'input_type': 'tensor(float)'}
OPTIONS = dict(saved.OPTIONS, engine='onnxruntime', openvino_device='CPU', cpu_threads=1)
CPU_SCOPE = saved.SCOPE + ' AFTER is CPUORT; cached original-author BEFORE is historical GPU. Backend and training both changed, so differences cannot be attributed solely to training.'
INFERENCE_SCOPE = ('Actual CubitConstructionModel.infer using CPUExecutionProvider/f32/intra1/inter1, '
    'not ConstructionProfile.process. Public CPU CAL checks full-profile parity separately. '
    'Private diagnostics are offline, unscored and provide no FPS qualification.')
sha, read = saved.sha, saved.read


def cpu_config_passes(config):
    passed = (config.get('providers') == ['CPUExecutionProvider']
        and config.get('intra_op_num_threads') == 1 and config.get('inter_op_num_threads') == 1)
    if not passed:
        raise ValueError('Actual CPUORT provider/intra1/inter1 execution differs')
    return True


def baseline_sources(paths):
    """Hash/JSON integrity only; no original video/image decode or inference."""
    for file, digest in [(BASE / 'baseline-plan-v2-before-inference.json', saved.BASELINE_PLAN_SHA),
                         (BASE / 'original-video-inventory.json', saved.INVENTORY_SHA),
                         (BASE / 'independent-private-baseline-audit-v1.json', saved.BASELINE_AUDIT_SHA)]:
        if sha(file) != digest:
            raise ValueError('Immutable private baseline changed')
        paths.append(file)
    archive_path = BASE / 'baseline-v2-executed-source-archive/manifest.json'
    audit, archive = read(BASE / 'independent-private-baseline-audit-v1.json'), read(archive_path)
    if audit['all_checks_passed'] is not True or audit['failures'] or sha(archive_path) != audit['source_pins']['source_archive_manifest_sha256']:
        raise ValueError('Verified original baseline/source archive required')
    if sha(ROOT / 'edge_runtime/profiles/construction_preprocessing.py') != archive['source_pins']['edge_runtime/profiles/construction_preprocessing.py']:
        raise ValueError('Shared native gain math changed from cached GPU BEFORE')
    paths.append(archive_path)
    for name, digest in archive['source_pins'].items():
        path = archive_path.parent / name
        if sha(path) != digest:
            raise ValueError('Original baseline archived source changed')
        paths.append(path)
    inventory = read(BASE / 'original-video-inventory.json')
    if len(inventory) != 4 or sum(row['decoded_frames'] for row in inventory) != 878:
        raise ValueError('Exact four complete original videos required')
    result = []
    for source in inventory:
        path = BASE / Path(source['file']).stem / 'before-three-models-v2.json'
        report = read(path)
        prior = next(row for row in audit['videos'] if row['source_name'] == Path(source['file']).stem)
        if sha(path) != prior['report_sha256'] or sha(source['file']) != source['sha256'] or report['source']['sha256'] != source['sha256']:
            raise ValueError('Private original file or audited cached GPU report changed')
        if len(report['frames']) != source['decoded_frames'] or len(source['all_frame_times']) != source['decoded_frames'] or report['selection_plan_sha256'] != saved.BASELINE_PLAN_SHA:
            raise ValueError('Complete source/PTS/baseline scope differs')
        for index, (frame, timestamp) in enumerate(zip(report['frames'], source['all_frame_times'])):
            if (frame['source_frame_index'] != index or frame['source_pts'] != timestamp['pts']
                or frame['source_time_base'] != timestamp['time_base'] or frame['evaluated'] is not False or frame['expected'] != []
                or any(row.get('evaluation_status') != 'unscored' for row in frame['models']['author_masks']['predictions'])):
                raise ValueError('Cached GPU BEFORE must retain original unscored source timeline')
        paths.append(path)
        result.append({'inventory': source, 'baseline_report': saved.rel(path), 'baseline_report_sha256': sha(path)})
    return result


def metadata_plan():
    if sha(ORIGINAL_HELPER) != ORIGINAL_HELPER_SHA:
        raise ValueError('Original prepared GPU full-video helper must remain unchanged')
    proposal, approval, choice = read(PROPOSAL), read(APPROVAL), read(CHOICE)
    fit_path, approved_path = TRAIN / 'fit-execution.json', TRAIN / 'approved-plan-before-probe-and-fit.json'
    fit, approved = read(fit_path), read(approved_path)
    if approval.get('parent_approved_network') is not True or approval['approved_proposal_sha256'] != sha(PROPOSAL):
        raise ValueError('Exact root CPU technical-recovery approval required')
    if (proposal['technical_recovery_only'] is not True or proposal['backend'] != BACKEND or choice['backend'] != BACKEND
        or proposal['policy'] != saved.POLICY or proposal['classes'] != saved.CLASSES
        or proposal['labels'] != ['author-baseline', 'step0300', 'step0600', 'step0900']
        or proposal['calibration_inputs'] != read(ORIGINAL)['calibration_inputs']
        or choice['recovery_protocol_sha256'] != sha(PROPOSAL)
        or choice['approved_fit_plan_sha256'] != sha(approved_path) or proposal['approved_fit_plan_sha256'] != sha(approved_path)
        or fit['state'] != 'completed' or fit['completed_steps'] != 900 or fit['plan_sha256'] != sha(approved_path)
        or approved['policy'] != saved.POLICY):
        raise ValueError('Same completed900 fit/assets/source/policy and CPU public global choice required')
    selected, passed = saved.selected_checkpoint(choice)
    paths = [Path(__file__), ORIGINAL_HELPER, PROPOSAL, APPROVAL, CHOICE, ORIGINAL, fit_path, approved_path,
        ROOT / 'scripts/evaluate_construction_round4_cpu_recovery.py',
        ROOT / 'scripts/evaluate_user_construction_videos_round4.py',
        ROOT / 'edge_runtime/profiles/construction_cubit.py', ROOT / 'edge_runtime/profiles/construction_model.py',
        ROOT / 'edge_runtime/profiles/construction_preprocessing.py']
    for name, digest in proposal['pins'].items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError('Executed CPU recovery source/asset changed: ' + name)
        paths.append(path)
    outcomes = dict(zip(['step0300', 'step0600', 'step0900'], choice['checkpoints']))
    outcomes['author-baseline'] = choice['before_same_cpu']
    if [choice['checkpoints'][i]['step'] for i in range(3)] != [300, 600, 900]:
        raise ValueError('Fixed ordered all-three CPU outcomes required')
    for label in proposal['labels']:
        folder = RECOVERY / label
        report_path, execution_path = folder / 'fixed-policy-calibration-report.json', folder / 'calibration-execution.json'
        records_path, numeric_path = folder / 'calibration-runtime-records.json', folder / 'first-cpu-numeric-report.json'
        report, execution, numeric, records = map(read, [report_path, execution_path, numeric_path, records_path])
        outcome, asset = outcomes[label], proposal['models'][label]
        if (execution['state'] != 'completed' or execution['completed_frames'] != 122
            or execution['protocol_sha256'] != sha(PROPOSAL) or execution['report_sha256'] != sha(report_path)
            or execution['model_sha256'] != asset['sha256'] or outcome['model_sha256'] != asset['sha256']
            or outcome['model_file'] != asset['file'] or outcome['report_sha256'] != sha(report_path)
            or report['protocol_sha256'] != sha(PROPOSAL) or report['model']['sha256'] != asset['sha256']
            or report['records_sha256'] != sha(records_path) or report['record_count'] != 122
            or report['full_profile_prediction_parity'] is not True or report['numeric_parity'] is not True
            or report['all_minimum_checks_pass'] is not outcome['all_minimum_checks_pass']
            or execution['numeric_report_sha256'] != sha(numeric_path) or numeric['pass'] is not True):
            raise ValueError('All three + same-CPU-author completed public numeric/CAL bindings required')
        cpu_config_passes(numeric['actual_execution_config'])
        stats = numeric['stats']
        if stats['all_finite'] is not True or any(type(stats[k]) not in (float, int) or not math.isfinite(stats[k]) or not 0 <= stats[k] <= limit for k, limit in proposal['numeric_guards'].items() if k != 'all_finite'):
            raise ValueError('Actual fixed CPU numeric limits failed')
        if len(records) != 122 or any(any(actual[k] != expected[k] for k in ['source', 'frame_index', 'image', 'expected', 'negative']) or actual['direct_and_profile_actual_inputs_identical'] is not True for actual, expected in zip(records, proposal['calibration_inputs'])):
            raise ValueError('Complete exact-source public CPU parity evidence required')
        paths.extend([report_path, execution_path, records_path, numeric_path])
    sources = baseline_sources(paths)
    asset = proposal['models'][f'step{selected["step"]:04d}']
    model = ROOT / asset['file']
    if sha(model) != selected['model_sha256']:
        raise ValueError('Same globally chosen CPU asset required')
    role = ('CPU AFTER / cached GPU BEFORE: public-CAL chosen research model' if passed else
            'CPU AFTER / cached GPU BEFORE: EXACT900 FAILED research diagnostic')
    return {'schema_version': 1, 'selected_step': selected['step'], 'global_choice_sha256': sha(CHOICE),
        'recovery_protocol_sha256': sha(PROPOSAL), 'candidate_passed_public_cal_guards': passed, 'candidate_role': role,
        'model': {'file': model.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(model),
            'input_size': 640, 'classes': saved.CLASSES, 'source': 'Unchanged public TRAIN weights; separate CPUORT public global choice',
            'license': proposal['license'], 'validation_scope': CPU_SCOPE},
        'options': OPTIONS, 'sources': sources, 'scope': CPU_SCOPE, 'inference_scope': INFERENCE_SCOPE,
        'before': {'role': 'Cached actual original-author GPU masks, no old NN',
            'backend_training_confound': 'Backend and learned weights both differ; not isolated training improvement.'},
        'network_calls': {'initialization': 3, 'private_source': 878, 'old_model': 0},
        'software': {'python': sys.version, 'executable': sys.executable,
            'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'opencv-python', 'av', 'onnxruntime']}},
        'timing_scope': 'Offline CPU core-adapter diagnostics, no full-profile or real-time FPS qualification.',
        'rendering': 'Same1400x1240 RAW/cachedGPU-BEFORE/actualCPU-AFTER/info, exact complete timeline; no GT.',
        'process_local_reuse': 'Only original-helper OUT/PLAN/metadata/config/scope globals rebound in this standalone process; runtime and original files unchanged.',
        'pins': {saved.rel(path): sha(path) for path in paths}}


def configured_reuse():
    saved.OUT, saved.PLAN = OUT, PLAN
    saved.metadata_plan, saved.config_passes = metadata_plan, cpu_config_passes
    saved.SCOPE, saved.INFERENCE_SCOPE = CPU_SCOPE, INFERENCE_SCOPE
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'run'], required=True)
    parser.add_argument('--root-network-release', action='store_true')
    args = parser.parse_args()
    if args.stage == 'run' and not args.root_network_release:
        parser.error('Separate root release required for all CPU private inference')
    helper = configured_reuse()
    (helper.freeze if args.stage == 'freeze' else helper.run)()
