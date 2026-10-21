"""Stage/apply nine completed FAILED construction diagnostics, never promote.

Preparation writes previews only after all evidence/audits complete. Application
requires a separate explicit root release and the root-installed runtime mapping
and required_execution guard. No inference, upload or default edits occur here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / 'output/evaluation/construction-round4-cpu-recovery-v1'
PRIVATE = ROOT / 'output/evaluation/user-construction-videos-round4/after-public-cal-choice-cpu-recovery-v1'
OUT = CPU / 'local-integration'
PLAN = OUT / 'reviewable-registration-plan.json'
VARIANT = 'round4_mixed_cpu_failed'
ASSET = ROOT / 'edge_runtime/profiles/construction_round4_cpu_asset.py'
MODEL = ROOT / 'models/construction/round4-mixed-v3/crack-spalling-step0900-640.onnx'
MODEL_SHA = '8bf254cc4e1079bc58bd9590f5e57630890f6d4048e2afc76a1ceb51dfaa1d2b'
PUBLIC_NAMES = ['crack_89', 'crack_129', 'spalling_75', 'ros_crack_21', 'spalling_76']
CONFIGS = [ROOT / 'config/construction-videos.json', ROOT / 'config/public-presets.json']
REQUIRED = {'engine': 'onnxruntime', 'openvino_device': 'CPU', 'cpu_threads': 1}
NOTE = ('FAILED experimental model; masks are actual model candidates, not verified damage or code acceptance. '
        'Historical cached GPU BEFORE versus trained CPU AFTER changes both backend and weights. '
        'Measured humidity, fastener adequacy and structural severity remain unsupported. No promotion/default change.')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def local(relative, parent=ROOT):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError('Evidence path escapes required local folder')
    return path


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def append_exact(existing, additions, key):
    result = list(existing)
    for row in additions:
        matches = [item for item in result if item.get(key) == row[key]]
        if len(matches) > 1 or matches and matches[0] != row:
            raise ValueError('Refuse to overwrite/duplicate existing entry: ' + row[key])
        if not matches:
            result.append(row)
    return result


def actual_cpu(report):
    model = report['model']
    config = model['actual_execution_config']
    if (model['sha256'] != MODEL_SHA or model['engine'] != 'onnxruntime' or model['device'] != 'CPU'
        or config['providers'] != ['CPUExecutionProvider'] or config['intra_op_num_threads'] != 1
        or config['inter_op_num_threads'] != 1):
        raise ValueError('Evidence must use exact final900 CPUORT1 model')


def preset(label, source, private=False):
    return {'label': label, 'profile': 'construction', 'path': str(source) if private else rel(source),
        'source_kind': 'file', 'input_fps': 2, 'deadline_ms': 1500, 'preview_fps': 2,
        'video_mode': 'every_frame', 'purpose': 'failure_diagnostic', 'independent_accuracy_credit': False,
        'private_local_input': private,
        'profile_options': {'mode': 'damage', 'damage_variant': VARIANT, **REQUIRED,
            'confidence': .25, 'nms_iou': .45, 'exposure_policy': 'low_light_gain',
            'mask_rendering': 'masks', 'damage_classes': ['crack', 'spalling'], 'confirm_frames': 1, 'stage': 'all'},
        'note': NOTE + ' Original raw camera input only. Requested2FPS replay is not measured capacity; every_frame may replay slowly.'}


def build(public_audit_path):
    choice_path = CPU / 'global-three-checkpoint-freeze.json'
    choice = read(choice_path)
    if choice['winner'] is not None or sorted(row['step'] for row in choice['checkpoints']) != [300, 600, 900] or any(row['all_minimum_checks_pass'] for row in choice['checkpoints']):
        raise ValueError('Integration is specifically the completed no-winner FAILED900 diagnostic')
    selected = next(row for row in choice['checkpoints'] if row['step'] == 900)
    if selected['model_sha256'] != MODEL_SHA or sha(MODEL) != MODEL_SHA or choice['production_qualification'] is not False:
        raise ValueError('Exact fixed900 failed model identity required')
    proposal_path, approval_path = CPU / 'recovery-proposal-before-cpu-network.json', CPU / 'approval-before-cpu-network.json'
    proposal, approval = read(proposal_path), read(approval_path)
    if choice['recovery_protocol_sha256'] != sha(proposal_path) or approval.get('parent_approved_network') is not True or approval['approved_proposal_sha256'] != sha(proposal_path):
        raise ValueError('Completed separately approved CPU recovery required')
    paths = [Path(__file__), choice_path, proposal_path, approval_path, MODEL, MODEL.with_suffix('.provenance.json'),
             ROOT / 'edge_runtime/profiles/construction_cubit.py', public_audit_path]
    wrapper = ROOT / 'edge_runtime/profiles/construction_cubit.py'
    if sha(wrapper) != proposal['pins'][rel(wrapper)]:
        raise ValueError('Actual frozen mask decoder changed')
    for candidate in choice['checkpoints']:
        folder = CPU / f'step{candidate["step"]:04d}'
        execution_path, report_path = folder / 'calibration-execution.json', folder / 'fixed-policy-calibration-report.json'
        execution = read(execution_path)
        numeric_path = folder / 'first-cpu-numeric-report.json'
        report = read(report_path)
        if (execution['state'] != 'completed' or execution['completed_frames'] != 122
            or execution['report_sha256'] != candidate['report_sha256'] or sha(report_path) != candidate['report_sha256']
            or execution['protocol_sha256'] != sha(proposal_path) or execution['model_sha256'] != candidate['model_sha256']
            or report['model']['sha256'] != candidate['model_sha256'] or report['full_profile_prediction_parity'] is not True
            or report['numeric_parity'] is not True or execution['numeric_report_sha256'] != sha(numeric_path)
            or read(numeric_path)['pass'] is not True):
            raise ValueError('All three actual CPU CAL outcomes required')
        paths.extend([execution_path, report_path, numeric_path])
    baseline_folder = CPU / 'author-baseline'
    baseline_execution, baseline_report = baseline_folder / 'calibration-execution.json', baseline_folder / 'fixed-policy-calibration-report.json'
    if read(baseline_execution)['state'] != 'completed' or read(baseline_execution)['completed_frames'] != 122 or read(baseline_execution)['report_sha256'] != sha(baseline_report) or choice['before_same_cpu']['report_sha256'] != sha(baseline_report):
        raise ValueError('One completed unchanged-author same-CPU122 baseline required')
    paths.extend([baseline_execution, baseline_report])
    audit = read(public_audit_path)
    if audit['passed'] is not True:
        raise ValueError('Independent public recording count audit must pass')
    audited = {Path(row['path']).resolve(): row for row in audit['recordings']}
    summary_path = CPU / 'videos/five-complete-video-summary.json'
    summary = read(summary_path)
    if [row['sequence'] for row in summary] != PUBLIC_NAMES:
        raise ValueError('All five actual complete public reports/media required')
    paths.append(summary_path)
    videos, presets = [], []
    for item in summary:
        name = item['sequence']; report_path, media = local(item['report']), local(item['video'])
        report, execution_path = read(report_path), CPU / 'videos' / (name + '-execution.json')
        execution = read(execution_path)
        if (execution['state'] != 'completed' or execution['report_sha256'] != sha(report_path) or execution['video_sha256'] != sha(media)
            or sha(media) != report['derived_video']['sha256'] or report['global_choice_sha256'] != sha(choice_path)
            or report['selected_step'] != 900 or report['purpose'] != 'failure_diagnostic' or report['candidate_status'] != 'FAILED CALIBRATION'
            or report['independent_accuracy_credit'] is not False or len(report['frames']) != report['source']['original_camera_frames']
            or report['derived_video']['encoded_frames'] != len(report['frames']) or report['derived_video']['all_original_microsecond_pts_exact'] is not True):
            raise ValueError('Actual full public FAILED900 media/report identity differs')
        actual_cpu(report)
        audited_row = audited.get(report_path.resolve())
        if not audited_row or audited_row['passed'] is not True or audited_row['sha256'] != sha(report_path) or audited_row['errors']:
            raise ValueError('Each new public report requires a matching passed independent audit')
        replay = report['source']['lossless_replay_input']; source = local(replay['path'], ROOT / 'datasets/construction')
        if (source == media or source.suffix.lower() != '.nut' or replay['codec'] != 'ffv1'
            or sha(source) != replay['sha256'] or replay['original_camera_exposures'] != len(report['frames'])
            or replay['all_opencv_bgr_pixels_equal_raw_gray_to_bgr'] is not True):
            raise ValueError('Source preset must use verified original lossless exposures, never annotated video')
        label = f'Construction CPU FAILED900 · {name} / original camera input'
        total = report['summary']
        description = (f'FAILED reused development case: {total["localized_matches"]} publisher-box matches, '
            f'{total["missed_defects"]} misses, {total["false_alerts"]} unmatched boxes atIoU.50; '
            'unmatched does not mean independently verified material false alarm. No video pixelGT. '
            + ('POST-HOC previously known-hit case; no independent credit. ' if name == 'spalling_76' else ''))
        videos.append({'id': 'round4-cpu-after-' + name, 'label': f'CPU AFTER · {name} / FAILED900 research',
            'path': rel(media), 'evidence_path': rel(report_path), 'purpose': 'failure_diagnostic',
            'independent_accuracy_credit': False, 'issue_summary': description, 'note': description + NOTE,
            'attribution': 'Gamage et al., ev-CIVIL (DTU), CC BY4.0; external publisher box annotations; actual local masks added.',
            'source_url': 'https://doi.org/10.11583/DTU.25924228', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
            'source_preset_label': label})
        presets.append(preset(label, source))
        paths.extend([report_path, media, source, execution_path])
    private_plan_path, private_execution_path = PRIVATE / 'plan-before-any-private-after-network.json', PRIVATE / 'execution.json'
    private_audit_path = PRIVATE / 'independent-native-evidence-audit.json'
    private_plan, execution, private_audit = map(read, [private_plan_path, private_execution_path, private_audit_path])
    if (execution['state'] != 'completed' or execution['private_source_calls'] != 878 or execution['initialization_calls'] != 3
        or execution['old_model_calls'] != 0 or execution['plan_sha256'] != sha(private_plan_path)
        or private_plan['selected_step'] != 900 or private_plan['candidate_passed_public_cal_guards'] is not False
        or private_plan['global_choice_sha256'] != sha(choice_path) or private_plan['model']['sha256'] != MODEL_SHA
        or private_audit['all_checks_passed'] is not True or private_audit['failures'] or private_audit['original_frames'] != 878
        or private_audit['plan_sha256'] != sha(private_plan_path) or len(private_audit['videos']) != 4):
        raise ValueError('Complete878 private AFTER and its independent native audit required')
    inventory_path = ROOT / 'output/evaluation/user-construction-videos-round4/original-video-inventory.json'
    inventory = read(inventory_path)
    completed = {row['source']: row for row in execution['completed_videos']}
    for original in inventory:
        stem = Path(original['file']).stem; report_path = PRIVATE / stem / 'after-public-cal-choice.json'; report = read(report_path)
        proof = next(row for row in private_audit['videos'] if row['source'] == stem)
        media = local(report['derived_video']['path'], PRIVATE / stem)
        source = Path(report['source']['original_path']).resolve()
        if (source != Path(original['file']).resolve() or sha(source) != original['sha256']
            or report['source']['original_camera_frames'] != original['decoded_frames'] or len(report['frames']) != original['decoded_frames']
            or report['plan_sha256'] != sha(private_plan_path) or report['selected_step'] != 900 or report['candidate_passed_public_cal_guards'] is not False
            or report['independent_accuracy_credit'] is not False or report['source']['sha256'] != original['sha256']
            or proof['report_sha256'] != sha(report_path) or proof['video_sha256'] != sha(media)
            or completed[stem]['report_sha256'] != sha(report_path) or completed[stem]['video_sha256'] != sha(media)):
            raise ValueError('Private original/report/media/audit identity differs')
        actual_cpu(report)
        if any(row['evaluated'] is not False or row['expected'] != [] or any(p['evaluation_status'] != 'unscored' for p in row['predictions']) for row in report['frames']):
            raise ValueError('Private outputs must remain unscored without fabricated GT')
        if any(report['summary'][key] is not None for key in ['localized_matches', 'missed_defects', 'false_alerts', 'annotated_frames']):
            raise ValueError('No private accuracy claims allowed')
        label = f'Owner CPU FAILED900 · {stem} / original private input'
        videos.append({'id': 'round4-cpu-after-' + stem.lower().replace('_', '-'),
            'label': f'Owner CPU AFTER · {stem} / FAILED900 unscored', 'path': rel(media), 'evidence_path': rel(report_path),
            'purpose': 'failure_diagnostic', 'independent_accuracy_credit': False, 'private_local_input': True,
            'issue_summary': report['issue_summary'], 'note': 'Every prediction unscored; no private frame GT. ' + NOTE,
            'attribution': 'Owner-provided private local recording; no redistribution license inferred. Local only, never training or upload.',
            'source_preset_label': label})
        presets.append(preset(label, source, private=True))
        paths.extend([report_path, media, source])
    pixel_execution = CPU / 'fixed55-pixel-diagnostic/execution.json'
    pixel_report = CPU / 'fixed55-pixel-diagnostic/paired55-semantic-pixel-report.json'
    pixel_protocol = CPU / 'fixed55-pixel-diagnostic/execution-protocol-before-pixel-network.json'
    if (read(pixel_execution)['state'] != 'completed' or read(pixel_execution)['completed_model_photos'] != 110
        or read(pixel_execution)['report_sha256'] != sha(pixel_report)
        or read(pixel_protocol)['global_choice_sha256'] != sha(choice_path) or read(pixel_protocol)['selected_step'] != 900
        or read(pixel_protocol)['both_models_same_cpu'] is not True or read(pixel_report)['before_after']['after']['model']['sha256'] != MODEL_SHA):
        raise ValueError('Fixed55 both-CPU diagnostic must finish before integration')
    paths.extend([private_plan_path, private_execution_path, private_audit_path, inventory_path, pixel_execution, pixel_report, pixel_protocol])
    metadata = read(MODEL.with_suffix('.provenance.json'))
    spec = {'file': MODEL.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': MODEL_SHA,
        'wrapper_sha256': sha(wrapper), 'classes': ['crack', 'spalling'], 'decoder': 'cubit_yolo_seg', 'input_size': 640,
        'capabilities': ['surface_crack', 'surface_spalling'], 'class_capabilities': {'crack': 'surface_crack', 'spalling': 'surface_spalling'},
        'confidence': .25, 'class_confidence': {'crack': .25, 'spalling': .25}, 'nms_iou': .45,
        'exposure_policy': 'low_light_gain', 'mask_rendering': 'masks', 'required_execution': REQUIRED,
        'source': metadata['source'], 'license': metadata['license'], 'diagnostic_only': True, 'experimental': True,
        'display_name': 'CPU crack/spall masks · FAILED900 research',
        'validation_scope': 'No checkpoint passed complete122 CPU calibration guards. Exact900 failure diagnostic; actual native masks retained. Private outputs unscored; no reliable humidity/fastener/structural/SBC assessment or15FPS qualification.'}
    if len(videos) != 9 or len(presets) != 9:
        raise ValueError('Exactly nine AFTER cases and nine matching original-input presets required')
    return {'model_variant': VARIANT, 'spec': spec, 'videos': videos, 'presets': presets,
        'input_pins': {str(path): sha(path) for path in paths}, 'public_count_audit': str(public_audit_path),
        'no_promotion_no_defaults_no_capability_changes': True}


def prepare(public_audit):
    value = build(public_audit)
    value['existing_config_pins'] = {rel(path): sha(path) for path in CONFIGS}
    value['proposed_videos'] = append_exact(read(CONFIGS[0]), value['videos'], 'id')
    value['proposed_presets'] = append_exact(read(CONFIGS[1]), value['presets'], 'label')
    value['asset_source'] = '"""Data-only FAILED CPU research asset; no model loading or defaults."""\nVARIANT = ' + repr(VARIANT) + '\nSPEC = ' + pprint.pformat(value['spec'], width=110, sort_dicts=True) + '\n'
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(PLAN, value)
    write_new(OUT / 'construction-videos.proposed.json', value['proposed_videos'])
    write_new(OUT / 'public-presets.proposed.json', value['proposed_presets'])
    with (OUT / 'construction_round4_cpu_asset.proposed.py').open('x', encoding='utf8') as stream:
        stream.write(value['asset_source'])
    print('Prepared nine FAILED diagnostics only:', PLAN, sha(PLAN))


def apply():
    plan = read(PLAN)
    current = build(Path(plan['public_count_audit']))
    if any(plan[key] != value for key, value in current.items()):
        raise ValueError('Reviewed completed evidence/audits/spec changed')
    for relative, digest in plan['existing_config_pins'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Existing registry changed; preserve reviewed append plan')
    backup = OUT / 'original-config-backups'
    backup.mkdir(exist_ok=False)
    for path in CONFIGS:
        target = backup / path.name
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
        if sha(target) != plan['existing_config_pins'][rel(path)]:
            raise ValueError('Byte-exact registry backup failed')
    # This is the sole new runtime-tree file: static data, never dispatcher edits.
    if ASSET.exists() and ASSET.read_text(encoding='utf8') != plan['asset_source']:
        raise ValueError('Refuse changed existing CPU asset data')
    if not ASSET.exists():
        with ASSET.open('x', encoding='utf8') as stream:
            stream.write(plan['asset_source'])
    sys.path.insert(0, str(ROOT))
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    runtime = ROOT / 'edge_runtime/profiles/construction_model.py'
    if DAMAGE_VARIANTS.get(VARIANT) != plan['spec'] or 'required_execution' not in runtime.read_text(encoding='utf8'):
        raise ValueError('Root must first install/review the import mapping and fail-closed required_execution guard; no configs changed')
    for path, value in zip(CONFIGS, [plan['proposed_videos'], plan['proposed_presets']]):
        temporary = path.with_name(path.name + '.round4-cpu-staging')
        with temporary.open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
        os.replace(temporary, path)
    write_new(OUT / 'applied.json', {'plan_sha256': sha(PLAN), 'new_videos': 9, 'new_raw_presets': 9,
        'registered_failed_variant': VARIANT, 'runtime_mapping_guard_source_sha256': sha(runtime),
        'config_sha256': {rel(path): sha(path) for path in CONFIGS}, 'no_default_or_capability_edits': True})
    print('Appended9 FAILED diagnostics/raw presets; existing BEFORE/defaults preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'apply'], required=True)
    parser.add_argument('--public-count-audit', type=Path)
    parser.add_argument('--root-integration-release', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.public_count_audit is None:
            parser.error('Exact completed independent public count-audit JSON required')
        prepare(args.public_count_audit.resolve())
    else:
        if not args.root_integration_release:
            parser.error('Separate root integration release required; preparation is preview-only')
        apply()
