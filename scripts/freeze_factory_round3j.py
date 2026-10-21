"""Freeze all three curvature TRAIN-augmentation candidates; no NN/test pixels."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3j_native_curvature'
EXPECTED_CAL_SOURCE_SHA = '7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(row):
    p = row['point']
    return (p['meaningful_iou_0_1_recall'], p['worst_positive_iou'], p['coverage_50pct_recall'],
            p['worst_positive_coverage'], p['micro_pixel_iou'], -p['normal_fp'], p['foreground_pixel_recall'])


def same_search_policy(left, right):
    # Per-checkpoint CPU contention is observational provenance, not a change
    # to the fixed search or ranking. Retain each actual execution scope.
    return ({key: value for key, value in left.items() if key != 'execution_scope'} ==
            {key: value for key, value in right.items() if key != 'execution_scope'})


def validate_report_set(protocol, reports, expected_source_sha=None):
    if protocol['save_at'] != [200, 400, 600] or sorted(row['step'] for row in reports) != [200, 400, 600]:
        raise ValueError('Exactly all three protected200/400/600 calibrations are mandatory')
    identity_sha = None
    for row in reports:
        identity = row['calibration_source_identity']
        sources = identity['ordered_sources']
        digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode('utf-8')).hexdigest()
        if digest != identity['sha256']:
            raise ValueError('Calibration source identity digest is invalid')
        if len(sources) != 80 or sum(bool(source['defective']) for source in sources) != 10 or len({source['image'] for source in sources}) != 80:
            raise ValueError('Protected80-view/10-positive original CAL partition changed')
        if identity_sha is not None and identity_sha != digest:
            raise ValueError('Calibration sources changed across curvature checkpoints')
        identity_sha = digest
    if expected_source_sha is not None and identity_sha != expected_source_sha:
        raise ValueError('Calibration identity is not the original protected80 source image/mask partition')
    return identity_sha


def main():
    protocol_path = DIRECTORY / (NAME + '-protocol-before-fit.json')
    report_path = DIRECTORY / (NAME + '-calibration.json')
    training_path = DIRECTORY / (NAME + '-training.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    training = json.loads(training_path.read_text(encoding='utf-8'))
    reports = json.loads(report_path.read_text(encoding='utf-8'))
    if training['protocol'] != protocol or training['history'][-1]['step'] != 600:
        raise ValueError('The protected fixed600-step fit must finish before global freeze')
    if training['calibration_images_opened'] or training['comparison_images_opened'] or not training['official_sdd2_test_still_sealed']:
        raise ValueError('Training report violates protected data separation')
    for path, field in [(Path(protocol['warmstart']), 'warmstart_sha256'),
                        (ROOT / 'scripts/factory_curvature_sampling.py', 'sampler_sha256'),
                        (ROOT / 'scripts/train_factory_round3j.py', 'wrapper_sha256'),
                        (ROOT / 'scripts/train_factory_round3c.py', 'trainer_helper_sha256'),
                        (ROOT / 'scripts/train_factory_round3d.py', 'source_model_implementation_sha256'),
                        (ROOT / 'scripts/train_factory_round3b.py', 'loss_implementation_sha256'),
                        (DIRECTORY / (NAME + '-pairing-fixtures.json'), 'pairing_fixtures_sha256')]:
        if sha(path) != protocol[field]:
            raise ValueError('Protected training source/fixture identity changed')
    identity_sha = validate_report_set(protocol, reports['reports'], EXPECTED_CAL_SOURCE_SHA)
    selection = reports['selection']
    if (selection['high_grid'], selection['area_grid'], selection['growth_grid'], selection['final_rank']) != (protocol['raw_high_grid'], protocol['area_grid'], protocol['growth_grid'], protocol['selection_rank']):
        raise ValueError('Calibration grid/rank changed')
    choices = []
    for row in reports['reports']:
        path = ROOT / 'models/factory' / f'{NAME}-step{row["step"]}.onnx'
        if Path(row['checkpoint']).resolve() != path.resolve() or sha(path) != row['model_sha256']:
            raise ValueError('Calibration report is not tied to this exact planned checkpoint')
        point = row['selected']
        if point is None:
            continue
        if point['normal_fp'] > 1 or point['foreground_pixel_precision'] < .65:
            raise ValueError('Candidate fails predeclared precision/normal-FP guard')
        metadata_path = path.with_suffix('.onnx.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if sha(metadata_path) != row['metadata_sha256'] or metadata['model_sha256'] != row['model_sha256']:
            raise ValueError('Calibrated metadata/model identity changed')
        if metadata['training_protocol'] != protocol or not same_search_policy(metadata['round3_calibration_protocol'], selection):
            raise ValueError('Calibrated metadata does not preserve training/search protocol')
        if metadata['round3_calibration_protocol']['execution_scope'] != row['execution_scope']:
            raise ValueError('Checkpoint execution provenance differs from its own actual report')
        if metadata['calibration_selected_point'] != point or metadata['calibration_source_identity_sha256'] != identity_sha:
            raise ValueError('Selected point/source identity changed')
        if (metadata['pixel_threshold'], metadata['minimum_component_pixels'], metadata.get('hysteresis_growth_threshold')) != (point['high'], point['area'], point['growth']):
            raise ValueError('Serving policy differs from actual calibrated policy')
        if metadata['input_shape'] != [1, 3, 1280, 512] or metadata['output_shape'] != [1, 1, 640, 256]:
            raise ValueError('Qualified C input/output geometry changed')
        if not metadata['training_only_curvature_augmentation'] or not metadata['evaluation_inputs_untouched_real']:
            raise ValueError('Missing distinction between augmentation and real evaluation evidence')
        choices.append({'step': row['step'], 'model_path': str(path), 'model_sha256': row['model_sha256'],
                        'metadata_sha256': row['metadata_sha256'], 'point': point})
    if not choices:
        raise ValueError('No candidate met the protected calibration guards; preserve failure without winner')
    choices.sort(key=rank, reverse=True)
    value = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'attempt': 'curvature-augmentation',
             'variant': NAME, 'winner': choices[0], 'ranked_candidates': choices,
             'selection_gate': protocol['selection_gate'], 'selection_rank': protocol['selection_rank'],
             'protocol_sha256': sha(protocol_path), 'calibration_report_sha256': sha(report_path),
             'training_report_sha256': sha(training_path), 'calibration_source_identity_sha256': identity_sha,
             'preserved_C_freeze_sha256': sha(DIRECTORY / 'winner-attempt3-phase-max-freeze-before-comparison.json'),
             'comparison_images_used_for_selection': 0, 'official_sdd2_test_still_sealed': True,
             'capacity_gate': protocol['capacity_gate'],
             'scope': 'Iterative exposed-development experiment; TRAIN-only augmentation, real evaluation inputs. No independent-test/production assurance.'}
    target = DIRECTORY / 'winner-curvature-augmentation-freeze-before-comparison.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if old['winner'] != value['winner'] or old['calibration_report_sha256'] != value['calibration_report_sha256'] or old['protocol_sha256'] != value['protocol_sha256']:
            raise ValueError('Refusing to replace a different immutable curvature freeze')
        value = old
    else:
        target.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({'freeze_path': str(target), 'freeze_sha256': sha(target), 'winner': value['winner']}, indent=2))


if __name__ == '__main__':
    main()
