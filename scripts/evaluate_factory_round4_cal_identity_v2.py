"""Metadata-only CAL identity correction; immutable Round4 fit/search stay intact.

The protected CAL digest uses historical JSON whitespace. The original Round4
evaluator accidentally called its trainer's compact serializer, and stopped
before constructing any CAL model. This wrapper changes only that serializer.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'output/optimization/factory-round4-supervised'
RECEIPT = OUT / 'CAL-identity-v2-correction-before-CAL.json'
DECISION = OUT / 'parent-approved-CAL-identity-v2-decision.json'
OLD = ROOT / 'scripts/evaluate_factory_round4.py'
OLD_SHA = '34bec29e967394e46a0f7503501749244659589723299d6155e3dda71afce160'
EXPECTED_CAL = '7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07'
REFERENCE = ROOT / 'output/optimization/factory-round3l9-learned-partitioned/checkpoint200-calibration.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def protected_cal_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def approval():
    decision = read(DECISION)
    if decision.get('approved_CAL_identity_metadata_only_recovery') is not True:
        raise ValueError('Root exact metadata-only recovery approval required')
    if decision.get('original_evaluator_sha256') != OLD_SHA or sha(OLD) != OLD_SHA:
        raise ValueError('Immutable executed evaluator changed')
    if decision.get('wrapper_sha256') != sha(Path(__file__)):
        raise ValueError('Root approval must bind the exact recovery wrapper')
    return decision


def record():
    approval()
    if RECEIPT.exists():
        raise ValueError('Preserve recorded correction')
    old_report = read(REFERENCE)
    rows = old_report['calibration_source_identity']['ordered_sources']
    historical = protected_cal_identity(rows)
    compact = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if len(rows) != 80 or sum(row['defective'] for row in rows) != 10 or historical != EXPECTED_CAL or historical == compact:
        raise ValueError('Historical protected CAL contract proof failed')
    import cv2
    from train_factory_surface import samples
    split = read(OUT / 'protocol-before-fit.json')['split_items']['calibration']
    base = ROOT / 'datasets/factory/KolektorSDD'
    actual = []
    for image, truth, defective in samples(base, split):
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError('Current protected source decode failed')
        actual.append({'image': str(image.relative_to(base)), 'defective': bool(defective),
                       'image_sha256': sha(image), 'mask_sha256': sha(image.with_name(image.stem + '_label.bmp')),
                       'native_image_shape': list(frame.shape), 'native_mask_shape': list(truth.shape)})
    if actual != rows:
        raise ValueError('Current ordered CAL paths/pixel hashes/GT hashes/native geometry differs from protected reference')
    for step in [300, 600, 900]:
        if any((OUT / f'checkpoint{step}-{suffix}').exists() for suffix in ['CAL-started.json', 'calibration.json', 'scorecache.npz', 'terminal.json']):
            raise ValueError('Recovery must precede any CAL forward/result')
        if read(OUT / f'checkpoint{step}-numeric.json')['all_checks_passed'] is not True:
            raise ValueError('All3 original numeric PASS reports must exist unchanged')
    archive = OUT / 'failed-pre-CAL-identity-v1'
    archive.mkdir()
    failed_log = OUT / 'calibrate-all3-console.log'
    for source in [OLD, failed_log]:
        target = archive / source.name
        target.write_bytes(source.read_bytes())
        if sha(target) != sha(source):
            raise ValueError('Failure archive mismatch')
    sources = [Path(__file__), ROOT / 'tests/test_factory_round4_cal_identity.py',
               OLD, ROOT / 'scripts/factory_stfpm_common.py', ROOT / 'scripts/train_factory_surface.py']
    executed = OUT / 'executed-CAL-identity-v2-sources'
    for source in sources:
        target = executed / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        if sha(target) != sha(source):
            raise ValueError('Recovery source archive mismatch')
    source_pins = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    manifest = executed / 'manifest.json'
    with manifest.open('x', encoding='utf8') as stream:
        json.dump({'source_pins': source_pins, 'original_fit_protocol_sha256': sha(OUT / 'protocol-before-fit.json')}, stream, indent=2)
    value = {'correction': 'Only historical protected CAL JSON identity serialization; no fit/weights/data/grid/rank/config/cutoff change',
             'parent_decision_sha256': sha(DECISION), 'original_evaluator_sha256': OLD_SHA,
             'source_pins': source_pins,
             'reference_report_sha256': sha(REFERENCE), 'historical_CAL_identity_sha256': historical,
             'wrong_compact_identity_sha256': compact, 'CAL_source_count': 80, 'CAL_defective_count': 10,
             'current_ordered_CAL_content_exactly_equals_protected_reference': True,
             'ordered_CAL_sources': actual,
             'all_three_numeric_PASS_preserved': True, 'CAL_model_initializations_before_correction': 0,
             'preserved_numeric_reports': {str(step): sha(OUT / f'checkpoint{step}-numeric.json') for step in [300, 600, 900]},
             'failed_log_sha256': sha(failed_log), 'failure_preserved_path': str(archive.relative_to(ROOT)),
             'failure_archive_pins': {str(path.relative_to(ROOT)): sha(path) for path in archive.iterdir()},
             'executed_source_archive': str(executed.relative_to(ROOT)),
             'executed_source_archive_manifest_sha256': sha(manifest),
             'original_fit_protocol_unchanged': True, 'no_new_training': True}
    with RECEIPT.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({'receipt': str(RECEIPT), 'sha256': sha(RECEIPT), 'proof': historical}, indent=2))


def verify():
    approval()
    receipt = read(RECEIPT)
    if receipt['parent_decision_sha256'] != sha(DECISION) or receipt['historical_CAL_identity_sha256'] != EXPECTED_CAL or sha(REFERENCE) != receipt['reference_report_sha256']:
        raise ValueError('Recovery decision/protected reference changed')
    rows = receipt['ordered_CAL_sources']
    if (rows != read(REFERENCE)['calibration_source_identity']['ordered_sources'] or protected_cal_identity(rows) != EXPECTED_CAL
        or len(rows) != 80 or sum(row['defective'] for row in rows) != 10 or receipt['CAL_source_count'] != 80 or receipt['CAL_defective_count'] != 10
        or receipt['current_ordered_CAL_content_exactly_equals_protected_reference'] is not True or receipt['all_three_numeric_PASS_preserved'] is not True
        or receipt['CAL_model_initializations_before_correction'] != 0 or receipt['original_evaluator_sha256'] != OLD_SHA):
        raise ValueError('Recovery must retain exact protected CAL identity and zero prior CAL initialization')
    for step in [300, 600, 900]:
        path = OUT / f'checkpoint{step}-numeric.json'
        if sha(path) != receipt['preserved_numeric_reports'][str(step)] or read(path)['all_checks_passed'] is not True:
            raise ValueError('Previously passed numeric report changed')
    manifest = ROOT / receipt['executed_source_archive'] / 'manifest.json'
    if sha(manifest) != receipt['executed_source_archive_manifest_sha256'] or read(manifest)['source_pins'] != receipt['source_pins']:
        raise ValueError('Recovery executed source manifest differs')
    for relative, digest in receipt['source_pins'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Recorded recovery source changed')
        if sha(ROOT / receipt['executed_source_archive'] / relative) != digest:
            raise ValueError('Recovery executed source archive changed')
    for relative, digest in receipt['failure_archive_pins'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Preserved failure archive changed')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['record-correction', 'calibrate-all3', 'freeze'], required=True)
    parser.add_argument('--root-released', action='store_true')
    args = parser.parse_args()
    if args.stage == 'record-correction':
        record()
        return
    if not args.root_released:
        raise ValueError('Separate root GPU release required')
    verify()
    import evaluate_factory_round4 as original
    # Both affected call sites validate the same historical CAL80 identity.
    # Original fit verification, all3 numerics, search and deployment stay exact.
    original.identity = protected_cal_identity
    original_write = original.write
    def write_with_recovery_lineage(path, value):
        if path.name.endswith('-calibration.json') or path.name == 'winner-frozen-before-root-capacity.json':
            value = dict(value, CAL_identity_v2_correction_sha256=sha(RECEIPT),
                         CAL_identity_v2_wrapper_source_sha256=sha(Path(__file__)))
        return original_write(path, value)
    original.write = write_with_recovery_lineage
    sys.argv = [str(OLD), '--stage', args.stage, '--root-released']
    original.main()


if __name__ == '__main__':
    main()
