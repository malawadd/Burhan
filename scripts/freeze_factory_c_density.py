"""Protect only a strict non-control winner from ALL nine actual C calibrations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import argparse
from factory_c_density_identity import validate_density_artifacts
from factory_c_density_common import verify_protocol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3k_c200_density_f16'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(point):
    return (point['meaningful_iou_0_1_recall'], point['worst_positive_iou'], point['coverage_50pct_recall'],
            point['worst_positive_coverage'], point['micro_pixel_iou'], -point['normal_fp'], point['foreground_pixel_recall'])


def main():
    global NAME
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    validate_density_artifacts(name=NAME)
    protocol_path,_=verify_protocol()
    report_path = OUT / (NAME + '-calibration.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    reports = json.loads(report_path.read_text(encoding='utf-8'))['reports']
    fit_path = OUT / (NAME + '-normal-fit.json')
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    parity = json.loads((OUT / (NAME + '-fitted-gpu-parity.json')).read_text(encoding='utf-8'))
    if not parity['all_checks_passed'] or parity['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Fitted GPU parity prerequisite drift')
    if sorted(row['step'] for row in reports) != protocol['save_at']:
        raise ValueError('All nine fixed candidates must finish before selection')
    if sha(OUT / (NAME + '-normal-bank.npz')) != fit['bank_sha256']:
        raise ValueError('TRAIN-only normal bank changed')
    exports_path=OUT/(NAME+'-fitted-export-manifest.json')
    exports=json.loads(exports_path.read_text(encoding='utf-8'))
    if parity['normal_bank_sha256']!=fit['bank_sha256'] or parity['fitted_export_manifest_sha256']!=sha(exports_path):
        raise ValueError('Parity/bank/all-nine export provenance changed')
    for candidate in exports['candidates']:
        if sha(Path(candidate['model_path']))!=candidate['model_sha256']:
            raise ValueError('A fitted candidate changed after numeric parity')
    guard_path=OUT/(NAME+'-guard-clipping-audit.json')
    guard=json.loads(guard_path.read_text(encoding='utf-8'))
    if guard['bank_sha256']!=fit['bank_sha256']:raise ValueError('Guard diagnostics bank drift')
    choices, identity_sha = [], None
    for row in reports:
        validate_density_artifacts(row,name=NAME)
        path = Path(row['checkpoint'])
        if sha(path) != row['model_sha256']:
            raise ValueError('A calibrated model changed, including a rejected candidate')
        identity = row['calibration_source_identity']
        records = identity['ordered_sources']
        actual = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        if actual != identity['sha256'] or len(records) != 80 or sum(r['defective'] for r in records) != 10:
            raise ValueError('Protected calibration source identity changed')
        if identity_sha is not None and identity_sha != actual:
            raise ValueError('Calibration source changed between candidates')
        if actual!='7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07':
            raise ValueError('Only protected original80 CAL inputs are permitted')
        identity_sha = actual
        if actual!=guard['calibration_source_identity_sha256']:raise ValueError('Calibration changed since guard diagnostics')
        point = row['selected']
        if point is None:
            continue
        meta_path = path.with_suffix('.onnx.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        expected_candidate = next(candidate for candidate in protocol['candidates'] if candidate['id'] == row['step'])
        if sha(meta_path) != row['metadata_sha256'] or meta['model_sha256'] != row['model_sha256']:
            raise ValueError('Calibrated metadata/model drift')
        if point['normal_fp'] > 1 or point['foreground_pixel_precision'] < .65:
            raise ValueError('Fixed final guard violated')
        if meta['training_protocol'] != protocol or meta['normal_density_candidate'] != expected_candidate:
            raise ValueError('Fitted source/fusion protocol changed')
        if meta['normal_density_bank_sha256'] != fit['bank_sha256'] or meta['normal_density_fit_report_sha256'] != sha(fit_path):
            raise ValueError('Normal-bank provenance changed')
        if meta['normal_density_guard_diagnostics_sha256']!=sha(guard_path):raise ValueError('Guard diagnostic provenance changed')
        if meta['calibration_selected_point'] != point or meta['calibration_source_identity_sha256'] != actual:
            raise ValueError('Selected point/source provenance changed')
        selection = meta['round3_calibration_protocol']
        if any(selection[k] != protocol[p] for k, p in [('high_grid', 'raw_high_grid'), ('area_grid', 'area_grid'), ('growth_grid', 'growth_grid'), ('final_rank', 'selection_rank')]):
            raise ValueError('Fixed threshold search/rank changed')
        if (meta['pixel_threshold'], meta['minimum_component_pixels'], meta.get('hysteresis_growth_threshold')) != (point['high'], point['area'], point['growth']):
            raise ValueError('Postprocessing policy changed')
        if meta['input_shape'] != [1, 3, 1280, 512] or meta['output_shape'] != [1, 1, 640, 256]:
            raise ValueError('Actual graph geometry description changed')
        choices.append({'candidate_id': row['step'], 'fusion': expected_candidate,
                        'model_path': str(path), 'model_sha256': row['model_sha256'],
                        'metadata_sha256': row['metadata_sha256'], 'point': point})
    control = next((row for row in choices if row['candidate_id'] == 0), None)
    choices.sort(key=lambda row: rank(row['point']), reverse=True)
    # If the control is ineligible, strict eligibility is itself a gain; no
    # retuning of an old comparison point is used to manufacture the control.
    accepted = bool(choices and choices[0]['candidate_id'] != 0 and
                    (control is None or rank(choices[0]['point']) > rank(control['point'])))
    value = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'attempt': 'C200-density',
             'variant': 'NewC200 sharedfine exact-half normal-density fusion AFTERsemanticmaxpool', 'accepted_for_independent_audit': accepted,
             'winner': choices[0] if accepted else None, 'ranked_candidates': choices,
             'control': control, 'protocol_sha256': sha(protocol_path), 'calibration_report_sha256': sha(report_path),
             'normal_fit_report_sha256': sha(fit_path), 'bank_sha256': fit['bank_sha256'],
             'calibration_source_identity_sha256': identity_sha,
             'comparison_gate': protocol['comparison_gate'], 'capacity_gate': protocol['capacity_gate'],
             'official_sdd2_test_still_sealed': True, 'comparison_images_used_for_selection': 0}
    target = OUT / 'winner-c200-density-freeze-before-comparison.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if old['winner'] != value['winner'] or old['calibration_report_sha256'] != value['calibration_report_sha256']:
            raise ValueError('A different density outcome is already frozen')
        value = old
    else:
        target.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
