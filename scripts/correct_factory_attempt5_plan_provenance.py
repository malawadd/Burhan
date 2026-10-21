"""Preserve historical E plans; attach observed workload outside declarations."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3g_native_width'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = OUT / (NAME + '-calibration.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    row = next(r for r in data['reports'] if r['step'] == 900)
    meta_path = Path(row['checkpoint']).with_suffix('.onnx.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    old_report = json.loads((OUT / (NAME + '-calibration-before-workload-description-correction.json')).read_text(encoding='utf-8'))
    old_row = next(r for r in old_report['reports'] if r['step'] == 900)
    old_meta = json.loads((OUT / (NAME + '-step900-metadata-before-workload-description-correction.json')).read_text(encoding='utf-8'))
    freeze_path = OUT / 'winner-attempt5-native-width-freeze-before-comparison.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    assert freeze['winner']['step'] == 900
    assert freeze['winner']['metadata_sha256'] == sha(meta_path) == row['metadata_sha256']
    assert freeze['winner']['model_sha256'] == sha(Path(row['checkpoint']))
    assert row['selected'] == old_row['selected'] == meta['calibration_selected_point']
    if (OUT / (NAME + '-calibration-first-description-correction.json')).exists():
        raise ValueError('Prior correction already preserved')
    shutil.copyfile(path, OUT / (NAME + '-calibration-first-description-correction.json'))
    shutil.copyfile(meta_path, OUT / (NAME + '-step900-metadata-first-description-correction.json'))
    scope = row['execution_scope']
    before = {'initial_freeze_sha256': sha(freeze_path),
              'model_sha256': sha(Path(row['checkpoint'])),
              'score_cache_sha256': sha(OUT / (NAME + '-step900-calibration-scoremaps.npz')),
              'protocol_sha256': sha(OUT / (NAME + '-protocol-before-fit.json')),
              'pre_scoring_protocol_sha256': sha(OUT / (NAME + '-calibration-protocol-before-scoring.json'))}
    # Restore exact historical declarations; actual observation is additive.
    row['execution_scope'] = old_row['execution_scope']
    row['observed_execution_scope'] = scope
    data['selection'] = old_report['selection']
    meta['round3_calibration_protocol'] = old_meta['round3_calibration_protocol']
    meta['observed_calibration_execution_scope'] = scope
    meta['workload_provenance_correction'] = {
        'historical_plan_restored_byte_equivalent_as_object': True,
        'observed_scope_is_additive_not_written_before_scoring': True,
        'initial_freeze_sha256': before['initial_freeze_sha256'],
        'weights_scores_policy_raw_timing_unchanged': True}
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    row['metadata_sha256'] = sha(meta_path)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    assert row['selected'] == old_row['selected']
    assert row['raw_predict_ms'] == old_row['raw_predict_ms']
    assert row['elapsed_seconds'] == old_row['elapsed_seconds']
    assert sha(Path(row['checkpoint'])) == before['model_sha256']
    assert sha(OUT / (NAME + '-step900-calibration-scoremaps.npz')) == before['score_cache_sha256']
    assert sha(OUT / (NAME + '-protocol-before-fit.json')) == before['protocol_sha256']
    assert sha(OUT / (NAME + '-calibration-protocol-before-scoring.json')) == before['pre_scoring_protocol_sha256']
    value = {'before': before, 'final_metadata_sha256': sha(meta_path),
             'final_calibration_report_sha256': sha(path),
             'historical_plan_and_observed_scope_separated': True,
             'model_scores_points_timing_protocol_files_unchanged': True,
             'scope': 'Pre-original-scoring provenance correction only; no inference, fitting or selection rerun'}
    (OUT / 'attempt5-additive-workload-provenance-correction.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
