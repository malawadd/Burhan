"""Correct E900 descriptive workload before freeze; no model/score/policy change."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3g_native_width'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (DIRECTORY / 'winner-attempt5-native-width-freeze-before-comparison.json').exists():
        raise ValueError('Cannot annotate a protected frozen winner')
    path = DIRECTORY / (NAME + '-calibration.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    row = next(r for r in data['reports'] if r['step'] == 900)
    metadata_path = Path(row['checkpoint']).with_suffix('.onnx.json')
    original = json.loads(metadata_path.read_text(encoding='utf-8'))
    if sha(metadata_path) != row['metadata_sha256']:
        raise ValueError('Metadata drift before descriptive provenance correction')
    before = {'model_sha256': sha(Path(row['checkpoint'])),
              'metadata_sha256': sha(metadata_path),
              'calibration_report_sha256': sha(path),
              'selected_point': row['selected'],
              'raw_predict_ms': row['raw_predict_ms'],
              'elapsed_seconds': row['elapsed_seconds']}
    backup = DIRECTORY / (NAME + '-calibration-before-workload-description-correction.json')
    metadata_backup = DIRECTORY / (NAME + '-step900-metadata-before-workload-description-correction.json')
    if backup.exists() or metadata_backup.exists():
        raise ValueError('Preserved pre-correction evidence already exists')
    shutil.copyfile(path, backup)
    shutil.copyfile(metadata_path, metadata_backup)
    scope = {'contended_calibration': True, 'opencv_threads': 1,
             'no_concurrent_fit_or_other_nn': True,
             'concurrent_workload': ('Project dataset download/hash/contact-sheet QA was active; '
                                     'ordinary host services also remained active.'),
             'raw_latency_capacity_evidence': False,
             'note': ('Descriptive workload correction before global freeze. No concurrent fits '
                      'or other NN, but project dataset I/O/QA was not quiet. Raw model/calibration '
                      'timing does not qualify complete-profile or sustained runtime speed.')}
    row['execution_scope'] = scope
    data['selection']['execution_scope'] = scope
    original['round3_calibration_protocol']['execution_scope'] = scope
    metadata_path.write_text(json.dumps(original, indent=2), encoding='utf-8')
    row['metadata_sha256'] = sha(metadata_path)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    corrected = json.loads(path.read_text(encoding='utf-8'))
    final = next(r for r in corrected['reports'] if r['step'] == 900)
    assert final['selected'] == before['selected_point']
    assert final['elapsed_seconds'] == before['elapsed_seconds']
    assert final['raw_predict_ms'] == before['raw_predict_ms']
    assert sha(Path(row['checkpoint'])) == before['model_sha256']
    report = {'before': before,
              'after_metadata_sha256': sha(metadata_path),
              'after_report_sha256': sha(path),
              'execution_scope': scope,
              'weights_scores_policy_raw_timing_unchanged': True,
              'scope': 'Descriptive provenance correction only; no inference or policy search'}
    target = DIRECTORY / 'attempt5-calibration-workload-description-correction.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
