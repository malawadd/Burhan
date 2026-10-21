"""Archive executed wall sources and emit a failed-research integration proposal.

No inference, threshold selection, configuration writes or source modification.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/construction-wall-five-class-v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    protocol = read(OUT / 'evaluation-protocol-before-calibration.json')
    plan_path = ROOT / 'output/training/construction-wall-five-class-v3/approved-fit-plan.json'
    plan = read(plan_path)
    point_path = OUT / 'operating-point.json'
    point = read(point_path)
    report_path = OUT / 'fixed18-diagnostic-report.json'
    report = read(report_path)
    if len(report['records']) != 18 or point['all_minimum_review_guards_pass']:
        raise RuntimeError('This proposal is specifically the complete failed V3 result')
    for stage, key, expected in [('calibration', 'operating_point_sha256', sha(point_path)),
                                 ('diagnostic', 'report_sha256', sha(report_path))]:
        receipt = read(OUT / (stage + '-execution.json'))
        if receipt['state'] != 'completed' or receipt[key] != expected:
            raise RuntimeError('Incomplete or unbound wall result: ' + stage)
    pinned = {
        'scripts/evaluate_mbdd_wall_detector.py': protocol['script_sha256'],
        'scripts/evaluate_mbdd_wall_detector_v3.py': protocol['versioned_wrapper']['wrapper_sha256'],
        'scripts/train_mbdd_wall_detector_v3.py': plan['trainer_sha256'],
        'edge_runtime/profiles/construction_model.py': protocol['construction_model_source_sha256'],
    }
    extras = ['scripts/evaluate_construction_defect_video.py',
              'edge_runtime/profiles/construction_preprocessing.py',
              'scripts/prepare_mbdd_wall_yolo.py',
              'scripts/diagnose_mbdd_wall_training_export.py',
              'scripts/archive_mbdd_wall_v3_result.py']
    archive = OUT / 'executed-source-archive'
    if archive.exists():
        raise RuntimeError('Preserve existing executed-source archive')
    archive.mkdir()
    records = []
    for relative in list(pinned) + extras:
        source = ROOT / relative
        actual = sha(source)
        if relative in pinned and actual != pinned[relative]:
            raise RuntimeError('Executed source changed before archive: ' + relative)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        records.append({'original_path': relative, 'archived_path': target.relative_to(ROOT).as_posix(),
                        'sha256': actual, 'identity_prebound_to_executed_protocol': relative in pinned})
    write(archive / 'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Exact current executed sources. Four core identities checked against pre-execution plans; auxiliary dependencies archived observationally, not retroactively pre-pinned.',
          'records': records, 'evaluation_protocol_sha256': sha(OUT / 'evaluation-protocol-before-calibration.json'),
          'point_sha256': sha(point_path), 'report_sha256': sha(report_path)})
    model = ROOT / 'models/construction/mbdd-wall-five-class-v3/wall-five-class-total16-last12-640.onnx'
    metadata = read(model.with_suffix('.provenance.json'))
    if sha(model) != protocol['model_sha256'] or sha(model.with_suffix('.provenance.json')) != protocol['model_metadata_sha256']:
        raise RuntimeError('Final wall model/metadata changed')
    mapping = protocol['class_capabilities']
    spec = {'file': model.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(model),
            'decoder': 'yolo_detect', 'input_size': 640, 'classes': protocol['classes'],
            'capabilities': protocol['capabilities'], 'class_capabilities': mapping,
            'confidence': min(point['class_confidence'].values()), 'class_confidence': point['class_confidence'],
            'source': metadata['source'], 'license': metadata['license'], 'exposure_policy': 'off',
            'diagnostic_only': True, 'display_name': 'Wall appearance research · FAILED calibration',
            'validation_scope': '181 TRAIN/48 CAL source-block-separated photos; 18 previously exposed diagnostic photos. All five minimum guards fail. No healthy-wall negatives, video, hidden moisture, paint-only material, depth, site-independent or SBC qualification.'}
    write(OUT / 'integration-proposal.json', {
        'schema_version': 1, 'status': 'PROPOSAL ONLY; root owns shared registry. Failed model must not become an automatic default or working capability.',
        'model_variants': {'mbdd_wall_total16_failed': spec},
        'model_metadata_sha256': sha(model.with_suffix('.provenance.json')),
        'source_archive_manifest_sha256': sha(archive / 'manifest.json'),
        'evaluation_protocol_sha256': sha(OUT / 'evaluation-protocol-before-calibration.json'),
        'point_sha256': sha(point_path), 'report_sha256': sha(report_path),
        'calibration': point['by_class'], 'fixed18_diagnostics': report['by_class'],
        'all_minimum_review_guards_pass': False, 'all_actual_serving_parity': point['all_actual_serving_parity'],
        'healthy_wall_fpr': None, 'photo_comparisons': [p.relative_to(ROOT).as_posix() for p in sorted(OUT.glob('*-raw-publisher-model.jpg'))],
        'limits': ['Still photographs remain photos; no fault video was created.',
                   'Leakage labels describe wet/stained appearance, not measured humidity or hidden leaks.',
                   'Abscission describes surface material loss, not verified paint or concrete substrate.',
                   'Bulge is a 2D appearance label, not measured flatness/depth.',
                   'Other-class-absent photos are not certified healthy negatives.'],
    })
    print('ARCHIVED', sha(archive / 'manifest.json'))
    print('PROPOSAL', sha(OUT / 'integration-proposal.json'))


if __name__ == '__main__':
    main()
