"""Failed-only registration proposal and executed-source snapshot; no NN/config edit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/construction-cubit-author-v2'
VIDEO = OUT / 'videos'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    candidate_path = OUT / 'candidate-protocol-before-network.json'
    candidate = read(candidate_path)
    video_path = VIDEO / 'fixed-five-video-protocol-before-forward.json'
    video_protocol = read(video_path)
    result = read(OUT / 'fixed-policy-calibration-report.json')
    if result['all_minimum_checks_pass']:
        raise RuntimeError('This proposal is specifically failed diagnostic registration')
    model = ROOT / result['model']['model_path']
    if sha(model) != result['model']['sha256']:
        raise RuntimeError('Frozen model differs')
    metadata = read(model.with_suffix('.provenance.json'))
    summaries = read(VIDEO / 'five-complete-video-summary.json')
    names = ['crack_89', 'crack_129', 'spalling_75', 'ros_crack_21', 'spalling_76']
    if [r['sequence'] for r in summaries] != names:
        raise RuntimeError('Complete original five cases required')
    if any(candidate['pinned_files'][key] != video_protocol['pinned_files'][key] for key in set(candidate['pinned_files']) & set(video_protocol['pinned_files'])):
        raise RuntimeError('Candidate/video dependency identities disagree')
    all_pins = candidate['pinned_files'] | video_protocol['pinned_files']
    archive = OUT / 'executed-source-archive'
    if archive.exists():
        raise RuntimeError('Preserve previous snapshot/proposal')
    archive.mkdir()
    records = []
    for relative, digest in all_pins.items():
        source = ROOT / relative
        if sha(source) != digest:
            raise RuntimeError('Executed candidate dependency changed before snapshot: ' + relative)
        if source.suffix != '.py':
            continue
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        records.append({'original_path': relative, 'archived_path': target.relative_to(ROOT).as_posix(), 'sha256': digest})
    write(archive / 'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Executed Python source bytes checked against approved candidate/video protocols, before root registration changes.',
            'candidate_protocol_sha256': sha(candidate_path), 'video_protocol_sha256': sha(video_path), 'records': records})
    spec = {'file': model.relative_to(ROOT / 'models/construction').as_posix(), 'sha256': sha(model),
            'wrapper_sha256': sha(ROOT / 'edge_runtime/profiles/construction_cubit.py'),
            'decoder': 'cubit_yolo_seg', 'input_size': 640, 'classes': ['crack', 'spalling'],
            'capabilities': ['surface_crack', 'surface_spalling'],
            'class_capabilities': {'crack': 'surface_crack', 'spalling': 'surface_spalling'},
            'confidence': .25, 'class_confidence': {'crack': .25, 'spalling': .25}, 'exposure_policy': 'low_light_gain',
            'source': metadata['source'], 'license': metadata['license'], 'diagnostic_only': True, 'experimental': True,
            'display_name': 'Author crack/spall masks · FAILED camera calibration',
            'validation_scope': 'Reused122 CAL: crack12/125+81unmatched, spall0/38 (abstains), thin7/62,5/10negativeflags. Five fixed genuine development recordings; source checkpoint training overlap unknown. Actual prototype masks, no video pixelGT/15FPS/SBC qualification.'}
    recordings, presets = [], []
    for row in summaries:
        name = row['sequence']
        report_path = ROOT / row['report']
        report = read(report_path)
        media = ROOT / row['video']
        receipt = read(VIDEO / (name + '-execution.json'))
        if receipt['state'] != 'completed' or receipt['report_sha256'] != sha(report_path) or receipt['video_sha256'] != sha(media) or report['selection_protocol_sha256'] != sha(video_path):
            raise RuntimeError('Only complete bound original media/report can be proposed')
        identity = 'cubit-author-v2-' + name.replace('_', '-')
        posthoc = name == 'spalling_76'
        meaning = 'surface material loss' if name.startswith('spalling') else 'night crack' if name.startswith('ros') else 'thin crack' if name == 'crack_89' else 'surface crack'
        summary = report['summary']
        note = f'FAILED author-mask research: {summary["localized_matches"]} box matches, {summary["missed_defects"]} misses, {summary["false_alerts"]} unmatched boxes. All original exposures and actual masks retained; no pixel GT or independent accuracy credit.'
        if posthoc:
            note = 'POST-HOC known-hit source chosen previously from old-model results. ' + note
        label = f'CUBIT author masks · {meaning} · FAILED research' + (' · POST-HOC' if posthoc else '')
        preset_label = label + ' / original lossless video'
        recordings.append({'id': identity, 'label': label, 'path': row['video'], 'evidence_path': row['report'],
              'purpose': 'failure_diagnostic', 'issue_summary': note, 'note': note,
              'source_preset_label': preset_label, 'source_url': 'https://doi.org/10.11583/DTU.25924228',
              'license_url': 'https://creativecommons.org/licenses/by/4.0/',
              'attribution': report['source']['authors'] + '; ev-CIVIL CC BY4.0. Local actual model masks and separate publisher boxes added.',
              'model_sha256': sha(model), 'video_sha256': sha(media), 'evidence_sha256': sha(report_path),
              'independent_accuracy_credit': False})
        replay = report['source']['lossless_replay_input']
        if Path(replay['path']).suffix.lower() != '.nut':
            raise RuntimeError('Expected verified original lossless video container')
        presets.append({'label': preset_label, 'profile': 'construction', 'path': replay['path'], 'source_kind': 'file',
              'input_fps': 15, 'deadline_ms': 500, 'preview_fps': 8, 'video_mode': 'every_frame',
              'purpose': 'failure_diagnostic', 'independent_accuracy_credit': False,
              'profile_options': {'mode': 'damage', 'damage_variant': 'cubit_author_v2_failed',
                  'confidence': .25, 'class_confidence': {'crack': .25, 'spalling': .25},
                  'mask_rendering': 'masks', 'exposure_policy': 'low_light_gain', 'engine': 'openvino',
                  'openvino_device': 'GPU', 'cpu_threads': 1, 'confirm_frames': 1,
                  'show_context_objects': False, 'stage': 'structure'},
              'note': note + ' Requested15FPS source replay is not measured capacity; model input is lossless original video.'})
    target = OUT / 'integration-proposal.json'
    write(target, {'schema_version': 1, 'status': 'PROPOSAL ONLY; parent owns registration; failed research must not become automatic default or qualified capability.',
          'model_variants': {'cubit_author_v2_failed': spec}, 'model_metadata_sha256': sha(model.with_suffix('.provenance.json')),
          'source_snapshot_manifest_sha256': sha(archive / 'manifest.json'), 'model_calibration_report_sha256': sha(OUT / 'fixed-policy-calibration-report.json'),
          'recordings': recordings, 'presets': presets, 'category_changes': [],
          'calibration_summary': result['summary'], 'geometric_thin': result['geometric_thin'],
          'all_minimum_checks_pass': False, 'full_profile_parity': True,
          'native_masks': 'Real prototype outputs retained. Existence of mask foreground is not pixel accuracy; publisher video GT has boxes only.',
          'documentation': 'docs/construction-cubit-author-candidate.md'})
    print('SNAPSHOT', sha(archive / 'manifest.json'))
    print('PROPOSAL', sha(target))


if __name__ == '__main__':
    main()
