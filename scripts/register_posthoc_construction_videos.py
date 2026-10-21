"""Register all four audited post-hoc illustrations, explicitly as failed research."""
import hashlib
import json
from pathlib import Path
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FOLDER = ROOT / 'output/evaluation/construction-posthoc-known-hit-videos'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def append_exact(rows, row, key):
    old = next((v for v in rows if v[key] == row[key]), None)
    if old is not None and old != row:
        raise ValueError('Refusing to replace a changed registration')
    if old is None:
        rows.append(row)


def main():
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    proposal = read(FOLDER / 'integration-proposal.json')
    audit_path = FOLDER / 'root-independent-media-audit.json'
    audit = read(audit_path)
    if not audit['all_checks_passed'] or audit['all_frame_records'] != 706 or audit['actual_NN_calls'] != 0:
        raise ValueError('Completed independent four-video audit required')
    protocol = FOLDER / 'two-posthoc-known-hit-protocol-before-forward.json'
    if sha(protocol) != proposal['selection_protocol_sha256']:
        raise ValueError('Post-hoc selection protocol changed')
    if set(proposal['new_damage_variants']) != {'evcivil_r4_last4_failed'}:
        raise ValueError('Unexpected research model')
    if proposal['reuse_existing_damage_variant'] not in DAMAGE_VARIANTS:
        raise ValueError('Before model unavailable')
    variants = proposal['new_damage_variants']
    for key, value in variants.items():
        if (not value['diagnostic_only'] or not value['experimental']
                or sha(ROOT / 'models/construction' / value['file']) != value['sha256']):
            raise ValueError('Failed model identity/status changed')
        if key in DAMAGE_VARIANTS and DAMAGE_VARIANTS[key] != value:
            raise ValueError('Changed model registration')
    videos_path, presets_path = ROOT / 'config/construction-videos.json', ROOT / 'config/public-presets.json'
    videos, presets = read(videos_path), read(presets_path)
    proposed = proposal['recordings_and_source_presets']
    if {v['id'] for v in proposed} != {f'posthoc-{tag}-{case}' for tag in ['before', 'after']
                                     for case in ['crack-129', 'spalling-76']}:
        raise ValueError('All four fixed recordings required')
    for row in proposed:
        video, evidence = ROOT / row['video'], ROOT / row['evidence_path']
        report = read(evidence)
        if (sha(video) != row['video_sha256'] or sha(evidence) != row['evidence_sha256']
                or report['independent_accuracy_credit'] or not report['posthoc_known_hit_selection']
                or report['selection_protocol_sha256'] != sha(protocol)):
            raise ValueError('Changed post-hoc evidence')
        match = next(v for v in audit['results'] if v['report_sha256'] == sha(evidence))
        if match['video_sha256'] != sha(video):
            raise ValueError('Independent media audit identity changed')
        counts = report['summary']
        note = (f"POST-HOC known-hit illustration / FAILED research: {counts['localized_matches']} matches, "
                f"{counts['missed_defects']} misses, {counts['false_alerts']} unmatched boxes. "
                'Selected after CAL hits were known; no independent accuracy credit. '
                'Material-loss substrate is unverified; all exposures and failures retained.')
        append_exact(videos, {'id': row['id'], 'label': row['title'], 'path': row['video'],
            'evidence_path': row['evidence_path'], 'purpose': 'failure_diagnostic',
            'issue_summary': note, 'note': note, 'source_url': 'https://doi.org/10.11583/DTU.25924228',
            'license_url': 'https://creativecommons.org/licenses/by/4.0/',
            'attribution': 'Gamage et al., ev-CIVIL, DTU, CC BY4.0. External annotations are separate from model input.',
            'source_preset_label': row['source_preset_label'], 'video_sha256': sha(video),
            'report_sha256': sha(evidence), 'independent_accuracy_credit': False}, 'id')
        source = ROOT / row['source_input']
        if sha(source) != report['source']['lossless_replay_input']['sha256']:
            raise ValueError('Original source preset identity changed')
        append_exact(presets, {'label': row['source_preset_label'], 'profile': 'construction',
            'path': row['source_input'], 'input_fps': 15, 'deadline_ms': 500, 'preview_fps': 8,
            'video_mode': 'every_frame', 'purpose': 'failure_diagnostic',
            'profile_options': {**row['options'], 'engine': 'openvino', 'openvino_device': 'GPU',
                                'cpu_threads': 1, 'confirm_frames': 1},
            'note': note + ' Requested15FPS replay is not a measured capacity claim.',
            'independent_accuracy_credit': False}, 'label')
    source_path = ROOT / 'edge_runtime/profiles/construction_model.py'
    source = source_path.read_text(encoding='utf-8')
    marker = 'DAMAGE_VARIANTS = {\n'
    missing = {key: value for key, value in variants.items() if key not in DAMAGE_VARIANTS}
    if missing:
        if source.count(marker) != 1:
            raise ValueError('Unexpected registry source')
        entries = ''.join('    ' + repr(key) + ': ' + pprint.pformat(value, width=110).replace('\n', '\n    ')
                          + ',\n' for key, value in missing.items())
        source = source.replace(marker, marker + entries, 1)
        compile(source, str(source_path), 'exec')
    receipt = FOLDER / 'registration.json'
    if receipt.exists():
        raise ValueError('Preserve the original registration receipt')
    for target in [videos_path, presets_path, source_path]:
        backup = FOLDER / ('before-registration-' + target.name)
        with backup.open('xb') as stream:
            stream.write(target.read_bytes())
    source_path.write_text(source, encoding='utf-8')
    videos_path.write_text(json.dumps(videos, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    presets_path.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump({'all_four_registered': True, 'independent_accuracy_credit': False,
                   'proposal_sha256': sha(FOLDER / 'integration-proposal.json'),
                   'audit_sha256': sha(audit_path), 'defaults_changed': False,
                   'model_registry_sha256': sha(source_path), 'video_registry_sha256': sha(videos_path),
                   'presets_sha256': sha(presets_path)}, stream, indent=2)
        stream.write('\n')
    print('Registered four complete audited post-hoc videos and raw-source presets as failed research.')


if __name__ == '__main__':
    main()
