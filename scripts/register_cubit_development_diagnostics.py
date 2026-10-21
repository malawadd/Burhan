"""Install the audited author-mask research option and its five failed diagnostics."""
import hashlib
import json
import pprint
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/construction-cubit-author-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    proposal_path = OUT / 'integration-proposal.json'
    proposal = read(proposal_path)
    assert sha(proposal_path) == 'deb71b1fc56a6e0f268459afc6e98df069bec767d01a3a41f54575d6a4948230'
    audit_path = OUT / 'videos/root-independent-five-video-audit.json'
    audit = read(audit_path)
    assert audit['all_checks_passed'] and audit['all_frame_records'] == 683
    assert audit['actual_NN_calls'] == 0
    assert sha(OUT / 'executed-source-archive/manifest.json') == proposal['source_snapshot_manifest_sha256']
    assert sha(OUT / 'fixed-policy-calibration-report.json') == proposal['model_calibration_report_sha256']
    assert set(proposal['model_variants']) == {'cubit_author_v2_failed'}
    for spec in proposal['model_variants'].values():
        assert spec['diagnostic_only'] and spec['experimental']
        assert spec['classes'] == ['crack', 'spalling'] and spec['decoder'] == 'cubit_yolo_seg'
        assert sha(ROOT / 'models/construction' / spec['file']) == spec['sha256']
        assert sha(ROOT / 'edge_runtime/profiles/construction_cubit.py') == spec['wrapper_sha256']
    videos_path = ROOT / 'config/construction-videos.json'
    presets_path = ROOT / 'config/public-presets.json'
    source_path = ROOT / 'edge_runtime/profiles/construction_model.py'
    videos, presets = read(videos_path), read(presets_path)
    assert len(proposal['recordings']) == len(proposal['presets']) == 5
    for video in proposal['recordings']:
        assert video['id'].startswith('cubit-author-v2-') and video['purpose'] == 'failure_diagnostic'
        assert not video['independent_accuracy_credit']
        match = next(r for r in audit['results'] if r['report_sha256'] == video['evidence_sha256'])
        assert match['video_sha256'] == video['video_sha256'] == sha(ROOT / video['path'])
        assert sha(ROOT / video['evidence_path']) == video['evidence_sha256']
        assert not any(r['id'] == video['id'] for r in videos)
        videos.append(video)
    for preset in proposal['presets']:
        assert preset['source_kind'] == 'file' and preset['profile'] == 'construction'
        assert preset['purpose'] == 'failure_diagnostic' and not preset['independent_accuracy_credit']
        assert preset['profile_options']['damage_variant'] == 'cubit_author_v2_failed'
        assert (ROOT / preset['path']).is_file()
        assert not any(r['label'] == preset['label'] for r in presets)
        presets.append(preset)
    source = source_path.read_text(encoding='utf-8')
    marker = 'DAMAGE_VARIANTS = {\n'
    assert source.count(marker) == 1 and "'cubit_author_v2_failed':" not in source
    entries = ''.join('    ' + repr(key) + ': ' + pprint.pformat(spec, width=110).replace('\n', '\n    ') + ',\n'
                      for key, spec in proposal['model_variants'].items())
    source = source.replace(marker, marker + entries, 1)
    compile(source, str(source_path), 'exec')
    receipt_path = OUT / 'registration.json'
    assert not receipt_path.exists()
    for target in [videos_path, presets_path, source_path]:
        with (OUT / ('before-registration-' + target.name)).open('xb') as backup:
            backup.write(target.read_bytes())
    source_path.write_text(source, encoding='utf-8')
    videos_path.write_text(json.dumps(videos, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    presets_path.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump({'proposal_sha256': sha(proposal_path), 'independent_audit_sha256': sha(audit_path),
                   'defaults_changed': False, 'all_five_registered_as_failure_diagnostics': True,
                   'independent_accuracy_credit': False, 'actual_NN_calls': 0}, stream, indent=2)
        stream.write('\n')
    print('Registered failed author-mask research and all five audited complete videos.')


if __name__ == '__main__':
    main()
