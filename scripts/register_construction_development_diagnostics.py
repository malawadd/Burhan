"""Register the eight completed, pinned BEFORE/AFTER development recordings."""
import hashlib
import json
from pathlib import Path
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    path = ROOT / 'output/evaluation/construction-round3/four-development-video-integration-proposal.json'
    proposal = json.loads(path.read_text(encoding='utf-8'))
    protocol = ROOT / 'output/evaluation/construction-round3/four-development-video-comparison-protocol.json'
    if sha(protocol) != proposal['case_selection_protocol_sha256'] or proposal['independent_accuracy_credit']:
        raise ValueError('Development comparison identity/scope changed')
    variants = proposal['damage_variants']
    if set(variants) != {'evcivil_r4_head_failed', 'evcivil_r4_real_lowlight_failed'}:
        raise ValueError('Unexpected model registration')
    for key, row in variants.items():
        model = ROOT / 'models/construction' / row['file']
        if not row['diagnostic_only'] or not row['experimental'] or sha(model) != row['sha256']:
            raise ValueError('Failed model must retain its exact identity and diagnostic status')
        if key in DAMAGE_VARIANTS and DAMAGE_VARIANTS[key] != row:
            raise ValueError('Refusing to overwrite changed model registration')
    if len(proposal['recording_rows']) != 8 or len(proposal['public_preset_rows']) != 8:
        raise ValueError('All four sources and both models are required')
    for row in proposal['recording_rows']:
        for field, digest in [('path', 'video_sha256'), ('evidence_path', 'report_sha256')]:
            target = (ROOT / row[field]).resolve()
            if not target.is_relative_to(ROOT) or sha(target) != row[digest]:
                raise ValueError(f'Changed/outside artifact: {row["id"]}/{field}')
        report = json.loads((ROOT / row['evidence_path']).read_text(encoding='utf-8'))
        if report['candidate_status'] != 'FAILED CALIBRATION' or report['selection_protocol_sha256'] != sha(protocol):
            raise ValueError('Recording must retain actual failed calibration and fixed source selection')
    updates = []
    for filename, identity, rows in [
        ('construction-videos.json', 'id', proposal['recording_rows']),
        ('public-presets.json', 'label', proposal['public_preset_rows']),
    ]:
        target = ROOT / 'config' / filename
        current = json.loads(target.read_text(encoding='utf-8'))
        existing = {row[identity]: row for row in current}
        for original in rows:
            row = {**original, 'independent_accuracy_credit': False}
            if row['purpose'] != 'failure_diagnostic':
                raise ValueError('No failed diagnostic may become a success preset')
            if row[identity] in existing and existing[row[identity]] != row:
                raise ValueError(f'Refusing to overwrite changed row: {row[identity]}')
            if row[identity] not in existing:
                current.append(row)
        updates.append((target, json.dumps(current, indent=2, ensure_ascii=False) + '\n'))
    new_models = {key: value for key, value in variants.items() if key not in DAMAGE_VARIANTS}
    if new_models:
        target = ROOT / 'edge_runtime/profiles/construction_model.py'
        source = target.read_text(encoding='utf-8')
        marker = 'DAMAGE_VARIANTS = {\n'
        if source.count(marker) != 1:
            raise ValueError('Unexpected registry source structure')
        entries = ''.join('    ' + repr(key) + ': ' + pprint.pformat(value, width=110).replace('\n', '\n    ') + ',\n'
                          for key, value in new_models.items())
        source = source.replace(marker, marker + entries, 1)
        compile(source, str(target), 'exec')
        updates.append((target, source))
    for target, source in updates:
        target.write_text(source, encoding='utf-8')
    print('Registered eight exact real recordings/raw-source presets and two failed model variants; defaults preserved.')


if __name__ == '__main__':
    main()
