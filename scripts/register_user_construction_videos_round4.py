"""Register local owner-video diagnostics without claiming accuracy or a new detector."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/user-construction-videos-round4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = OUT / 'local-registration-v1.json'
    if receipt.exists():
        raise ValueError('Preserve previous registration receipt')
    plan = json.loads((OUT / 'baseline-plan-v2-before-inference.json').read_text())
    inventory = json.loads((OUT / 'original-video-inventory.json').read_text())
    videos_path = ROOT / 'config/construction-videos.json'
    presets_path = ROOT / 'config/public-presets.json'
    capabilities_path = ROOT / 'config/construction-capabilities.json'
    paths = [videos_path, presets_path, capabilities_path]
    backup = OUT / 'config-before-local-registration'
    backup.mkdir()
    before = {}
    for path in paths:
        before[path.relative_to(ROOT).as_posix()] = digest(path)
        (backup / path.name).write_bytes(path.read_bytes())
    videos = json.loads(videos_path.read_text(encoding='utf-8'))
    presets = json.loads(presets_path.read_text(encoding='utf-8'))
    capabilities = json.loads(capabilities_path.read_text(encoding='utf-8'))
    titles = {
        'VID-20261020-WA0000': 'Owner wall crack',
        'VID-20261020-WA0001': 'Owner fastener/anchorage issue · detector unsupported',
        'VID-20261020-WA0002': 'Owner wall humidity report · visible appearance only',
        'VID_20261020_075653953': 'Owner upper-wall cracks',
    }
    ids = {}
    for source in inventory:
        stem = Path(source['file']).stem
        identifier = 'owner-round4-' + stem.lower().replace('_', '-')
        assert not any(row['id'] == identifier for row in videos)
        ids[stem] = identifier
        folder = OUT / stem
        report_path = folder / 'before-three-models-v2.json'
        report = json.loads(report_path.read_text())
        assert report['source']['sha256'] == digest(Path(source['file']))
        assert all(report['summary'][key] is None for key in
                   ['annotated_frames', 'localized_matches', 'missed_defects', 'false_alerts'])
        labels = {}
        for name, model in plan['models'].items():
            label = f"Owner local video · {stem} / {name} · unscored research"
            assert not any(row['label'] == label for row in presets)
            labels[name] = label
            presets.append({
                'label': label, 'profile': 'construction', 'path': source['file'],
                'source_kind': 'file', 'input_fps': 15, 'deadline_ms': 500,
                'preview_fps': 8, 'video_mode': 'every_frame',
                'purpose': 'failure_diagnostic', 'independent_accuracy_credit': False,
                'private_local_input': True, 'profile_options': model['options'],
                'note': 'Original owner video; local only, diagnostic only, never training. '
                        '15FPS is a requested replay rate, not measured model capacity. '
                        'No verified frame annotations, moisture measurement or fastener inspection.',
            })
        preferred = 'broad_masks' if stem.endswith('WA0002') else 'author_masks'
        video = folder / 'before-three-models-v2.mp4'
        assert video.is_file()
        videos.append({
            'id': identifier, 'label': titles[stem] + ' / three-model BEFORE',
            'path': video.relative_to(ROOT).as_posix(),
            'evidence_path': report_path.relative_to(ROOT).as_posix(),
            'purpose': 'failure_diagnostic', 'independent_accuracy_credit': False,
            'private_local_input': True,
            'issue_summary': report['issue_summary'],
            'note': 'All original frames and timing; RAW / crack-spall masks / wall boxes / broad masks. '
                    'Every prediction is unscored. Empty output is not an inspection pass. '
                    'The source button loads one matching model, not the four-pane export. '
                    'Fastener adequacy and measured humidity are unsupported.',
            'attribution': 'Owner-provided private local camera recording; local model overlays added. '
                           'No public redistribution license is inferred.',
            'source_preset_label': labels[preferred],
        })
    fastener = next(row for row in capabilities['categories'] if row['id'] == 'finish_installation_fault')
    assert fastener['model_status'] == 'unsupported' and not fastener.get('model_ref')
    fastener['label'] = 'Fastener/anchorage and other installation defects'
    fastener['condition'] = 'Owner-reported incorrect fastening or another installation detail that differs from the required design.'
    fastener['validation_scope'] = ('No installed model verifies screw engagement, torque, anchor embedment, '
                                    'required fastening arrangement or capacity. Object/crack detections do not establish fastening adequacy.')
    fastener['video_evidence'].append({
        'case_id': ids['VID-20261020-WA0001'],
        'scope': 'Owner-reported fastening issue; three existing damage models only, unscored. Enable failure diagnostics to view.',
    })
    for path, value in zip(paths, [videos, presets, capabilities]):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    receipt.write_text(json.dumps({
        'before': before, 'after': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        'recording_ids': ids, 'new_original_video_presets': 12,
        'private_inputs_uploaded': False, 'trained_on_private_inputs': False,
        'accuracy_credit': False, 'new_fastener_detector': False,
    }, indent=2) + '\n')
    print(receipt)


if __name__ == '__main__':
    main()
