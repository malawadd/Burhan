"""Verify nine audited local AFTER recordings and original-input presets, no NN."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/construction-round4-cpu-recovery-v1/local-integration'
BASE = 'http://127.0.0.1:8765'


def get(route):
    with urlopen(BASE + route, timeout=30) as response:
        assert response.status == 200
        return json.load(response)


def main():
    target = OUT / 'localhost-nine-recording-delivery.json'
    if target.exists():
        raise ValueError('Preserve first delivery verification')
    local = json.loads((ROOT / 'config/construction-videos.json').read_text(encoding='utf8'))
    served = get('/api/construction/videos')
    rows = [r for r in local if r['id'].startswith('round4-cpu-after-')]
    assert len(rows) == 9
    records = []
    for row in rows:
        actual = next(v for v in served if v['id'] == row['id'])
        assert actual['available'] and actual['evidence_available']
        report = get(actual['evidence_url'])
        assert report == json.loads((ROOT / row['evidence_path']).read_text())
        if row.get('private_local_input'):
            for key in ['annotated_frames', 'localized_matches', 'missed_defects', 'false_alerts', 'unsupported_annotations']:
                assert report['summary'][key] is None
            assert all(not frame['evaluated'] and not frame['expected'] for frame in report['frames'])
        with urlopen(Request(BASE + actual['url'], method='HEAD'), timeout=30) as response:
            assert response.status == 200
            assert int(response.headers['Content-Length']) == (ROOT / row['path']).stat().st_size
        with urlopen(Request(BASE + actual['url'], headers={'Range': 'bytes=0-31'}), timeout=30) as response:
            assert response.status == 206
            with (ROOT / row['path']).open('rb') as stream:
                assert response.read() == stream.read(32)
        records.append({'id': row['id'], 'report_equals_saved': True, 'head_and_range_ok': True})
    presets = [p for p in get('/api/presets') if p.get('profile_options', {}).get('damage_variant') == 'round4_mixed_cpu_failed']
    configs = json.loads((ROOT / 'config/public-presets.json').read_text(encoding='utf8'))
    assert len(presets) == 9
    for preset in presets:
        original = next(p for p in configs if p['label'] == preset['label'])
        assert preset['available'] and original['source_kind'] == 'file'
        assert preset['source'].startswith('file:') and Path(original['path']).is_file()
        assert Path(preset['source'][5:]).resolve() == Path(original['path']).resolve()
        assert original['profile_options']['engine'] == 'onnxruntime'
        assert original['profile_options']['openvino_device'] == 'CPU'
        assert original['profile_options']['cpu_threads'] == 1
        assert 'raw-cached-before-actual-after' not in original['path']
        assert not any(Path(original['path']).resolve() == (ROOT / row['path']).resolve() for row in rows)
    fixing = next(r for r in get('/api/construction/capabilities')['categories'] if r['id'] == 'finish_installation_fault')
    assert fixing['effective_status'] == 'unsupported' and not fixing['model_installed']
    assert not get('/api/state')['running']
    value = {'scope': 'Delivery/configuration only, no NN, quality or FPS credit.', 'recordings': records,
             'original_input_presets': 9, 'actual_annotated_video_as_model_input': False,
             'fastener_category_unsupported': True, 'station_running': False,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with target.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
    print(target)


if __name__ == '__main__':
    main()
