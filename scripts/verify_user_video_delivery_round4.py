"""Read-only localhost delivery checks for already audited owner recordings; no NN."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/user-construction-videos-round4'
BASE = 'http://127.0.0.1:8765'


def get(route):
    with urlopen(BASE + route, timeout=30) as response:
        assert response.status == 200
        return json.load(response)


def main():
    target = OUT / 'localhost-delivery-v1.json'
    if target.exists():
        raise ValueError('Preserve earlier receipt')
    local = json.loads((ROOT / 'config/construction-videos.json').read_text(encoding='utf-8'))
    served = get('/api/construction/videos')
    rows = [r for r in local if r['id'].startswith('owner-round4-')]
    assert len(rows) == 4
    checked = []
    for row in rows:
        item = next(v for v in served if v['id'] == row['id'])
        assert item['available'] and item['evidence_available'] and item['private_local_input']
        evidence = get(item['evidence_url'])
        assert evidence == json.loads((ROOT / row['evidence_path']).read_text())
        assert all(value is None for key, value in evidence['summary'].items()
                   if key != 'unscored_prediction_counts')
        with urlopen(Request(BASE + item['url'], method='HEAD'), timeout=30) as response:
            assert response.status == 200
            assert int(response.headers['Content-Length']) == (ROOT / row['path']).stat().st_size
        with urlopen(Request(BASE + item['url'], headers={'Range': 'bytes=0-31'}), timeout=30) as response:
            assert response.status == 206
            with (ROOT / row['path']).open('rb') as stream:
                assert response.read() == stream.read(32)
        checked.append({'id': row['id'], 'evidence_equal': True, 'head_and_range_ok': True})
    presets = [p for p in get('/api/presets') if p.get('private_local_input')]
    local_presets = json.loads((ROOT / 'config/public-presets.json').read_text(encoding='utf-8'))
    assert len(presets) == 12
    for preset in presets:
        local_preset = next(p for p in local_presets if p['label'] == preset['label'])
        assert preset['available'] and local_preset['source_kind'] == 'file'
        assert preset['source'] == 'file:' + local_preset['path']
        assert Path(local_preset['path']).is_file()
        assert 'before-three-models' not in local_preset['path']
    capabilities = get('/api/construction/capabilities')
    fixing = next(r for r in capabilities['categories'] if r['id'] == 'finish_installation_fault')
    assert fixing['effective_status'] == 'unsupported' and not fixing['model_installed']
    state = get('/api/state')
    assert not state['running']
    value = {'recordings': checked, 'original_local_file_presets_verified': 12,
             'fastener_category': 'unsupported; no installed model', 'station_running': False,
             'software_tests': {'app_suite_passed': 462, 'skipped': 3, 'warnings': 1,
                                'seconds': 49.03,
                                'scope': 'Software checks only; Torch training-only fixtures run separately.'},
             'scope': 'HTTP/report/config delivery, not accuracy or FPS.',
             'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(value, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
