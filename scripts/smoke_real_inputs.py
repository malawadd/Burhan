"""Exercise installed real models through the shared runtime with outbound sockets denied."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/evaluation/real-runtime-smoke.json')
    args = parser.parse_args()
    presets = json.loads((ROOT / 'config/public-presets.json').read_text(encoding='utf-8'))
    selected = {p['profile']: p for p in reversed(presets)}
    results = []
    def deny_network(*args, **kwargs):
        raise AssertionError('Outbound network was attempted during installed-model operation')
    with tempfile.TemporaryDirectory(prefix='inspectflow-real-') as data_dir:
        edge = EdgeRuntime(data_dir)
        try:
            with patch('socket.socket.connect', deny_network), patch('socket.create_connection', deny_network):
                for name in ('factory', 'freight', 'roads'):
                    preset = selected[name]
                    options = dict(preset.get('profile_options', {}))
                    for key in ('model_path', 'defect_model_path', 'reference_path'):
                        if isinstance(options.get(key), str):
                            options[key] = str(ROOT / options[key])
                    config = RuntimeConfig(profile=name,
                        source=preset.get('source_kind', 'image') + ':' + str(ROOT / preset['path']),
                        input_fps=preset.get('input_fps', 2),
                        deadline_ms=preset.get('deadline_ms', 5000),
                        profile_options=options)
                    begun = time.perf_counter()
                    edge.start(config)
                    end = time.perf_counter() + 45
                    while time.perf_counter() < end:
                        state = edge.snapshot()
                        if state['error']:
                            raise RuntimeError(f"{name}: {state['error']}")
                        result = state['last_result'] or {}
                        ready = state['metrics']['processed'] >= 4
                        if name == 'freight':
                            ready = ready and bool(result.get('domain_state', {}).get('ocr'))
                        if ready:
                            break
                        time.sleep(.04)
                    else:
                        raise RuntimeError(f'{name}: actual model result not produced within 45 seconds')
                    edge.stop()
                    state = edge.snapshot()
                    assert state['metrics']['inspection_errors'] == 0, state
                    assert state['last_result']['provenance'] == 'local_document'
                    assert edge.preview().startswith(b'\xff\xd8')
                    results.append({'profile': name, 'label': preset['label'],
                        'wall_ms': round((time.perf_counter()-begun)*1000, 2),
                        'config': state['config'], 'description': state['description'],
                        'metrics': state['metrics'], 'last_result': state['last_result'],
                        'outbound_socket_connections_denied': True, 'saved_media': False})
        finally:
            edge.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'note': 'Repeated public stills through real model paths. Not independent accuracy samples or a physical network-disconnection test.', 'results': results}, indent=2), encoding='utf-8')
    print(json.dumps([{'profile': r['profile'], 'processed': r['metrics']['processed'],
                       'status': r['last_result']['status'], 'p95_ms': r['metrics']['p95_ms']} for r in results], indent=2))


if __name__ == '__main__':
    main()
