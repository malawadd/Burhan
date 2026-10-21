"""Capture host workload before root speed benchmarks; no NN or setting changes."""
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round4-root-benchmark/host-before-first-benchmark.json'


def main():
    if OUT.exists():
        raise ValueError('Preserve first workload snapshot')
    with urlopen('http://127.0.0.1:8765/api/state', timeout=20) as response:
        station = json.load(response)
    if station['running']:
        raise ValueError('Stop the project inference station before this benchmark')
    blockers = []
    processes = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            arguments = process.info['cmdline'] or []
            code = ' '.join(arguments)
            name = process.info['name'] or ''
            project_script = any(
                argument.lower().endswith('.py')
                and any(token in argument.replace('\\', '/').lower() for token in ['scripts/'])
                for argument in arguments[1:])
            # Inspect only Python/ffmpeg project jobs; never record command lines
            # or stop unrelated applications/services on the user's computer.
            if (name.lower().startswith(('python', 'ffmpeg'))
                    and str(ROOT).lower() in code.lower()
                    and (project_script or name.lower().startswith('ffmpeg'))
                    and 'capture_round4_benchmark_host.py' not in code):
                blockers.append({'pid': process.pid, 'name': name})
            process.cpu_percent(None)
            processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if blockers:
        raise ValueError('Project worker processes still active: ' + repr(blockers))
    psutil.cpu_percent(None)
    time.sleep(1)
    workload = []
    for process in processes:
        try:
            workload.append({'pid': process.pid, 'name': process.name(),
                             'cpu_percent': process.cpu_percent(None),
                             'rss_bytes': process.memory_info().rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    memory = psutil.virtual_memory()
    value = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Observed host workload only. No process stopped/suspended and no power setting changed.',
             'station_running': False,
             'root_required_check': 'Root separately confirms all agents idle, all exports/NN/encoding/tests finished and browser playback paused.',
             'platform': platform.platform(), 'python': platform.python_version(),
             'cpu_physical': psutil.cpu_count(logical=False), 'cpu_logical': psutil.cpu_count(),
             'cpu_percent_observed_one_second': psutil.cpu_percent(None),
             'memory_total_bytes': memory.total, 'memory_available_bytes': memory.available,
             'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'openvino']},
             'top_observed_processes': sorted(workload, key=lambda row: row['cpu_percent'], reverse=True)[:15],
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(OUT)


if __name__ == '__main__':
    main()
