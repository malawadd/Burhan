"""Repair explicit source types; video files must reach the video reader."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation/construction-source-type-repair'
VIDEO_SUFFIXES = {'.mp4', '.mkv', '.mov', '.avi', '.nut', '.webm'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = ROOT / 'config/public-presets.json'
    rows = json.loads(config.read_text(encoding='utf-8'))
    repaired = []
    for row in rows:
        if row['profile'] == 'construction' and Path(row['path']).suffix.lower() in VIDEO_SUFFIXES:
            if row.get('source_kind', 'image') != 'file':
                repaired.append({'label': row['label'], 'path': row['path'],
                                 'before': row.get('source_kind', 'image'), 'after': 'file'})
                row['source_kind'] = 'file'
    if not repaired:
        raise ValueError('No repair needed; preserve the first receipt')
    with (OUT / 'before-public-presets.json').open('xb') as backup:
        backup.write(config.read_bytes())
    before = hashlib.sha256(config.read_bytes()).hexdigest()
    config.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (OUT / 'repair.json').open('x', encoding='utf-8') as stream:
        json.dump({'changed': repaired, 'before_sha256': before,
                   'after_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                   'actual_NN_calls': 0, 'model_or_accuracy_changes': False}, stream, indent=2)
        stream.write('\n')
    print(f'Repaired {len(repaired)} construction video source types.')


if __name__ == '__main__':
    main()
