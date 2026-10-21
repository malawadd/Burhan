"""Real video presets must select the actual video reader, including lossless NUT."""
import json
from pathlib import Path


def test_registered_construction_video_sources_use_video_reader():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / 'config/public-presets.json').read_text(encoding='utf-8'))
    videos = [r for r in rows if r['profile'] == 'construction'
              and Path(r['path']).suffix.lower() in {'.mp4', '.mkv', '.mov', '.avi', '.nut', '.webm'}]
    assert videos
    for preset in videos:
        assert preset.get('source_kind') == 'file', preset['label']
