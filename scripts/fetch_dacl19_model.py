"""Acquire the pinned dataset-author DACL19 demo checkpoint, without unpickling."""
from pathlib import Path
import hashlib
import json
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'f5bee34782e7e02393ba4f206a95ba5ed67eb8a5'
BASE = f'https://huggingface.co/spaces/phiyodr/dacl-challenge/resolve/{COMMIT}/'
FILES = [
    ('app.py', 'author-app.py.txt', '30590ca5e7c8c7aa574a6405418158339361a4c9a158f40dccb4079f151aa6ba'),
    ('README.md', 'author-README.md', 'fac797c332a7c428c296aad127cf8c103c491ce13cb3ed34286a846ea734ef32'),
    ('runs/2023-08-31_rich-paper-12/best_model_cpu.pth', 'author-segformer-mit-b1.pth', '166b778bdbc57ef2bfd5f71bc6ae724bf598d1a4c6a1b69afa4fbd936f2d965b'),
]


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    directory = ROOT / 'models/construction/dacl19'
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for remote, name, expected in FILES:
        path = directory / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            with urlopen(BASE + remote, timeout=90) as incoming, temporary.open('wb') as outgoing:
                while block := incoming.read(1024 * 1024):
                    outgoing.write(block)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned author asset hash mismatch: ' + name)
        artifacts.append({'path': name, 'sha256': expected, 'bytes': path.stat().st_size})
    report = {'source': 'https://huggingface.co/spaces/phiyodr/dacl-challenge', 'source_commit': COMMIT,
              'assets': artifacts, 'dataset_license': 'CC BY-NC4.0 per original authors',
              'model_license': 'No separately declared checkpoint license in reviewed author Space README; commercial clearance unresolved',
              'checkpoint_handling': 'Full model pickle retained as an inert source file; use export_dacl19.py weights_only=True extraction, never unrestricted unpickling',
              'local_validation': 'Candidate acquisition only; no accuracy/speed/category support is established by download'}
    (directory / 'download-manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
