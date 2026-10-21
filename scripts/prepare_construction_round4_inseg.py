"""Bounded public author-TRAIN ZIP ranges; no model, private data or validation archive."""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/construction/round4-pixel-domain/cubit-inseg'
IDS = {'images': '1LEaYyASd7nHL-RryO1i-ucF2X5NAikqo', 'labels': '1KEYRaHPRZ6G3ScoOOsFjzDYDzOzVXeUg'}
MAX_BYTES = 250_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with path.open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


class Ranges(io.RawIOBase):
    """Seekable read-only ZIP facade, strict response range and persistent byte ledger."""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.url = 'https://drive.usercontent.google.com/download?id=' + IDS[kind] + '&export=download&confirm=t'
        self.pos = 0
        self.size = None
        self.request(-65536, None)

    def request(self, first, last):
        ledger_path = OUT / 'range-ledger.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'requests': [], 'received_bytes': 0}
        header = 'bytes=' + (str(first) if first < 0 else f'{first}-{last}')
        expected_max = -first if first < 0 else last - first + 1
        if expected_max > 20_000_000 or ledger['received_bytes'] + expected_max > MAX_BYTES:
            raise ValueError('Strict per-request/total acquisition bound would be exceeded')
        if (OUT / 'PAUSE').exists():
            raise RuntimeError('Cooperative PAUSE requested before next verified range')
        with urlopen(Request(self.url, headers={'Range': header, 'User-Agent': 'Mozilla/5.0'}), timeout=60) as response:
            if response.status != 206:
                raise ValueError('Server ignored bounded range; no full archive read')
            content_range = response.headers['Content-Range']
            span, total = content_range.removeprefix('bytes ').split('/')
            got_first, got_last = map(int, span.split('-'))
            total = int(total)
            actual_first = total + first if first < 0 else first
            actual_last = total - 1 if first < 0 else last
            if (got_first, got_last) != (actual_first, actual_last):
                raise ValueError('Incorrect remote byte range')
            if self.size is not None and total != self.size:
                raise ValueError('Remote ZIP size changed')
            self.size = total
            data = response.read(expected_max + 1)
            if len(data) != expected_max:
                raise ValueError('Truncated or oversized range')
        ledger['requests'].append({'archive': self.kind, 'url': self.url, 'range': content_range,
                                  'bytes': len(data), 'sha256': digest(data)})
        ledger['received_bytes'] += len(data)
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= self.pos <= self.size:
            raise ValueError('ZIP seek outside archive')
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size == 0:
            return b''
        data = self.request(self.pos, self.pos + size - 1)
        self.pos += len(data)
        return data


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'source': 'https://github.com/BenyunZhao/CUBIT-InSeg',
        'author_folder': 'https://drive.google.com/drive/folders/1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b',
        'source_role': 'Publisher TRAIN archives ONLY; no author val/test acquisition.',
        'class_mapping_status': 'Must inspect author annotation metadata before assigning numeric class meaning.',
        'rights': 'Publisher repository declares CC BY 4.0 images/annotations/metadata; author checkpoint separate rights remain unresolved.',
        'byte_bound': MAX_BYTES, 'selection': 'Metadata SHA256 order of label member name. First128 paired labels, before any image pixels; '
                     'then up to32 positives per verified crack/spalling label. Exact image selection frozen separately before images.',
        'private_videos': 'Excluded from acquisition, fitting and calibration; no upload.',
        'calibration': 'Existing122 ev-CIVIL/negative calibration identities unchanged; no training selection from their predictions.',
        'pause': 'Create PAUSE in dataset root; next range fails before download, preserving completed members.'}
    write_new(OUT / 'acquisition-protocol-before-members.json', protocol)
    inventories = {}
    for kind in ['labels', 'images']:
        with zipfile.ZipFile(Ranges(kind)) as archive:
            records = [{'name': i.filename, 'compressed_bytes': i.compress_size, 'bytes': i.file_size,
                        'crc32': i.CRC, 'header_offset': i.header_offset, 'compression': i.compress_type}
                       for i in archive.infolist() if not i.is_dir()]
        inventories[kind] = records
        write_new(OUT / (kind + '-zip-member-inventory.json'), records)
    images = {Path(r['name']).stem: r for r in inventories['images']}
    paired = [r for r in inventories['labels'] if Path(r['name']).stem in images]
    ordered = sorted(paired, key=lambda r: digest(('CUBIT-InSeg-TRAIN-v1:' + r['name']).encode()))
    chosen = [{'label': r, 'image': images[Path(r['name']).stem]} for r in ordered[:128]]
    write_new(OUT / 'label-candidate-selection-before-labels.json', {'count': len(chosen), 'records': chosen})
    print({k: len(v) for k, v in inventories.items()}, 'paired', len(paired), 'candidate_labels', len(chosen), flush=True)


def labels():
    plan = json.loads((OUT / 'label-candidate-selection-before-labels.json').read_text())
    folder = OUT / 'candidate-labels'
    folder.mkdir(exist_ok=True)
    records = []
    with zipfile.ZipFile(Ranges('labels')) as archive:
        for record in plan['records']:
            expected = record['label']
            info = archive.getinfo(expected['name'])
            if info.CRC != expected['crc32'] or info.compress_size != expected['compressed_bytes']:
                raise ValueError('Publisher member inventory changed')
            target = folder / Path(info.filename).name
            if target.exists():
                raise ValueError('Preserve previous label acquisition; no silent overwrite')
            # zipfile verifies original uncompressed CRC before returning.
            data = archive.read(info)
            target.write_bytes(data)
            records.append({**record, 'label_path': target.relative_to(ROOT).as_posix(), 'label_sha256': digest(data)})
            print('verified', info.filename, len(data), flush=True)
    write_new(OUT / 'label-candidate-assets.json', {'records': records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inventory', 'labels'], required=True)
    args = parser.parse_args()
    (inventory if args.stage == 'inventory' else labels)()
