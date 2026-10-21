"""Bounded original SDNET2025 ZIP member acquisition through its public file link."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, io, json, struct, urllib.request, zipfile, zlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/construction/expansion-public-v1/sdnet2025'
URL = 'https://data.mendeley.com/public-files/datasets/cxnj6cp4rb/files/8227ae89-9e21-4880-ae64-f150c10180df/file_downloaded'
MAX_BYTES = 800_000_000


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')


class RemoteZip(io.RawIOBase):
    def __init__(self):
        super().__init__()
        self.pos = 0
        self.size = None
        self.url = URL
        self.request(-65536, None)

    def request(self, start, end):
        ledger_path = OUT / 'range-ledger.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'received_bytes': 0, 'requests': []}
        count = -start if start < 0 else end - start + 1
        if count > 32_000_000 or ledger['received_bytes'] + count > MAX_BYTES:
            raise ValueError('Strict range download budget exceeded')
        header = str(start) if start < 0 else f'{start}-{end}'
        request = urllib.request.Request(self.url, headers={'Range': 'bytes=' + header, 'User-Agent': 'InspectFlow-public-research/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError('Publisher did not honor bounded range')
            span, total = response.headers['Content-Range'].removeprefix('bytes ').split('/')
            actual_start, actual_end = map(int, span.split('-'))
            total = int(total)
            if (actual_start, actual_end) != (total + start if start < 0 else start, total - 1 if start < 0 else end):
                raise ValueError('Incorrect response range')
            if self.size is not None and total != self.size:
                raise ValueError('Publisher archive changed')
            self.size = total
            self.url = response.url  # Normal publisher redirect to the public S3 object.
            data = response.read(count + 1)
            if len(data) != count:
                raise ValueError('Truncated or oversized response')
        ledger['received_bytes'] += len(data)
        ledger['requests'].append({'range': f'{actual_start}-{actual_end}/{total}', 'bytes': len(data),
                                   'sha256': hashlib.sha256(data).hexdigest(), 'resolved_url': self.url})
        write(ledger_path, ledger)
        return data

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        return self.pos
    def read(self, size=-1):
        size = min(self.size - self.pos, self.size - self.pos if size < 0 else size)
        if size == 0: return b''
        data = self.request(self.pos, self.pos + size - 1)
        self.pos += len(data)
        return data


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZip()) as archive:
        rows = [{'name': i.filename, 'bytes': i.file_size, 'compressed_bytes': i.compress_size,
                 'crc32': i.CRC, 'header_offset': i.header_offset, 'compression': i.compress_type}
                for i in archive.infolist() if not i.is_dir()]
    write(OUT / 'zip-inventory.json', rows)
    print(json.dumps({'members': len(rows), 'first_members': rows[:15]}), flush=True)


def members(names):
    records = []
    with zipfile.ZipFile(RemoteZip()) as archive:
        for name in names:
            pure = PurePosixPath(name.replace('\\', '/'))
            if pure.is_absolute() or '..' in pure.parts or ':' in name:
                raise ValueError('Unsafe member name')
            target = OUT / 'original' / pure
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(name))  # zipfile verifies source CRC.
            row = {'member': name, 'path': target.relative_to(ROOT).as_posix(),
                   'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size}
            records.append(row)
            write(OUT / 'acquired-members.json', {'records': records})
            print('ACQUIRED', name, row['bytes'], flush=True)


def batches(names):
    """Read contiguous compressed ranges once, verifying every original CRC."""
    inventory_rows = json.loads((OUT / 'zip-inventory.json').read_text())
    selected = sorted([r for r in inventory_rows if r['name'] in set(names)], key=lambda r: r['header_offset'])
    if len(selected) != len(set(names)): raise ValueError('Selection contains unknown ZIP members')
    remote = RemoteZip()
    groups = []
    for row in selected:
        target = OUT / 'original' / row['name']
        if target.exists(): continue
        end = row['header_offset'] + 30 + len(row['name'].encode()) + 65535 + row['compressed_bytes']
        if groups and row['header_offset'] - groups[-1]['end'] < 16384 and end - groups[-1]['start'] < 28_000_000:
            groups[-1]['rows'].append(row); groups[-1]['end'] = max(groups[-1]['end'], end)
        else: groups.append({'start': row['header_offset'], 'end': end, 'rows': [row]})
    for group in groups:
        end = min(group['end'], remote.size - 1)
        blob = remote.request(group['start'], end)
        for row in group['rows']:
            offset = row['header_offset'] - group['start']
            if blob[offset:offset + 4] != b'PK\x03\x04': raise ValueError('Incorrect local ZIP header')
            name_len, extra_len = struct.unpack_from('<HH', blob, offset + 26)
            start = offset + 30 + name_len + extra_len
            compressed = blob[start:start + row['compressed_bytes']]
            raw = zlib.decompress(compressed, -15) if row['compression'] == 8 else compressed if row['compression'] == 0 else None
            if raw is None or len(raw) != row['bytes'] or zlib.crc32(raw) != row['crc32']:
                raise ValueError('Source CRC/size mismatch: ' + row['name'])
            pure = PurePosixPath(row['name'])
            if pure.is_absolute() or '..' in pure.parts or ':' in row['name']: raise ValueError('Unsafe path')
            target = OUT / 'original' / pure; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        print('ACQUIRED_BATCH', len(group['rows']), len(blob), flush=True)
    records = []
    for row in selected:
        target = OUT / 'original' / row['name']
        records.append({'member': row['name'], 'path': target.relative_to(ROOT).as_posix(), 'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    write(OUT / 'image-assets.json', {'records': records, 'neural_training_executed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inventory', 'members', 'batches'])
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.stage == 'inventory': inventory()
    else:
        if args.selection is None: parser.error('--selection JSON list of exact member names is required')
        (batches if args.stage == 'batches' else members)(json.loads(args.selection.read_text()))
