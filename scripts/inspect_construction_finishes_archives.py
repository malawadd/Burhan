"""Publisher-only metadata inventory for finishes research; no NN or full ZIP."""
from pathlib import Path
import hashlib
import json
import zipfile
from urllib.request import urlopen
from fetch_construction_bridge_video import RangeFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/construction/finishes-expansion/source-records'


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    OUT.mkdir(parents=True, exist_ok=True)
    with urlopen('https://zenodo.org/api/records/15622584', timeout=60) as source:
        record = json.load(source)
    if record['metadata']['license']['id'] != 'cc-by-4.0':
        raise RuntimeError('Original dataset license changed')
    archive = next(r for r in record['files'] if r['key'] == 'MBDD2025.zip')
    with zipfile.ZipFile(RangeFile(archive['links']['self'], archive['size'])) as z:
        entries = [{'name': r.filename, 'bytes': r.file_size, 'compressed_bytes': r.compress_size,
                    'header_offset': r.header_offset, 'compression': r.compress_type, 'crc32': f'{r.CRC:08x}'} for r in z.infolist()]
    result = {'publisher': record, 'archive_checksum_scope': 'Publisher fullZIP MD5 recorded; only central directory downloaded, not fullarchive checksum verified',
              'entries': entries, 'raw_video_members': [r['name'] for r in entries if Path(r['name']).suffix.lower() in {'.mp4','.mov','.avi','.mkv'}]}
    target = OUT / 'mbdd2025-publisher-inventory.json'
    payload = (json.dumps(result, indent=2) + '\n').encode()
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError('Preserve existing metadata inventory')
    target.write_bytes(payload)
    print(json.dumps({'path': str(target), 'sha256': hashlib.sha256(payload).hexdigest(), 'entries': len(entries),
                      'first_members': [r['name'] for r in entries[:45]], 'raw_videos': result['raw_video_members']}, indent=2))


if __name__ == '__main__':
    main()
