"""Reproducible public source audit; never trains from inaccessible metadata.

Usage: python scripts/audit_construction_expansion_sources.py probe
       python scripts/audit_construction_expansion_sources.py archive SOURCE PATH
The archive command inventories safely without extraction. It deliberately does
not infer physical-scene independence or convert unknown label taxonomies.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/construction/expansion-public-v1'
SOURCES = {
    'sdnet2025': {
        'record': 'https://data.mendeley.com/datasets/cxnj6cp4rb/1',
        'doi': '10.17632/cxnj6cp4rb.1', 'license': 'CC BY 4.0',
        'file': 'https://data.mendeley.com/public-files/datasets/cxnj6cp4rb/files/8227ae89-9e21-4880-ae64-f150c10180df/file_downloaded',
        'publisher_archive_sha256': '5be49a5ac5fa9c7ea5b3d0de03eca8b9f3d4f337b96a89c4c7f9d7ecce385ec0',
        'publisher_original_count': 826,
        'advertised_classes': ['fixed', 'loosened', 'missing', 'corroded'],
        'taxonomy_warning': 'Data article reports 302 fixed, 324 loosened, 200 missing original images; verify actual files before enabling corrosion.',
        'derivatives_warning': 'Publisher describes augmentation to 1500; original and resized versions must not cross splits.',
    },
    'phele': {
        'record': 'https://data.mendeley.com/datasets/yzsxcvc8dp/4',
        'doi': '10.17632/yzsxcvc8dp.4', 'license': 'CC BY 4.0',
        'file': 'https://data.mendeley.com/public-files/datasets/yzsxcvc8dp/files/010f225b-ec1e-4a66-adf2-1579028acb03/file_downloaded',
        'advertised_classes': ['Mold/dampness', 'Peeling paint'],
        'taxonomy_warning': 'Preserve combined Mold/dampness; appearance cannot establish moisture quantity or mold organism.',
        'derivatives_warning': 'Counts, original provenance, annotation format and scene groups unverified until actual archive audit.',
    },
    'indoor_defects': {
        'record': 'https://universe.roboflow.com/cuhk-ldjub/indoor_defects-logzf',
        'license': 'CC BY 4.0', 'publisher_image_count': 209,
        'advertised_classes': ['crack', 'paint_bubble', 'paint_peel', 'spalling'],
        'taxonomy_warning': 'Instance segmentation advertised; genuine polygon files and original image provenance require archive verification.',
        'derivatives_warning': 'Do not treat augmented export images as independent photographs.',
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def fetch_receipt(url, target):
    request = urllib.request.Request(url, headers={'User-Agent': 'InspectFlow-public-research/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            # Metadata only. Never accidentally acquire a multi-gigabyte ZIP.
            body = response.read(2 * 1024 * 1024 + 1)
            result = {'status': response.status, 'content_type': response.headers.get('Content-Type'),
                      'content_range': response.headers.get('Content-Range'), 'url': url}
    except urllib.error.HTTPError as error:
        body = error.read(2 * 1024 * 1024)
        result = {'status': error.code, 'content_type': error.headers.get('Content-Type'),
                  'challenge': error.headers.get('Cf-Mitigated'), 'url': url}
    except (urllib.error.URLError, TimeoutError) as error:
        return {'url': url, 'error': str(error), 'acquired': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    result.update({'body': target.relative_to(ROOT).as_posix(), 'body_sha256': sha(target),
                   'body_bytes': len(body), 'acquired_training_data': False})
    return result


def probe():
    results = {}
    for key, source in SOURCES.items():
        receipt = {'source': source, 'record': fetch_receipt(source['record'], OUT / key / 'record-response.bin')}
        if source.get('doi'):
            receipt['datacite'] = fetch_receipt('https://api.datacite.org/dois/' + source['doi'], OUT / key / 'datacite.json')
        receipt['training_status'] = 'unavailable_until_original_files_annotations_and_group_splits_verified'
        results[key] = receipt
    value = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'script_sha256': sha(Path(__file__)), 'sources': results,
             'private_owner_videos_used': False, 'neural_training_executed': False,
             'release_gate': {'precision': .95, 'recall': .90, 'independent_positives': 50, 'independent_negatives': 50}}
    write(OUT / 'source-audit.json', value)
    print(json.dumps({key: row['record'].get('status', row['record'].get('error')) for key, row in results.items()}))


def archive(key, path):
    source = SOURCES[key]
    digest = sha(path)
    expected = source.get('publisher_archive_sha256')
    if expected and digest != expected:
        raise ValueError('Publisher SHA256 mismatch; original archive not accepted')
    records = []
    with zipfile.ZipFile(path) as archive_file:
        for member in archive_file.infolist():
            name = member.filename.replace('\\', '/')
            pure = PurePosixPath(name)
            if pure.is_absolute() or '..' in pure.parts or ':' in name or '\x00' in name:
                raise ValueError('Unsafe archive member: ' + name)
            if member.flag_bits & 1:
                raise ValueError('Encrypted archive member: ' + name)
            records.append({'name': name, 'bytes': member.file_size, 'compressed_bytes': member.compress_size,
                            'crc32': f'{member.CRC:08x}', 'suffix': pure.suffix.lower()})
    counts = Counter(row['suffix'] for row in records)
    report = {'source': source, 'archive': str(Path(path).resolve()), 'archive_sha256': digest,
              'publisher_checksum_verified': bool(expected), 'members': records, 'suffix_counts': counts,
              'total_uncompressed_bytes': sum(row['bytes'] for row in records),
              'training_authorized': False,
              'next_gate': 'Inspect actual annotations, original/derivative identity, source groups and negatives before preparing a frozen split.'}
    write(OUT / key / 'archive-inventory.json', report)
    print(json.dumps({'members': len(records), 'suffix_counts': counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('probe')
    archive_parser = commands.add_parser('archive')
    archive_parser.add_argument('source', choices=SOURCES)
    archive_parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.command == 'probe':
        probe()
    else:
        archive(args.source, args.path)
