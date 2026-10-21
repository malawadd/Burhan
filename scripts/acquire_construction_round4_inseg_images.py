"""Freeze annotation-only source selection, then bounded exact author ZIP image members."""
from pathlib import Path
import argparse
import binascii
import hashlib
import json
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_construction_round4_inseg import Ranges, OUT, MAX_BYTES, digest, write_new


def polygons(path):
    result = []
    for index, line in enumerate(path.read_text().splitlines()):
        fields = line.split()
        category, coords = int(fields[0]), list(map(float, fields[1:]))
        if category not in {0, 1} or len(coords) < 6 or len(coords) % 2 or any(not 0 <= x <= 1 for x in coords):
            raise ValueError('Invalid source polygon: ' + str(path))
        result.append({'source_polygon_index': index, 'class_id': category, 'vertices': len(coords) // 2})
    return result


def selection():
    manifest_path = OUT / 'label-candidate-assets.json'
    manifest = json.loads(manifest_path.read_text())
    remaining = MAX_BYTES - json.loads((OUT / 'range-ledger.json').read_text())['received_bytes']
    records, counts, planned_bytes = [], {0: 0, 1: 0}, 0
    # The candidate order was frozen before labels or pixels. A source can count
    # toward both cohorts, but is downloaded exactly once.
    for record in manifest['records']:
        path = ROOT / record['label_path']
        if digest(path.read_bytes()) != record['label_sha256']:
            raise ValueError('Acquired author annotation changed')
        rows = polygons(path)
        classes = sorted({row['class_id'] for row in rows})
        if not any(counts[c] < 32 for c in classes):
            continue
        cost = 30 + len(record['image']['name'].encode()) + 65535 + record['image']['compressed_bytes']
        if planned_bytes + cost + 65536 > remaining:
            raise ValueError('Declared positive quotas exceed remaining strict budget; no image acquisition')
        planned_bytes += cost
        records.append({**record, 'source_polygons': rows, 'class_ids_present': classes})
        for c in classes:
            counts[c] += 1
        if all(counts[c] >= 32 for c in counts):
            break
    if any(counts[c] < 32 for c in counts):
        raise ValueError('Frozen candidate pool did not provide both declared cohorts')
    value = {'source_role': 'Publisher TRAIN only; unknown physical-site/author-checkpoint overlap',
        'class_mapping': {'0': 'crack', '1': 'spalling'},
        'class_mapping_evidence': 'Publisher author checkpoint names plus acquired CR-class0/SP-class1 source annotation examples; no prediction-driven relabelling.',
        'selection_basis': 'First frozen metadata-hash-order candidate positives until32 photos per class; no image/model outcomes viewed.',
        'candidate_manifest_sha256': digest(manifest_path.read_bytes()),
        'acquisition_source_sha256': digest(Path(__file__).read_bytes()),
        'range_reader_source_sha256': digest((ROOT / 'scripts/prepare_construction_round4_inseg.py').read_bytes()),
        'budget_total_bytes': MAX_BYTES, 'planned_max_additional_image_bytes': planned_bytes + 65536,
        'photos': len(records), 'positive_photo_counts': counts, 'records': records}
    write_new(OUT / 'image-selection-before-pixels.json', value)
    print('FROZEN', len(records), counts, 'maximagebytes', planned_bytes + 65536,
          digest((OUT / 'image-selection-before-pixels.json').read_bytes()), flush=True)


def images():
    plan_path = OUT / 'image-selection-before-pixels.json'
    plan = json.loads(plan_path.read_text())
    if digest(Path(__file__).read_bytes()) != plan['acquisition_source_sha256']:
        raise ValueError('Frozen image acquisition source changed')
    if digest((ROOT / 'scripts/prepare_construction_round4_inseg.py').read_bytes()) != plan['range_reader_source_sha256']:
        raise ValueError('Frozen range reader changed')
    folder = OUT / 'original-images'
    folder.mkdir(exist_ok=True)
    records = []
    remote = Ranges('images')
    for record in plan['records']:
        target = folder / Path(record['image']['name']).name
        receipt_path = target.with_suffix(target.suffix + '.source.json')
        if target.exists() or receipt_path.exists():
            raise ValueError('Preserve prior image acquisition; no silent overwrite')
        image = record['image']
        start = image['header_offset']
        end = min(remote.size - 1, start + 30 + len(image['name'].encode()) + 65535 + image['compressed_bytes'] - 1)
        block = remote.request(start, end)
        fields = struct.unpack('<4s5H3L2H', block[:30])
        if fields[0] != b'PK\x03\x04' or fields[3] != 8:
            raise ValueError('Unexpected original member compression/header')
        name_length, extra_length = fields[-2:]
        if block[30:30 + name_length].decode('utf8') != image['name']:
            raise ValueError('Original member name differs')
        payload = 30 + name_length + extra_length
        compressed = block[payload:payload + image['compressed_bytes']]
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(compressed, image['bytes'] + 1)
        if not decoder.eof or decoder.unconsumed_tail or len(data) != image['bytes'] or binascii.crc32(data) != image['crc32']:
            raise ValueError('Original image CRC/size/decompression failed')
        target.write_bytes(data)
        receipt = {**record, 'image_path': target.relative_to(ROOT).as_posix(), 'image_sha256': digest(data),
            'image_selection_sha256': digest(plan_path.read_bytes()), 'original_member_crc_verified': True,
            'source': 'https://github.com/BenyunZhao/CUBIT-InSeg', 'license': 'CC BY4.0 publisher dataset declaration',
            'source_role': 'Publisher TRAIN; no model prediction used for acquisition'}
        write_new(receipt_path, receipt)
        records.append(receipt)
        print('VERIFIED', target.name, len(data), digest(data), flush=True)
    write_new(OUT / 'image-assets.json', {'selection_sha256': digest(plan_path.read_bytes()),
        'records': records, 'bytes_received_all_stages': json.loads((OUT / 'range-ledger.json').read_text())['received_bytes']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['selection', 'images'], required=True)
    args = parser.parse_args()
    (selection if args.stage == 'selection' else images)()
