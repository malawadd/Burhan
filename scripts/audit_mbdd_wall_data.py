"""Independent wall source/XML/split/CRC audit; no model or neural operation."""
from pathlib import Path
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
import zlib

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'datasets/construction/finishes-expansion'
DATA = BASE / 'mbdd-wall-five-class-v1'
CLASSES = ['crack', 'leakage', 'abscission', 'corrosion', 'bulge']
MANIFEST_SHA = 'be5dec6e2022a065875f92b53a7841e244612dc32d3fa218fff1db2b71a33fa5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block(name):
    match = re.fullmatch(r'Hefei(\d+)\.[^.]+', Path(name).name)
    if not match:
        raise ValueError('Unexpected original source naming')
    return (int(match.group(1)) - 1) // 200


def main():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    manifest_path = DATA / 'asset-manifest.json'
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('Exact parent-declared230 source manifest required')
    m = json.loads(manifest_path.read_text()); ppath = DATA / 'protocol-before-source-labels-pixels.json'
    p = json.loads(ppath.read_text()); spath = DATA / 'selection-before-photo-acquisition.json'; s = json.loads(spath.read_text())
    invpath = BASE / 'source-records/mbdd2025-publisher-inventory.json'
    diagpath = BASE / 'mbdd2025-wall-photos/asset-manifest.json'
    if sha(ppath) != m['protocol_sha256'] or sha(spath) != m['selection_sha256'] or sha(invpath) != p['source_inventory_sha256'] or sha(diagpath) != p['protected_diagnostic_manifest_sha256']:
        raise ValueError('Original publisher/selection/diagnostic identities changed')
    if m['classes'] != CLASSES or p['classes'] != CLASSES or m['source'] != 'https://zenodo.org/records/15622584' or m['license'] != 'CC BY4.0':
        raise ValueError('Declared publisher class/source/license mapping differs')
    inventory = {r['name']: r for r in json.loads(invpath.read_text())['entries']}
    parsed = {}; empty = 0; invalid_unused = 0
    selected_names = {r['source_name'] for r in m['records']}
    archive = (DATA / 'original-annotation-range.bin').read_bytes()
    if hashlib.sha256(archive).hexdigest() != s['annotation_range_sha256']:
        raise ValueError('Pinned original annotation range bytes differ')
    entries = sorted([r for r in inventory.values() if '/Annotations/' in r['name'] and r['name'].endswith('.xml')], key=lambda r: r['name'])
    for entry in entries:
        path = Path(entry['name']); offset = entry['header_offset'] - p['annotation_range']['start']
        header = archive[offset:offset + 30]
        if header[:4] != b'PK\x03\x04':
            raise ValueError('Publisher original local ZIP header differs')
        nl, el = struct.unpack_from('<HH', header, 26)
        payload = archive[offset + 30 + nl + el:offset + 30 + nl + el + entry['compressed_bytes']]
        raw = zlib.decompress(payload, -15) if entry['compression'] == 8 else payload if entry['compression'] == 0 else b''
        if len(raw) != entry['bytes'] or f'{zlib.crc32(raw):08x}' != entry['crc32']:
            raise ValueError('Original annotation CRC/size differs from publisher inventory')
        root = ET.fromstring(raw); wh = [int(root.findtext('size/width')), int(root.findtext('size/height'))]; boxes = []
        for obj in root.findall('object'):
            label = obj.findtext('name')
            if label not in CLASSES:
                raise ValueError('Unknown publisher class')
            bounds = [float(obj.findtext('bndbox/' + k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
            if not np.isfinite(bounds).all() or not (0 <= bounds[0] < bounds[2] <= wh[0] and 0 <= bounds[1] < bounds[3] <= wh[1]):
                if path.stem + '.jpg' in selected_names:
                    raise ValueError('SELECTED publisher XML contains invalid geometry: ' + path.name)
                invalid_unused += 1
            boxes.append({'label': label, 'class_id': CLASSES.index(label), 'source_xyxy': bounds})
        name = path.stem + '.jpg'; filename = root.findtext('filename')
        if filename not in {name, name.removeprefix('Hefei')}:
            raise ValueError('XML filename cannot be associated with original publisher photo')
        parsed[name] = {'boxes': boxes, 'wh': wh, 'block': block(name), 'sha256': hashlib.sha256(raw).hexdigest()}
        empty += not boxes
    if len(parsed) != s['parsed_original_xml'] or len(parsed) != 14471 or empty != 8 or empty != m['empty_original_xml']:
        raise ValueError('Full XML inventory/empty-label scope differs')
    diag = json.loads(diagpath.read_text()); excluded = sorted({block(r['annotation_member']) for r in diag['records']})
    if excluded != p['diagnostic_blocks_excluded']:
        raise ValueError('Original18 diagnostic block exclusion differs')
    seed = p['seed']; digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    blocks = sorted({r['block'] for r in parsed.values()} - set(excluded), key=lambda b: digest(seed + 'block:' + str(b)))
    cut = int(.75 * len(blocks)); split = {'train': blocks[:cut], 'calibration': blocks[cut:]}
    if split != p['split_blocks']:
        raise ValueError('Entire-block seeded partition differs')
    expected = []
    for role, groups in split.items():
        candidates = sorted([name for name, row in parsed.items() if row['block'] in groups], key=lambda name: digest(seed + role + name))
        chosen = {}
        for label in CLASSES:
            for name in [name for name in candidates if any(b['label'] == label for b in parsed[name]['boxes'])][:40 if role == 'train' else 10]:
                chosen[name] = role
        expected.extend((role, name) for name in chosen)
    if expected != [(r['split'], r['source_name']) for r in s['selected']] or expected != [(r['split'], r['source_name']) for r in m['records']] or len(expected) != 230:
        raise ValueError('Class-balanced pre-photo seeded source selection differs')
    signatures = []; counts = {role: {label: 0 for label in CLASSES} for role in split}
    for row in m['records']:
        native = parsed[row['source_name']]
        if row['boxes'] != native['boxes'] or row['frame_size_wh'] != native['wh'] or row['capture_block_200'] != native['block'] or row['annotation_sha256'] != native['sha256'] or sha(DATA / row['annotation']) != native['sha256']:
            raise ValueError('Manifest labels/coordinates/class IDs differ from actual XML')
        path = DATA / row['image']; raw = path.read_bytes(); entry = inventory[row['image_member']]
        if sha(path) != row['image_sha256'] or len(raw) != entry['bytes'] or f'{zlib.crc32(raw):08x}' != entry['crc32']:
            raise ValueError('Actual photo SHA/CRC/size differs')
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or list(frame.shape[1::-1]) != native['wh']:
            raise ValueError('Actual native photo and XML geometry differ')
        tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = cv2.dct(tiny)[:8, :8].ravel()[1:]
        signature = ''.join('1' if value > float(np.median(low)) else '0' for value in low)
        if signature != row['phash63']:
            raise ValueError('Actual native photo similarity signature differs')
        signatures.append({**row, 'phash63': signature})
        if row['status'] == 'eligible_pending_independent_review':
            for label in CLASSES:
                counts[row['split']][label] += any(b['label'] == label for b in row['boxes'])
    known = []
    for row in diag['records']:
        path = diagpath.parent / row['image']; frame = cv2.imread(str(path))
        if sha(path) != row['image_sha256'] or frame is None:
            raise ValueError('Preserved18 diagnostic photo changed')
        tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = cv2.dct(tiny)[:8, :8].ravel()[1:]
        known.append({'split': 'diagnostic', 'source_name': row['filename'], 'image_sha256': row['image_sha256'], 'phash63': ''.join('1' if x > float(np.median(low)) else '0' for x in low)})
    flags = []; quarantine = set()
    for i, a in enumerate(signatures):
        for b in [r for r in signatures[i + 1:] if r['split'] != a['split']] + known:
            distance = sum(x != y for x, y in zip(a['phash63'], b['phash63'])); exact = a['image_sha256'] == b['image_sha256']
            if exact or distance <= 6:
                endpoint = a if a['split'] == 'train' or b['split'] == 'diagnostic' else b
                quarantine.add(endpoint['source_name'])
                flags.append({'a': a['source_name'], 'a_split': a['split'], 'b': b['source_name'], 'b_split': b['split'], 'exact': exact, 'phash_hamming': distance, 'quarantined': endpoint['source_name']})
    if flags != m['crosssplit_flags'] or sorted(quarantine) != m['quarantined'] or any((r['status'] == 'quarantined_crosssplit_similarity') != (r['source_name'] in quarantine) for r in m['records']):
        raise ValueError('Conservative similarity quarantine differs')
    result = {'source_manifest_sha256': MANIFEST_SHA, 'audit_source_sha256': sha(Path(__file__)), 'metadata_source_contract_passed': True,
              'original_xml_crc_verified': len(parsed), 'original_photos_crc_geometry_sha_verified': 230,
              'empty_original_xml': empty, 'source_class_order': CLASSES, 'post_quarantine_positive_photo_counts': counts,
              'unselected_original_objects_outside_native_bounds': invalid_unused,
              'invalid_selected_objects': 0,
              'full_geometry_inventory_sha256': sha(DATA / 'independent-invalid-xml-geometry-inventory.json'),
              'quarantined': sorted(quarantine), 'diagnostic18_whole_blocks_excluded': True,
              'visual_review_pending': True, 'approved_for_train_probe': False,
              'scope': 'Source/label/split contract only. Blocks and pHash cannot prove independent physical sites; empty XML is not healthy negative evidence. No NN, accuracy, capacity or fit approval.'}
    target = DATA / 'independent-metadata-audit.json'
    if target.exists():
        raise ValueError('Preserve previous independent receipt')
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
