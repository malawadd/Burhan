"""Bounded primary Batalha efflorescence archive audit; no training or evaluation."""
from pathlib import Path
import hashlib
import json
import urllib.request
from collections import Counter

import acquire_sdnet_expansion as remote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/construction/expansion-public-v1/additional-leads/batalha'
URL = 'https://zenodo.org/api/records/21453438/files/Efflorescence_Limesstone.zip/content'


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def main():
    remote.OUT, remote.URL, remote.MAX_BYTES = OUT, URL, 80_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT.parent / 'batalha-record.json'
    if not metadata_path.exists():
        request = urllib.request.Request('https://zenodo.org/api/records/21453438',
                                         headers={'User-Agent': 'InspectFlow-public-research/1.0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            payload = response.read(1_000_001)
        if len(payload) > 1_000_000:
            raise ValueError('Publisher metadata response exceeds audit bound')
        record = json.loads(payload)
        metadata_path.write_bytes(payload)
    record = json.loads(metadata_path.read_text())
    if record['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Re-audit publisher rights before acquiring photographs')
    if not (OUT / 'zip-inventory.json').exists():
        remote.inventory()
    rows = json.loads((OUT / 'zip-inventory.json').read_text())
    images = {Path(r['name']).stem: r['name'] for r in rows
              if r['name'].startswith('Efflorescence_imagens/')
              and Path(r['name']).suffix.lower() in ('.jpg', '.jpeg')}
    masks = {Path(r['name']).stem: r['name'] for r in rows
             if r['name'].startswith('Efflorescence_masks/') and r['name'].endswith('.png')}
    paired = sorted(set(images) & set(masks))
    selected = sorted(paired, key=lambda n: hashlib.sha256(('audit-v1:' + n).encode()).digest())[:4]
    selection = {'policy': 'all masks; four paired originals with smallest SHA256(audit-v1:stem), selected before pixels',
                 'selected_photo_stems': selected, 'training_executed': False,
                 'members': list(masks.values()) + [images[n] for n in selected]}
    if not (OUT / 'selection-before-pixels.json').exists():
        write('selection-before-pixels.json', selection)
    elif json.loads((OUT / 'selection-before-pixels.json').read_text()) != selection:
        raise ValueError('Frozen audit selection changed')
    remote.batches(selection['members'])

    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    mask_stats = []
    hashes = Counter()
    for stem, member in sorted(masks.items()):
        path = OUT / 'original' / member
        raw = path.read_bytes()
        hashes[hashlib.sha256(raw).hexdigest()] += 1
        with Image.open(path) as image:
            a = np.asarray(image)
            # Exact 8-bit histogram avoids sorting tens of millions of mask pixels.
            if a.dtype == np.uint8:
                histogram = np.bincount(a.reshape(-1), minlength=256)
                values = np.flatnonzero(histogram)
                counts = histogram[values]
            else:
                values, counts = np.unique(a, return_counts=True)
            mask_stats.append({'stem': stem, 'mode': image.mode, 'size': list(image.size),
                               'values': {str(v): int(c) for v, c in zip(values, counts)},
                               'nonzero_fraction': float(np.count_nonzero(a) / a.size)})
    panels, photo_stats = [], []
    for stem in selected:
        photo_path, mask_path = OUT / 'original' / images[stem], OUT / 'original' / masks[stem]
        photo, mask = Image.open(photo_path).convert('RGB'), Image.open(mask_path).convert('L')
        if photo.size != mask.size:
            raise ValueError('Photo/mask dimensions differ: ' + stem)
        tint = Image.new('RGB', photo.size, (255, 40, 40))
        overlay = Image.composite(Image.blend(photo, tint, .45), photo, mask.point(lambda x: 255 if x else 0))
        panel = Image.new('RGB', (1000, 380), 'white')
        for i, picture in enumerate((photo, overlay)):
            fitted = ImageOps.contain(picture, (495, 345))
            panel.paste(fitted, (i * 500, 30))
        ImageDraw.Draw(panel).text((4, 5), stem + ' | original / publisher mask overlay', fill='black')
        panels.append(panel)
        photo_stats.append({'stem': stem, 'photo_size': list(photo.size), 'mask_size': list(mask.size),
                            'photo_sha256': hashlib.sha256(photo_path.read_bytes()).hexdigest(),
                            'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()})
    sheet = Image.new('RGB', (1000, len(panels) * 380), 'white')
    for i, panel in enumerate(panels): sheet.paste(panel, (0, i * 380))
    sheet.save(OUT / 'actual-photo-mask-audit.jpg', quality=88)
    ledger = json.loads((OUT / 'range-ledger.json').read_text())
    write('actual-file-audit.json', {
        'source': 'https://zenodo.org/records/21453438', 'license': 'CC BY 4.0',
        'publisher_archive_bytes': 662950017, 'publisher_md5_not_whole_verified': 'b095764ee06a7b52bdf6e7ec31b0d7ee',
        'archive_members': len(rows), 'photographs': len(images), 'masks': len(masks),
        'paired_stems': len(paired), 'unpaired_images': sorted(set(images) - set(masks)),
        'unpaired_masks': sorted(set(masks) - set(images)),
        'acquired_masks': len(mask_stats), 'acquired_photographs': len(photo_stats),
        'exact_duplicate_mask_files': sum(n - 1 for n in hashes.values()),
        'mask_statistics': mask_stats, 'photo_statistics': photo_stats,
        'received_range_bytes_including_inventory': ledger['received_bytes'],
        'known_physical_sites': 1, 'site_identity': 'Monastery of Batalha, Portugal',
        'split_created': False, 'training_executed': False, 'model_quality_evaluated': False,
        'release_status': 'unavailable: no trained/validated local category; one known source site; no video or healthy-negative qualification',
    })
    print(json.dumps({'photos': len(images), 'masks': len(masks), 'pairs': len(paired),
                      'unpaired_images': sorted(set(images) - set(masks)),
                      'acquired_photos': len(photo_stats), 'received_bytes': ledger['received_bytes']}))


if __name__ == '__main__':
    main()
