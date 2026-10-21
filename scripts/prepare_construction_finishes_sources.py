"""Freeze and acquire a small real MVTec tile photo diagnostic; no inference.

Official manufacturer terms prevail over the research-author mirror. Existing
construction data, model weights, operating points and videos are never edited.
"""
from pathlib import Path, PurePosixPath
import concurrent.futures
import hashlib
import json
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/construction/finishes-expansion/mvtec-tile'
COMMIT = '74db7cacf256eb62b5ce87094f83e9201fb7ac0a'
REPO = 'jiang-cc/MMAD'
SEED = 'construction-tile-normal-only-20261020:'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'InspectFlow-research/1.0'}), timeout=90) as response:
        return response.read()


def write_once(path, value):
    content = (json.dumps(value, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f'Preserve already frozen file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    api = f'https://huggingface.co/api/datasets/{REPO}/tree/{COMMIT}/MVTec-AD/tile?recursive=true&limit=1000'
    inventory = json.loads(fetch(api))
    files = {r['path']: r for r in inventory if r['type'] == 'file' and r['path'].endswith('.png')}
    normals = sorted((p for p in files if '/train/good/' in p), key=lambda p: sha((SEED + p).encode()))
    if len(normals) != 58:
        raise RuntimeError('Insufficient original normal training photos')
    groups = {'normal_fit': normals[:48], 'normal_calibration': normals[48:58],
              'test': sorted(p for p in files if '/test/' in p)}
    if len(groups['test']) != 117 or sum('/crack/' in p for p in groups['test']) != 17:
        raise RuntimeError('Expected full original 117-photo tile test, including17 cracks')
    chosen = []
    for partition, paths in groups.items():
        for path in paths:
            mask = None
            defect = PurePosixPath(path).parent.name
            if partition == 'test' and defect != 'good':
                mask = path.replace('/test/', '/ground_truth/').replace('.png', '_mask.png')
                if mask not in files:
                    raise RuntimeError('Missing original pixel mask')
            chosen.append({'partition': partition, 'source_path': path, 'defect_type': defect,
                           'sha256': files[path]['lfs']['oid'], 'bytes': files[path]['size'],
                           'mask_source_path': mask, 'mask_sha256': files[mask]['lfs']['oid'] if mask else None})
    protocol = {'schema_version': 1, 'status': 'Frozen before pixel viewing or model inference',
                'official_source': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
                'mirror': f'https://huggingface.co/datasets/{REPO}/tree/{COMMIT}',
                'mirror_role': 'MMAD research-author redistribution; identity checked against pinned mirror LFS, not independent manufacturer archive checksum',
                'license': 'Original MVTec AD CC BY-NC-SA4.0; noncommercial/share-alike',
                'media_type': 'real_manufacturing_still_photos', 'video_available': False,
                'scope': 'Tile texture manufacturing surface anomalies. Not installed flooring, grout, lippage, adhesion, finish acceptance or SBC compliance.',
                'fit_rule': 'Mirror contains58 original train/good PNGs (not the full230 official training set). First48 by SHA256(seed+source_path) fit; next10 normal-only calibration. Mirror captions/derivatives excluded.',
                'seed': SEED, 'test_rule': 'ALL117 original tile test photos; all17 crack photos and all other defect types, no test selection/tuning',
                'model_plan': {'encoder': 'Existing safely loaded official MobileNetV3-small ImageNet early layers0,3; new256 geometry',
                               'bank': 'Deterministic up-to1024 normal patch descriptors from48 fit photos; no test/defect fitting',
                               'threshold': '95th percentile per-photo maximum anomaly score over10 normal calibration photos; fixed before test',
                               'output_label': 'tile_surface_anomaly (not semantic crack classification)',
                               'metrics': 'Pixel foreground P/R/IoU and derived region IoU.50 by defect; normal-image FPR; image AUROC; actual overlays; speed separately'}
                , 'records': chosen}
    write_once(OUT / 'protocol-before-acquisition.json', protocol)
    jobs = {p for r in chosen for p in [r['source_path'], r['mask_source_path']] if p}

    def acquire(path):
        rel = PurePosixPath(path).relative_to('MVTec-AD/tile')
        target = OUT / 'original' / str(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = target.read_bytes() if target.exists() else fetch(f'https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}/{path}')
        if len(blob) != files[path]['size'] or sha(blob) != files[path]['lfs']['oid']:
            raise RuntimeError('Mirror original size/hash mismatch: ' + path)
        if not target.exists():
            target.write_bytes(blob)
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(acquire, sorted(jobs)), 1):
            if index % 25 == 0:
                print('verified originals', index, '/', len(jobs), flush=True)
    manifest = {**protocol, 'status': 'Acquired original photos; no model fit/inference/evaluation yet',
                'protocol_sha256': sha((OUT / 'protocol-before-acquisition.json').read_bytes()),
                'files': len(jobs), 'photo_count': len(chosen), 'mask_count': len(jobs) - len(chosen),
                'total_original_bytes': sum(files[p]['size'] for p in jobs)}
    write_once(OUT / 'asset-manifest.json', manifest)
    print(json.dumps({'manifest': str(OUT / 'asset-manifest.json'), 'sha256': sha((OUT / 'asset-manifest.json').read_bytes()),
                      'photos': len(chosen), 'files': len(jobs)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
