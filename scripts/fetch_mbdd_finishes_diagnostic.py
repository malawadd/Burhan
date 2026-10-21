"""Acquire bounded original publisher-labelled wall photos by ZIP ranges.

No fitting or inference, no relabelling leakage as measured moisture, and no
photo-to-video conversion. Dataset source building/session grouping is unknown.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import struct
import time
import xml.etree.ElementTree as ET
import zlib
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from fetch_construction_bridge_video import RangeFile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/construction/finishes-expansion'
OUT = DATA / 'mbdd2025-wall-photos'
SEED = 'wall-finishes-20261020:'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write_once(path, value):
    blob = (json.dumps(value, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() != blob:
        raise RuntimeError('Preserve existing frozen wall data: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    inventory_path = DATA / 'source-records/mbdd2025-publisher-inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    archive = inventory['publisher']['files'][0]
    entries = {r['name']:r for r in inventory['entries']}
    candidates = sorted((r for r in entries if '/Annotations/' in r and r.endswith('.xml')),
                        key=lambda p:sha((SEED+p).encode()))[:128]
    protocol = {'schema_version':1, 'source':'https://zenodo.org/records/15622584', 'license':'CC BY4.0',
                'inventory_sha256':sha(inventory_path.read_bytes()),
                'selection':'First128 originalXMLs by SHA256(seed+member), before reading annotations or pixels; then first4 hashordered images per actual discovered publisher class, deduplicated. All labels preserved; no inference-based selection.',
                'seed':SEED,'annotation_candidates':candidates,'media_type':'original UAV still photos, not original video',
                'purpose':'External wall defect/data geometry diagnostic; not active fit or independent all-stage qualification',
                'label_policy':'Keep original crack/leakage/corrosion/abscission/bulge names exactly. Leakage means publisher visible appearance, not verified active leak or moisture content. Abscission is broad detachment, not verified paint-only defect.',
                'grouping':'Original image identity separated, but building/session identifiers not verified; do not randomsplit adjacent views for independent claims',
                'checksum_scope':'Each acquired originalZIPmember size+CRC32+SHA256 checked. Publisher fullarchive MD5 recorded, not downloaded/verified.'}
    write_once(OUT/'protocol-before-acquisition.json',protocol)

    def acquire(name):
        entry=entries[name]
        path=OUT/'original'/Path(name).relative_to('MBDD2025')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raw=path.read_bytes()
        else:
            # One bounded range includes localheader+payload. Avoid three tiny
            # requests/member and respect publisher429 Retry-After backoff.
            start=entry['header_offset']
            offsets=sorted(r['header_offset'] for r in entries.values() if r['header_offset']>start)
            end=offsets[0]-1 if offsets else start+entry['compressed_bytes']+65535
            for attempt in range(4):
                try:
                    request=Request(archive['links']['self'],headers={'Range':f'bytes={start}-{end}','User-Agent':'InspectFlow-research/1.0'})
                    with urlopen(request,timeout=90) as response:
                        if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                            raise RuntimeError('Publisher range not honored; refusing fullarchive')
                        block=response.read(end-start+2)
                    if len(block)!=end-start+1:raise RuntimeError('Truncated original member range')
                    break
                except HTTPError as error:
                    if error.code!=429 or attempt==3:raise
                    delay=max(20,min(60,int(error.headers.get('Retry-After','30'))))
                    print('Publisher429; waiting',delay,'seconds before bounded retry',flush=True)
                    time.sleep(delay)
            header=block[:30]
            if header[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid ZIP source header')
            nl,el=struct.unpack_from('<HH',header,26)
            compressed=block[30+nl+el:30+nl+el+entry['compressed_bytes']]
            raw=zlib.decompress(compressed,-15) if entry['compression']==8 else compressed if entry['compression']==0 else None
            if raw is None:raise RuntimeError('Unsupported compression')
        if len(raw)!=entry['bytes'] or f'{zlib.crc32(raw):08x}'!=entry['crc32']:
            raise RuntimeError('Publisher original member integrity failed')
        if not path.exists():path.write_bytes(raw)
        return name,raw,path

    annotations=[]
    for candidate in candidates:
        annotations.append(acquire(candidate))
        time.sleep(.35)
    parsed=[]
    for name,blob,path in annotations:
        source=ET.fromstring(blob)
        size=source.find('size')
        labels=[]
        for obj in source.findall('object'):
            bounds=obj.find('bndbox')
            labels.append({'label':obj.findtext('name'),'source_xyxy':[float(bounds.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]})
        parsed.append({'annotation_member':name,'annotation':path.relative_to(OUT).as_posix(),'annotation_sha256':sha(blob),
                       'filename':source.findtext('filename'),'frame_size_wh':[int(size.findtext('width')),int(size.findtext('height'))],
                       'boxes':labels})
    classes=sorted({r['label'] for row in parsed for r in row['boxes']})
    chosen={}
    for label in classes:
        for row in [r for r in parsed if any(b['label']==label for b in r['boxes'])][:4]:
            chosen[row['annotation_member']]=row
    write_once(OUT/'selection-after-annotations-before-pixels.json',{'protocol_sha256':sha((OUT/'protocol-before-acquisition.json').read_bytes()),
                                                                 'discovered_classes':classes,'selected':list(chosen.values())})
    assets=[]
    for row in chosen.values():
        name='MBDD2025/JPEGImages/'+row['filename']
        if name not in entries:
            matches=[p for p in entries if '/JPEGImages/' in p and Path(p).stem==Path(row['annotation_member']).stem]
            if len(matches)!=1:raise RuntimeError('Ambiguous original source image')
            name=matches[0]
        _,blob,path=acquire(name)
        assets.append({**row,'image':path.relative_to(OUT).as_posix(),'image_sha256':sha(blob),'image_member':name})
        print('preserved',name,sorted({b['label'] for b in row['boxes']}),flush=True)
    manifest={'protocol_sha256':sha((OUT/'protocol-before-acquisition.json').read_bytes()),'source':protocol['source'],'license':protocol['license'],
              'media_type':protocol['media_type'],'status':'Original photos/GT acquired, no model inference/fitting',
              'candidate_annotations':len(parsed),'classes':classes,'photos':len(assets),'records':assets,
              'video_members_in_archive':inventory['raw_video_members']}
    write_once(OUT/'asset-manifest.json',manifest)
    print(json.dumps({'path':str(OUT/'asset-manifest.json'),'sha256':sha((OUT/'asset-manifest.json').read_bytes()),'photos':len(assets),'classes':classes},indent=2))


if __name__=='__main__':main()
