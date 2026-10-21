"""Fetch five additional real Dec27 training-only contingency member triples."""
from pathlib import Path
import hashlib
import json
import struct
import zipfile
import zlib
from fetch_construction_bridge_video import RangeFile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'datasets/construction/evcivil-round3-contingency'


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    original = ROOT / 'datasets/construction/evcivil'
    record_path = original / 'publisher-record.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record['license']['name'] != 'CC BY 4.0':
        raise RuntimeError('Publisher source license changed')
    archive = next(row for row in record['files'] if row['name'] == 'evCIVIL_Field_dataset.zip')
    selected = json.loads((original / 'night-dec27-contingency-inventory.json').read_text(encoding='utf-8'))['candidates']
    if {row['id'] for row in selected} != {6,7,8,13,15}:
        raise RuntimeError('Frozen candidate inventory differs')
    protocol = {'source':'https://doi.org/10.11583/DTU.25924228', 'license':'CC BY4.0',
                'source_record_sha256':hashlib.sha256(record_path.read_bytes()).hexdigest(),
                'archive_name':archive['name'], 'archive_url':archive['download_url'], 'archive_size':archive['size'],
                'scope':'Five additional genuine Dec27 camera/frame-label/event-label triples. TRAIN-only contingency; unused, no active protocol, calibration or heldout edits. No model-based selection, no decoding/exposure selection or neural inference.',
                'selected_global_ids':[6,7,8,13,15], 'explicit_excluded_ids':[10,21,22,32,43,44,89],
                'selection':'Complete triples in saved central-directory inventory within requested IDs6–20/23–31, excluding named exposed/calibration/heldout IDs; no model output selection.',
                'integrity':'Byte ranges only; original ZIP central-directory size and CRC32 checked per member, SHA256 recorded. Full10.87GB archive checksum is not verified.',
                'candidate_inventory':selected}
    BASE.mkdir(parents=True,exist_ok=True)
    target = BASE / 'night-dec27-acquisition-protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Contingency protocol changed')
    target.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    rows=[]
    with zipfile.ZipFile(RangeFile(archive['download_url'],archive['size'])) as source:
        for candidate in selected:
            for row in candidate['members']:
                entry=source.getinfo(row['name'])
                if entry.file_size!=row['bytes'] or entry.compress_size!=row['compressed_bytes']:
                    raise RuntimeError('Pinned source member directory changed')
                folder=BASE/'raw/night_dec27'/f"27_crack_{candidate['id']}"
                folder.mkdir(parents=True,exist_ok=True)
                destination=folder/Path(entry.filename).name
                if not destination.exists():
                    member=RangeFile(archive['download_url'],archive['size'])
                    member.seek(entry.header_offset)
                    header=member.read(30)
                    if header[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid ZIP local header')
                    nl,el=struct.unpack_from('<HH',header,26)
                    member.seek(entry.header_offset+30+nl+el)
                    blob=member.read(entry.compress_size)
                    raw=zlib.decompress(blob,-15) if entry.compress_type==8 else blob if entry.compress_type==0 else None
                    if raw is None or len(raw)!=entry.file_size or zlib.crc32(raw)!=entry.CRC:
                        raise RuntimeError('Source ZIP member integrity failed')
                    destination.write_bytes(raw)
                raw=destination.read_bytes()
                if len(raw)!=entry.file_size or zlib.crc32(raw)!=entry.CRC:
                    raise RuntimeError('Existing member integrity failed')
                provenance={'source':protocol['source'],'license':protocol['license'],'archive_url':archive['download_url'],
                            'zip_member':entry.filename,'zip_crc32':f'{entry.CRC:08x}',
                            'bytes':entry.file_size,'sha256':hashlib.sha256(raw).hexdigest(),
                            'relative_path':destination.relative_to(BASE).as_posix(),
                            'scope':'Original unmodified data, training-only unused contingency; no timestamp association or label conversion performed.'}
                destination.with_suffix(destination.suffix+'.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
                rows.append(provenance)
                print(destination.name,entry.file_size,flush=True)
    manifest={'protocol_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'status':'UNUSED TRAINING CONTINGENCY',
              'member_count':len(rows),'sequences':5,'total_source_bytes':sum(row['bytes'] for row in rows),'members':rows}
    path=BASE/'night-dec27-asset-manifest.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'manifest':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'member_count':len(rows)},indent=2))


if __name__=='__main__':
    main()
