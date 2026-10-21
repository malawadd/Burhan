"""Prefrozen additional real low-light recordings: heldout only, never fit/calibration."""
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import urlopen
import zipfile
import zlib
from fetch_construction_bridge_video import RangeFile
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'datasets/construction/evcivil-round3'

if __name__=='__main__':
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    rows=[{'group':'night_jan2','sequence':f'2_jan_crack_{i}','prefix':f'Field_dataset/Night_outdoor/night_outdoor_jan_2/2_jan_crack_{i}/'} for i in [32,43,44]]
    protocol={'frozen_before_new_model_training_or_inference':True,'split':'additional_fresh_night_heldout_only','source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0',
              'selection':'All three genuine Jan2 low-light recordings, selected by capture condition before new NN; no model-based sample choice. Global crack IDs32/43/44 absent fit/cal sources.',
              'training_or_calibration':False,'thin_box_criterion':'short side <=30 original pixels and long side >=2*short side; no physical-width/severity inference','sequences':rows}
    target=BASE/'night-holdout-protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Frozen night protocol differs')
    target.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    with urlopen('https://api.figshare.com/v2/articles/25924228') as response:record=json.load(response)
    archive=next(f for f in record['files'] if f['name']=='evCIVIL_Field_dataset.zip')
    with zipfile.ZipFile(RangeFile(archive['download_url'],archive['size'])) as source:
        for row in rows:
            for entry in source.infolist():
                if not entry.filename.startswith(row['prefix']) or not entry.filename.endswith(('_frames.h5','_label.npy','_frame_label.npy')):continue
                destination=BASE/'raw'/row['group']/row['sequence']/Path(entry.filename).name;destination.parent.mkdir(exist_ok=True,parents=True)
                if not destination.exists():
                    member=RangeFile(archive['download_url'],archive['size']);member.seek(entry.header_offset);header=member.read(30)
                    nl,el=struct.unpack_from('<HH',header,26);member.seek(entry.header_offset+30+nl+el);blob=member.read(entry.compress_size)
                    raw=zlib.decompress(blob,-15) if entry.compress_type==8 else blob if entry.compress_type==0 else None
                    if raw is None or len(raw)!=entry.file_size or zlib.crc32(raw)!=entry.CRC:raise RuntimeError('Night member integrity failure')
                    destination.write_bytes(raw)
                raw=destination.read_bytes()
                if len(raw)!=entry.file_size or zlib.crc32(raw)!=entry.CRC:raise RuntimeError('Existing night member integrity failure')
                destination.with_suffix(destination.suffix+'.provenance.json').write_text(json.dumps({'source':protocol['source'],'split':protocol['split'],'zip_member':entry.filename,'zip_crc32':f'{entry.CRC:08x}','sha256':hashlib.sha256(raw).hexdigest()},indent=2),encoding='utf8')
                print(destination,flush=True)
