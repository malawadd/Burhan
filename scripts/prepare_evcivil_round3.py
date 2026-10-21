"""Fetch and prepare source-group-separated, real crack/spall video supervision.

No challenge frame enters train/calibration. This script never trains a model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from urllib.request import urlopen
import zipfile
import zlib
from fetch_construction_bridge_video import RangeFile
from construction_annotations import publisher_labels_at_or_after

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/construction/evcivil-round3'
SELECTION={
    'train': [('dset_3',n) for n in ['crack_118','crack_119','crack_120','crack_122','crack_123','crack_125','crack_126','spalling_54','spalling_55','spalling_56']]
             + [('dset_4',n) for n in ['crack_140','crack_141','crack_142','crack_143','crack_144','crack_145','crack_146','crack_147','spalling_72']]
             + [('night_dec27',f'27_crack_{i}') for i in [1,2,4,5]],
    'calibration': [('dset_6',n) for n in ['crack_129','crack_130','spalling_75','spalling_76','spalling_77']]
                   + [('night_nov18',f'ros_crack_{i}') for i in [21,22]],
    'heldout': [('dset_5',n) for n in ['crack_196','crack_197','crack_198','spalling_70','spalling_71']],
}


def protocol():
    result={'schema_version':3,'annotation_association':'Publisher first-at-or-after camera timestamp, gap <= median frame period/2 +1000 microseconds; no past-nearest labels', 'prefit_amendment':'Publisher forward-label integrity corrected, then actual Dec27 night fit / Nov18 night calibration capture groups added BEFORE ANY round3 fit/calibration/NN. Jan2 night heldout untouched. No model metric used.', 'dataset':'https://doi.org/10.11583/DTU.25924228','license':'CC BY 4.0',
            'classes':{'0':'spalling','1':'crack'},'frozen_before_round3_model_training':True,
            'excluded_source_groups':['field_data_day/dset_1','field_data_day/dset_2','day_time_challenge','field_data_day/at_eth'],
            'exposed_challenges_never_train_or_calibrate':['crack_10','crack_89','spalling_35'],
            'selection_reason':'Predeclared source-folder and global defect-ID separation, diverse real field video labels; excludes synthetic-labelled records and known duplicate crack_121.',
            'sample_rule':'Uniformly choose at most 24 publisher-annotated original exposures per train sequence, 16 per calibration/heldout sequence; no score-based sampling.',
            'split_scope':'Capture-folder and conservatively global class+numeric-defect-ID separation, not a verified independent project/bridge split. Geographic/physical identity across publisher folders is not fully documented.',
            'sequences':{split:[{'group':group,'sequence':name,'prefix':f'Field_dataset/Night_outdoor/night_outdoor_27th_dec/{name}/' if group=='night_dec27' else f'Field_dataset/Night_outdoor/night_outdoor_18th_Nov/{name}/' if group=='night_nov18' else f'Field_dataset/field_data_day/{group}/{name}/'} for group,name in rows] for split,rows in SELECTION.items()}}
    seen={}
    for split,rows in SELECTION.items():
        for group,name in rows:
            identity=re.search(r'(crack|spalling)_(\d+)',name)
            identity=identity.group(0) if identity else name
            if identity in seen:raise RuntimeError(f'Global defect ID overlaps: {identity}')
            if identity in {'crack_10','crack_89','spalling_35','crack_32','crack_43','crack_44'}:raise RuntimeError('Heldout global defect ID cannot be fit/calibration')
            seen[identity]=split
    result["global_identity_split"] = seen
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--amend-annotation-convention','--amend-prefit-protocol',dest='amend_annotation_convention',action='store_true');args=parser.parse_args()
    BASE.mkdir(exist_ok=True,parents=True);data=protocol();p=BASE/'protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf8'))!=data:
        if not args.amend_annotation_convention:raise RuntimeError('Existing frozen protocol differs; explicit pre-fit annotation amendment required')
        if list((ROOT/'models/construction').glob('evcivil-yolov8n-r3*.onnx')):raise RuntimeError('Cannot amend as pre-fit after a fitted round3 asset exists')
        (BASE/f"protocol-prefit-history-v{json.loads(p.read_text(encoding='utf8')).get('schema_version',1)}.json").write_text(p.read_text(encoding='utf8'),encoding='utf8')

    p.write_text(json.dumps(data,indent=2),encoding='utf8')
    if args.fetch:
        try:
            from pip._vendor import truststore
            truststore.inject_into_ssl()
        except ImportError:pass
        with urlopen('https://api.figshare.com/v2/articles/25924228',timeout=30) as response:record=json.load(response)
        if record['license']['name']!='CC BY 4.0':raise RuntimeError('Publisher license changed')
        archive=next(f for f in record['files'] if f['name']=='evCIVIL_Field_dataset.zip')
        (BASE/'publisher-record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        with zipfile.ZipFile(RangeFile(archive['download_url'],archive['size'])) as source:
            for split,rows in data['sequences'].items():
                for row in rows:
                    for entry in source.infolist():
                        if not entry.filename.startswith(row['prefix']) or not entry.filename.endswith(('_frames.h5','_label.npy','_frame_label.npy')):continue
                        target=BASE/'raw'/row['group']/row['sequence']/Path(entry.filename).name;target.parent.mkdir(exist_ok=True,parents=True)
                        if not target.exists():
                            temporary=target.with_suffix(target.suffix+'.part')
                            # Fetch one complete compressed member in a single bounded range.
                            # ZipExtFile's small read loops otherwise create hundreds of HTTP requests.
                            member=RangeFile(archive['download_url'],archive['size']);member.seek(entry.header_offset)
                            header=member.read(30)
                            if header[:4]!=b'PK\x03\x04' or entry.flag_bits&1:raise RuntimeError('Unsupported/encrypted ZIP member')
                            name_len,extra_len=struct.unpack_from('<HH',header,26)
                            member.seek(entry.header_offset+30+name_len+extra_len)
                            compressed=member.read(entry.compress_size)
                            raw=zlib.decompress(compressed,-15) if entry.compress_type==zipfile.ZIP_DEFLATED else compressed if entry.compress_type==zipfile.ZIP_STORED else None
                            if raw is None or len(raw)!=entry.file_size or zlib.crc32(raw)!=entry.CRC:raise RuntimeError('Raw member CRC/size mismatch')
                            temporary.write_bytes(raw)
                            temporary.replace(target)
                        crc=0;digest=hashlib.sha256()
                        with target.open('rb') as local:
                            while block:=local.read(1024*1024):crc=zlib.crc32(block,crc);digest.update(block)
                        if crc!=entry.CRC or target.stat().st_size!=entry.file_size:raise RuntimeError('Member CRC/size mismatch')
                        target.with_suffix(target.suffix+'.provenance.json').write_text(json.dumps({'source':data['dataset'],'license':data['license'],'split':split,'source_group':row['group'],'global_defect_id':row['sequence'],'zip_member':entry.filename,'zip_crc32':f'{entry.CRC:08x}','sha256':digest.hexdigest()},indent=2),encoding='utf8')
                        print(split,row['sequence'],target.name,flush=True)
    if args.prepare:
        import cv2
        import h5py
        import numpy as np
        cv2.setNumThreads(1);records=[];hashes={}
        previous=BASE/'exposures.json'
        if previous.exists():
            prior=json.loads(previous.read_text(encoding='utf8'))
            if args.amend_annotation_convention:(BASE/'exposures-before-annotation-amendment.json').write_text(previous.read_text(encoding='utf8'),encoding='utf8')
            for row in prior:
                image=(ROOT/row['image']).resolve();label=(BASE/'labels'/row['split']/(image.stem+'.txt')).resolve()
                if not image.is_relative_to((BASE/'images').resolve()) or not label.is_relative_to((BASE/'labels').resolve()):raise RuntimeError('Generated path leaves intended dataset root')
                if not image.name.startswith(('dset_','night_dec27-','night_nov18-')) or image.suffix!='.png' or label.suffix!='.txt':raise RuntimeError('Refusing cleanup of an unowned generated sample')
                image.unlink(missing_ok=True);label.unlink(missing_ok=True)

        for split,rows in data['sequences'].items():
            for row in rows:
                folder=BASE/'raw'/row['group']/row['sequence'];label=folder/f"{row['sequence']}_frame_label.npy"
                if not label.exists():label=folder/f"{row['sequence']}_label.npy"
                labels=np.load(label,allow_pickle=False)
                with h5py.File(folder/f"{row['sequence']}_frames.h5",'r') as source:
                    frames=source['frames'][:];times=source['frame_timestamp'][:].astype(np.int64)
                period=float(np.median(np.diff(times)));eligible=[]
                for index,timestamp in enumerate(times):
                    boxes=publisher_labels_at_or_after(labels,timestamp,period)
                    if not len(boxes):continue
                    valid=[];h,w=frames.shape[1:]
                    for box in boxes:
                        category='crack' if int(box[1])==0 else 'spalling'
                        x,y,bw,bh=map(float,box[2:6]);x1=max(0,x);y1=max(0,y);x2=min(w,x+bw);y2=min(h,y+bh)
                        if x2-x1>=2 and y2-y1>=2:valid.append({'class_id':1 if category=='crack' else 0,'label':category,'bbox':[x1,y1,x2-x1,y2-y1],'annotation_timestamp_us':int(box[0])})
                    if valid:eligible.append((index,valid))
                maximum=24 if split=='train' else 16
                selected=[eligible[i] for i in np.unique(np.round(np.linspace(0,len(eligible)-1,min(maximum,len(eligible)))).astype(int))] if eligible else []
                for index,boxes in selected:
                    frame=frames[index];digest=hashlib.sha256(frame.tobytes()).hexdigest()
                    if digest in hashes and hashes[digest]!=split:raise RuntimeError('Pixel-identical exposure leaks across splits')
                    hashes[digest]=split;stem=f"{row['group']}-{row['sequence']}-frame{index:05d}"
                    image=BASE/'images'/split/(stem+'.png');annotation=BASE/'labels'/split/(stem+'.txt');image.parent.mkdir(exist_ok=True,parents=True);annotation.parent.mkdir(exist_ok=True,parents=True)
                    cv2.imwrite(str(image),frame);lines=[]
                    for box in boxes:
                        x,y,bw,bh=box['bbox'];lines.append(f"{box['class_id']} {(x+bw/2)/w:.8f} {(y+bh/2)/h:.8f} {bw/w:.8f} {bh/h:.8f}")
                    annotation.write_text('\n'.join(lines)+'\n',encoding='utf8')
                    records.append({'split':split,'source_group':row['group'],'global_defect_id':row['sequence'],'canonical_defect_id':re.search(r'(crack|spalling)_(\d+)',row['sequence']).group(0),'source_frame_period_us':period,'original_frame_index':index,'original_timestamp_us':int(times[index]),'image':str(image.relative_to(ROOT)),'sha256_pixels':digest,'boxes':boxes,'label_scope':'Publisher timestamped boxes; no pixel-mask GT. Unannotated exposures excluded, not treated as sound negatives.'})
                print(split,row['sequence'],'eligible',len(eligible),'selected',len(selected),flush=True)
        yaml=f"path: {BASE.as_posix()}\ntrain: images/train\nval: images/calibration\ntest: images/heldout\nnames:\n  0: spalling\n  1: crack\n"
        (BASE/'dataset.yaml').write_text(yaml,encoding='utf8');(BASE/'exposures.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        print('Prepared',len(records),'original selected exposures')


if __name__=='__main__':main()
