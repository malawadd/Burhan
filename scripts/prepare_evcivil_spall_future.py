"""Separate future real-video spall experiment; no NN or active dataset edits."""
from pathlib import Path
import hashlib,json,struct,zipfile,zlib
import cv2
import h5py
import numpy as np
from fetch_construction_bridge_video import RangeFile
from construction_annotations import publisher_labels_at_or_after

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction'
OUT=DATA/'evcivil-spall-future-r4'
SEED='evcivil-spall-future-20261020:'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl();cv2.setNumThreads(1)
    proposal=json.loads((DATA/'evcivil/unused-train-day-inventory-proposal-round3.json').read_text(encoding='utf-8'))
    candidates=[r for r in proposal['all_day_sequence_inventory'] if r['class']=='spalling' and r['status']=='conditional_exposed_capture_group_not_allowed_current_protocol']
    ordered={group:sorted([r for r in candidates if r['group']==group],key=lambda r:hashlib.sha256((SEED+r['group']+'/'+r['canonical_id']).encode()).hexdigest()) for group in ['dset_1','dset_2']}
    selected=[row for group in ordered.values() for row in group[:6]]
    if len(selected)!=12:raise RuntimeError('Need exactly six metadata-selected spall IDs per source folder')
    if sum(r['compressed_bytes'] for r in selected)>350_000_000:raise RuntimeError('Bounded acquisition exceeds350MB')
    record_path=DATA/'evcivil/publisher-record.json';publisher=json.loads(record_path.read_text(encoding='utf-8'))
    if publisher['license']['name']!='CC BY 4.0':raise RuntimeError('Publisher license differs')
    archive=next(row for row in publisher['files'] if row['name']=='evCIVIL_Field_dataset.zip')
    protocol={'experiment':'future-spall-data-r4','created_date':'2026-10-20','status':'DATA PROPOSAL ONLY; never active fit/calibration/heldout',
              'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0','seed':SEED,
              'selection':'Six IDs per dset1/2, first by fixed SHA256(seed+group/ID), before pixels/inference; full ordered candidates preserved. Acquisition itself does not clear duplicates.',
              'reserved_canonical_ids':proposal['reserved_canonical_ids'],
              'whole_eval_capture_groups_excluded':['dset_5','dset_6','night_outdoor_jan_2','night_outdoor_18th_Nov'],
              'old_exposed_cases':'crack10/crack89/spall35 never fit. Other1/2 cases share exposed capture domains, so old exposed clips are DEVELOPMENT ONLY in any future fit using these assets.',
              'independence':'Different canonical IDs do not prove distinct physical defects/projects. Pixel/perceptual duplicate checks and visual review required; no metadata-only clearance.',
              'annotation_rule':'Shared publisher first-at-or-after timestamp association, gap<=median camera period/2+1000us; clip actual boxes to source pixels; qualify only frames with spalling target. Preserve all crack/spall boxes in those frames.',
              'sample_rule':'At most24 uniformly selected qualified original camera exposures per sequence. Original H5/NPY and all qualified indices/PTS retained. No photo-to-video encoding or synthetic data.',
              'archive_url':archive['download_url'],'archive_size':archive['size'],'source_record_sha256':sha(record_path),
              'candidate_order':{key:[r['canonical_id'] for r in rows] for key,rows in ordered.items()},
              'selected':selected,'compressed_byte_budget':350_000_000}
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=OUT/'protocol-before-acquisition.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Frozen future-data plan changed')
    write(protocol_path,protocol)
    manifest_path=OUT/'asset-manifest.json'
    if manifest_path.exists():raise RuntimeError('Preserve completed future-data acquisition')
    assets=[]
    with zipfile.ZipFile(RangeFile(archive['download_url'],archive['size'])) as source:
        for row in selected:
            members=[]
            for expected in row['members']:
                entry=source.getinfo(expected['name'])
                if (entry.file_size,entry.compress_size)!=(expected['bytes'],expected['compressed_bytes']):raise RuntimeError('Publisher member metadata changed')
                target=OUT/'raw'/row['group']/row['canonical_id']/Path(entry.filename).name
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    stream=RangeFile(archive['download_url'],archive['size']);stream.seek(entry.header_offset);header=stream.read(30)
                    if header[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid original ZIP header')
                    nl,el=struct.unpack_from('<HH',header,26);stream.seek(entry.header_offset+30+nl+el);blob=stream.read(entry.compress_size)
                    raw=zlib.decompress(blob,-15) if entry.compress_type==8 else blob if entry.compress_type==0 else None
                    if raw is None or len(raw)!=entry.file_size or zlib.crc32(raw)!=entry.CRC:raise RuntimeError('Original member integrity failed')
                    target.write_bytes(raw)
                raw=target.read_bytes()
                if len(raw)!=entry.file_size or zlib.crc32(raw)!=entry.CRC:raise RuntimeError('Existing original member differs')
                members.append({'path':target.relative_to(OUT).as_posix(),'sha256':sha(target),'bytes':len(raw),'zip_member':entry.filename,'crc32':f'{entry.CRC:08x}'})
            assets.append({**row,'original_members':members})
            print('Downloaded',row['group'],row['canonical_id'],flush=True)
    exposures=[];sequence_summary=[]
    for row in assets:
        folder=OUT/'raw'/row['group']/row['canonical_id'];name=row['canonical_id']
        camera=folder/(name+'_frames.h5');label=folder/(name+'_label.npy')
        with h5py.File(camera,'r') as source:
            frames=source['frames'][:];times=source['frame_timestamp'][:].astype(np.int64)
        if len(frames)!=len(times) or len(times)<2 or np.any(np.diff(times)<=0):raise RuntimeError('Original camera timing must be strictly increasing')
        period=float(np.median(np.diff(times)));labels=np.load(label,allow_pickle=False);qualified=[]
        for index,timestamp in enumerate(times):
            h,w=frames[index].shape[:2];boxes=[]
            for b in publisher_labels_at_or_after(labels,timestamp,period):
                if int(b[1]) not in {0,1}:raise RuntimeError('Unsupported publisher class ID')
                x,y,bw,bh=map(float,b[2:6]);right=min(w,x+bw);bottom=min(h,y+bh);x=max(0,x);y=max(0,y)
                if right-x>=2 and bottom-y>=2:
                    boxes.append({'class_id':1 if int(b[1])==0 else 0,'label':'crack' if int(b[1])==0 else 'spalling','bbox':[x,y,right-x,bottom-y],'annotation_timestamp_us':int(b[0])})
            if any(b['class_id']==0 for b in boxes):qualified.append({'original_frame_index':index,'timestamp_us':int(timestamp),'source_pts_us':int(timestamp-times[0]),'boxes':boxes})
        chosen=[qualified[i] for i in np.unique(np.linspace(0,len(qualified)-1,min(24,len(qualified))).round().astype(int))] if qualified else []
        for item in chosen:
            frame=frames[item['original_frame_index']];stem=row['group']+'-'+name+f"-frame{item['original_frame_index']:05d}"
            image=OUT/'images/proposed_train'/(stem+'.png');annotation=OUT/'labels/proposed_train'/(stem+'.txt')
            image.parent.mkdir(parents=True,exist_ok=True);annotation.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(image),frame):raise RuntimeError('Failed to preserve camera exposure')
            h,w=frame.shape[:2];lines=[]
            for b in item['boxes']:
                x,y,bw,bh=b['bbox'];lines.append(f"{b['class_id']} {(x+bw/2)/w:.10f} {(y+bh/2)/h:.10f} {bw/w:.10f} {bh/h:.10f}")
            annotation.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            exposures.append({**item,'source_group':row['group'],'canonical_id':name,'image':image.relative_to(OUT).as_posix(),
                              'image_sha256':sha(image),'label':annotation.relative_to(OUT).as_posix(),'label_sha256':sha(annotation),
                              'pixel_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'source_shape':list(frame.shape),
                              'camera_source_sha256':sha(camera),'label_source_sha256':sha(label),'period_us':period})
        sequence_summary.append({'group':row['group'],'canonical_id':name,'original_camera_frames':len(frames),'qualified_spall_exposures':len(qualified),
                                 'selected_exposures':len(chosen),'original_start_timestamp_us':int(times[0]),'median_period_us':period,'all_qualified':qualified})
        print('Qualified',name,len(qualified),'selected',len(chosen),flush=True)
    report={'protocol_sha256':sha(protocol_path),'status':'UNUSED proposed future fitting data; duplicate/visual review pending',
            'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0','classes':{'0':'spalling','1':'crack'},
            'source_assets':assets,'sequence_summary':sequence_summary,'exposures':exposures}
    write(manifest_path,report)
    print(json.dumps({'manifest':str(manifest_path),'sha256':sha(manifest_path),'sequences':len(assets),'qualified_selected_exposures':len(exposures)},indent=2))


if __name__=='__main__':main()
