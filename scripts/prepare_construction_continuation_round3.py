"""Version a train-only continuation after the failed first calibration.

The original fit, calibration, test and exposed video files stay intact. This
script adds actual supplementary fit data; it performs no neural inference and
must run outside any exclusive speed benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'datasets/construction/evcivil-round3'
NEW=ROOT/'datasets/construction/evcivil-round3-continuation'
SUP=ROOT/'datasets/construction/dacl10k/round3-positive-contingency'
NIGHT=ROOT/'datasets/construction/evcivil-round3-contingency/raw/night_dec27'
POSITIVE_SHA='2dc962b359bba28ba8eb1871184c2c2c5d8aa841cf4540d393fccd150276c360'
NIGHT_SHA='9b876fac3b7f2a103527e13fa6b355bcc5e0ce96d83c212cdd2b8f9cdb85f558'
NIGHT_IDS=[6,7,8,13,15]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')


def main():
    import cv2
    import h5py
    import numpy as np
    from construction_annotations import publisher_labels_at_or_after
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--closeup-manifest',type=Path,required=True);parser.add_argument('--closeup-sha',required=True);args=parser.parse_args()
    cv2.setNumThreads(1);NEW.mkdir(exist_ok=True,parents=True)
    first_op=ROOT/'models/construction/evcivil-yolov8n-r3.operating-point.json'
    diagnosis=ROOT/'output/evaluation/construction-round3/evcivil-yolov8n-r3/fit-diagnostics/summary.json'
    if not first_op.exists() or not diagnosis.exists():raise RuntimeError('Actual first fit/calibration diagnosis required before continuation')
    protocol=json.loads((OLD/'protocol.json').read_text(encoding='utf8'))
    previous=json.loads((OLD/'exposures.json').read_text(encoding='utf8'))
    # Canonical identity, not file prefix: all old fit/cal/test IDs, plus heldout
    # Jan2 IDs and the exposed original challenges, are forbidden additions.
    reserved=set(protocol['global_identity_split'])|{'crack_10','crack_89','spalling_35'}
    night_holdout=json.loads((OLD/'night-holdout-protocol.json').read_text(encoding='utf8'))
    reserved|={re.search(r'(crack|spalling)_\d+',r['sequence']).group(0) for r in night_holdout['sequences']}
    if digest(SUP/'asset-manifest.json')!=POSITIVE_SHA:raise RuntimeError('Supplement manifest checksum mismatch')
    night_manifest=NIGHT.parents[1]/'night-dec27-asset-manifest.json'
    if digest(night_manifest)!=NIGHT_SHA:raise RuntimeError('Night supplement manifest checksum mismatch')
    for member in json.loads(night_manifest.read_text(encoding='utf8'))['members']:
        path=(NIGHT.parents[1]/member['relative_path']).resolve()
        if not path.is_relative_to(NIGHT.parents[1].resolve()) or digest(path)!=member['sha256']:raise RuntimeError('Night source path/checksum mismatch')
    positives=json.loads((SUP/'asset-manifest.json').read_text(encoding='utf8'))
    if len(positives['records'])!=80:raise RuntimeError('Frozen positive selection must contain all80 photos')
    closeup_manifest=args.closeup_manifest.resolve()
    if not closeup_manifest.is_relative_to((ROOT/'datasets/construction').resolve()) or digest(closeup_manifest)!=args.closeup_sha:raise RuntimeError('Closeup manifest path/checksum mismatch')
    closeups=json.loads(closeup_manifest.read_text(encoding='utf8'))
    if len(closeups['records'])!=40:raise RuntimeError('Need all40 frozen real crack closeup views')
    original_sources={r['original_image_sha256'] for r in positives['records']}
    if not all(r['original_image_sha256'] in original_sources for r in closeups['records']):raise RuntimeError('Closeup introduced an unselected original photograph')
    positive_rows=[(SUP,r) for r in positives['records']]+[(closeup_manifest.parent,r) for r in closeups['records']]
    if any(r.get('source_split')!='train' for _,r in positive_rows):raise RuntimeError('Photograph supplement must remain official train only')
    if len({r['original_image_sha256'] for r in positives['records']})!=80:raise RuntimeError('Original positive sources are not unique')
    if len({r['original_image_sha256'] for r in closeups['records']})!=40:raise RuntimeError('Closeup selection must use40 distinct source photos')
    photo_dirs=sorted({(base/r['derived']['image']).parent.as_posix() for base,r in positive_rows})
    plan={'schema_version':1,'candidate':'evcivil-yolov8n-r3-adapted','frozen_before_candidate_training':True,
          'frozen_before_round3_model_training':False,'prior_failed_candidate_sha256':'19137c9ff183d65f389c27abdcc8536a4b1aa79578054c55ce8d7b64b0824410',
          'selection_basis':'Actual fit/calibration domain gap; no heldout/challenge model predictions opened',
          'source_protocol_sha256':digest(OLD/'protocol.json'),'source_exposures_sha256':digest(OLD/'exposures.json'),
          'first_operating_point_sha256':digest(first_op),'fit_cal_diagnosis_sha256':digest(diagnosis),
          'positive_manifest_sha256':POSITIVE_SHA,'closeup_manifest_sha256':args.closeup_sha,'closeup_views':40,'unique_original_positive_photos':80,'night_asset_manifest_sha256':NIGHT_SHA,'night_ids':NIGHT_IDS,'reserved_canonical_ids':sorted(reserved),
          'configuration':{'input_size':416,'freeze':4,'batch':4,'nominal_batch':4,'cpu_threads':3,'learning_rate':.0005,'warmup_epochs':0,'max_epochs':12,'patience':8,'wall_time_cap':None,'training_contention':'May overlap fixed-step one-thread factory fit; no inference timing claim from training duration'},
          'split_scope':'Only training expanded; original calibration/heldout and exposed exclusions retained exactly. Source-folder/global-ID split; physical project independence unverified.',
          'licenses':'ev-CIVIL CC BY4.0; DACL10k author CC BY-NC4.0; initial Ultralytics/crack-joint checkpoint lineage AGPL and source-training-license caveats remain.'}
    plan_path=NEW/'protocol.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf8'))!=plan:raise RuntimeError('Refusing to change frozen continuation plan')
    write(plan_path,plan);records=list(previous);cross_split={r['sha256_pixels'] for r in previous if r['split']!='train'};night_counts={}
    for number in NIGHT_IDS:
        name=f'27_crack_{number}';canonical=f'crack_{number}'
        if canonical in reserved:raise RuntimeError('Supplemental global defect ID overlaps an existing split/challenge')
        folder=NIGHT/name
        with h5py.File(folder/f'{name}_frames.h5','r') as source:frames=source['frames'][:];times=source['frame_timestamp'][:].astype(np.int64)
        label_path=folder/f'{name}_frame_label.npy';labels=np.load(label_path,allow_pickle=False)
        period=float(np.median(np.diff(times)));eligible=[]
        for index,timestamp in enumerate(times):
            boxes=[];h,w=frames.shape[1:]
            for b in publisher_labels_at_or_after(labels,timestamp,period):
                x,y,bw,bh=map(float,b[2:6]);right=min(w,x+bw);bottom=min(h,y+bh);x=max(0,x);y=max(0,y)
                if right-x>=2 and bottom-y>=2:boxes.append({'label':'crack' if int(b[1])==0 else 'spalling','class_id':1 if int(b[1])==0 else 0,'bbox':[x,y,right-x,bottom-y],'annotation_timestamp_us':int(b[0])})
            if boxes:eligible.append((index,boxes))
        selected=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(24,len(eligible))).round().astype(int))] if eligible else []
        night_counts[name]={'original_camera_frames':len(frames),'labelled_exposures':len(eligible),'selected_exposures':len(selected),'median_frame_period_us':period}
        for index,boxes in selected:
            frame=frames[index];sha=hashlib.sha256(frame.tobytes()).hexdigest()
            if sha in cross_split:raise RuntimeError('Supplemental pixels duplicate calibration/heldout')
            stem=f'night_dec27-{name}-frame{index:05d}';image=NEW/'images/train'/f'{stem}.png';label=NEW/'labels/train'/f'{stem}.txt'
            image.parent.mkdir(exist_ok=True,parents=True);label.parent.mkdir(exist_ok=True,parents=True)
            if not cv2.imwrite(str(image),frame):raise RuntimeError('Failed to save actual night exposure')
            lines=[]
            for b in boxes:
                x,y,bw,bh=b['bbox'];lines.append(f"{b['class_id']} {(x+bw/2)/w:.8f} {(y+bh/2)/h:.8f} {bw/w:.8f} {bh/h:.8f}")
            label.write_text('\n'.join(lines)+'\n',encoding='utf8')
            records.append({'split':'train','source_group':'night_dec27_supplement','global_defect_id':name,'canonical_defect_id':canonical,'original_frame_index':index,'original_timestamp_us':int(times[index]),'source_frame_period_us':period,'image':str(image.relative_to(ROOT)),'sha256_pixels':sha,'boxes':boxes,'source_kind':'genuine camera exposure','annotation_sha256':digest(label_path)})
    for positive_root,row in positive_rows:
        derived=row['derived'];image=positive_root/derived['image'];label=positive_root/derived['label']
        if digest(image)!=derived['image_sha256'] or digest(label)!=derived['label_sha256']:raise RuntimeError('Supplement image/label checksum mismatch')
        boxes=[]
        for item in derived['instances']:
            x,y,right,bottom=item['output_projected_source_bbox_xyxy'];category=int(item['class_id'])
            if right<=x or bottom<=y:raise RuntimeError('Invalid projected publisher-polygon box')
            boxes.append({'class_id':category,'label':['spalling','crack'][category],'bbox':[x,y,right-x,bottom-y],'source_polygon_id':item['source_polygon_id'],'source_label':item['source_label']})
        gray=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise RuntimeError('Unreadable real positive crop')
        records.append({'split':'train','source_group':'dacl_official_train_positive','global_defect_id':row['source_name'],'original_frame_index':0,'image':str(image.relative_to(ROOT)),'sha256_pixels':hashlib.sha256(gray.tobytes()).hexdigest(),'image_sha256':derived['image_sha256'],'boxes':boxes,'source_kind':'Real photograph training crop; never video evidence','source_image_sha256':row['original_image_sha256'],'annotation_scope':derived['annotation_scope']})
    for split in ['calibration','heldout']:
        old_rows=[r for r in previous if r['split']==split];new_rows=[r for r in records if r['split']==split]
        if old_rows!=new_rows:raise RuntimeError('Continuation changed evaluation examples')
    write(NEW/'exposures.json',records)
    for name in ['background-import.json','night-holdout-protocol.json']:(NEW/name).write_bytes((OLD/name).read_bytes())
    train=[(OLD/'images/train').as_posix()]+photo_dirs
    if any(r['selected_exposures'] for r in night_counts.values()):train.append((NEW/'images/train').as_posix())
    import yaml
    (NEW/'dataset.yaml').write_text(yaml.safe_dump({'path':NEW.as_posix(),'train':train,'val':(OLD/'images/calibration').as_posix(),'test':(OLD/'images/heldout').as_posix(),'names':{0:'spalling',1:'crack'}}),encoding='utf8')
    result={'night':night_counts,'unique_positive_photographs':80,'derived_positive_training_views':120,'video_exposures':sum(r.get('source_kind')!='Real photograph training crop; never video evidence' for r in records),'total_positive_fit_exposures':sum(r['split']=='train' for r in records),'calibration_exposures':sum(r['split']=='calibration' for r in records),'heldout_sampled_exposures':sum(r['split']=='heldout' for r in records),'protocol_sha256':digest(plan_path),'exposures_sha256':digest(NEW/'exposures.json')}
    write(NEW/'preparation-summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
