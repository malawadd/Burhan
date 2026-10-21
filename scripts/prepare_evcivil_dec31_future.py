"""Separate real Dec31 TRAIN-candidate acquisition; no NN or active data edits."""
from pathlib import Path
import argparse, hashlib, json, re, struct, zipfile, zlib
import cv2, h5py, numpy as np
from fetch_construction_bridge_video import RangeFile
from construction_annotations import publisher_labels_at_or_after

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/construction'
OUT=BASE/'evcivil-dec31-future'
GROUP='night_outdoor_31st_dec'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def canonical(value):
    m=re.fullmatch(r'(?:31_)?(crack|spalling)_(\d+)',str(value))
    return f'{m[1]}_{int(m[2])}' if m else None

def inputs():
    paths=sorted(set(BASE.rglob('exposures.json'))|set(BASE.rglob('night-holdout-protocol.json')))
    paths += [BASE/'evcivil/unused-train-day-inventory-proposal-round3.json', BASE/'evcivil/publisher-record.json',
              BASE/'evcivil/field-member-inventory.json',BASE/'evcivil-round4-broad-adapted-gain/data-protocol.json',
              ROOT/'scripts/construction_annotations.py',ROOT/'edge_runtime/profiles/construction_preprocessing.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def reserved_ids():
    ids={'crack_10','crack_89','spalling_35','crack_121','crack_21','crack_22','crack_32','crack_43','crack_44',
         *[f'crack_{i}' for i in (1,2,4,5,6,7,8,13,15)]}
    def visit(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in {'canonical_defect_id','global_defect_id','canonical_id','sequence'} and canonical(x):ids.add(canonical(x))
                if k in {'reserved_canonical_ids','explicit_excluded_ids'}:
                    for s in x:
                        if canonical(s):ids.add(canonical(s))
                visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    for p in BASE.rglob('exposures.json'):visit(json.loads(p.read_text()))
    for p in BASE.rglob('night-holdout-protocol.json'):visit(json.loads(p.read_text()))
    proposal=json.loads((BASE/'evcivil/unused-train-day-inventory-proposal-round3.json').read_text())
    ids.update(proposal['reserved_canonical_ids'])
    return sorted(ids)

def record_protocol():
    publisher_path=BASE/'evcivil/publisher-record.json';record=json.loads(publisher_path.read_text())
    if record['license']['name']!='CC BY 4.0':raise ValueError('Publisher license changed')
    archive=next(r for r in record['files'] if r['name']=='evCIVIL_Field_dataset.zip')
    inventory=json.loads((BASE/'evcivil/field-member-inventory.json').read_text());reserved=reserved_ids()
    folders=sorted({str(Path(r['name']).parent).replace('\\','/') for r in inventory
                    if f'/{GROUP}/' in r['name'] and r['name'].endswith('_frames.h5')})
    selected=[];excluded=[]
    with zipfile.ZipFile(RangeFile(archive['download_url'],archive['size'])) as z:
        for folder in folders:
            sequence=folder.rsplit('/',1)[-1];ident=canonical(sequence)
            if ident is None:raise ValueError('Unexpected sequence ID')
            if ident in reserved:excluded.append({'sequence':sequence,'canonical_id':ident,'reason':'Protected existing TRAIN/CAL/heldout/exposed or conservative alias union'});continue
            members=[]
            for suffix in ['_frames.h5','_label.npy']:
                e=z.getinfo(f'{folder}/{sequence}{suffix}')
                if e.compress_type not in (0,8):raise ValueError('Unsupported member compression')
                members.append({'name':e.filename,'bytes':e.file_size,'compressed_bytes':e.compress_size,
                                'crc32':f'{e.CRC:08x}','header_offset':e.header_offset,'compress_type':e.compress_type})
            selected.append({'sequence':sequence,'canonical_id':ident,'members':members})
    budget=sum(m['compressed_bytes'] for s in selected for m in s['members'])
    if budget>85_000_000:raise ValueError('Exceeded bounded camera-only acquisition')
    value={'schema_version':1,'experiment':'future-dec31-real-night-train-candidates','status':'DATA PROPOSAL ONLY; no automatic acceptance/fit inclusion',
        'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0','archive_url':archive['download_url'],'archive_size':archive['size'],
        'protected_input_sha256':inputs(),'script_sha256':sha(Path(__file__)),'reserved_canonical_ids':reserved,
        'selected':selected,'excluded':excluded,'compressed_bytes':budget,
        'selection':'ALL complete Dec31 camera+publisher-label pairs with genuinely unused class-specific global IDs, selected before any pixels/darkness/model output. No replacement/relaxed exclusions.',
        'sampling':'At most24 unique uniformly spaced qualified real camera exposures per sequence, regardless of darkness or content. No negatives inferred from unannotated exposures.',
        'association':'Existing shared first-at-or-after original _label.npy timestamp, delta<=median camera period/2+1000us. Dec31 has no separate _frame_label.npy in the original member inventory; no missing frame labels are invented.',
        'pixel_analysis':'Actual native uint8 camera p99 and immutable shared low_light_gain; no modification to raw assets/saved proposed PNGs.',
        'duplicate_analysis':'Exact image/pixel identities against protected fit/CAL/heldout metadata; perceptual screen only already-used TRAIN pixels and cross-proposed sequences. Unresolved flags require review; unknown physical sites remain a limitation.',
        'timing':'Original camera timestamps strictly increasing; source PTS relative to original first camera exposure. No fabricated temporal frames.',
        'scope':'No NN, event streams, active832/880 dataset changes, CAL/heldout pixel opening, weight load or training.'}
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'protocol-before-acquisition.json'
    if target.exists():raise ValueError('Protocol already recorded; preserve immutable history')
    write(target,value);print(json.dumps({'protocol':str(target),'sha256':sha(target),'selected':len(selected),'excluded':len(excluded),'compressed_bytes':budget},indent=2))

def acquire(expected):
    p=OUT/'protocol-before-acquisition.json'
    if sha(p)!=expected:raise ValueError('Protocol SHA mismatch')
    protocol=json.loads(p.read_text())
    if inputs()!=protocol['protected_input_sha256'] or sha(Path(__file__))!=protocol['script_sha256']:raise ValueError('Protected source/code identity changed')
    if (OUT/'asset-manifest.json').exists():raise ValueError('Completed acquisition already exists')
    assets=[];exposures=[];summaries=[]
    for row in protocol['selected']:
        members=[];folder=OUT/'raw'/row['sequence'];folder.mkdir(parents=True,exist_ok=True)
        for entry in row['members']:
            target=folder/Path(entry['name']).name
            if not target.exists():
                stream=RangeFile(protocol['archive_url'],protocol['archive_size']);stream.seek(entry['header_offset']);header=stream.read(30)
                if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid original ZIP header')
                nl,el=struct.unpack_from('<HH',header,26);stream.seek(entry['header_offset']+30+nl+el);blob=stream.read(entry['compressed_bytes'])
                raw=zlib.decompress(blob,-15) if entry['compress_type']==8 else blob
                if len(raw)!=entry['bytes'] or f'{zlib.crc32(raw):08x}'!=entry['crc32']:raise ValueError('Publisher member integrity failed')
                target.write_bytes(raw)
            raw=target.read_bytes()
            if len(raw)!=entry['bytes'] or f'{zlib.crc32(raw):08x}'!=entry['crc32']:raise ValueError('Existing member integrity failed')
            members.append({**entry,'path':target.relative_to(OUT).as_posix(),'sha256':sha(target)})
        assets.append({**row,'original_members':members});print('Original assets',row['sequence'],flush=True)
    # All original assets verified before any candidate pixel/annotation analysis.
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    for row in assets:
        folder=OUT/'raw'/row['sequence'];name=row['sequence']
        labels=np.load(folder/(name+'_label.npy'),allow_pickle=False)
        for a in (labels,):
            if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all():raise ValueError('Invalid publisher annotation array')
        qualified=[];frame_stats=[]
        with h5py.File(folder/(name+'_frames.h5'),'r') as f:
            times=f['frame_timestamp'][:].astype(np.int64);frames=f['frames']
            if len(times)<2 or len(times)!=len(frames) or np.any(np.diff(times)<=0):raise ValueError('Invalid camera timeline')
            period=float(np.median(np.diff(times)))
            for i,t in enumerate(times):
                raw=frames[i]
                if raw.dtype!=np.uint8 or raw.ndim!=2:raise ValueError('Expected original uint8 grayscale camera')
                enhanced,details=camera_exposure(cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR),'low_light_gain')
                stats={'original_frame_index':i,'timestamp_us':int(t),'source_pts_us':int(t-times[0]),'p99':float(np.percentile(raw,99)),
                       'gain':details,'pixel_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'source_shape':list(raw.shape)}
                frame_stats.append(stats);boxes=[];h,w=raw.shape
                for b in publisher_labels_at_or_after(labels,t,period):
                    if int(b[1]) not in (0,1):raise ValueError('Unknown publisher class')
                    x,y,bw,bh=map(float,b[2:]);right=min(w,x+bw);bottom=min(h,y+bh);x=max(0,x);y=max(0,y)
                    if right-x>=2 and bottom-y>=2:boxes.append({'class_id':1 if int(b[1])==0 else 0,'label':'crack' if int(b[1])==0 else 'spalling','bbox':[x,y,right-x,bottom-y],'annotation_timestamp_us':int(b[0]),'association_delta_us':float(b[0]-t)})
                if any(b['label']=='crack' for b in boxes):qualified.append({**stats,'boxes':boxes})
            selected=[qualified[i] for i in np.unique(np.linspace(0,len(qualified)-1,min(24,len(qualified))).round().astype(int))] if qualified else []
            for q in selected:
                image=OUT/'images/proposed_train'/f"{name}-frame{q['original_frame_index']:05d}.png";image.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(image),frames[q['original_frame_index']]):raise ValueError('Failed original exposure write')
                exposures.append({**q,'source_group':GROUP,'sequence':name,'canonical_id':row['canonical_id'],'image':image.relative_to(OUT).as_posix(),'image_sha256':sha(image),'annotation_scope':'Publisher timestamped boxes, no pixel masks; unannotated frames never assumed negative'})
        summaries.append({'sequence':name,'canonical_id':row['canonical_id'],'original_camera_frames':len(times),'median_period_us':period,'label_rows':len(labels),'separate_frame_label_available':False,'qualified_crack_exposures':len(qualified),'selected_exposures':len(selected),
            'all_camera_statistics':frame_stats,'all_qualified':qualified,'gain_applied_camera_exposures':sum(s['gain']>1 for s in frame_stats),'gain_applied_qualified_exposures':sum(s['gain']>1 for s in qualified),
            'gain_applied_selected_exposures':sum(s['gain']>1 for s in selected)})
        print('Qualified',name,len(qualified),'selected',len(selected),'dark',sum(s['gain']>1 for s in selected),flush=True)
    write(OUT/'asset-manifest.json',{'protocol_sha256':sha(p),'status':'UNUSED future proposal; exact/near duplicate and visual acceptance pending','source':protocol['source'],'license':protocol['license'],
        'source_assets':assets,'sequence_summary':summaries,'exposures':exposures})
    print(json.dumps({'manifest_sha256':sha(OUT/'asset-manifest.json'),'source_sequences':len(assets),'proposed_views':len(exposures)},indent=2))

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    from pip._vendor import truststore
    truststore.inject_into_ssl();cv2.setNumThreads(1)
    p=argparse.ArgumentParser();p.add_argument('--record-protocol',action='store_true');p.add_argument('--acquire',action='store_true');p.add_argument('--protocol-sha256');a=p.parse_args()
    if a.record_protocol==a.acquire:p.error('Choose exactly one phase')
    record_protocol() if a.record_protocol else acquire(a.protocol_sha256)
