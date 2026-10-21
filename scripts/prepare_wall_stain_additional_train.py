"""Acquire only MBDD TRAIN-exclusive-block additions; freeze CAL/TEST unchanged.

No owner pixels, neural inference or fitting. Publisher leakage is a visible stain
annotation, not a humidity measurement. Numeric200 blocks are only scene proxies.
"""
from __future__ import annotations
import concurrent.futures,hashlib,json,sys,zlib
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_mbdd_wall_training as source
BASE=ROOT/'datasets/construction/finishes-expansion'
OLD=ROOT/'datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json'
OUT=ROOT/'datasets/construction/expansion-public-v1/wall-stain-additional-train-v1'
SEED='wall-stain-train-exclusive-20261021-v1:'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_once(path,value):
    blob=json.dumps(value,indent=2)+'\n';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists()and path.read_text()!=blob:raise ValueError('Preserve frozen record: '+str(path))
    path.write_text(blob)
def valid(r):
    w,h=r['frame_size_wh']
    return w>0 and h>0 and bool(r['boxes'])and all(np.isfinite(b['source_xyxy']).all()and 0<=b['source_xyxy'][0]<b['source_xyxy'][2]<=w and 0<=b['source_xyxy'][1]<b['source_xyxy'][3]<=h for b in r['boxes'])
def main():
    if (OUT/'prepared/manifest.json').exists():raise ValueError('Addition already prepared; preserve experiment')
    cv2.setNumThreads(1)
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    inv_path=BASE/'source-records/mbdd2025-publisher-inventory.json';old_asset_path=BASE/'mbdd-wall-five-class-v1/asset-manifest.json';diag_path=BASE/'mbdd2025-wall-photos/asset-manifest.json';ann_path=BASE/'mbdd-wall-five-class-v1/original-annotation-range.bin'
    inventory=json.loads(inv_path.read_text());old=json.loads(OLD.read_text());old_assets=json.loads(old_asset_path.read_text());diag=json.loads(diag_path.read_text())
    old_protocol=json.loads((old_asset_path.parent/'protocol-before-source-labels-pixels.json').read_text());ann=ann_path.read_bytes()
    if sha(ann_path)!='2c8f63ac5569e8fc7f495e6c2de76a648457dd7e071441d502340a853fc31a73':raise ValueError('Frozen annotation bytes changed')
    entries={e['name']:e for e in inventory['entries']};offsets=sorted({e['header_offset']for e in entries.values()});roles=defaultdict(set)
    for r in old['records']+old_assets['records']:roles[r['split']].add(r['capture_block_200'])
    diag_blocks={source.block_id(r['annotation_member'])for r in diag['records']}
    forbidden=roles['calibration']|roles['test']|diag_blocks;allowed=roles['train']-forbidden
    exposed={Path(r['original']).name for r in old['records']}|{r['source_name']for r in old_assets['records']}|{Path(r['annotation_member']).stem+'.jpg'for r in diag['records']}
    candidates=[];annotation_receipts=[]
    for entry in entries.values():
        if '/Annotations/'not in entry['name']or not entry['name'].endswith('.xml'):continue
        raw=source.unpack(entry,ann,old_protocol['annotation_range']['start']);r=source.parsed(entry,raw)
        annotation_receipts.append({'member':entry['name'],'sha256':hashlib.sha256(raw).hexdigest(),'crc32':entry['crc32']})
        if r['capture_block_200']not in allowed or r['source_name']in exposed or not valid(r):continue
        r['image_member']='MBDD2025/JPEGImages/'+r['source_name']
        if r['image_member']in entries:candidates.append(r)
    order=lambda r:hashlib.sha256((SEED+r['source_name']).encode()).hexdigest()
    positives=sorted([r for r in candidates if any(b['label']=='leakage'for b in r['boxes'])],key=order)[:200]
    negatives=sorted([r for r in candidates if not any(b['label']=='leakage'for b in r['boxes'])],key=order)[:len(positives)]
    selection=[{**r,'split':'train'}for r in positives+negatives]
    protocol={'source':old['source'],'license':old['license'],'script_sha256':sha(__file__),'old_manifest_sha256':sha(OLD),'old_assets_sha256':sha(old_asset_path),'protected_diagnostics_sha256':sha(diag_path),'source_inventory_sha256':sha(inv_path),'annotation_range_sha256':sha(ann_path),
        'initial_strict_policy_result':'No blocks unused by all prior TRAIN/CAL/TEST/protected roles: all73 source blocks already represented.',
        'approved_revised_policy':'Use additional original photos ONLY from old TRAIN blocks absent from ALL old CAL/TEST/protected blocks. No independence gain claimed. Preserve old roles/bytes; no model-based selection.',
        'allowed_train_exclusive_blocks':sorted(allowed),'forbidden_blocks':sorted(forbidden),'existing_train_cal_overlap_blocks':sorted(roles['train']&roles['calibration']),'existing_train_test_overlap_blocks':sorted(roles['train']&roles['test']),
        'available_valid_unused_positive':sum(any(b['label']=='leakage'for b in r['boxes'])for r in candidates),'available_valid_unused_other_defect_negative':sum(not any(b['label']=='leakage'for b in r['boxes'])for r in candidates),
        'seed':SEED,'selection_policy':'Allup to200 validunusedleakagephotos, equalcountotherdefectnegatives orderedbySHA256(seed+source_name); frozen before pixels. No backfill after quarantine.',
        'quarantine':'New TRAIN endpoint quarantined on originalSHA or63bitDCTpHash distance<=6 to ANY oldCAL/TEST/protected; duplicate newTRAIN endpoints removed bysame rule. Other-defect negatives are not verified healthywalls.',
        'cal_test_unchanged':True,'private_owner_videos_used':False,'selection':selection,'original_xml_crc_verified_members':len(annotation_receipts)}
    write_once(OUT/'protocol-before-photo-acquisition.json',protocol);write_once(OUT/'annotation-crc-sha-receipts.json',annotation_receipts)
    url=inventory['publisher']['files'][0]['links']['self']
    def acquire(row):
        entry=entries[row['image_member']];target=OUT/'original'/row['source_name'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raw=target.read_bytes()
        else:
            start=entry['header_offset'];end=min(o for o in offsets if o>start)-1
            raw=source.unpack(entry,source.byte_range(url,start,end),start);target.write_bytes(raw)
        if len(raw)!=entry['bytes']or f'{zlib.crc32(raw):08x}'!=entry['crc32']:raise ValueError('OriginalimageCRC/size mismatch')
        frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if frame is None or list(frame.shape[1::-1])!=row['frame_size_wh']:raise ValueError('XML/source geometry mismatch')
        return {**row,'original':target.relative_to(ROOT).as_posix(),'original_sha256':sha(target),'phash63':source.visual_hash(frame),'crc32':entry['crc32'],'original_bytes':len(raw)}
    acquired=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        for row in pool.map(acquire,selection):
            acquired.append(row);print('NEW_TRAIN_CRC_VERIFIED',len(acquired),'/',len(selection),flush=True)
    protected=[]
    for r in old['records']:
        if r['split']=='train':continue
        p=ROOT/r['original'];protected.append({'id':r['id'],'split':r['split'],'sha256':sha(p),'phash63':source.visual_hash(cv2.imread(str(p)))})
    for r in diag['records']:
        p=diag_path.parent/r['image'];protected.append({'id':r['filename'],'split':'protected_diagnostic','sha256':sha(p),'phash63':source.visual_hash(cv2.imread(str(p)))})
    quarantine=[];kept=[]
    for r in acquired:
        hit=None
        refs=protected+[{'id':k['source_name'],'split':'new_train_duplicate','sha256':k['original_sha256'],'phash63':k['phash63']}for k in kept]
        for ref in refs:
            distance=sum(a!=b for a,b in zip(r['phash63'],ref['phash63']))
            if r['original_sha256']==ref['sha256']or distance<=6:
                hit={'source_name':r['source_name'],'reference':ref['id'],'reference_role':ref['split'],'phash_distance':distance,'exact_sha':r['original_sha256']==ref['sha256']};break
        if hit:quarantine.append(hit)
        else:kept.append(r)
    additions=[]
    for row in kept:
        image=cv2.imread(str(ROOT/row['original']));h,w=image.shape[:2];ratio=min(640/w,640/h);nw,nh=round(w*ratio),round(h*ratio);px,py=(640-nw)//2,(640-nh)//2
        canvas=np.full((640,640,3),114,np.uint8);canvas[py:py+nh,px:px+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
        path=OUT/'prepared/images/train'/row['source_name'];path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
        boxes=[]
        for b in row['boxes']:
            if b['label']!='leakage':continue
            x1,y1,x2,y2=b['source_xyxy'];boxes.append({'class_id':0,'bbox':[x1*nw/w+px,y1*nh/h+py,(x2-x1)*nw/w,(y2-y1)*nh/h],'source_label':'leakage'})
        additions.append({'id':path.stem,'image':path.relative_to(ROOT).as_posix(),'source_sha256':sha(path),'split':'train','boxes':boxes,'original':row['original'],'original_sha256':row['original_sha256'],'capture_block_200':row['capture_block_200'],'source_size_wh':[w,h],'phash63':row['phash63'],'publisher_crc32':row['crc32'],'annotation_sha256':row['annotation_sha256'],'source_boxes':row['boxes']})
    records=old['records']+additions
    for r in old['records']:
        if sha(ROOT/r['image'])!=r['source_sha256']:raise ValueError('Existing frozen prepared image changed')
    report={**{k:v for k,v in old.items()if k not in ['records','split_counts','positive_images']},'records':records,'split_counts':dict(Counter(r['split']for r in records)),'positive_images':dict(Counter(r['split']for r in records if r['boxes'])),
            'addition_protocol_sha256':sha(OUT/'protocol-before-photo-acquisition.json'),'additional_training':{'accepted_images':len(additions),'positive':sum(bool(r['boxes'])for r in additions),'other_defect_negative':sum(not r['boxes']for r in additions),'blocks':sorted({r['capture_block_200']for r in additions}),'quarantine':quarantine,'cal_test_records_unchanged':True,'no_independence_gain_claimed':True},'validation_status':'research_only'}
    write_once(OUT/'acquired-original-assets.json',acquired);write_once(OUT/'prepared/manifest.json',report)
    print(json.dumps({k:report[k]for k in ['split_counts','positive_images','additional_training']}),flush=True)
if __name__=='__main__':main()
