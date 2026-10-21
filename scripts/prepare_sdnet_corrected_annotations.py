"""Reconcile exact-byte SDNET annotation fragments; inherit frozen source splits.

Near-duplicate photographs NEVER exchange labels. Opposite-category boxes with
IoU>=.5 on an exact photograph quarantine that whole photograph. Same-category
overlapping fragments from different source rows retain an actual larger source
annotation, with all alternatives recorded. No model inference or training.
"""
from __future__ import annotations
import copy,hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json'
OUT=ROOT/'datasets/construction/expansion-public-v2/sdnet2025-corrected'
EVIDENCE=ROOT/'output/evaluation/construction-resumption-fastening-diagnosis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,indent=2)+'\n')
def iou(a,b):
    inter=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return inter/max(1e-9,a[2]*a[3]+b[2]*b[3]-inter)
def provenance(row,box):
    return {'source_record_id':row['id'],'publisher_label':row['publisher_label'],'source_annotation_id':box.get('publisher_annotation_id'),'annotation_sha256':row['annotation_sha256'],'source_image_sha256':row['source_sha256']}
def card(row):
    image=cv2.imread(str(ROOT/row['image']))
    for b in row['boxes']:
        x,y,w,h=map(round,b['bbox']);color=(0,255,0)if b['class_id']==0 else(255,180,0)
        cv2.rectangle(image,(x,y),(x+w,y+h),color,3)
        cv2.putText(image,'Loosen'if b['class_id']==0 else'Missing',(x,max(20,y)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
    cv2.rectangle(image,(0,610),(640,640),(0,0,0),-1);cv2.putText(image,row['id']+' publisher annotations',(6,632),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    return image
def main():
    if (OUT/'manifest.json').exists():raise ValueError('Preserve corrected annotations; new version required')
    cv2.setNumThreads(1);source=json.loads(SOURCE.read_text());groups=defaultdict(list)
    for row in source['records']:groups[row['source_sha256']].append(row)
    protocol={'source_manifest_sha256':sha(SOURCE),'script_sha256':sha(__file__),'source':source['source'],'license':source['license'],'scope':'Original public annotations only; no private owner frames, no model selection/inference/fitting',
        'grouping':'Exact original imageSHA256 only; never propagate labels between near-duplicate differentpixels. Inherit original split and duplicategroup; refuse inconsistent inheritance.',
        'contradiction':'Whole exact-photo group quarantined if opposite-class boxes overlapIoU>=.5. This is ambiguous sourceGT, not an automatic class correction.',
        'same_class_dedup':'Across different source records only, overlappingIoU>=.5 same-class fragments retain the actual largest-area sourcebox+polygon (ties provenancelexical). Record alternative annotations. Never synthesize averagedboxes/masks.',
        'split_policy':'Inherit TRAIN/CAL/TEST unchangedperexactphotogroup; noallocationusingmodelresults. TEST remains previouslyexposedreuseddevelopment, not freshqualification.',
        'completeness':'Union resolves some knownpartialannotations. Completenessofallobjects/othernearbyconnections stillunverified; normalFixed annotations lack torque/physicalverification.',
        'private_videos_used':False,'crossclass_iou_cut':.5,'sameclass_iou_cut':.5}
    write(OUT/'protocol-before-reconciliation.json',protocol);records=[];quarantine=[];reconciliations=[]
    for image_sha,rows in sorted(groups.items()):
        if len({r['split']for r in rows})!=1 or len({r['duplicate_group']for r in rows})!=1:raise ValueError('Exactimage crossesoriginalsplit/group')
        for r in rows:
            if sha(ROOT/r['image'])!=image_sha:raise ValueError('Frozensourcepixelschanged')
        candidates=[{**copy.deepcopy(b),'annotation_sources':[provenance(r,b)]}for r in sorted(rows,key=lambda x:x['id'])for b in r['boxes']]
        conflicts=[]
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                overlap=iou(a['bbox'],b['bbox'])
                if a['class_id']!=b['class_id']and overlap>=.5:conflicts.append({'iou':overlap,'annotations':[a,b]})
        if conflicts:
            quarantine.append({'image_sha256':image_sha,'source_ids':[r['id']for r in rows],'split':rows[0]['split'],'reason':'crossclasssamepixeloverlap','conflicts':conflicts});continue
        chosen=[];dedup=[]
        candidates.sort(key=lambda b:(-b['bbox'][2]*b['bbox'][3],json.dumps(b['annotation_sources'],sort_keys=True)))
        for b in candidates:
            match=next((k for k in chosen if k['class_id']==b['class_id']and iou(k['bbox'],b['bbox'])>=.5 and
                        not any(x['source_record_id']==y['source_record_id']for x in k['annotation_sources']for y in b['annotation_sources'])),None)
            if match is None:chosen.append(b)
            else:
                dedup.append({'kept_source_annotation':copy.deepcopy(match['annotation_sources'][0]),'alternative':copy.deepcopy(b),'iou':iou(match['bbox'],b['bbox'])});match['annotation_sources'].extend(b['annotation_sources'])
        representative=sorted(rows,key=lambda r:r['id'])[0];split=representative['split'];identity='exact-'+image_sha[:20]
        target=OUT/'images'/split/(identity+'.jpg');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/representative['image'],target)
        if sha(target)!=image_sha:raise ValueError('Copied sourceimage changed')
        labels=OUT/'labels'/split/(identity+'.txt');labels.parent.mkdir(parents=True,exist_ok=True)
        labels.write_text(''.join(f"{b['class_id']} {(b['bbox'][0]+b['bbox'][2]/2)/640:.8f} {(b['bbox'][1]+b['bbox'][3]/2)/640:.8f} {b['bbox'][2]/640:.8f} {b['bbox'][3]/640:.8f}\n"for b in chosen))
        mask=np.zeros((640,640),np.uint8)
        for b in chosen:
            for poly in b['polygons']:cv2.fillPoly(mask,[np.rint(np.array(poly).reshape(-1,2)).astype(np.int32)],b['class_id']+1)
        mask_path=OUT/'masks'/split/(identity+'.png');mask_path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(mask_path),mask)
        row={**copy.deepcopy(representative),'id':identity,'image':target.relative_to(ROOT).as_posix(),'mask':mask_path.relative_to(ROOT).as_posix(),'mask_sha256':sha(mask_path),'boxes':chosen,'publisher_label':'exact_photo_annotation_union'if len(rows)>1 else representative['publisher_label'],
             'source_ids':[r['id']for r in rows],'source_records':[{'id':r['id'],'publisher_label':r['publisher_label'],'annotation_sha256':r.get('annotation_sha256'),'source_image':r['source_image']}for r in rows],
             'evaluation_role':'reused_development'if split=='test'else split}
        records.append(row)
        if len(rows)>1:reconciliations.append({'image_sha256':image_sha,'split':split,'source_ids':row['source_ids'],'before_annotation_count':sum(len(r['boxes'])for r in rows),'after_annotation_count':len(chosen),'sameclass_dedup':dedup})
    yaml=OUT/'dataset.yaml';yaml.write_text(f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/calibration\nnames:\n  0: visibly_unseated_connection\n  1: fastener_absence_appearance\n")
    result={**{k:v for k,v in source.items()if k not in ['records','photos','split_counts','instances','prepared_script_sha256','dataset_yaml_sha256']},'records':records,'photos':len(records),'split_counts':dict(Counter(r['split']for r in records)),
        'instances':dict(Counter(b['label']for r in records for b in r['boxes'])),'dataset_yaml_sha256':sha(yaml),'prepared_script_sha256':sha(__file__),'parent_manifest_sha256':sha(SOURCE),'reconciliation_protocol_sha256':sha(OUT/'protocol-before-reconciliation.json'),
        'quarantined_exact_groups':quarantine,'reconciled_exact_groups':reconciliations,'validation_status':'research_only','target_scope':'Visible nut/bolt absence or loosening regions inSDNET; not plate-to-wall gaps, torque, embedment or approvedfasteningcorrectness.'}
    write(OUT/'manifest.json',result)
    byid={r['id']:r for r in source['records']};pairs=[('Loosen-158','Missing-126'),('Loosen-161','Missing-60')]
    evidence=np.vstack([np.hstack([card(byid[a]),card(byid[b])])for a,b in pairs]);EVIDENCE.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(EVIDENCE/'annotation-pairs-conflict-and-partial.jpg'),evidence)
    print(json.dumps({k:result[k]for k in ['photos','split_counts','instances']}));print('quarantine_groups',len(quarantine),'reconciled_groups',len(reconciliations))
if __name__=='__main__':main()
