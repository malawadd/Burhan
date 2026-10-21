"""TRAIN-only SDNET localization/scale diagnosis; never trains or tunes CAL/TEST."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_task_detector import ConstructionTaskDetector

DATA=ROOT/'datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json'
MODEL=ROOT/'models/construction/expansion-sdnet-v2-performance/yolov8n/manifest.json'
OUT=ROOT/'output/evaluation/construction-resumption-fastening-diagnosis'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iou(a,b):
    x,y=max(a[0],b[0]),max(a[1],b[1]);xx,yy=min(a[0]+a[2],b[0]+b[2]),min(a[1]+a[3],b[1]+b[3])
    inter=max(0,xx-x)*max(0,yy-y);return inter/max(1e-9,a[2]*a[3]+b[2]*b[3]-inter)
def evaluate(rows,predictions,threshold):
    result={}
    for label in ['visibly_unseated_connection','fastener_absence_appearance']:
        tp=fp=fn=0; overlaps=[];positive_areas=[];miss_areas=[]
        for row,preds in zip(rows,predictions):
            truth=[b['bbox']for b in row['boxes']if b['label']==label];used=set()
            for p in sorted([p for p in preds if p['label']==label and p['score']>=threshold],key=lambda p:-p['score']):
                best=max([(iou(p['bbox'],b),i)for i,b in enumerate(truth)if i not in used],default=(0,None))
                if best[0]>=.5:used.add(best[1]);tp+=1;overlaps.append(best[0])
                else:fp+=1
            fn+=len(truth)-len(used)
            for i,b in enumerate(truth):
                positive_areas.append((b[2]*b[3])**.5*.8)
                if i not in used:miss_areas.append((b[2]*b[3])**.5*.8)
        result[label]={'tp':tp,'annotation_unmatched':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'mean_matched_iou':float(np.mean(overlaps))if overlaps else None,
                       'median_gt_sqrt_area_at512':float(np.median(positive_areas)),'median_missed_sqrt_area_at512':float(np.median(miss_areas))if miss_areas else None}
    return result
def nms(items):
    kept=[]
    for p in sorted(items,key=lambda x:-x['score']):
        if not any(k['label']==p['label']and iou(k['bbox'],p['bbox'])>.45 for k in kept):kept.append(p)
    return kept
def main():
    if (OUT/'diagnosis.json').exists():raise ValueError('Preserve diagnosis; use versioned output for another experiment')
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    manifest=json.loads(DATA.read_text());train=[r for r in manifest['records']if r['split']=='train']
    selected=[]
    for publisher in ['Loosened.json','Missing.json','Fixed']:
        selected.extend(sorted([r for r in train if r['publisher_label']==publisher],key=lambda r:hashlib.sha256(('diagnosis-v1:'+r['id']).encode()).hexdigest())[:8])
    protocol={'scope':'TRAIN-only diagnostics; no CAL/TEST inference or threshold selection; no optimizer updates','selected_ids':[r['id']for r in selected],
              'selection':'First8 per publisher class by SHA256(diagnosis-v1:id); chosen before predictions','dataset_sha256':sha(DATA),'model_manifest_sha256':sha(MODEL),'script_sha256':sha(__file__),
              'grid':[.025,.1,.5],'tile_geometry':'four fixed320quadrants per640photo, each letterboxed512; globalclassNMS.45; no GT-assisted cropping'}
    (OUT/'protocol-before-forward.json').write_text(json.dumps(protocol,indent=2))
    sizes={}
    for label in manifest['classes']:
        boxes=[b for r in train for b in r['boxes']if b['label']==label]
        area=np.array([(b['bbox'][2]*b['bbox'][3])**.5*.8 for b in boxes]);short=np.array([min(b['bbox'][2:])*.8 for b in boxes])
        sizes[label]={'instances':len(boxes),'sqrt_area_pixels_at512_percentiles':np.percentile(area,[0,10,25,50,75,90,100]).tolist(),'short_side_pixels_at512_percentiles':np.percentile(short,[0,10,25,50,75,90,100]).tolist(),'fraction_short_side_under16':float(np.mean(short<16)),'fraction_short_side_under32':float(np.mean(short<32))}
    detector=ConstructionTaskDetector(MODEL);detector.thresholds={c:.01 for c in manifest['classes']}
    full=[];tiled=[];receipts=[];cards=[]
    for row in selected:
        path=ROOT/row['image']
        if sha(path)!=row['source_sha256']:raise ValueError('Frozen TRAIN bytes changed')
        image=cv2.imread(str(path));native=detector.infer(image);tiles=[]
        for y in [0,320]:
            for x in [0,320]:
                for p in detector.infer(image[y:y+320,x:x+320]):
                    p=dict(p);b=p['bbox'];p['bbox']=[b[0]+x,b[1]+y,b[2],b[3]];tiles.append(p)
        tiles=nms(tiles);full.append(native);tiled.append(tiles)
        receipts.append({'id':row['id'],'image':row['image'],'sha256':sha(path),'publisher_label':row['publisher_label'],'ground_truth':row['boxes'],'full_predictions':native,'four_tile_predictions':tiles})
        canvas=image.copy()
        for b in row['boxes']:
            x,y,w,h=map(round,b['bbox']);cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,255,0),2)
        for p in native:
            if p['score']<.1:continue
            x,y,w,h=map(round,p['bbox']);cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,150,255),2);cv2.putText(canvas,f"{p['label'].split('_')[0]} {p['score']:.2f}",(x,max(14,y)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,150,255),1)
        card=cv2.resize(canvas,(320,320));cv2.rectangle(card,(0,0),(320,22),(0,0,0),-1);cv2.putText(card,row['id']+' GREEN=GT ORANGE=pred>=.1',(4,15),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1);cards.append(card)
    curves={}
    for name,path in [('initial','output/training/construction-expansion-sdnet-v2/fit/results.csv'),('resumed','output/training/construction-expansion-sdnet-v2-performance/fit/results.csv')]:
        curves[name]=list(csv.DictReader((ROOT/path).open()))
    report={'protocol':protocol,'train_object_sizes':sizes,'sample_counts':dict(Counter(r['publisher_label']for r in selected)),
            'whole_frame':{str(t):evaluate(selected,full,t)for t in protocol['grid']},'four_quadrants':{str(t):evaluate(selected,tiled,t)for t in protocol['grid']},
            'learning_curves':curves,'records':receipts,'interpretation':'TRAIN fit/scale diagnostics only; not held-out accuracy. Oppositeclass annotation completeness unverified. Cropping inference4xmodelcalls may hurt latency.'}
    cv2.imwrite(str(OUT/'train-localization-contact-sheet.jpg'),np.vstack([np.hstack(cards[i:i+4])for i in range(0,len(cards),4)]))
    (OUT/'diagnosis.json').write_text(json.dumps(report,indent=2,default=lambda x:float(x)if isinstance(x,np.generic)else x))
    print(json.dumps({k:v for k,v in report.items()if k not in ['records','learning_curves']}))
if __name__=='__main__':main()
