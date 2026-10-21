"""Compare actual fit vs calibration localization, without opening any heldout input.

Use only after an explicit compute handoff. These are training diagnostics, never
independent accuracy claims. Original pixels, publisher labels and predictions
are separate panels; no video is made from these photographs/exposures.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from evaluate_construction_round3 import ROOT,OUT,load_new,sampled,score_frame,summarize,ConstructionModel


def panel(frame,record,name):
    raw=frame.copy();gt=frame.copy();pred=frame.copy()
    for e in record['expected']:
        x,y,w,h=map(round,e['bbox']);cv2.rectangle(gt,(x,y),(x+w,y+h),(240,210,30),1)
        cv2.putText(gt,e['label'],(x,max(12,y-3)),0,.35,(240,210,30),1)
    for p in record['predictions']:
        x,y,w,h=map(round,p['bbox']);color=(60,205,90) if p.get('matched_annotation_id') else (30,170,245)
        cv2.rectangle(pred,(x,y),(x+w,y+h),color,1)
        cv2.putText(pred,f"{p['label']} {p['score']:.2f}",(x,max(12,y-3)),0,.35,color,1)
    images=[]
    for title,image in [('REAL INPUT',raw),('PUBLISHER GT (external)',gt),('ACTUAL MODEL',pred)]:
        image=cv2.resize(image,(692,520),interpolation=cv2.INTER_NEAREST)
        canvas=np.full((590,692,3),24,np.uint8);canvas[65:585]=image
        cv2.putText(canvas,title,(12,24),0,.6,(240,240,240),1)
        cv2.putText(canvas,name[:72],(12,51),0,.42,(230,230,230),1);images.append(canvas)
    return np.hstack(images)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path);parser.add_argument('--baseline',action='store_true');parser.add_argument('--max-train',type=int,default=0);args=parser.parse_args()
    if bool(args.model)==args.baseline:raise ValueError('Select exactly one model path or --baseline')
    if args.baseline:
        path=None;model=ConstructionModel({'mode':'crack','crack_variant':'joint_aware_candidate','engine':'openvino','openvino_device':'GPU','mask_rendering':'boxes'});name='baseline-joint-aware';supported={'crack'};meta={}
    else:
        path=args.model.resolve();op=json.loads(path.with_suffix('.operating-point.json').read_text(encoding='utf8'))
        model=load_new(path,op['class_confidence']);name=path.stem;supported={'crack','spalling'};meta=json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf8'))
    directory=OUT/name/'fit-diagnostics';directory.mkdir(exist_ok=True,parents=True)
    summary={}
    for split in ['train','calibration']:
        rows=sampled(split,Path(meta['training_data_root'])) if meta.get('training_data_root') else sampled(split)
        if split=='train' and args.max_train and len(rows)>args.max_train:rows=[rows[i] for i in np.unique(np.linspace(0,len(rows)-1,args.max_train).round().astype(int))]
        records=[];frames=[]
        for row in rows:
            frame=cv2.imread(str(row['image']));record=score_frame(model.infer(frame),row['expected'],supported,row['negative'])
            record.update(source=row['source'],image=str(row['image']),negative=row['negative']);records.append(record);frames.append(frame)
        summary[split]=summarize(records);summary[split]['model']=model.describe();summary[split]['input_images']=len(rows)
        summary[split]['scope']='Fit/calibration diagnostic, not independent heldout accuracy'
        (directory/f'{split}-records.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        groups={'crack':lambda r:any(e['label']=='crack' for e in r['expected']),
                'spalling':lambda r:any(e['label']=='spalling' for e in r['expected']),
                'night':lambda r:r['source'].startswith(('27_','ros_')),
                'negative':lambda r:r['negative']}
        for group,accept in groups.items():
            eligible=[i for i,r in enumerate(records) if accept(r)]
            if not eligible:continue
            chosen=[('first',eligible[0]),('worst',max(eligible,key=lambda i:(records[i]['missed_defects'],records[i]['false_alerts'] or 0,-records[i]['localized_matches'])))]
            for status,i in chosen:
                name=f"{split} {group} {status}: {records[i]['source']} TP{records[i]['localized_matches']} FN{records[i]['missed_defects']} FP{records[i]['false_alerts']}"
                cv2.imwrite(str(directory/f'{split}-{group}-{status}.jpg'),panel(frames[i],records[i],name))
        print(split,json.dumps(summary[split]),flush=True)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')


if __name__=='__main__':
    cv2.setNumThreads(1);main()
