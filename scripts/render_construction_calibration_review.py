"""Render actual RAW/GT/gained-input/model panels from cached calibration only.

No neural forward or threshold change. Fixed controls and posthoc diagnostic
examples are labelled separately; none provides independent evaluation credit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_preprocessing import camera_exposure


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(array,box,color,label,dashed=False):
    x,y,w,h=map(round,box);right=x+w;bottom=y+h
    if dashed:
        for q in range(x,right,7):
            cv2.line(array,(q,y),(min(q+4,right),y),color,1);cv2.line(array,(q,bottom),(min(q+4,right),bottom),color,1)
        for q in range(y,bottom,7):
            cv2.line(array,(x,q),(x,min(q+4,bottom)),color,1);cv2.line(array,(right,q),(right,min(q+4,bottom)),color,1)
    else:cv2.rectangle(array,(x,y),(right,bottom),color,1)
    cv2.putText(array,label,(max(0,x),max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.34,color,1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path,required=True);args=parser.parse_args()
    cv2.setNumThreads(1);model=args.model.resolve();op_path=model.with_suffix('.operating-point.json')
    op=json.loads(op_path.read_text(encoding='utf8'))
    if sha(model)!=op['model_sha256'] or not op['serving_decoder_parity_verified']:raise RuntimeError('Exact calibrated model/parity required')
    audit=ROOT/'output/evaluation/construction-round3'/model.stem
    source=audit/'calibration-runtime-records.json';records=json.loads(source.read_text(encoding='utf8'))
    controls=[('fixed-first-crack',next(r for r in records if any(e['label']=='crack' for e in r['expected']))),
        ('fixed-first-material-loss',next(r for r in records if any(e['label']=='spalling' for e in r['expected'])))]
    dark=[r for r in records if r['input_exposure_gain']>1 and r['expected']]
    if dark:controls.append(('fixed-first-low-light',dark[0]))
    hits=[r for r in records if any(p['label']=='spalling' and p.get('matched_annotation_id') for p in r['predictions'])]
    if hits:controls.append(('posthoc-first-material-loss-hit',hits[0]))
    controls.append(('posthoc-most-unmatched',max(records,key=lambda r:r['false_alerts'] or 0)))
    out=audit/'review';out.mkdir(exist_ok=True);manifest=[]
    for name,record in controls:
        raw=cv2.imread(record['image'])
        if raw is None or raw.shape!=(260,346,3):raise RuntimeError('Expected genuine native346x260 camera calibration input')
        corrected,gain=camera_exposure(raw,record['input_exposure_policy'])
        if gain!=record['input_exposure_gain']:raise RuntimeError('Actual cached input gain differs')
        gt=raw.copy();pred=corrected.copy();matched={p.get('matched_annotation_id') for p in record['predictions']}
        for e in record['expected']:
            draw(gt,e['bbox'],(245,210,30) if e['id'] in matched else (70,70,245),'PUBLISHER '+e['label'],True)
        for p in record['predictions']:
            draw(pred,p['bbox'],(60,205,90) if p.get('matched_annotation_id') else (30,170,245),f"MODEL {p['label']} {p['score']:.2f}")
        canvas=np.full((1260,1410,3),(24,27,30),np.uint8)
        header=f"CALIBRATION ONLY / {name} / source {record['source']} / original frame {record['original_frame_index']}"
        cv2.putText(canvas,header,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.56,(235,235,235),1)
        cv2.putText(canvas,f"Actual gain {gain:.3f}x; matched {record['localized_matches']}, missed {record['missed_defects']}, unmatched {record['false_alerts']}; box IoU>=.50",(12,60),cv2.FONT_HERSHEY_SIMPLEX,.52,(235,235,235),1)
        for array,title,x,y in [(raw,'RAW CAMERA: UNCHANGED',8,110),(gt,'EXTERNAL PUBLISHER BOXES: NOT MODEL',710,110),
                                 (corrected,'ACTUAL NATIVE MODEL INPUT: BEFORE LETTERBOX',8,690),(pred,'ACTUAL MODEL BOXES ON MODEL INPUT',710,690)]:
            cv2.putText(canvas,title,(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1)
            canvas[y:y+520,x:x+692]=cv2.resize(array,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas,'Solid green=matched model; amber=unmatched model. Dashed cyan/red=external GT matched/missed. No pixel-GT accuracy claim.',(10,1240),cv2.FONT_HERSHEY_SIMPLEX,.44,(205,215,220),1)
        path=out/f'{name}.jpg'
        if path.exists():raise RuntimeError('Refusing to replace a previously exported calibration panel')
        cv2.imwrite(str(path),canvas)
        manifest.append({'selection':name,'selection_scope':'fixed control' if name.startswith('fixed') else 'posthoc calibration diagnosis, not validation',
            'source':record['source'],'original_frame_index':record['original_frame_index'],'source_raw_image':record['image'],
            'source_raw_file_sha256':sha(Path(record['image'])),'raw_bgr_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
            'actual_native_model_input_bgr_sha256':hashlib.sha256(corrected.tobytes()).hexdigest(),'gain':gain,
            'panel':str(path.relative_to(ROOT)),'panel_sha256':sha(path),'counts':record['by_class']})
    result={'model_sha256':sha(model),'operating_point_sha256':sha(op_path),'runtime_records_sha256':sha(source),
        'independent_accuracy_credit':False,'no_nn_or_threshold_change':True,'panels':manifest}
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'panels':len(manifest),'path':str(out.relative_to(ROOT))},indent=2))


if __name__=='__main__':main()
