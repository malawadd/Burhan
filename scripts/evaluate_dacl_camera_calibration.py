"""Installed DACL512 crack/spall camera comparator, unchanged calibration only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
from edge_runtime.profiles.construction_dacl import DaclConstructionModel,decode_masks
from evaluate_construction_round3 import BASE,sampled,score_frame,summarize

OUT=ROOT/'output/evaluation/construction-round3/dacl512-camera-calibration'
GRID=[.10,.15,.20,.25,.35,.45,.55,.65,.75,.85]
CHANNELS={'crack':0,'spalling':8}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    spec=DAMAGE_VARIANTS['dacl_segformer_b1'];rows=sampled('calibration')
    protocol={'schema_version':1,'frozen_before_camera_calibration_forward':True,
        'variant':'dacl_segformer_b1','model_sha256':spec['sha256'],'classes':['crack','spalling'],
        'scope':'Entire unchanged122image calibration set; no exposed or heldout forward; author checkpoint source-data overlap unknown',
        'preprocessing':'Unchanged installed author DACL512 full-frame square resize, RGB/ImageNet normalization in graph',
        'decoder':'Exact installed independent-channel sigmoid/nearest-grid projection; min4grid-pixel8-connected regions; cap20/class then100global; no morphology/grouping/GTROI',
        'threshold_grid':GRID,'matching_iou_threshold':.5,
        'metric_scope':'Derived component-box IoU versus publisher boxes only, not pixel-mask accuracy; per-class annotation frames+same10negativephotos',
        'selection':'Perclass maxrecall withP>=.8 andnegativeFPR<=.1; else bestF1 explicitfailure. ActualcombinednegativeFPR reported separately',
        'thin_definition':'Native publisher crack box short<=30px andlong>=2*short; not actual width',
        'timing_scope':'Accuracy may overlapCPUfit; no FPS/capacity claim,512 alreadymeasuredslowphoto model',
        'manifest_sha256':sha(BASE/'exposures.json'),'negative_import_sha256':sha(BASE/'background-import.json'),
        'ordered_images':[{'source':r['source'],'image':str(r['image'].relative_to(ROOT)),'sha256':sha(r['image'])}for r in rows]}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'preregistered-protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Frozen source/protocol changed')
    else:path.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    return path


def sparse_maps(two):
    result=np.zeros((19,*two.shape[1:]),dtype=np.float32)
    result[0],result[8]=two[0],two[1]
    return result


def evaluate():
    protocol=freeze();rows=sampled('calibration')
    spec=DAMAGE_VARIANTS['dacl_segformer_b1']
    model=DaclConstructionModel({'damage_variant':'dacl_segformer_b1','engine':'openvino','openvino_device':'GPU',
                                 'cpu_threads':1,'damage_classes':['crack','spalling'],'mask_rendering':'masks'},spec)
    maps=[]
    for index,row in enumerate(rows):
        frame=cv2.imread(str(row['image']));prob=model.predict_maps(frame)
        maps.append(prob[[0,8]].copy())
        if index%20==0:print(f'Camera calibration maps{index+1}/{len(rows)}',flush=True)
    np.savez_compressed(OUT/'two-channel-probabilities.npz',probabilities=np.stack(maps))
    selected={};search={}
    for label in CHANNELS:
        trials=[]
        for threshold in GRID:
            records=[];negative_flags=[]
            for row,prob in zip(rows,maps):
                frame=cv2.imread(str(row['image']))
                predictions=decode_masks(sparse_maps(prob),frame.shape[:2],[label],threshold,4,'boxes')
                expected=[e for e in row['expected']if e['label']==label]
                record=score_frame(predictions,expected,{label},negative=row['negative']);records.append(record)
                if row['negative']:negative_flags.append(bool(predictions))
            metrics=summarize(records)['by_class'][label];p,r=metrics['precision'] or 0,metrics['recall'] or 0
            fpr=sum(negative_flags)/len(negative_flags)
            trials.append({'threshold':threshold,**metrics,'negative_image_fpr':fpr,'f1':2*p*r/(p+r) if p+r else 0,
                           'precision_and_negative_gate':p>=.8 and fpr<=.1})
        passed=[r for r in trials if r['precision_and_negative_gate']]
        point=max(passed,key=lambda r:(r['recall'] or 0,r['precision'] or 0,r['threshold'])) if passed else max(trials,key=lambda r:(r['f1'],r['precision'] or 0,r['threshold']))
        selected[label]=dict(point,status='calibration_gate_passed'if passed else'FAILED_calibration_gate');search[label]=trials
    records=[];native_masks={}
    model.class_thresholds={label:point['threshold'] for label,point in selected.items()}
    original_predict=model.predict_maps
    parity_frames=0
    for index,(row,prob)in enumerate(zip(rows,maps)):
        frame=cv2.imread(str(row['image']));all_predictions=[]
        for label in CHANNELS:
            preds=decode_masks(sparse_maps(prob),frame.shape[:2],[label],selected[label]['threshold'],4,'masks')
            union=np.zeros(frame.shape[:2],dtype=np.uint8)
            for prediction in preds:union[prediction['mask']]=1
            native_masks[f'{label}_{index}']=union
            all_predictions.extend(preds)
        # Exact serving infer decoder on identical cached numerical maps. The
        # network is independent of cutoffs; no second stochastic or timed run.
        model.predict_maps=lambda _frame,p=prob:sparse_maps(p)
        serving=model.infer(frame)
        signature=lambda values:sorted((p['label'],tuple(p['bbox']),round(p['score'],7))for p in values)
        if signature(serving)!=signature(all_predictions):raise RuntimeError('DACL serving operating-point parity failed')
        parity_frames+=1
        record=score_frame(all_predictions,row['expected'],set(CHANNELS),negative=row['negative'])
        record.update(source=row['source'],frame_index=row['frame_index'],negative=row['negative'])
        records.append(record)
    model.predict_maps=original_predict
    np.savez_compressed(OUT/'selected-native-masks.npz',**native_masks)
    negatives=[r for r in records if r['negative']]
    thin_ids={e['id']for r in rows for e in r['expected']if e['label']=='crack'and min(e['bbox'][2:])<=30 and max(e['bbox'][2:])>=2*min(e['bbox'][2:])}
    hits=sum(p.get('matched_annotation_id')in thin_ids for r in records for p in r['predictions'])
    result={'model':model.describe(),'protocol_sha256':sha(protocol),'selected':selected,'search':search,
            'actual_combined_negative_image_fpr':sum(bool(r['predictions'])for r in negatives)/len(negatives),
            'thin_geometry_annotations':len(thin_ids),'thin_geometry_localized':hits,
            'pixel_metrics':None,'pixel_scope':'No publisher pixel masks; predicted native semantic masks retained separately',
            'serving_postprocessing_parity':{'frames':parity_frames,'scope':'Actualinfer on exactcached networkmaps atselectedclasscutoffs; identicallabels/boxes/scores; no new inference or speedclaim'},
            'validation_scope':'Correlated unchanged calibration only; checkpoint training/selection overlapunknown; not15FPSvideoqualification',
            'paired_class_scope':'Threshold metrics use same-class-annotatedframes+10negatives; combinedrecords additionally score allsupported annotations andunmatchedboxes on annotatedframes'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (OUT/'records.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    render(rows,records,native_masks,selected)
    print(json.dumps(result,indent=2),flush=True)


def render(rows,records,masks,selected):
    selectors={'first-crack':lambda r:any(e['label']=='crack'for e in r['expected']),
               'first-spalling':lambda r:any(e['label']=='spalling'for e in r['expected']),
               'first-night':lambda r:'night_nov18'in str(r['image']),
               'first-verified-negative':lambda r:r['negative']}
    for name,selector in selectors.items():
        index=next(i for i,r in enumerate(rows)if selector(r));row=rows[index];frame=cv2.imread(str(row['image']))
        gt=frame.copy();pred=frame.copy();height,width=frame.shape[:2]
        for e in row['expected']:
            x,y,w,h=map(round,e['bbox']);cv2.rectangle(gt,(x,y),(x+w,y+h),(255,255,0),1)
        for label,color in [('crack',(0,0,255)),('spalling',(255,0,200))]:
            mask=masks[f'{label}_{index}'].astype(bool);pred[mask]=(pred[mask]*.25+np.array(color)*.75).astype(np.uint8)
        panel=np.hstack([frame,gt,pred]);panel=cv2.copyMakeBorder(panel,75,40,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
        cv2.putText(panel,f'DACL512 CALIBRATION ONLY: {name}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(panel,f'Crack cutoff>{selected["crack"]["threshold"]}; spall>{selected["spalling"]["threshold"]}; no pixel ground truth',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1)
        for j,text in enumerate(['RAW CAMERA','PUBLISHER BOXES (cyan)','MODEL MASK: crack red / spall pink']):
            cv2.putText(panel,text,(j*width+5,65),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.putText(panel,'Actual retained components; semantic masks are model predictions, not publisher masks.',(8,panel.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1)
        cv2.imwrite(str(OUT/f'{name}.jpg'),panel)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    if args.freeze_only:print(freeze())
    else:evaluate()
