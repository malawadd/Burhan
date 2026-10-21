"""Score real captured frame sequences against publisher defect localization, never against predictions.

Export environment has PyAV for timestamp-faithful annotated video. No new training occurs here.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_model import ConstructionModel, prepare, decode
from edge_runtime.profiles.roads import _iou


class JointAwareCandidate:
    """Explicit experimental model, separate from the application's pinned default."""
    def __init__(self, engine, device, confidence):
        import hashlib
        path=ROOT/'models/construction/crack-joints-yolov8n-seg.onnx'
        self.provenance=json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf8'))
        self.sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha!=self.provenance['onnx_sha256']:raise RuntimeError('Candidate ONNX hash mismatch')
        if engine!='openvino':raise ValueError('Experimental joint-aware benchmark uses explicit OpenVINO only')
        import openvino as ov
        self.device=device; self.confidence=confidence;self.engine=engine;self.engine_version=ov.get_version()
        started=time.perf_counter()
        core=ov.Core();m=core.read_model(str(path))
        self.compiled=core.compile_model(m,device,{'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32'})
        self.request=self.compiled.create_infer_request();self.path=path;self.load_ms=(time.perf_counter()-started)*1000
    def infer(self,frame):
        tensor,transform=prepare(frame,'crack')
        result=self.request.infer({0:tensor})
        outputs=[result[self.compiled.output(i)] for i in range(len(self.compiled.outputs))]
        rows=decode(outputs,'crack',transform,frame.shape[:2],self.confidence,.45,'boxes',classes=['joint','crack'])
        # Joint objects are explicit non-issue context; no joint is relabelled as a crack.
        return [r for r in rows if r['label']=='crack']
    def describe(self):
        return {'name':'Experimental joint-aware YOLOv8n crack segmentation','sha256':self.sha,
                'model_source':self.provenance['source'],'license_status':self.provenance['license'],
                'model_path':str(self.path),'engine':self.engine,'device':self.device,'engine_version':self.engine_version,'model_load_ms':self.load_ms,
                'confidence_threshold':self.confidence,'nms_iou':.45,'matching_iou_threshold':.5,
                'input_shape':[1,3,640,640],'mask_rendering':'boxes','supported_capabilities':['surface_crack'],'crack_variant':'joint_aware_candidate','preprocessing':'letterbox RGB /255','classes':['joint','crack'],
                'experimental':True,'validation_scope':'Exposed public diagnostic videos; no independent Saudi validation'}


def match(predictions, expected):
    """Class-specific greedy confidence matching at preregistered IoU >= .50."""
    remaining = set(range(len(expected))); matches = []
    for prediction in predictions:
        choices = [(float(_iou(prediction["bbox"], truth["bbox"])), i)
                   for i,truth in enumerate(expected) if i in remaining and truth["label"] == prediction["label"]]
        overlap, index = max(choices,default=(0,None))
        if index is not None and overlap >= .5:
            remaining.remove(index); prediction["matched_annotation_id"] = expected[index]["id"]
            prediction["iou"] = overlap; matches.append(expected[index]["id"])
        else:
            prediction["matched_annotation_id"] = None
            # Keep one-to-one assignment, but report actual best geometry even
            # when a higher-confidence box already claimed that annotation.
            prediction["iou"] = max((float(_iou(prediction["bbox"], truth["bbox"]))
                                     for truth in expected if truth["label"] == prediction["label"]), default=0.)
    return matches, [expected[i] for i in sorted(remaining)]


def annotate(frame, predictions, sequence, index, record):
    """Show external annotation and model output with visibly separate identities."""
    scale = 2; image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((640, 1100, 3),np.uint8); canvas[:]=(20,25,30)
    canvas[90:90+image.shape[0],:image.shape[1]] = image
    def box(bbox,color,dashed=False):
        x,y,w,h=[round(v*scale) for v in bbox];y+=90
        x=max(0,x);y=max(90,y);right=min(691,x+w);bottom=min(609,y+h)
        if dashed:
            for px in range(x,right,14):
                cv2.line(canvas,(px,y),(min(px+8,right),y),color,2)
                cv2.line(canvas,(px,bottom),(min(px+8,right),bottom),color,2)
            for py in range(y,bottom,14):
                cv2.line(canvas,(x,py),(x,min(py+8,bottom)),color,2)
                cv2.line(canvas,(right,py),(right,min(py+8,bottom)),color,2)
        else:cv2.rectangle(canvas,(x,y),(right,bottom),color,2)
        return x,y
    matched={p['matched_annotation_id'] for p in predictions if p.get('matched_annotation_id')}
    for expected in record['expected']:
        unsupported=expected.get('evaluation_status')=='unsupported'
        color=(155,155,155) if unsupported else (240,210,30) if expected['id'] in matched else (70,70,245)
        x,y=box(expected['bbox'],color,True)
        cv2.putText(canvas,'PUBLISHER '+('SPALLING (UNSUPPORTED)' if unsupported else 'CRACK GT'),(x,min(602,max(105,y+15))),cv2.FONT_HERSHEY_SIMPLEX,.40,color,1)
    for prediction in predictions:
        matched_prediction=bool(prediction.get('matched_annotation_id'))
        color=(60,205,90) if matched_prediction else (30,170,245)
        x,y=box(prediction['bbox'],color)
        cv2.putText(canvas,f"MODEL #{prediction['id']} {prediction['score']:.2f}",(x,max(108,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1)
    cv2.putText(canvas,'REAL DEFECT: EXTERNAL LABELS vs LOCAL MODEL',(15,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1)
    cv2.putText(canvas,f"{sequence} | original frame {index} | {record['video_time_s']:.3f}s",(15,57),cv2.FONT_HERSHEY_SIMPLEX,.58,(240,240,240),1)
    x=710; evaluated=record.get('evaluated',False)
    notes=[('THIS FRAME',(240,240,240)),
           (f"Localized matches: {record['localized_matches']}",(60,205,90)),
           (f"Missed crack annotations: {record['missed_defects']}",(70,70,245)),
           (f"Unmatched model boxes: {record['false_alerts']}" if evaluated else 'Crack accuracy: not scored',(30,170,245)),
           ('Box IoU >= .50; partial coverage fails.',(205,205,205)),
           ('', (0,0,0)),('LEGEND',(240,240,240)),
           ('Solid GREEN: localized model match',(60,205,90)),
           ('Solid AMBER: unmatched candidate',(30,170,245)),
           ('Dashed RED: publisher missed crack',(70,70,245)),
           ('Dashed CYAN: matched publisher GT',(240,210,30)),
           ('Dashed GRAY: unsupported damage',(155,155,155)),
           ('', (0,0,0)),('NEXT CHECK',(240,240,240)),
           ('Inspect crack vs joint, stain or shadow.',(225,225,225)),
           ('Measure width; obtain engineer review.',(225,225,225)),
           ('Model output is not code approval.',(225,225,225))]
    if sequence.startswith('spalling'):
        notes[2]=('Unsupported spall annotations: 1',(155,155,155))
        notes[14]=('Spalling model is NOT installed.',(225,225,225))
        notes[15]=('Check missing/loose concrete and steel.',(225,225,225))
    for i,(line,color) in enumerate(notes):cv2.putText(canvas,line,(x,110+i*27),cv2.FONT_HERSHEY_SIMPLEX,.44,color,1)
    cv2.putText(canvas,'Gamage / Xuanni Huo, ev-CIVIL (DTU), CC BY 4.0 | Labels are external truth, never model inputs',(12,630),cv2.FONT_HERSHEY_SIMPLEX,.40,(170,190,205),1)
    return canvas


def main():
    from prepare_evcivil import encode_video
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--engine',choices=['onnxruntime','openvino'],default='openvino')
    a.add_argument('--device',choices=['CPU','GPU'],default='GPU');a.add_argument('--sequences',nargs='+',default=['crack_10','crack_89','spalling_35'])
    a.add_argument('--candidate-joints',action='store_true');a.add_argument('--output-tag',default='baseline');a.add_argument('--confidence',type=float,default=.25)
    args=a.parse_args();cv2.setNumThreads(1)
    model=(JointAwareCandidate(args.engine,args.device,args.confidence) if args.candidate_joints else
           ConstructionModel({'mode':'crack','engine':args.engine,'openvino_device':args.device,'mask_rendering':'boxes','confidence':args.confidence}))
    output=ROOT/'output';results=[]
    for name in args.sequences:
        folder=ROOT/'datasets/construction/evcivil'/name
        ground=json.loads((folder/'annotations.json').read_text(encoding='utf8'))
        frames=np.load(folder/'frames.npy',allow_pickle=False);times=np.load(folder/'timestamps-us.npy',allow_pickle=False)
        records=[];durations=[];annotated=[];tp=fp=fn=unsupported=annotated_frames=0
        for index,(frame,truth) in enumerate(zip(frames,ground['frames'])):
            bgr=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            started=time.perf_counter();rows=model.infer(bgr);durations.append((time.perf_counter()-started)*1000)
            predictions=[{'id':i+1,'label':r['label'],'bbox':list(r['bbox']),'score':r['score']} for i,r in enumerate(rows)]
            expected=truth['expected'];supported=[t for t in expected if t['label']=='crack'];other=[t for t in expected if t['label']!='crack']
            for t in other:t['evaluation_status']='unsupported'
            matches,missed=match(predictions,supported)
            for prediction in predictions:
                prediction['evaluation_status']='matched' if prediction['matched_annotation_id'] else 'false_alert' if supported else 'unscored'
            # Frames without annotation are not sound negatives. Unsupported damage is a separate miss category.
            if expected:
                annotated_frames+=1;unsupported+=len(other)
            if supported:
                tp+=len(matches);fp+=sum(p['matched_annotation_id'] is None for p in predictions);fn+=len(missed)
            record={**truth,'frame_size_wh':[int(frames.shape[2]),int(frames.shape[1])], 'evaluated':bool(supported),'predictions':predictions,'localized_matches':len(matches),'missed_defects':len(missed),
                    'unsupported_defects':len(other),'false_alerts':sum(p['matched_annotation_id'] is None for p in predictions) if supported else None,
                    'evaluation_scope':'Annotated crack frame, class-specific IoU >= 0.50' if supported else 'Unsupported damage excluded from crack accuracy' if other else 'Unlabelled frame excluded from accuracy'}
            records.append(record);annotated.append(annotate(frame,predictions,name,index,record))
        summary={'annotated_frames':annotated_frames,'localized_matches':tp,'missed_defects':fn,'false_alerts':fp,'unsupported_defects':unsupported,'unsupported_annotations':unsupported,
                 'pixel_mask_iou':None,'pixel_mask_scope':'Publisher supplies boxes, not pixel masks; no pixel-mask localization score is available',
                 'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                 'evaluation_scope':'Publisher boxes, class-specific IoU >= .50; exposed reused public diagnostic subset, unknown model-training overlap; no Saudi validation',
                 'warm_model_fps':1000/float(np.mean(durations[1:])), 'model_p95_ms':float(np.percentile(durations[1:],95)),
                 'speed_scope':'Preprocess + local inference + candidate decode only, excludes video decode, annotation and encoding'}
        target=output/f'construction-{name}-{args.output_tag}'
        preview_index=len(annotated)//2
        cv2.imwrite(str(target.with_name(target.name+'-frame').with_suffix('.jpg')),annotated[preview_index])
        encoded=encode_video(target.with_suffix('.mp4'),np.asarray(annotated),times,max(1,round(1e6/np.median(np.diff(times)))))
        result={'schema_version':1,'matching_iou_threshold':.5,'purpose':'defect_evidence','sequence':name,
                'issue_summary': 'Visible branching surface crack in real field camera footage' if name=='crack_10' else 'Visible dark surface crack under difficult exposure' if name=='crack_89' else 'Visible chipped/spalled concrete edge; detector does not support spalling',
                'why_it_matters':'Surface damage needs close inspection and cause assessment; pixels alone cannot determine structural severity.',
                'next_check':'Inspect the actual damaged location, measure defect dimensions with calibrated tools, and obtain competent assessment.',
                'annotation_scope':ground['annotation_scope'],'frame_size_wh':[int(frames.shape[2]),int(frames.shape[1])],'source':{k:v for k,v in ground.items() if k!='frames'},
                'model':model.describe(),'summary':summary,'frames':records,'derived_video':encoded}
        target.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf8');results.append({'sequence':name,**summary,'video':str(target.with_suffix('.mp4'))})
        print(json.dumps(results[-1]),flush=True)
    (output/f'construction-evcivil-{args.output_tag}-summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')


if __name__=='__main__':main()
