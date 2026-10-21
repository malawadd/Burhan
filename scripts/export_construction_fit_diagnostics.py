"""Real complete seen-fit AND fixed-calibration videos; never independent evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from evaluate_construction_round3 import ROOT,BASE,OUT,load_new,score_frame,summarize,render,save_review_panels,ensure_lossless_sources
from construction_annotations import publisher_labels_at_or_after

SELECTION=[('dset_3','spalling_54','SEEN DURING TRAINING; fixed first training spalling sequence; correlated development diagnostic'),
           ('dset_6','spalling_75','CALIBRATION FAILURE AUDIT; fixed first calibration spalling sequence; not independent validation'),
           ('dset_6','crack_129','CALIBRATION THIN-CRACK AUDIT; fixed first calibration crack sequence; not independent validation')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    target=OUT/'seen-fit-and-calibration-video-protocol.json'
    protocol={'schema_version':1,'selected_before_diagnostic_video_inference':True,
              'selection':[{'source_group':g,'sequence':n,'scope':s}for g,n,s in SELECTION],
              'selection_basis':'Fixed first training spall and fixed first calibration spall/crack; no favorable-model-output selection',
              'operating_point':'Actual final-model calibration-frozen per-class point; no training-video threshold tuning',
              'scope':'Development only; all source exposures retained; no fresh heldout or oldexposed inputs',
              'ground_truth':'External publisher boxes drawn separately; no pixel accuracy claim; unlabelled exposures unscored',
              'timing':'Accuracy/export can overlap fixed CPU fit; no inference capacity claim'}
    OUT.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if json.loads(target.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Frozen diagnosticselection changed')
    else:target.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    return target


def cases():
    import h5py
    result=[]
    for group,name,scope in SELECTION:
        folder=BASE/'raw'/group/name;h5=folder/f'{name}_frames.h5'
        with h5py.File(h5,'r')as src:
            frames=src['frames'][:];times=src['frame_timestamp'][:].astype(np.int64)
        label=folder/f'{name}_frame_label.npy'
        if not label.exists():label=folder/f'{name}_label.npy'
        labels=np.load(label,allow_pickle=False);period=float(np.median(np.diff(times)));truth=[]
        for index,timestamp in enumerate(times):
            expected=[]
            for j,b in enumerate(publisher_labels_at_or_after(labels,timestamp,period)):
                x,y,w,h=map(float,b[2:6]);right=min(frames.shape[2],x+w);bottom=min(frames.shape[1],y+h);x=max(0,x);y=max(0,y)
                if right>x and bottom>y:
                    expected.append({'id':f'{name}:{index}:{j}','label':'crack'if int(b[1])==0 else'spalling',
                                     'bbox':[x,y,right-x,bottom-y],'annotation_timestamp_us':int(b[0])})
            truth.append({'source_frame_index':index,'video_time_s':(int(timestamp)-int(times[0]))/1e6,
                          'source_timestamp_us':int(timestamp),'expected':expected})
        result.append({'name':name,'scope':scope,'frames':frames,'times':times,'truth':truth,
                       'source':{'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0','source_group':group,
                                 'original_camera_frames':len(frames),'raw_h5_sha256':sha(h5),'annotation_sha256':sha(label),
                                 'timebase':'Original microsecond exposure PTS shifted tozero; one frame record per original exposure',
                                 'evaluation_role':'seen_training'if group=='dset_3'else'calibration','independent_accuracy_credit':False}})
    return result


def export(path):
    from prepare_evcivil import encode_video
    protocol=freeze();op=json.loads(path.with_suffix('.operating-point.json').read_text(encoding='utf8'))
    if op['model_sha256']!=sha(path):raise RuntimeError('Calibration point/model identity changed')
    model=load_new(path,op['class_confidence']);sources=cases();ensure_lossless_sources(sources)
    publisher=json.loads((BASE/'publisher-record.json').read_text(encoding='utf8'))
    reports=[]
    for case in sources:
        case['source']['authors']='; '.join(r['full_name']for r in publisher['authors'])+'; annotation contributor XuanniHuo'
        records=[];canvases=[]
        for gray,truth in zip(case['frames'],case['truth']):
            frame=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            predictions=model.infer(frame)
            record={**truth,**score_frame(predictions,truth['expected'],{'crack','spalling'}),
                    'input_exposure_gain':float(getattr(model,'last_input_exposure_gain',1.0)),
                    'input_exposure_policy':getattr(model,'exposure_policy','off'),
                    'frame_size_wh':[frame.shape[1],frame.shape[0]]}
            records.append(record);canvases.append(render(gray,record,case['name'],'SEEN FIT'if case['source']['evaluation_role']=='seen_training'else'CALIBRATION'))
        target=ROOT/f'output/construction-round3-diagnostic-{path.stem}-{case["name"]}'
        report={'schema_version':1,'purpose':'development_diagnostic','matching_iou_threshold':.5,'sequence':case['name'],
                'issue_summary':case['scope'],'annotation_scope':case['scope'],'independent_accuracy_credit':False,
                'why_it_matters':'Actual publisher-labelled surface appearance; seen-fit success does not establish generalization.',
                'next_check':'Compare fixed calibration failures and unseen source validation before any deployment claim.',
                'frame_size_wh':[int(case['frames'].shape[2]),int(case['frames'].shape[1])],
                'source':case['source'],'model':model.describe(),'class_confidence':op['class_confidence'],
                'selection_protocol_sha256':sha(protocol),'candidate_status':'historical_failed_calibration' if path.stem=='evcivil-yolov8n-r3' else 'failed_adapted_calibration',
                'summary':summarize(records),'frames':records}
        report['derived_video']=encode_video(target.with_suffix('.mp4'),np.asarray(canvases),case['times'],max(1,round(1e6/np.median(np.diff(case['times'])))))
        target.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        save_review_panels(case,records,target)
        result={'sequence':case['name'],'evaluation_role':case['source']['evaluation_role'],'summary':report['summary'],
                'video':str(target.with_suffix('.mp4').relative_to(ROOT)),'video_sha256':sha(target.with_suffix('.mp4')),
                'report':str(target.with_suffix('.json').relative_to(ROOT)),'report_sha256':sha(target.with_suffix('.json'))}
        reports.append(result);print(json.dumps(result,indent=2),flush=True)
    (OUT/f'seen-fit-and-calibration-video-manifest-{path.stem}.json').write_text(json.dumps(reports,indent=2),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--model',type=Path)
    args=parser.parse_args();cv2.setNumThreads(1)
    if args.freeze_only:print(freeze())
    elif not args.model:parser.error('--model required for actual export')
    else:export(args.model.resolve())
