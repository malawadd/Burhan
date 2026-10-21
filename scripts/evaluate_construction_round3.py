"""Freeze new-class thresholds on new calibration data, then compare unseen source videos.

No exposed challenge recording can enter calibration. Scores are box IoU .50;
publisher boxes do not justify pixel-mask accuracy claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_model import ConstructionModel,DAMAGE_VARIANTS,prepare,decode_damage_boxes
from evaluate_construction_defect_video import match
from construction_annotations import publisher_labels_at_or_after

BASE=ROOT/'datasets/construction/evcivil-round3'
OUT=ROOT/'output/evaluation/construction-round3'


def load_new(path,thresholds=None,minimum=.05):
    meta=json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['onnx_sha256']:raise RuntimeError('Model checksum mismatch')
    if meta.get('exposure_policy','off')!='off':
        prepared_path=Path(meta['training_data_root'])/'data-protocol.json'
        if hashlib.sha256(prepared_path.read_bytes()).hexdigest()!=meta['prepared_data_protocol_sha256']:
            raise RuntimeError('Prepared input provenance changed; calibration is forbidden')
        prepared=json.loads(prepared_path.read_text(encoding='utf8'))
        shared_sha=hashlib.sha256((ROOT/'edge_runtime/profiles/construction_preprocessing.py').read_bytes()).hexdigest()
        if prepared['policy']!=meta['exposure_policy'] or prepared['preprocessing_source_sha256']!=shared_sha:
            raise RuntimeError('Training/serving camera preprocessing identity differs; calibration is forbidden')
    spec={'file':str(path.relative_to(ROOT/'models/construction')),'sha256':meta['onnx_sha256'],'source':meta['source'],
          'classes':meta['classes'],'capabilities':meta['capabilities'],'decoder':'yolo_detect','input_size':meta['input_size'],
          'confidence':minimum,'class_confidence':thresholds or {},'license':meta['license'],
          'exposure_policy':meta.get('exposure_policy','off'),
          'validation_scope':meta.get('validation_scope','Known current-fit source-folder/global-defect-ID exclusions; physical project independence and initial pretrained-checkpoint data overlap unverified; no Saudi-site qualification')}
    DAMAGE_VARIANTS['round3_evaluation']=spec
    return ConstructionModel({'mode':'damage','damage_variant':'round3_evaluation','engine':'openvino','openvino_device':'GPU','mask_rendering':'boxes'})


def sampled(partition,base=BASE):
    data=json.loads((base/'exposures.json').read_text(encoding='utf8'))
    rows=[]
    for r in data:
        if r['split']==partition:
            rows.append({'source':r['global_defect_id'],'image':ROOT/r['image'],'frame_index':r['original_frame_index'],
                         'expected':[{'id':f"{r['global_defect_id']}:{r['original_frame_index']}:{i}",**b} for i,b in enumerate(r['boxes'])],
                         'negative':False})
    for r in json.loads((base/'background-import.json').read_text(encoding='utf8'))['records']:
        if r['split']==partition:rows.append({'source':r['source_name'],'image':ROOT/r['derived_image'],'frame_index':0,'expected':[],'negative':True})
    return rows


def score_frame(rows,expected,supported,negative=False):
    known=[e for e in expected if e['label'] in supported]
    unsupported=[e|{'evaluation_status':'unsupported'} for e in expected if e['label'] not in supported]
    predictions=[{'id':i+1,'label':r['label'],'bbox':list(r['bbox']),'score':float(r['score'])} for i,r in enumerate(rows) if r['label'] in supported]
    matches,missed=match(predictions,known);evaluated=bool(known) or negative
    for p in predictions:p['evaluation_status']='matched' if p.get('matched_annotation_id') else 'false_alert' if evaluated else 'unscored'
    counts={}
    for label in ['crack','spalling']:
        ps=[p for p in predictions if p['label']==label];es=[e for e in known if e['label']==label]
        tp=sum(bool(p.get('matched_annotation_id')) for p in ps)
        counts[label]={'tp':tp,'fn':len(es)-tp,'fp':sum(not p.get('matched_annotation_id') for p in ps) if evaluated else 0,
                       'unsupported_annotations':sum(e['label']==label for e in unsupported)}
    return {'expected':known+unsupported,'predictions':predictions,'evaluated':evaluated,'by_class':counts,
            'localized_matches':len(matches),'missed_defects':len(missed),'false_alerts':sum(not p.get('matched_annotation_id') for p in predictions) if evaluated else None,
            'unsupported_defects':len(unsupported)}


def summarize(records,timings=None):
    classes={}
    for label in ['crack','spalling']:
        c={k:sum(r['by_class'][label][k] for r in records) for k in ['tp','fn','fp','unsupported_annotations']}
        c['precision']=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None
        c['recall']=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None;classes[label]=c
    tp=sum(c['tp'] for c in classes.values());fn=sum(c['fn'] for c in classes.values());fp=sum(c['fp'] for c in classes.values())
    result={'annotated_frames':sum(bool(r['expected']) for r in records),'evaluated_frames':sum(r['evaluated'] for r in records),
            'localized_matches':tp,'missed_defects':fn,'false_alerts':fp,'unsupported_annotations':sum(r['unsupported_defects'] for r in records),
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'all_annotated_class_localization_coverage':tp/(tp+fn+sum(r['unsupported_defects'] for r in records)) if tp+fn+sum(r['unsupported_defects'] for r in records) else None,
            'recall_scope':'recall field scores supported classes; all_annotated_class_localization_coverage additionally credits unsupported annotations as zero coverage, never as a claimed model miss','by_class':classes,
            'evaluation_scope':'Class-specific one-to-one publisher box IoU >= .50; unlabelled exposures unscored; no pixel mask metric; frame instances correlated.'}
    if timings and len(timings)>1:result.update(warm_model_fps=1000/float(np.mean(timings[1:])),model_p95_ms=float(np.percentile(timings[1:],95)),speed_scope='Preprocessing + local inference + boxes; excludes decode, video output and UI')
    return result


def class_annotation_scope(records,label,supported):
    """Same annotated-frame/class subset for paired precision/recall comparisons."""
    return [score_frame([p for p in r['predictions'] if p['label']==label],
                        [e for e in r['expected'] if e['label']==label],
                        {label} if label in supported else set(),negative=r.get('negative',False))
            for r in records if any(e['label']==label for e in r['expected']) or r.get('negative',False)]


def calibrate(path):
    model=load_new(path);records=[];raw_cache=[]
    for row in sampled('calibration'):
        frame=cv2.imread(str(row['image']))
        tensor,transform=model.prepare_input(frame)
        response=model.request.infer({0:tensor})
        outputs=[np.array(response[model.compiled.output(i)],copy=True) for i in range(len(model.compiled.outputs))]
        raw_cache.append((outputs,transform,frame.shape[:2]))
        predictions=decode_damage_boxes(outputs,transform,frame.shape[:2],model.input_size,model.spec['classes'],.05,.45)
        records.append({k:v for k,v in row.items() if k!='image'}|{'predictions':[{k:v for k,v in p.items() if k!='mask'} for p in predictions],
                        'input_exposure_gain':float(getattr(model,'last_input_exposure_gain',1.0))})
    grid=[.10,.15,.20,.25,.35,.45,.55,.65,.75,.85];choices={};search={}
    for label in ['crack','spalling']:
        trials=[]
        for threshold in grid:
            scored=[];negative_flags=[]
            for row,(outputs,transform,shape) in zip(records,raw_cache):
                # Re-run the real decoder on RAW outputs: score>NMS threshold and
                # top-100 ordering are exactly the serving implementation.
                cutoffs={name:(threshold if name==label else 1.0) for name in model.spec['classes']}
                predictions=decode_damage_boxes(outputs,transform,shape,model.input_size,model.spec['classes'],threshold,.45,cutoffs)
                expected=[p for p in row['expected'] if p['label']==label]
                r=score_frame(predictions,expected,{label},negative=True);scored.append(r)
                if row['negative']:negative_flags.append(bool(predictions))
            summary=summarize(scored)['by_class'][label];p=summary['precision'] or 0;r=summary['recall'] or 0
            fpr=sum(negative_flags)/len(negative_flags)
            trials.append({'threshold':threshold,**summary,'negative_image_fpr':fpr,'f1':2*p*r/(p+r) if p+r else 0,
                           'precision_and_negative_gate':p>=.8 and fpr<=.1})
        gated=[t for t in trials if t['precision_and_negative_gate']]
        selected=max(gated,key=lambda t:(t['recall'] or 0,t['precision'] or 0,t['threshold'])) if gated else max(trials,key=lambda t:(t['f1'],t['precision'] or 0,t['threshold']))
        choices[label]=selected['threshold'];search[label]={'selected':selected,'grid':trials}
    # Verify the selected joint operating point against actual infer(), not a
    # filtered post-NMS candidate cache. Also report combined negative-image FPR.
    model.class_thresholds=choices;model.threshold=min(choices.values())
    final_records=[];negative_flags=[]
    for row,original,(outputs,transform,shape) in zip(records,sampled('calibration'),raw_cache):
        expected_predictions=decode_damage_boxes(outputs,transform,shape,model.input_size,model.spec['classes'],model.threshold,.45,choices)
        actual=model.infer(cv2.imread(str(original['image'])))
        def signature(predictions):return [(p['label'],tuple(p['bbox']),round(float(p['score']),6)) for p in predictions]
        if signature(expected_predictions)!=signature(actual):raise RuntimeError('Calibration/actual serving decoder parity failed; heldout evaluation is forbidden')
        scored=score_frame(actual,row['expected'],set(model.spec['classes']),negative=row['negative']);final_records.append({**scored,'source':row['source'],'image':str(original['image']),'negative':row['negative'],
            'original_frame_index':original['frame_index'],'input_exposure_policy':model.exposure_policy,
            'input_exposure_gain':float(getattr(model,'last_input_exposure_gain',1.0))})
        if row['negative']:negative_flags.append(bool(actual))
    final_summary=summarize(final_records)
    for label in choices:
        final_class=final_summary['by_class'][label]
        if any(final_class[k]!=search[label]['selected'][k] for k in ['tp','fn','fp']):raise RuntimeError('Global top-100 changed selected class counts; recalibrate before heldout')
    metadata=json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf8'))
    thin_ids={e['id'] for r in final_records for e in r['expected']
              if e['label']=='crack' and min(e['bbox'][2:])<=30 and max(e['bbox'][2:])>=2*min(e['bbox'][2:])}
    thin_matches=sum(p.get('matched_annotation_id') in thin_ids for r in final_records for p in r['predictions'])
    result={'model_sha256':model.sha256,'class_confidence':choices,'matching_iou_threshold':.5,'nms_iou':.45,'training_protocol_sha256':metadata['split_protocol_sha256'],
            'selection_rule':'Per-class max recall subject to precision >=.80 and 10 true two-class-negative images FPR <=.10; if infeasible highest F1 with explicit gate failure. No exposed/heldout input used.',
            'serving_decoder_parity_verified':True,'selected_runtime_summary':final_summary,'combined_negative_image_false_alarm_rate':sum(negative_flags)/len(negative_flags),'new_calibration_only':True,'source_protocol_sha256':hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),'search':search,
            'input_preprocessing':{'policy':model.exposure_policy,'source_sha256':hashlib.sha256((ROOT/'edge_runtime/profiles/construction_preprocessing.py').read_bytes()).hexdigest(),
                                   'raw_calibration_source_images':True,'gain_applied_frames':sum(r['input_exposure_gain']>1 for r in final_records),
                                   'scope':'Full native image before letterbox, no annotation-derived ROI. Raw and enhanced input identities remain separate.'}}
    result['geometric_thin_crack_audit']={'annotations':len(thin_ids),'localized_matches':thin_matches,
        'recall':thin_matches/len(thin_ids) if thin_ids else None,
        'definition':'Native publisher box short side<=30px and long side>=2*short; not physical crack width or pixel-mask accuracy'}
    result['common_class_annotation_scopes']={label:summarize(class_annotation_scope(final_records,label,set(model.spec['classes']))) for label in choices}
    result['common_class_annotation_scope_note']='Paired comparisons must use the same class-annotated exposures plus10 negative images. Full two-class operating-point guard still scores all122 images, including wrong-class predictions on the other damage class.'
    guard_path=Path(metadata['training_data_root'])/'stage2-continuation-guard.json'
    if guard_path.exists():
        guard=json.loads(guard_path.read_text(encoding='utf8'))
        if guard['candidate']==metadata['experiment']:
            criteria=guard['minimum_to_avoid_automatic_stage2']
            checks={f'{label}_recall':(final_summary['by_class'][label]['recall'] or 0)>=criteria[f'{label}_localization_recall'] for label in choices}
            checks.update({f'{label}_precision':(final_summary['by_class'][label]['precision'] or 0)>=criteria['each_class_precision'] for label in choices})
            checks.update(geometric_thin_recall=(result['geometric_thin_crack_audit']['recall'] or 0)>=criteria['geometric_thin_crack_localization_recall'],
                combined_negative_alarm_rate=result['combined_negative_image_false_alarm_rate']<=criteria['combined_negative_image_alarm_rate_max'],
                serving_parity=result['serving_decoder_parity_verified'])
            result['minimum_continuation_guard']={'protocol_sha256':hashlib.sha256(guard_path.read_bytes()).hexdigest(),'checks':checks,
                'requires_distinct_stage2':not all(checks.values()),'production_qualification':False,
                'scope':'Minimum continuation decision only; even50% recall does not establish a solved inspection task.'}
    review_path=Path(metadata['training_data_root'])/'evaluation-minimum-guard.json'
    if review_path.exists():
        review=json.loads(review_path.read_text(encoding='utf8'))
        if review['candidate']!=metadata['experiment'] or review['frozen_fit_protocol_sha256']!=metadata['split_protocol_sha256'] or review['original_calibration_source_protocol_sha256']!=result['source_protocol_sha256']:
            raise RuntimeError('Additive evaluation minimum guard identity differs')
        criteria=review['minimum_accuracy_review']
        checks={f'{label}_recall':(final_summary['by_class'][label]['recall'] or 0)>=criteria[f'{label}_localization_recall'] for label in choices}
        checks.update({f'{label}_precision':(final_summary['by_class'][label]['precision'] or 0)>=criteria['each_class_precision'] for label in choices})
        checks.update(geometric_thin_recall=(result['geometric_thin_crack_audit']['recall'] or 0)>=criteria['geometric_thin_crack_localization_recall'],
            combined_negative_alarm_rate=result['combined_negative_image_false_alarm_rate']<=criteria['combined_negative_image_alarm_rate_max'],
            serving_parity=result['serving_decoder_parity_verified'])
        result['minimum_accuracy_review_guard']={'receipt_sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),'checks':checks,
            'all_minimum_guards_pass':all(checks.values()),'root_approval_still_required_before_heldout':True,
            'production_qualification':False,'scope':review['success_scope']}
    target=path.with_suffix('.operating-point.json');target.write_text(json.dumps(result,indent=2),encoding='utf8')
    audit=OUT/path.stem;audit.mkdir(exist_ok=True,parents=True)
    (audit/'calibration-candidates.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    (audit/'calibration-runtime-records.json').write_text(json.dumps(final_records,indent=2),encoding='utf8')
    np.savez_compressed(audit/'calibration-raw-outputs.npz',**{f'frame_{i:04d}':raw[0][0] for i,raw in enumerate(raw_cache)})
    print(json.dumps(result,indent=2))


def video_cases():
    import h5py
    cases=[]
    for name in ['crack_10','crack_89','spalling_35']:
        folder=ROOT/'datasets/construction/evcivil'/name;truth=json.loads((folder/'annotations.json').read_text(encoding='utf8'))
        cases.append({'name':name,'scope':'Exposed reused development challenge','frames':np.load(folder/'frames.npy',allow_pickle=False),
                      'times':np.load(folder/'timestamps-us.npy',allow_pickle=False),'truth':truth['frames'],'source':truth})
    protocol=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
    extra=json.loads((BASE/'night-holdout-protocol.json').read_text(encoding='utf8'))['sequences'] if (BASE/'night-holdout-protocol.json').exists() else []
    for row in protocol['sequences']['heldout']+extra:
        name=row['sequence'];folder=BASE/'raw'/row['group']/name
        with h5py.File(folder/f'{name}_frames.h5','r') as source:frames=source['frames'][:];times=source['frame_timestamp'][:].astype(np.int64)
        label=folder/f'{name}_frame_label.npy'
        if not label.exists():label=folder/f'{name}_label.npy'
        labels=np.load(label,allow_pickle=False);period=float(np.median(np.diff(times)));truth=[];unique=np.unique(labels[:,0])
        for index,timestamp in enumerate(times):
            boxes=publisher_labels_at_or_after(labels,timestamp,period)
            expected=[]
            for j,b in enumerate(boxes):
                x,y,w,h=map(float,b[2:6]);right=min(frames.shape[2],x+w);bottom=min(frames.shape[1],y+h);x=max(0,x);y=max(0,y)
                if right>x and bottom>y:expected.append({'id':f'{name}:{index}:{j}','label':'crack' if int(b[1])==0 else 'spalling','bbox':[x,y,right-x,bottom-y],'annotation_timestamp_us':int(b[0])})
            truth.append({'source_frame_index':index,'video_time_s':(int(timestamp)-int(times[0]))/1e6,'source_timestamp_us':int(timestamp),'expected':expected})
        cases.append({'name':name,'scope':'New NIGHT source-folder/global-ID-heldout recording; physical project identity unverified' if row['group']=='night_jan2' else 'New DAY source-folder/global-ID-heldout recording; physical project identity unverified','frames':frames,'times':times,'truth':truth,
                      'source':{'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0','source_group':row['group'],'raw_h5_sha256':hashlib.sha256((folder/f'{name}_frames.h5').read_bytes()).hexdigest(),'annotation_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'original_camera_frames':len(frames),'timebase':'Original camera exposure timestamps in microseconds, shifted to zero; one prediction record per original exposure'}})
    return cases


def ensure_lossless_sources(cases):
    from prepare_evcivil_lossless import encode_and_verify
    record=json.loads((BASE/'publisher-record.json').read_text(encoding='utf8'))
    authors='; '.join(a['full_name'] for a in record['authors'])+'; annotation contributor Xuanni Huo'
    for case in cases:
        if case['scope'].startswith('Exposed'):continue
        folder=BASE/'replay'/case['name'];folder.mkdir(exist_ok=True,parents=True)
        np.save(folder/'frames.npy',case['frames'],allow_pickle=False);np.save(folder/'timestamps-us.npy',case['times'],allow_pickle=False)
        annotations={'source':'https://doi.org/10.11583/DTU.25924228','authors':authors,'license':'CC BY4.0','frames':case['truth']}
        (folder/'annotations.json').write_text(json.dumps(annotations,indent=2),encoding='utf8')
        proof=folder/f"{folder.name}-lossless-frames.provenance.json"
        verification=json.loads(proof.read_text(encoding='utf8')) if proof.exists() else encode_and_verify(folder)
        case['source']['lossless_replay_input']=verification



def compare(path):
    from prepare_evcivil import encode_video
    op=json.loads(path.with_suffix('.operating-point.json').read_text(encoding='utf8'))
    if op['model_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise RuntimeError('Operating point does not match frozen model')
    cases=video_cases()
    publisher=json.loads((BASE/'publisher-record.json').read_text(encoding='utf8'))
    for case in cases:
        case['source']['original_camera_frames']=len(case['frames'])
        case['source']['authors']='; '.join(a['full_name'] for a in publisher['authors'])+'; annotation contributor Xuanni Huo'
    ensure_lossless_sources(cases);aggregate={}
    for tag in ['before','after']:
        model=ConstructionModel({'mode':'crack','crack_variant':'joint_aware_candidate','engine':'openvino','openvino_device':'GPU','mask_rendering':'boxes'}) if tag=='before' else load_new(path,op['class_confidence'])
        supported={'crack'} if tag=='before' else {'crack','spalling'};all_records=[]
        for case in cases:
            records=[];timings=[];canvases=[]
            for index,(gray,truth) in enumerate(zip(case['frames'],case['truth'])):
                frame=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);started=time.perf_counter();predictions=model.infer(frame);timings.append((time.perf_counter()-started)*1000)
                record={**truth,**score_frame(predictions,truth['expected'],supported),'frame_size_wh':[frame.shape[1],frame.shape[0]],
                        'input_exposure_gain':float(getattr(model,'last_input_exposure_gain',1.0)),
                        'input_exposure_policy':getattr(model,'exposure_policy','off')}
                records.append(record);canvases.append(render(gray,record,case['name'],tag))
            target=ROOT/f"output/construction-round3-{case['name']}-{tag}";summary=summarize(records,timings)
            report={'schema_version':1,'purpose':'defect_evidence','matching_iou_threshold':.5,'sequence':case['name'],'issue_summary':'Actual publisher-labelled surface crack/spalling, with localized matches and misses',
                    'why_it_matters':'Real surface damage warrants close inspection; appearance alone does not establish structural severity.',
                    'next_check':'Verify the exact surface, damage type and dimensions with a qualified inspector.','annotation_scope':case['scope'],
                    'frame_size_wh':[int(case['frames'].shape[2]),int(case['frames'].shape[1])],'model':model.describe(),'source':{k:v for k,v in case['source'].items() if k!='frames'},'summary':summary,'frames':records}
            if len(records)!=len(case['frames']):raise RuntimeError('Missing source exposure record')
            report['derived_video']=encode_video(target.with_suffix('.mp4'),np.asarray(canvases),case['times'],max(1,round(1e6/np.median(np.diff(case['times'])))))
            target.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8');save_review_panels(case,records,target);cv2.imwrite(str(target.with_name(target.name+'-frame').with_suffix('.jpg')),canvases[len(canvases)//2])
            for record in records:
                record['case_name']=case['name'];record['case_scope']=case['scope']
                record['thin_annotations']=[e['id'] for e in record['expected'] if e['label']=='crack' and min(e['bbox'][2:])<=30 and max(e['bbox'][2:])>=2*min(e['bbox'][2:])]
                record['thin_localized_matches']=sum(p.get('matched_annotation_id') in record['thin_annotations'] for p in record['predictions'])
            all_records.extend(records);print(tag,case['name'],json.dumps(summary),flush=True)
        negatives=[]
        for row in sampled('heldout'):
            if not row['negative']:continue
            frame=cv2.imread(str(row['image']));record=score_frame(model.infer(frame),[],supported,True);record['source']=row['source'];record['negative']=True;negatives.append(record)
        aggregate[tag]={'exposed':summarize([r for r in all_records if r['case_name'] in {'crack_10','crack_89','spalling_35'}]),
                        'fresh_video':summarize([r for r in all_records if r['case_name'] not in {'crack_10','crack_89','spalling_35'}]),
                        'fresh_day_video':summarize([r for r in all_records if r['case_scope'].startswith('New DAY')]),
                        'fresh_night_video':summarize([r for r in all_records if r['case_scope'].startswith('New NIGHT')]),
                        'true_two_class_negative_stills':summarize(negatives),'negative_images':len(negatives),
                        'negative_image_false_alarm_rate':sum(bool(r['false_alerts']) for r in negatives)/len(negatives),
                        'common_annotation_scopes':{label:{**{scope:summarize(class_annotation_scope([r for r in all_records if r['case_scope'].startswith(prefix)],label,supported)) for scope,prefix in [('fresh_day','New DAY'),('fresh_night','New NIGHT'),('exposed','Exposed')]},'all_video_plus_true_negatives':summarize(class_annotation_scope(all_records+negatives,label,supported))} for label in ['crack','spalling']},
                        'thin_box_audit':{scope:{'annotations':sum(len(r['thin_annotations']) for r in all_records if r['case_scope'].startswith(prefix)),'localized_matches':sum(r['thin_localized_matches'] for r in all_records if r['case_scope'].startswith(prefix))} for scope,prefix in [('fresh_day','New DAY'),('fresh_night','New NIGHT'),('exposed','Exposed')]}}
        (OUT/f'{tag}-negative-frames.json').write_text(json.dumps(negatives,indent=2),encoding='utf8')
        del model
    (OUT/'paired-summary.json').write_text(json.dumps(aggregate,indent=2),encoding='utf8');print(json.dumps(aggregate,indent=2))


def save_review_panels(case,records,target):
    eligible=[i for i,r in enumerate(records) if r['expected']]
    if not eligible:return
    best=max(eligible,key=lambda i:(records[i]['localized_matches'],-records[i]['missed_defects'],-(records[i]['false_alerts'] or 0),sum(p.get('iou',0) for p in records[i]['predictions'] if p.get('matched_annotation_id'))))
    worst=max(eligible,key=lambda i:(records[i]['missed_defects'],records[i]['false_alerts'] or 0,-records[i]['localized_matches']))
    for name,index in [('best',best),('worst',worst)]:
        record=records[index];raw=cv2.cvtColor(case['frames'][index],cv2.COLOR_GRAY2BGR);truth=raw.copy();prediction=raw.copy()
        for e in record['expected']:
            x,y,w,h=map(round,e['bbox']);cv2.rectangle(truth,(x,y),(x+w,y+h),(255,220,40),1)
            cv2.putText(truth,e['label'],(x,max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,220,40),1)
        for p in record['predictions']:
            x,y,w,h=map(round,p['bbox']);color=(60,205,90) if p.get('matched_annotation_id') else (30,170,245)
            cv2.rectangle(prediction,(x,y),(x+w,y+h),color,1);cv2.putText(prediction,p['label'],(x,max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        for panel,array in [('raw',raw),('publisher-gt',truth),('model-prediction',prediction)]:cv2.imwrite(str(target.with_name(target.name+f'-{name}-{panel}').with_suffix('.png')),array)
        cv2.imwrite(str(target.with_name(target.name+f'-{name}-review').with_suffix('.jpg')),render(case['frames'][index],record,case['name'],name))



def render(gray,record,name,tag):
    image=cv2.resize(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    canvas=np.full((640,1100,3),(24,27,30),np.uint8);canvas[90:610,:692]=image
    matched={p.get('matched_annotation_id') for p in record['predictions'] if p.get('matched_annotation_id')}
    def box(b,color,dashed):
        x,y,w,h=[round(v*2) for v in b];y+=90;right=min(691,x+w);bottom=min(609,y+h)
        if dashed:
            for q in range(max(0,x),right,14):cv2.line(canvas,(q,y),(min(q+8,right),y),color,2);cv2.line(canvas,(q,bottom),(min(q+8,right),bottom),color,2)
            for q in range(max(90,y),bottom,14):cv2.line(canvas,(x,q),(x,min(q+8,bottom)),color,2);cv2.line(canvas,(right,q),(right,min(q+8,bottom)),color,2)
        else:cv2.rectangle(canvas,(x,y),(right,bottom),color,2)
        return max(0,x),max(110,y)
    for truth in record['expected']:
        color=(150,150,150) if truth.get('evaluation_status')=='unsupported' else (245,210,30) if truth['id'] in matched else (70,70,245)
        x,y=box(truth['bbox'],color,True);cv2.putText(canvas,'PUBLISHER '+truth['label'],(x,min(605,y+12)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    for p in record['predictions']:
        color=(60,205,90) if p.get('matched_annotation_id') else (30,170,245);x,y=box(p['bbox'],color,False)
        prefix='MODEL UNVERIFIED' if p.get('evaluation_status')=='unscored' else 'MODEL'
        cv2.putText(canvas,f"{prefix} {p['label']} {p['score']:.2f}",(x,max(110,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1)
    lines=[f"REAL DAMAGE / {tag.upper()} / {name}",f"Source frame {record['source_frame_index']} at {record['video_time_s']:.6f}s"]
    for i,line in enumerate(lines):cv2.putText(canvas,line,(14,28+i*28),cv2.FONT_HERSHEY_SIMPLEX,.58,(235,235,235),1)
    scoring=[f"Matched {record['localized_matches']} / missed {record['missed_defects']}",f"Unmatched predictions {record['false_alerts']}"] if record['evaluated'] else ['UNSCORED: no supported GT','Matches/misses/unmatched','are not scored on this exposure']
    if not record['expected']:scoring[0]='UNSCORED: no publisher annotations'
    notes=['LOCALIZATION: BOX IoU >= .50']+scoring+[f"Unsupported annotations {record['unsupported_defects']}",'Solid green = model matched','Amber = unmatched/unverified','Dashed cyan/red = external GT','Gray = unsupported category','','NEXT INSPECTION','Crack: distinguish joint/shadow;','measure width with a real scale.','Publisher-labelled surface loss.','Verify substrate/coating first.','Inspect loose material/steel.','No automatic code acceptance.']
    for i,line in enumerate(notes):cv2.putText(canvas,line,(710,112+i*27),cv2.FONT_HERSHEY_SIMPLEX,.44,(225,225,225),1)
    cv2.putText(canvas,'Gamage et al., ev-CIVIL (DTU), CC BY4.0 | external labels never fed to model',(12,630),cv2.FONT_HERSHEY_SIMPLEX,.42,(180,190,200),1)
    if record.get('input_exposure_policy')=='low_light_gain':
        # The camera/source pane stays original. This separate pane shows the
        # exact native gained pixels before the declared model letterbox.
        gain=float(record['input_exposure_gain'])
        gained=np.clip(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR).astype(np.float32)*gain,0,255).astype(np.uint8)
        expanded=np.full((640,1808,3),(24,27,30),np.uint8)
        expanded[:,:1100]=canvas
        expanded[90:610,1110:1802]=cv2.resize(gained,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        cv2.putText(expanded,'ACTUAL NATIVE MODEL INPUT',(1110,29),cv2.FONT_HERSHEY_SIMPLEX,.60,(235,235,235),1)
        cv2.putText(expanded,f'Gain {gain:.3f}x; before letterbox; raw camera is at left',(1110,57),cv2.FONT_HERSHEY_SIMPLEX,.46,(235,235,235),1)
        cv2.putText(expanded,'Global photometric gain only; source/raw video is unchanged.',(1110,630),cv2.FONT_HERSHEY_SIMPLEX,.40,(180,190,200),1)
        return expanded
    return canvas


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--phase',choices=['calibrate','compare'],required=True);a.add_argument('--model',type=Path,required=True);args=a.parse_args()
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True,parents=True)
    (calibrate if args.phase=='calibrate' else compare)(args.model.resolve())
