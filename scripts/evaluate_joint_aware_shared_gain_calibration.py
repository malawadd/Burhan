"""Preregistered CAL-only original joint-aware model + external shared gain.

Record metadata-only protocol first. Never changes active models/profiles/fit.
Actual execution requires a pinned protocol and coordinator GPU authorization.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'datasets/construction/evcivil-round3'
GAIN=ROOT/'datasets/construction/evcivil-round4-gain'
OUT=ROOT/'output/evaluation/construction-round3/joint-aware-shared-gain-calibration'
GRID=[.10,.15,.20,.25,.35,.45,.55,.65,.75,.85]
MODEL_SHA='4d48232d7b1427ac468022d2cc11a344885629cac71ad48bd2ff4c6659302246'
GAIN_SHA='451b3f899fee6217066b425c664e3b3b9689cf81fd67dc1450eb5d9c90ad8707'
MANIFEST_SHA='e1de302d901f4994359bbc780f7f0e3a34d6408b5ae63842d37eaa5491467dd4'
RAW_BASE=ROOT/'output/evaluation/construction-round3/baseline-joint-aware/fit-diagnostics'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def relative(path):return Path(path).relative_to(ROOT).as_posix()

def protocol_and_cases():
    # No image/mask bytes are read here; identities come from preserved metadata.
    manifest_path=GAIN/'native-input-manifest.json'
    if sha(manifest_path)!=MANIFEST_SHA:raise ValueError('Preserved native input manifest changed')
    shared=ROOT/'edge_runtime/profiles/construction_preprocessing.py'
    if sha(shared)!=GAIN_SHA:raise ValueError('Shared camera policy changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    mapping={str((ROOT/r['raw_source_image']).resolve()):r for r in manifest if r['split']=='calibration'}
    sources=json.loads((BASE/'exposures.json').read_text(encoding='utf-8'))
    backgrounds=json.loads((BASE/'background-import.json').read_text(encoding='utf-8'))
    cases=[]
    for row in sources:
        if row['split']!='calibration':continue
        expected=[{'id':f"{row['global_defect_id']}:{row['original_frame_index']}:{i}",**b} for i,b in enumerate(row['boxes'])]
        cases.append({'source':row['global_defect_id'],'image':relative(ROOT/row['image']),
                      'frame_index':row['original_frame_index'],'expected':expected,'negative':False})
    for row in backgrounds['records']:
        if row['split']=='calibration':cases.append({'source':row['source_name'],'image':relative(ROOT/row['derived_image']),
                                                   'frame_index':0,'expected':[],'negative':True})
    for case in cases:
        path=(ROOT/case['image']).resolve()
        if not path.is_relative_to((BASE/'images/calibration').resolve()):raise ValueError('Comparator source leaves original raw calibration')
        identity=mapping[str(path)]
        case.update(raw_image_sha256=identity['raw_source_file_sha256'],native_shape_wh=identity['shape_wh'],
                    gained_native_bgr_sha256=identity['native_model_input_bgr_sha256'],expected_gain=identity['gain'])
    if len(cases)!=122 or sum(c['negative'] for c in cases)!=10:raise ValueError('Exact122 CAL sources/10 negatives required')
    counts={label:sum(b['label']==label for c in cases for b in c['expected']) for label in ['crack','spalling']}
    if counts!={'crack':125,'spalling':38}:raise ValueError('Original publisher CAL label membership changed')
    model=ROOT/'models/construction/crack-joints-yolov8n-seg.onnx'
    if sha(model)!=MODEL_SHA:raise ValueError('Original joint-aware ONNX changed')
    preview=[]
    filters=[('first',lambda c:True),('night',lambda c:c['source'].startswith(('27_','ros_'))),
             ('unsupported',lambda c:bool(c['expected']) and not any(b['label']=='crack' for b in c['expected'])),
             ('negative',lambda c:c['negative'])]
    for role,predicate in filters:
        index=next(i for i,c in enumerate(cases) if predicate(c));preview.append({'role':role,'case_index':index,'image':cases[index]['image']})
    protocol={'experiment':'original-joint-aware-plus-shared-gain-CAL-v1','frozen_before_forward':True,
        'scope':'Exact122 original construction CAL sources only, no fitting, exposed recordings or heldout inference; known calibration reuse, not blind validation.',
        'model':{'path':relative(model),'sha256':MODEL_SHA,'classes':['joint','crack'],'input_size':640,'engine':'openvino','device':'GPU','precision':'f32','mask_rendering':'boxes'},
        'preprocessing':{'external_policy':'low_light_gain','source_sha256':GAIN_SHA,'internal_exposure_policy':'off',
            'order':'camera_exposure(original_raw) ONCE, then UNCHANGED original crack model.infer(enhanced) with internalpolicyoff; letterbox/RGB255 unchanged',
            'rule':'Nativegrayp99<64 -> multiplyallBGRbymin(8,128/max(p99,1)),clip0..255,uint8truncation;otherwiseunchanged',
            'roi':None,'dilation':False,'box_inflation':False,'ground_truth_enters_model':False},
        'threshold_grid':GRID,'nms_iou':.45,'matching_iou':.5,'maximum_global_detections':100,
        'joint_context':'Keep original classargmax/perclassNMS/globaltop100 includingjoint; hidejoint onlyafterUNCHANGEDdecoder. No class relabeling or crack score recovery from joint-winning anchors.',
        'scoring_scope':'Run all122; primary crack precision/recall scores only75 crack-annotated frames+10 verified negatives.37 unsupported-only spall frames remain unscored for crack TP/FN/FP;38spallannotations separatelyunsupported. Same fair corrected rawbaseline scope.',
        'selection':'Maxcrackrecall underprecision>=.80 and10verifiednegativeimageFPR<=.10; tiesprecision,threshold. Ifinfeasible fixedbestF1,precision,threshold retainedexplicitFAILED.',
        'minimum_crack_continuation_guard':{'recall_min':.5,'precision_min':.8,'geometric_thin_recall_min':.5,'negative_image_fpr_max':.1,'selected_serving_decoder_parity_all122':True},
        'unsupported_spalling':'Alwaysunsupported. Passingcrackguard cannotavoidspallcontinuation or qualifycombinedconstructioninspection.',
        'thin_definition':'Nativepublisherbox shortside<=30px andlongside>=2*shortside; not physicalwidth or pixelmask accuracy.',
        'historical_raw_baseline':{'confidence':.25,'model_sha256':MODEL_SHA,'records_path':relative(RAW_BASE/'calibration-records.json'),
            'records_sha256':sha(RAW_BASE/'calibration-records.json'),'summary_sha256':sha(RAW_BASE/'summary.json'),
            'use':'RescoreactualsavedRAWmodelboxes withsameGT/matching/supportedscope; verify8TP/117FN/10unmatched. CompareGAINatfixed.25separatelyfromGAINselectedthreshold.'},
        'execution':{'accuracy_only':True,'construction_cpu_fit_threads':3,'opencv_threads':1,
            'forward_budget':'122gain/mincutoff forwards plus122gain/selectedpoint actualinfer parity forwards =244; no rawbaseline NNrerun',
            'cache':'SaveallactualNNoutputs and source/gainedidentity for122; offlinegrid rerunsUNCHANGEDdecoder fromrawoutputs, never filterspriorNMSresults.',
            'no_speed_claim':'ConcurrentCPUfit; no runtime/capacity/15FPS conclusion.'},
        'fixed_previews':preview,'ordered_cases':cases,'input_manifest_sha256':MANIFEST_SHA,
        'source_exposures_sha256':sha(BASE/'exposures.json'),'source_background_manifest_sha256':sha(BASE/'background-import.json'),
        'script_sha256':sha(Path(__file__)),'serving_decoder_source_sha256':sha(ROOT/'edge_runtime/profiles/construction_model.py'),
        'scorer_source_sha256':sha(ROOT/'scripts/evaluate_construction_round3.py')}
    return protocol,cases

def run(protocol,cases,directory):
    import cv2,numpy as np
    from edge_runtime.profiles.construction_model import ConstructionModel,prepare,decode
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from evaluate_construction_round3 import score_frame,summarize
    cv2.setNumThreads(1)
    if (directory/'summary.json').exists() or (directory/'run-journal.json').exists():raise ValueError('Existing comparator run preserved; use a separately reviewed version after any failure')
    write(directory/'run-journal.json',{'protocol_sha256':sha(directory/'protocol-before-forward.json'),'started':True,'completed':False})
    baseline=json.loads((ROOT/protocol['historical_raw_baseline']['records_path']).read_text(encoding='utf-8'))
    old_by_image={str(Path(r['image']).resolve()):r for r in baseline}
    raw_records=[]
    for case in cases:
        old=old_by_image[str((ROOT/case['image']).resolve())]
        expected=lambda rows:sorted((b['id'],b['label'],tuple(b['bbox'])) for b in rows)
        if expected(old['expected'])!=expected(case['expected']):raise ValueError('Historical raw baseline GT differs from exact comparator inputs')
        raw_records.append(score_frame(old['predictions'],copy.deepcopy(case['expected']),{'crack'},case['negative']))
    raw_summary=summarize(raw_records)
    if (raw_summary['localized_matches'],raw_summary['missed_defects'],raw_summary['false_alerts'])!=(8,117,10):raise ValueError('Historical baseline does not reproduce corrected8/125 same-criterion scope')
    model=ConstructionModel({'mode':'crack','crack_variant':'joint_aware_candidate','confidence':min(GRID),
        'nms_iou':.45,'engine':'openvino','openvino_device':'GPU','mask_rendering':'boxes','cpu_threads':1,'exposure_policy':'off'})
    if model.sha256!=MODEL_SHA or model.input_size!=640 or model.exposure_policy!='off':raise ValueError('Unchanged original model contract differs')
    original_request=model.request
    class Capture:
        def infer(self,inputs):
            self.tensor_sha256=hashlib.sha256(next(iter(inputs.values())).tobytes()).hexdigest()
            result=original_request.infer(inputs)
            self.outputs=[np.array(result[model.compiled.output(i)],copy=True) for i in range(len(model.compiled.outputs))]
            return result
    capture=Capture();model.request=capture;cache=[];input_records=[]
    raw_cache=directory/'raw-outputs';raw_cache.mkdir()
    for index,case in enumerate(cases):
        path=ROOT/case['image']
        if sha(path)!=case['raw_image_sha256']:raise ValueError('Actual raw CAL source identity changed')
        frame=cv2.imread(str(path));enhanced,gain=camera_exposure(frame,'low_light_gain')
        if [frame.shape[1],frame.shape[0]]!=case['native_shape_wh'] or gain!=case['expected_gain'] or hashlib.sha256(enhanced.tobytes()).hexdigest()!=case['gained_native_bgr_sha256']:raise ValueError('Shared native gain differs from frozen prepared identity')
        actual=model.infer(enhanced)
        tensor,transform=prepare(enhanced,'crack',size=640)
        if hashlib.sha256(tensor.tobytes()).hexdigest()!=capture.tensor_sha256 or model.last_input_exposure_gain!=1.:raise ValueError('Externalgain/internaloff preprocessing parity failed')
        decoded=decode(capture.outputs,'crack',transform,frame.shape[:2],min(GRID),.45,'boxes',classes=['joint','crack'])
        signature=lambda rows:[(r['label'],tuple(r['bbox']),round(float(r['score']),6)) for r in rows]
        if signature(actual)!=signature(decoded):raise ValueError('Minimum-threshold actual serving decoder differs')
        np.savez(raw_cache/f'frame-{index:04d}.npz',**{f'output_{i}':value for i,value in enumerate(capture.outputs)})
        cache.append({'transform':transform,'shape':frame.shape[:2]})
        input_records.append({'image':case['image'],'raw_image_sha256':case['raw_image_sha256'],'external_gain':gain,
                              'native_model_input_bgr_sha256':hashlib.sha256(enhanced.tobytes()).hexdigest(),'input_tensor_sha256':capture.tensor_sha256})
        if (index+1)%20==0:print(f'Joint+gain originalCAL input pass {index+1}/122',flush=True)
    model.request=original_request
    trials=[];trial_records={}
    thin_ids={b['id'] for c in cases for b in c['expected'] if b['label']=='crack' and min(b['bbox'][2:])<=30 and max(b['bbox'][2:])>=2*min(b['bbox'][2:])}
    for threshold in GRID:
        rows=[];negative=[]
        for index,(case,geometry) in enumerate(zip(cases,cache)):
            with np.load(raw_cache/f'frame-{index:04d}.npz') as data:outputs=[data[f'output_{i}'] for i in range(2)]
            predictions=decode(outputs,'crack',geometry['transform'],geometry['shape'],threshold,.45,'boxes',classes=['joint','crack'])
            scored=score_frame(predictions,copy.deepcopy(case['expected']),{'crack'},case['negative'])
            rows.append(scored)
            if case['negative']:negative.append(any(p['label']=='crack' for p in predictions))
        summary=summarize(rows);c=summary['by_class']['crack'];precision=c['precision'] or 0;recall=c['recall'] or 0
        thin_tp=sum(p.get('matched_annotation_id') in thin_ids for row in rows for p in row['predictions'])
        point={'threshold':threshold,**c,'negative_image_fpr':sum(negative)/len(negative),'f1':2*precision*recall/(precision+recall) if precision+recall else 0,
               'thin_annotations':len(thin_ids),'thin_localized':thin_tp,'thin_recall':thin_tp/len(thin_ids),
               'precision_and_negative_gate':precision>=.8 and sum(negative)/len(negative)<=.1}
        trials.append(point);trial_records[threshold]=rows
    eligible=[r for r in trials if r['precision_and_negative_gate']]
    selected=max(eligible,key=lambda r:(r['recall'] or 0,r['precision'] or 0,r['threshold'])) if eligible else max(trials,key=lambda r:(r['f1'],r['precision'] or 0,r['threshold']))
    model.threshold=selected['threshold'];final=[]
    for index,case in enumerate(cases):
        raw=cv2.imread(str(ROOT/case['image']));enhanced,gain=camera_exposure(raw,'low_light_gain');actual=model.infer(enhanced)
        actual=[r for r in actual if r['label']=='crack'];expected=trial_records[selected['threshold']][index]['predictions']
        if signature(actual)!=signature(expected):raise ValueError('Selected-point actual serving parity failed')
        scored=score_frame(actual,copy.deepcopy(case['expected']),{'crack'},case['negative'])
        final.append({'source':case['source'],'image':case['image'],'frame_index':case['frame_index'],'negative':case['negative'],**input_records[index],**scored})
        for preview in protocol['fixed_previews']:
            if preview['case_index']==index:save_preview(directory/('fixed-'+preview['role']+'.jpg'),raw,enhanced,scored,case['source'])
    guards={'crack_recall_ge50pct':(selected['recall'] or 0)>=.5,'crack_precision_ge80pct':(selected['precision'] or 0)>=.8,
            'thin_recall_ge50pct':selected['thin_recall']>=.5,'negative_image_fpr_le10pct':selected['negative_image_fpr']<=.1,'selected_actual_serving_parity_all122':True}
    result={'protocol_sha256':sha(directory/'protocol-before-forward.json'),'model':model.describe(),
        'external_preprocessing':protocol['preprocessing'],'historical_raw_baseline_same_criterion':raw_summary,
        'gain_fixed_baseline_cutoff_0_25':next(r for r in trials if r['threshold']==.25),'selected':selected,'grid':trials,
        'selected_summary':summarize(final),'crack_specialist_minimum_guard':{'checks':guards,'passed':all(guards.values()),'production_qualification':False},
        'spalling':'UNSUPPORTED38annotations; no spallcontinuation avoided or spall/localization claim',
        'accuracy_scope':'CAL-only reused data;85scored75crack+10verifiednegative,37unsupported-onlyspallframes unscored; no independent validation. FP=unmatchedboxes underpublisherannotation scope.',
        'pixel_mask_metrics':None,'mask_scope':'Actual detector boxes versus publisherboxesIoU.50; not derivedsemanticcomponent boxes and no pixel-mask/physicalwidth metric.',
        'timing_scope':protocol['execution'],'records':final,'input_records':input_records}
    write(directory/'summary.json',result)
    write(directory/'run-journal.json',{'protocol_sha256':sha(directory/'protocol-before-forward.json'),'completed':True,'summary_sha256':sha(directory/'summary.json')})
    print(json.dumps({k:v for k,v in result.items() if k not in {'records','input_records','model','grid'}},indent=2))

def save_preview(path,raw,enhanced,record,name):
    import cv2,numpy as np
    truth=raw.copy();pred=enhanced.copy()
    for b in record['expected']:
        x,y,w,h=map(round,b['bbox']);color=(160,160,160) if b.get('evaluation_status')=='unsupported' else (255,210,0)
        cv2.rectangle(truth,(x,y),(x+w,y+h),color,1)
    for p in record['predictions']:
        x,y,w,h=map(round,p['bbox']);cv2.rectangle(pred,(x,y),(x+w,y+h),(60,205,90) if p.get('matched_annotation_id') else (30,170,245),1)
    panels=[]
    for title,image in [('ORIGINAL RAW',raw),('ACTUAL SHARED GAIN INPUT',enhanced),('PUBLISHER GT (gray unsupported)',truth),('ACTUAL CRACK MODEL BOXES',pred)]:
        image=cv2.resize(image,(692,520));header=np.full((60,692,3),24,np.uint8)
        cv2.putText(header,title,(8,24),0,.6,(245,245,245),1);cv2.putText(header,name+' CAL only; no code approval',(8,48),0,.5,(245,245,245),1)
        panels.append(np.vstack([header,image]))
    cv2.imwrite(str(path),np.hstack(panels))

def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record-protocol',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--protocol-sha256');args=parser.parse_args();protocol,cases=protocol_and_cases()
    path=OUT/'protocol-before-forward.json'
    if args.record_protocol:
        OUT.mkdir(parents=True,exist_ok=True)
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=protocol:raise ValueError('Existing preregistration differs; preserve it and create a distinct version')
        if not path.exists():write(path,protocol)
        print(json.dumps({'protocol':str(path),'sha256':sha(path),'forward_budget':protocol['execution']['forward_budget']},indent=2));return
    if not args.protocol_sha256 or sha(path)!=args.protocol_sha256 or json.loads(path.read_text(encoding='utf-8'))!=protocol:raise ValueError('Exact pre-forward protocol/script/source identity is required')
    run(protocol,cases,OUT)

if __name__=='__main__':main()
