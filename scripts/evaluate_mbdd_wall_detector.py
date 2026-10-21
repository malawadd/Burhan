"""Five-class wall CAL/freeze and fixed18photo diagnostics; no heldout loader.

All source classes retain publisher names. Empty/absent labels do not establish
healthy buildings or hidden physical condition. No pixel IoU is computed.
"""
from pathlib import Path
import argparse,hashlib,json,sys,traceback
from datetime import datetime,timezone
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_model import ConstructionModel,DAMAGE_VARIANTS,decode_damage_boxes
from evaluate_construction_defect_video import match
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd-wall-five-class-v1/prepared-yolo'
MODEL=ROOT/'models/construction/mbdd-wall-five-class-v2/wall-five-class-last4-640.onnx'
OUT=ROOT/'output/evaluation/construction-wall-five-class-v2'
CLASSES=['crack','leakage','abscission','corrosion','bulge']
CAPABILITIES=['surface_crack','surface_leakage_appearance','surface_detachment','surface_corrosion_appearance','surface_bulge_appearance']
GRID=[.10,.15,.20,.25,.35,.45,.55,.65,.75,.85]
ACTIVE_RECEIPT=None
PLAIN={'crack':'Visible crack','leakage':'Wet/stained appearance','abscission':'Surface material detachment',
       'corrosion':'Corrosion appearance','bulge':'Surface bulge appearance'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def truth(row):
    return [{'id':row['source_name']+':'+str(i),'label':b['label'],
             'bbox':[b['source_xyxy'][0],b['source_xyxy'][1],b['source_xyxy'][2]-b['source_xyxy'][0],b['source_xyxy'][3]-b['source_xyxy'][1]]}
            for i,b in enumerate(row['boxes'])]


def score(predictions,expected):
    predictions=[{'id':i+1,'label':p['label'],'bbox':list(p['bbox']),'score':float(p['score'])} for i,p in enumerate(predictions)]
    matches,misses=match(predictions,expected)
    counts={name:{'tp':sum(bool(p.get('matched_annotation_id')) for p in predictions if p['label']==name),
                  'fp':sum(not p.get('matched_annotation_id') for p in predictions if p['label']==name),
                  'fn':sum(e['label']==name for e in misses),'annotations':sum(e['label']==name for e in expected)} for name in CLASSES}
    return {'expected':expected,'predictions':predictions,'by_class':counts}


def summary(records):
    output={}
    for name in CLASSES:
        counts={key:sum(r['by_class'][name][key] for r in records) for key in ['tp','fp','fn','annotations']}
        counts['precision']=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None
        counts['recall']=counts['tp']/counts['annotations'] if counts['annotations'] else None
        absent=[r for r in records if not r['by_class'][name]['annotations']]
        counts['class_absent_images']=len(absent)
        counts['class_absent_images_flagged']=sum(any(p['label']==name for p in r['predictions']) for r in absent)
        output[name]=counts
    return output


def load_model():
    meta=json.loads(MODEL.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    if sha(MODEL)!=meta['onnx_sha256'] or meta['classes']!=CLASSES:raise RuntimeError('Frozen fitted wall identity differs')
    spec={'file':MODEL.relative_to(ROOT/'models/construction').as_posix(),'sha256':sha(MODEL),'classes':CLASSES,
          'capabilities':CAPABILITIES,'class_capabilities':dict(zip(CLASSES,CAPABILITIES)),'source':meta['source'],'decoder':'yolo_detect',
          'input_size':640,'confidence':.05,'license':meta['license'],'exposure_policy':'off','diagnostic_only':True,
          'validation_scope':'Sourceblock-separated MBDD photos with unknown physicalproject grouping; visibleclass appearance only, healthy-wall FPR unmeasured'}
    DAMAGE_VARIANTS['mbdd_wall_evaluation_only']=spec
    return ConstructionModel({'mode':'damage','damage_variant':'mbdd_wall_evaluation_only','engine':'openvino','openvino_device':'GPU','cpu_threads':1,'mask_rendering':'boxes'})


def render(frame,record,target,guard_pass):
    panels=[frame.copy(),frame.copy(),frame.copy()]
    matched={p['matched_annotation_id'] for p in record['predictions'] if p.get('matched_annotation_id')}
    for e in record['expected']:
        x,y,w,h=e['bbox'];color=(220,190,40) if e['id'] in matched else (40,40,240)
        cv2.rectangle(panels[1],(round(x),round(y)),(round(x+w),round(y+h)),color,2)
        cv2.putText(panels[1],e['label'],(round(x),max(15,round(y)-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    for p in record['predictions']:
        x,y,w,h=p['bbox'];color=(50,190,70) if p.get('matched_annotation_id') else (20,180,240)
        cv2.rectangle(panels[2],(round(x),round(y)),(round(x+w),round(y+h)),color,2)
        cv2.putText(panels[2],p['label']+f" {p['score']:.2f}",(round(x),max(15,round(y)-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    canvas=np.full((410,1440,3),22,np.uint8)
    titles=['ORIGINAL RAW PHOTO','PUBLISHER BOXES: cyan hit / red missed','ACTUAL LOCAL MODEL: green hit / amber unmatched']
    for i,panel in enumerate(panels):
        canvas[48:318,i*480:(i+1)*480]=cv2.resize(panel,(480,270),interpolation=cv2.INTER_AREA)
        cv2.putText(canvas,titles[i],(i*480+4,40),cv2.FONT_HERSHEY_SIMPLEX,.37,(235,235,235),1)
    cv2.putText(canvas,'FIXED PHOTO DEVELOPMENT DIAGNOSTIC | '+('Minimum CAL guard passed; unqualified prototype' if guard_pass else 'FAILED CALIBRATION - preserve all misses/alerts'),
                (5,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1)
    for i,name in enumerate(CLASSES):
        values=record['by_class'][name];text=f"{PLAIN[name]}: {values['tp']} matched, {values['fn']} missed, {values['fp']} unmatched"
        cv2.putText(canvas,text,(5+(i%2)*720,340+(i//2)*20),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1)
    cv2.putText(canvas,'Verify material/source with close inspection; use moisture instrument or calibrated geometry where needed. No hidden condition or code decision.',
                (5,404),cv2.FONT_HERSHEY_SIMPLEX,.40,(210,210,210),1)
    cv2.imwrite(str(target),canvas)


def checked_point(point,protocol_path):
    """Rebind frozen selection to actual retained CAL records before diagnostics."""
    if point['model_sha256']!=sha(MODEL) or not point['all_actual_serving_parity'] or point['evaluation_protocol_sha256']!=sha(protocol_path):
        raise RuntimeError('Wall point model/protocol/parity identity differs')
    if set(point['class_confidence'])!=set(CLASSES) or set(point['search'])!=set(CLASSES):
        raise RuntimeError('Wall selected class scope differs')
    records_path=OUT/'calibration-runtime-records.json'
    raw_path=OUT/'calibration-raw-outputs.npz'
    if sha(records_path)!=point['calibration_records_sha256'] or sha(raw_path)!=point['calibration_raw_outputs_sha256']:
        raise RuntimeError('Retained actual CAL outputs changed')
    records=json.loads(records_path.read_text(encoding='utf-8'))
    recomputed=summary([score(row['predictions'],row['expected']) for row in records])
    if recomputed!=point['by_class']:raise RuntimeError('Actual CAL match counts do not bind to selected point')
    for name in CLASSES:
        trials=point['search'][name]['grid']
        if [row['threshold'] for row in trials]!=GRID:raise RuntimeError('Frozen threshold grid changed')
        for row in trials:
            p=row['tp']/(row['tp']+row['fp']) if row['tp']+row['fp'] else None
            r=row['tp']/row['annotations'] if row['annotations'] else None
            f1=2*(p or 0)*(r or 0)/((p or 0)+(r or 0)) if (p or 0)+(r or 0) else 0
            if row['precision']!=p or row['recall']!=r or row['f1']!=f1 or row['precision_gate']!=((p or 0)>=.8):
                raise RuntimeError('Frozen search arithmetic differs')
        eligible=[row for row in trials if row['precision_gate']]
        selected=max(eligible,key=lambda row:(row['recall'] or 0,row['precision'] or 0,row['threshold'])) if eligible else max(trials,key=lambda row:(row['f1'],row['precision'] or 0,row['threshold']))
        if selected!=point['search'][name]['selected'] or point['class_confidence'][name]!=selected['threshold']:
            raise RuntimeError('Frozen operating-point rule differs')
        if any(recomputed[name][key]!=selected[key] for key in ['tp','fp','fn']):raise RuntimeError('Selected joint CAL count parity differs')
    checks={name:(recomputed[name]['precision'] or 0)>=.8 and (recomputed[name]['recall'] or 0)>=.5 for name in CLASSES}
    if point['per_class_minimum_review_guard']!=checks or point['all_minimum_review_guards_pass']!=all(checks.values()):
        raise RuntimeError('Frozen minimum review guard differs')


def main():
    global ACTIVE_RECEIPT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['freeze','calibration','diagnostic'],required=True);args=parser.parse_args()
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=DATA/'input-manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol_path=OUT/'evaluation-protocol-before-calibration.json'
    if args.stage=='freeze':
        protocol={'script_sha256':sha(Path(__file__)),'model_sha256':sha(MODEL),'model_metadata_sha256':sha(MODEL.with_suffix('.provenance.json')),
                  'source_input_manifest_sha256':sha(manifest_path),'construction_model_source_sha256':sha(ROOT/'edge_runtime/profiles/construction_model.py'),
                  'classes':CLASSES,'capabilities':CAPABILITIES,'class_capabilities':dict(zip(CLASSES,CAPABILITIES)),
                  'confidence_grid':GRID,'matching_iou_threshold':.5,'nms_iou':.45,
                  'selection':'Perclass maximumCALrecall subjectprecision>=.80; ifinfeasible usehighestF1 asFAILEDdiagnostic. Actualjointdecoder top100 parity andperclass countparity required before any18diagnostic forwards.',
                  'minimum_review':{'each_class_precision':.8,'each_class_box_recall':.5,'serving_parity':True,'production_qualification':False},
                  'negative_scope':'Other-class-absent images are classconfusion diagnostics, NOT healthy-wall FPR. EightemptyXML notassumedhealthy andnotimported.',
                  'protected_diagnostic_manifest_sha256':sha(ROOT/'datasets/construction/finishes-expansion/mbdd2025-wall-photos/asset-manifest.json'),
                  'diagnostic_rule':'ALL18 originalpreselected wallphotos after model/pointfreeze, includingallmisses/wrongclassalerts; correlateddevelopmentstillphotos, no video/independent-site claim',
                  'prohibited':'No oldcameraheldout loader, no test-driven weights/thresholdchanges, no maskIoUfromboxes, no paint-only/hiddenmoisture/depth/SBCclaim'}
        if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Preserve frozen wall evaluation protocol')
        write(protocol_path,protocol);print('FROZEN',sha(protocol_path),flush=True);return
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    for path,expected in [(Path(__file__),protocol['script_sha256']),(MODEL,protocol['model_sha256']),
                          (MODEL.with_suffix('.provenance.json'),protocol['model_metadata_sha256']),
                          (manifest_path,protocol['source_input_manifest_sha256']),(ROOT/'edge_runtime/profiles/construction_model.py',protocol['construction_model_source_sha256'])]:
        if sha(path)!=expected:raise RuntimeError('Frozen wall evaluation source/model changed')
    receipt=OUT/(args.stage+'-execution.json')
    if receipt.exists():raise RuntimeError('Preserve attempted wall stage, including partial/failed executions; no silent retry')
    if args.stage=='calibration' and any((OUT/name).exists() for name in ['operating-point.json','calibration-raw-outputs.npz','calibration-runtime-records.json']):
        raise RuntimeError('Preserve existing partial wall calibration artifacts')
    if args.stage=='diagnostic' and ((OUT/'fixed18-diagnostic-report.json').exists() or list(OUT.glob('*-raw-publisher-model.jpg'))):
        raise RuntimeError('Preserve existing partial wall diagnostic artifacts')
    ACTIVE_RECEIPT=receipt
    write(receipt,{'stage':args.stage,'state':'started','started_utc':datetime.now(timezone.utc).isoformat(),
                   'evaluation_protocol_sha256':sha(protocol_path),'script_sha256':sha(Path(__file__))})
    model=load_model()
    point_path=OUT/'operating-point.json'
    if args.stage=='calibration':
        if point_path.exists():raise RuntimeError('Preserve completed wall calibration point')
        rows=[r for r in manifest['records'] if r['split']=='calibration'];cache=[]
        for row in rows:
            image=ROOT/row['prepared_image']
            if sha(image)!=row['prepared_image_sha256']:raise RuntimeError('Frozen CAL wall photo differs')
            frame=cv2.imread(str(image));tensor,transform=model.prepare_input(frame);response=model.request.infer({0:tensor})
            outputs=[np.array(response[model.compiled.output(i)],copy=True) for i in range(len(model.compiled.outputs))]
            cache.append((outputs,transform,frame.shape[:2]))
        raw_path=OUT/'calibration-raw-outputs.npz'
        np.savez_compressed(raw_path,**{f'frame_{i:04d}':r[0][0] for i,r in enumerate(cache)})
        choices={};search={}
        for name in CLASSES:
            trials=[]
            for cut in GRID:
                records=[]
                for row,(outputs,transform,shape) in zip(rows,cache):
                    cuts={label:cut if label==name else 1. for label in CLASSES}
                    ps=decode_damage_boxes(outputs,transform,shape,640,CLASSES,cut,.45,cuts)
                    records.append(score(ps,[e for e in truth(row) if e['label']==name]))
                result=summary(records)[name];p=result['precision'] or 0;r=result['recall'] or 0
                trials.append({'threshold':cut,**result,'precision_gate':p>=.8,'f1':2*p*r/(p+r) if p+r else 0})
            eligible=[r for r in trials if r['precision_gate']]
            selected=max(eligible,key=lambda r:(r['recall'] or 0,r['precision'] or 0,r['threshold'])) if eligible else max(trials,key=lambda r:(r['f1'],r['precision'] or 0,r['threshold']))
            choices[name]=selected['threshold'];search[name]={'grid':trials,'selected':selected}
        model.class_thresholds=choices;model.threshold=min(choices.values());records=[]
        for row,(outputs,transform,shape) in zip(rows,cache):
            expected=decode_damage_boxes(outputs,transform,shape,640,CLASSES,model.threshold,.45,choices)
            actual=model.infer(cv2.imread(str(ROOT/row['prepared_image'])))
            signature=lambda ps:[(p['label'],list(p['bbox']),round(float(p['score']),6)) for p in ps]
            if signature(expected)!=signature(actual):
                write(OUT/'calibration-serving-parity-failure.json',{'source_name':row['source_name'],'cached':signature(expected),'actual':signature(actual),
                      'completed_records':records,'failing_actual_record':score(actual,truth(row))})
                raise RuntimeError('Actual wall serving/cache predictions differ; diagnostics forbidden')
            records.append({'source_name':row['source_name'],**score(actual,truth(row))})
        results=summary(records)
        records_path=OUT/'calibration-runtime-records.json'
        write(records_path,records)
        for name in CLASSES:
            if any(results[name][key]!=search[name]['selected'][key] for key in ['tp','fp','fn']):raise RuntimeError('Globalcap changed wall class counts; preserve failure, do not open diagnostics')
        checks={name:(results[name]['precision'] or 0)>=.8 and (results[name]['recall'] or 0)>=.5 for name in CLASSES}
        point={'model_sha256':sha(MODEL),'evaluation_protocol_sha256':sha(protocol_path),'class_confidence':choices,
               'matching_iou_threshold':.5,'nms_iou':.45,'all_actual_serving_parity':True,'by_class':results,'search':search,
               'calibration_records_sha256':sha(records_path),'calibration_raw_outputs_sha256':sha(raw_path),
               'all_minimum_review_guards_pass':all(checks.values()),'per_class_minimum_review_guard':checks,'healthy_wall_fpr':None,
               'scope':protocol['negative_scope'],'production_qualification':False}
        write(point_path,point);checked_point(point,protocol_path)
        completed=json.loads(receipt.read_text());completed.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),operating_point_sha256=sha(point_path));write(receipt,completed)
        print(json.dumps(point,indent=2),flush=True);return
    point=json.loads(point_path.read_text(encoding='utf-8'))
    checked_point(point,protocol_path)
    completed_cal=json.loads((OUT/'calibration-execution.json').read_text(encoding='utf-8'))
    if completed_cal.get('state')!='completed' or completed_cal.get('operating_point_sha256')!=sha(point_path):
        raise RuntimeError('Only completed hash-bound wall calibration may open diagnostic photos')
    if (OUT/'fixed18-diagnostic-report.json').exists():raise RuntimeError('Preserve completed fixed wall diagnostics')
    diagroot=ROOT/'datasets/construction/finishes-expansion/mbdd2025-wall-photos';diagpath=diagroot/'asset-manifest.json'
    if sha(diagpath)!=protocol['protected_diagnostic_manifest_sha256']:raise RuntimeError('Original18 diagnostic selection changed')
    original=json.loads(diagpath.read_text(encoding='utf-8'));model.class_thresholds=point['class_confidence'];model.threshold=min(model.class_thresholds.values());records=[]
    for row in original['records']:
        image=diagroot/row['image']
        if sha(image)!=row['image_sha256']:raise RuntimeError('Original18 wall source differs')
        source={**row,'source_name':Path(row['image']).name};frame=cv2.imread(str(image));record=score(model.infer(frame),truth(source))
        comparison=OUT/(Path(row['image']).stem+'-raw-publisher-model.jpg')
        render(frame,record,comparison,point['all_minimum_review_guards_pass'])
        records.append({'image':row['image'],'image_sha256':row['image_sha256'],'comparison_image':comparison.relative_to(ROOT).as_posix(),
                        'comparison_image_sha256':sha(comparison),**record})
    write(OUT/'fixed18-diagnostic-report.json',{'model':model.describe(),'point_sha256':sha(point_path),'model_guard_pass':point['all_minimum_review_guards_pass'],
                                            'purpose':'photo_evidence' if point['all_minimum_review_guards_pass'] else 'failure_diagnostic',
                                            'by_class':summary(records),'records':records,'scope':protocol['diagnostic_rule']})
    completed=json.loads(receipt.read_text());completed.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(OUT/'fixed18-diagnostic-report.json'));write(receipt,completed)


if __name__=='__main__':
    try:main()
    except Exception:
        if ACTIVE_RECEIPT is not None:
            failed=json.loads(ACTIVE_RECEIPT.read_text());failed.update(state='failed',failed_utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc());write(ACTIVE_RECEIPT,failed)
        raise
