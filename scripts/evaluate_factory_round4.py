"""Round4: actual fixed3 numeric, passing-only CAL80, all3 freeze. No original NN."""
import argparse,json,math
from datetime import datetime,timezone
import cv2,numpy as np
from factory_round4_supervised import ROOT,OUT,NAME,PROTOCOL,SAVE_AT,sha,read,write,verify_plan,identity
from factory_stfpm_common import CAL_SHA,rank
from factory_mask_metrics import original_pixel_weights,component_mask_statistics,at_area,aggregate
from edge_runtime.profiles.factory_encoder import FactorySegmentation,seeded_growth_mask
ARMS={'primary':1,'secondary':3}

def summarize(rows):
    result=aggregate(rows);positive=[r for r in rows if r['defective']]
    result.update(meaningful_iou_0_1_recall=sum((r['pixel_iou'] or 0)>=.1 for r in positive)/len(positive),
        coverage_50pct_recall=sum((r['foreground_pixel_recall'] or 0)>=.5 for r in positive)/len(positive),
        worst_positive_iou=min((r['pixel_iou'] or 0) for r in positive),worst_positive_coverage=min((r['foreground_pixel_recall'] or 0) for r in positive))
    return result

def raw_candidates(rows,protocol):
    raw=[]
    for high in protocol['raw_high_grid']:
        per_area={area:[] for area in protocol['area_grid']}
        for row in rows:
            stats=component_mask_statistics(row['scores'],high,row['truth'],row['projection_weights'])
            for area in per_area:per_area[area].append({'defective':row['defective'],**at_area(stats,area)})
        for area,values in per_area.items():raw.append({'high':high,'area':area,**summarize(values)})
    return raw

def seeds_for_budget(raw,protocol,budget):
    seeds=[]
    for high in protocol['raw_high_grid']:
        eligible=[p for p in raw if p['high']==high and p['normal_fp']<=budget]
        if eligible:seeds.append(max(eligible,key=rank))
    return sorted(seeds,key=rank,reverse=True)[:6]

def select_round4(rows,protocol,budget,raw=None):
    raw=raw_candidates(rows,protocol) if raw is None else raw;grown=[];records={}
    for seed in seeds_for_budget(raw,protocol,budget):
        for low in protocol['growth_grid']:
            if low is not None and low>seed['high']:continue
            values=[]
            for row in rows:
                mask=seeded_growth_mask(row['scores'],seed['high'],seed['area'],low)
                stats=at_area(component_mask_statistics(mask.astype(np.float32),.5,row['truth'],row['projection_weights']),1)
                values.append({'image':row['image'],'defective':row['defective'],**stats})
            point={'high':seed['high'],'area':seed['area'],'growth':low,**summarize(values)}
            if point['normal_fp']!=seed['normal_fp']:raise ValueError('Growth changed flagged normal count')
            grown.append(point);records[(seed['high'],seed['area'],low)]=values
    eligible=[p for p in grown if p['normal_fp']<=budget and p['foreground_pixel_precision']>=.65]
    point=max(eligible,key=rank) if eligible else None
    return {'normal_FP_budget':budget,'grown_grid':grown,'selected':point,'selected_records':records.get((point['high'],point['area'],point['growth']),[]) if point else []}

def validate_search(raw,arm,protocol,budget):
    if arm['normal_FP_budget']!=budget or [(r['high'],r['area']) for r in raw]!=[(h,a) for h in protocol['raw_high_grid'] for a in protocol['area_grid']]:raise ValueError('Exact protected raw grid/budget required')
    keys=['meaningful_iou_0_1_recall','worst_positive_iou','coverage_50pct_recall','worst_positive_coverage','micro_pixel_iou','foreground_pixel_recall','foreground_pixel_precision']
    for point in raw+arm['grown_grid']:
        if type(point['normal_fp']) is not int or not 0<=point['normal_fp']<=70 or any(not math.isfinite(float(point[k])) or not 0<=point[k]<=1 for k in keys):raise ValueError('Invalid search metrics')
    seeds=seeds_for_budget(raw,protocol,budget)
    expected=[(s['high'],s['area'],low) for s in seeds for low in protocol['growth_grid'] if low is None or low<=s['high']]
    if [(p['high'],p['area'],p['growth']) for p in arm['grown_grid']]!=expected:raise ValueError('Exact independent seed/growth arm search required')
    for point in arm['grown_grid']:
        seed=next(s for s in seeds if s['high']==point['high'] and s['area']==point['area'])
        if point['normal_fp']!=seed['normal_fp']:raise ValueError('Growth changed normal flags')
    eligible=[p for p in arm['grown_grid'] if p['normal_fp']<=budget and p['foreground_pixel_precision']>=.65]
    if arm['selected']!=(max(eligible,key=rank) if eligible else None):raise ValueError('Frozen arm winner changed')

def configuration(compiled):
    result={}
    for key in ['EXECUTION_DEVICES','INFERENCE_PRECISION_HINT','NUM_STREAMS','PERFORMANCE_HINT','GPU_DISABLE_WINOGRAD_CONVOLUTION']:
        try:
            value=compiled.get_property(key)
            result[key]=list(value) if key=='EXECUTION_DEVICES' else value if isinstance(value,(str,bool,int,float)) else str(value)
        except Exception:result[key]=None
    return result

def recipe_passes(properties):
    devices=properties.get('EXECUTION_DEVICES');precision=properties.get('INFERENCE_PRECISION_HINT')
    return (isinstance(devices,list) and len(devices)==1 and str(devices[0]).split('.')[0]=='GPU'
        and precision in ['f16','float16',"<Type: 'float16'>"] and str(properties.get('NUM_STREAMS'))=='1'
        and str(properties.get('PERFORMANCE_HINT')) in ['LATENCY','PerformanceMode.LATENCY']
        and properties.get('GPU_DISABLE_WINOGRAD_CONVOLUTION') is False)

def numeric_passes(report):
    if len(report.get('cases',[]))!=3 or not recipe_passes(report.get('configuration',{})):return False
    finite=lambda x:type(x) in (int,float) and math.isfinite(x)
    for row in report['cases']:
        error=row.get('max_abs_error');bounds=row.get('score_range')
        if row.get('numeric_failure') is not None or not finite(error) or not 0<=error<=.01:return False
        if not isinstance(bounds,list) or len(bounds)!=2 or not all(finite(x) for x in bounds) or not 0<=bounds[0]<=bounds[1]<=1:return False
    return report['cases'][0]['score_range'][1]-report['cases'][0]['score_range'][0]>=1e-7

def checkpoint(step):
    completion=read(OUT/'training-complete.json')
    if completion['protocol_sha256']!=sha(PROTOCOL) or completion['saved_all_three']!=SAVE_AT or completion['fine_steps']!=900 or completion['transfer_steps']!=4096:
        raise ValueError('Complete protected4996 updates/all3 checkpoints required before any GPU feedback')
    path=ROOT/f'models/factory/{NAME}-step{step}.onnx';r=read(OUT/f'checkpoint{step}-training.json')
    if r['protocol_sha256']!=sha(PROTOCOL) or r['model_sha256']!=sha(path) or r['state_sha256']!=sha(path.with_suffix('.pt')):
        raise ValueError('Exact learned checkpoint lineage required')
    ref=OUT/f'checkpoint{step}-torch-reference.npz'
    if r['reference_sha256']!=sha(ref) or r['all_state_finite'] is not True or r['frozen_BN_normalization_unchanged'] is not True:raise ValueError('TRAIN/reference checks failed')
    return path,ref,r

def checked_numeric(step):
    path,ref,training=checkpoint(step);nr=OUT/f'checkpoint{step}-numeric.json';n=read(nr)
    if (n['step']!=step or n['model_sha256']!=sha(path) or n['protocol_sha256']!=sha(PROTOCOL) or n['reference_sha256']!=sha(ref)
        or n['initial_metadata_sha256']!=training['initial_metadata_sha256'] or n['training_receipt_sha256']!=sha(OUT/f'checkpoint{step}-training.json')
        or n['all_checks_passed']!=numeric_passes(n) or sha(path.with_suffix('.onnx.json'))!=training['initial_metadata_sha256']):raise ValueError('Exact numeric/model/reference/training/metadata lineage required')
    return n,nr

def numeric(step):
    path,ref,training=checkpoint(step);target=OUT/f'checkpoint{step}-numeric.json';started=OUT/f'checkpoint{step}-numeric-started.json'
    if any(p.exists() for p in [target,started]):raise ValueError('Preserve first numeric attempt')
    if sha(path.with_suffix('.onnx.json'))!=training['initial_metadata_sha256']:raise ValueError('Initial metadata changed')
    write(started,{'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'fixed_cases':3})
    model=FactorySegmentation(path,output_logits=True);props=configuration(model.compiled)
    if not recipe_passes(props):raise ValueError('Actual C-default GPUf16/1stream/LATENCY/defaultWinograd recipe differs')
    cases=[]
    with np.load(ref,allow_pickle=False) as data:
        inputs=data['inputs_bgr'];expected=data['scores']
        if inputs.shape!=(3,1280,512,3) or inputs.dtype!=np.uint8 or inputs[1].any() or not (inputs[2]==255).all():raise ValueError('Exact reference input schema differs')
        if expected.shape!=(3,640,256) or not np.isfinite(expected).all() or expected.min()<0 or expected.max()>1:raise ValueError('Invalid learned Torch score reference')
        for frame,truth in zip(inputs,expected):
            try:
                score=model.predict(frame,.05,4)[0]
                cases.append({'max_abs_error':float(np.max(np.abs(score-truth))),'score_range':[float(score.min()),float(score.max())],'numeric_failure':None})
            except ValueError as error:
                if str(error)!='Factory neural model produced invalid scores':raise
                cases.append({'max_abs_error':None,'score_range':None,'numeric_failure':str(error)})
    report={'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),
            'reference_sha256':sha(ref),'training_receipt_sha256':sha(OUT/f'checkpoint{step}-training.json'),'configuration':props,'cases':cases}
    report['all_checks_passed']=numeric_passes(report);write(target,report)
    if not report['all_checks_passed']:write(OUT/f'checkpoint{step}-terminal.json',{'step':step,'status':'numeric_failed_no_CAL','CAL_inputs_opened':0,'numeric_sha256':sha(target),'protocol_sha256':sha(PROTOCOL)})
    print(json.dumps({'step':step,'numeric_pass':report['all_checks_passed'],'cases':cases},indent=2),flush=True)

def calibration(step,protocol):
    from train_factory_surface import samples
    for other in SAVE_AT:
        checked_numeric(other)
    n,nr=checked_numeric(step)
    if not numeric_passes(n):return
    target=OUT/f'checkpoint{step}-calibration.json';attempt=OUT/f'checkpoint{step}-CAL-started.json'
    if any(p.exists() for p in [target,attempt,OUT/f'checkpoint{step}-terminal.json',OUT/f'checkpoint{step}-scorecache.npz']):raise ValueError('Preserve partial/failed CAL')
    path,ref,training=checkpoint(step);base=ROOT/'datasets/factory/KolektorSDD';sources=[];rows=[]
    for image,truth,defective in samples(base,protocol['split_items']['calibration']):
        frame=cv2.imread(str(image));mask=image.with_name(image.stem+'_label.bmp')
        if frame is None:raise ValueError('Protected CAL frame decode failed')
        sources.append({'image':str(image.relative_to(base)),'defective':bool(defective),'image_sha256':sha(image),'mask_sha256':sha(mask),
                        'native_image_shape':list(frame.shape),'native_mask_shape':list(truth.shape)})
        rows.append({'image':sources[-1]['image'],'truth':truth,'defective':bool(defective),'frame':frame})
    if len(sources)!=80 or sum(r['defective'] for r in sources)!=10 or identity(sources)!=CAL_SHA:raise ValueError('Exact protected80CAL identity required before anyCALNN')
    write(attempt,{'protocol_sha256':sha(PROTOCOL),'numeric_sha256':sha(nr),'CAL_source_identity_sha256':CAL_SHA})
    model=FactorySegmentation(path,output_logits=True);props=configuration(model.compiled)
    if not recipe_passes(props):raise ValueError('Actual CAL recipe differs')
    for row in rows:
        row['scores']=model.predict(row.pop('frame'),.05,4)[0];row['projection_weights']=original_pixel_weights(row['truth'],row['scores'].shape)
    for row in sources:
        image=base/row['image']
        if sha(image)!=row['image_sha256'] or sha(image.with_name(image.stem+'_label.bmp'))!=row['mask_sha256']:raise ValueError('CAL source changed')
    cache=OUT/f'checkpoint{step}-scorecache.npz'
    with cache.open('xb') as stream:np.savez_compressed(stream,scores=np.stack([r['scores'] for r in rows]),sources_json=np.asarray(json.dumps(sources)),model_sha256=np.asarray(sha(path)))
    raw=raw_candidates(rows,protocol);arms={}
    for arm,budget in ARMS.items():
        result=select_round4(rows,protocol,budget,raw);validate_search(raw,result,protocol,budget);arms[arm]=result
    report={'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'numeric_sha256':sha(nr),'scorecache_sha256':sha(cache),
        'calibration_source_identity':{'ordered_sources':sources,'sha256':CAL_SHA},'raw_grid':raw,'arms':arms,
        'configuration':props,'scope':'Actual full-frame calibration-only GPU scores; ordinaryhost workload uncontrolled, no inferenceFPS qualification'}
    for arm,result in arms.items():
        point=result['selected']
        if point is None:continue
        arm_path=path.with_name(path.stem+'-'+arm+'.onnx')
        with arm_path.open('xb') as stream:stream.write(path.read_bytes())
        if sha(arm_path)!=sha(path):raise ValueError('Policy arm must retain identical learned weights')
        meta=read(path.with_suffix('.onnx.json'))
        meta.update(profile_version=NAME+'-'+arm,pixel_threshold=point['high'],minimum_component_pixels=point['area'],hysteresis_growth_threshold=point['growth'],
            calibration_selected_point=point,calibration_source_identity_sha256=CAL_SHA,calibration_status='selected_on_protected_CAL80',round4_policy_arm=arm,
            round4_normal_FP_budget=ARMS[arm],threshold_selection='Actual Round4 GPU CAL80 prospective dual-arm fixed search; no original comparison selection')
        write(arm_path.with_suffix('.onnx.json'),meta);result['deployment']={'model_path':str(arm_path),'model_sha256':sha(arm_path),'metadata_sha256':sha(arm_path.with_suffix('.onnx.json'))}
    write(target,report);write(OUT/f'checkpoint{step}-terminal.json',{'step':step,'status':'calibration_complete','CAL_inputs_opened':80,
        'protocol_sha256':sha(PROTOCOL),'numeric_sha256':sha(nr),'calibration_sha256':sha(target)})
    print(json.dumps({'step':step,'selected':{arm:r['selected'] for arm,r in arms.items()}},indent=2),flush=True)

def freeze(protocol):
    choices={arm:[] for arm in ARMS};outcomes={}
    for step in SAVE_AT:
        path,ref,training=checkpoint(step);n,nr=checked_numeric(step);terminal=read(OUT/f'checkpoint{step}-terminal.json')
        if terminal['numeric_sha256']!=sha(nr) or terminal['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Numeric/terminal lineage differs')
        outcomes[str(step)]={'status':terminal['status'],'numeric_pass':numeric_passes(n),'terminal_sha256':sha(OUT/f'checkpoint{step}-terminal.json')}
        if not numeric_passes(n):
            if terminal['status']!='numeric_failed_no_CAL' or terminal['CAL_inputs_opened']!=0 or any((OUT/f'checkpoint{step}-{suffix}').exists() for suffix in ['CAL-started.json','calibration.json','scorecache.npz']):raise ValueError('Numericfailed cannot enterCAL')
            if sha(path.with_suffix('.onnx.json'))!=training['initial_metadata_sha256']:raise ValueError('Excluded metadata changed')
            continue
        rpath=OUT/f'checkpoint{step}-calibration.json';r=read(rpath)
        if terminal['status']!='calibration_complete' or terminal['CAL_inputs_opened']!=80 or terminal['calibration_sha256']!=sha(rpath):raise ValueError('All3 terminal exactCAL required')
        if (r['step']!=step or r['model_sha256']!=sha(path) or r['numeric_sha256']!=sha(nr) or r['protocol_sha256']!=sha(PROTOCOL)
            or not recipe_passes(r['configuration']) or r['scorecache_sha256']!=sha(OUT/f'checkpoint{step}-scorecache.npz')):raise ValueError('CAL model/cache/config differs')
        if identity(r['calibration_source_identity']['ordered_sources'])!=CAL_SHA or r['calibration_source_identity']['sha256']!=CAL_SHA:raise ValueError('CALsource identity differs')
        if list(r['arms'])!=list(ARMS):raise ValueError('Both prospective policy arms required')
        for arm,budget in ARMS.items():
            result=r['arms'][arm];validate_search(r['raw_grid'],result,protocol,budget);point=result['selected']
            if point is None:continue
            deployed=result['deployment'];arm_path=path.with_name(path.stem+'-'+arm+'.onnx');meta=arm_path.with_suffix('.onnx.json');m=read(meta)
            if (deployed['model_path']!=str(arm_path) or sha(arm_path)!=sha(path) or deployed['model_sha256']!=sha(path) or sha(meta)!=deployed['metadata_sha256']
                or m['model_sha256']!=sha(path) or m['round4_training_protocol_sha256']!=sha(PROTOCOL) or m['calibration_selected_point']!=point
                or m['split_items']!=protocol['split_items'] or m['calibration_status']!='selected_on_protected_CAL80' or m['round4_policy_arm']!=arm or m['round4_normal_FP_budget']!=budget
                or (m['pixel_threshold'],m['minimum_component_pixels'],m['hysteresis_growth_threshold'])!=(point['high'],point['area'],point['growth'])):raise ValueError('Frozen selected arm metadata differs')
            choices[arm].append({'step':step,**deployed,'point':point})
    for candidates in choices.values():candidates.sort(key=lambda r:rank(r['point']),reverse=True)
    result={'frozen_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(PROTOCOL),'all_three_terminal_outcomes':outcomes,'all_three_required_and_complete':True,
        'arms':{arm:{'normal_FP_budget':ARMS[arm],'winner':candidates[0] if candidates else None,'ranked_candidates':candidates} for arm,candidates in choices.items()},
        'both_arms_frozen_before_any_original':True,'accepted_for_root_fitted_capacity':any(choices.values()),'accepted_for_independent_audit':False,
        'original_audit_prerequisite':'Root fresh fitted complete-profile>=15 before exposedoriginalaudit; sustained16FPS/fullcounters beforeacceptance',
        'MTD_training_only_no_accuracy_claim':True,'SDD2_reserved466_and_official_TEST_unopened':True,'production_qualified':False}
    write(OUT/'winner-frozen-before-root-capacity.json',result);print(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['numeric-all3','calibrate-all3','freeze'],required=True);parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if not args.root_released:raise ValueError('Root GPU release required; no automatic postfit inference')
    cv2.setNumThreads(1);protocol,_=verify_plan()
    if args.stage=='numeric-all3':
        for step in SAVE_AT:numeric(step)
    elif args.stage=='calibrate-all3':
        for step in SAVE_AT:calibration(step,protocol)
    else:freeze(protocol)

if __name__=='__main__':main()
