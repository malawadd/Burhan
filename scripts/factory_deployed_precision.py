"""Conditional L10 empirical deployed-precision model; separate from parity claims.

Source preparation only until reviewed root decisions exist. No model fitting,
graph changes, automatic retry, configuration search or shared-runtime edits.
"""
import argparse,hashlib,json,math,sys,time,platform,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_factory_stfpm_partition import write_new,sha
from record_factory_partitioned_method_amendment import verify_amendment,AMENDMENT
from factory_partitioned_segmentation import adapter_type
from probe_factory_stfpm_partition import L4_REF,L4_REF_SHA,configuration_audit
from factory_stfpm_learn_common import gpu_configuration_passes
from factory_stfpm_common import CAL_SHA,identity,rank,validate_search_report
OUT=ROOT/'output/optimization/factory-round3l10-deployed-precision';NAME='spatial_round3l10_deployed_precision';STEPS=[200,400,600]
PLAN=OUT/'source-only-method-plan-before-old-outcomes.json';PROTOCOL=OUT/'protocol-before-empirical-NN.json'
OLD=ROOT/'output/optimization/factory-round3l9-learned-partitioned'
OLD_FREEZE=OLD/'winner-frozen-before-root-capacity.json'
FILES=['scripts/factory_deployed_precision.py','tests/test_factory_deployed_precision.py',
       'scripts/prepare_factory_partitioned_learned.py','scripts/calibrate_factory_partitioned_learned.py','scripts/freeze_factory_partitioned_learned.py',
       'scripts/factory_partitioned_segmentation.py','scripts/factory_stfpm_partition_math.py','scripts/prepare_factory_stfpm_partition.py',
       'scripts/probe_factory_stfpm_partition.py','scripts/record_factory_partitioned_method_amendment.py',
       'scripts/calibrate_factory_stfpm_learn.py','scripts/factory_stfpm_learn_common.py','scripts/factory_stfpm_common.py',
       'scripts/factory_mask_metrics.py','scripts/train_factory_surface.py','scripts/diagnose_factory_stfpm_stages.py',
       'edge_runtime/profiles/factory.py','edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py']


def model_path(step):return ROOT/f'models/factory/{NAME}-step{step}.onnx'


def software():
    from importlib.metadata import version
    return {'Python':platform.python_version(),'NumPy':np.__version__,'OpenVINO':version('openvino'),
            'platform':platform.platform(),'definition':'Actual fixed GPU feature graph plus fixed NumPy-f32 CPU score function; no Torch-equivalence claim'}


def source_plan():
    amendment=verify_amendment()
    return {'name':NAME,'checkpoints':STEPS,'amendment_sha256':sha(AMENDMENT),'source_bindings':amendment['source_bindings'],
        'prerequisite':'All3 strict learnedL9 numeric_failed_no_CAL terminal outcomes and null globalwinner; oldparity failures preserved; separate ROOTrelease',
        'definition':'Same3 actual learnedL9 six-feature ONNX bytes/weights/inputnorm, sameGPUf16noWinograd/oneLATENCYstream, sameNumPyf32head. No graph/weights/BN/math/config/ROI/fit changes.',
        'equivalence_claim':False,'old_Torch_errors':'Retain/log; never relabel oldfailedparity as pass or apply newtolerance',
        'numeric_calls':['normal_TRAIN','zero','255','normal_TRAIN_repeat1','normal_TRAIN_repeat2','normal_TRAIN_repeat3'],
        'empirical_numeric':{'all6finite_bounded':True,'normal_nonconstant_min_span':1e-7,'max_pairwise_repeat_difference':1e-6,'actual_GPU_recipe_required':True},
        'capacity':{'sources':'Sameprotected16normalTRAIN','init':3,'real_warmup':4,'first_measured_calls':64,'minimum_mean_calls_per_second':15,
                    'policy':'Unchanged default .05/area4/no-growth beforeCAL; actualselectedpolicy rootbench required again','all_tails':True,'no_repeat':True},
        'order':'All3 empiricalnumeric first; each passingcandidate first64capacity; all3 gatingoutcomes before ANYCAL. Failednumeric/speed opens0CAL; preserve all3 terminal outcomes.',
        'CAL_policy':amendment['CAL_policy'],'freeze':'All3terminal outcomes; onlyeligiblecompletedCAL, sameweakestpositive rank/normalFP<=1/70/pixelP>=.65; oneglobalwinner',
        'acceptance':'Rootselectedfittedpolicy>=15 beforeoriginal, sustained16inputFPS+allruntimecounters beforeacceptance; originalexposeddevelopmentnotblind',
        'software':software(),'source_pins':{p:sha(ROOT/p) for p in FILES},'new_fit_or_BN_updates':0,'new_NN_calls_during_preparation':0}


def empirical_acceptance(maps,audit):
    """Recompute empirical validity; Torch closeness intentionally absent."""
    if (len(maps)!=6 or audit.get('actual_configuration_passed') is not True
            or not gpu_configuration_passes(audit.get('properties',{}))):return False,None
    for value in maps:
        if (not isinstance(value,np.ndarray) or value.shape!=(640,256) or value.dtype!=np.float32
                or not np.isfinite(value).all() or value.min()<0 or value.max()>1):return False,None
    if float(np.ptp(maps[0]))<1e-7:return False,None
    repeated=np.stack([maps[i] for i in [0,3,4,5]])
    difference=float(np.max(repeated.max(axis=0)-repeated.min(axis=0)))
    return difference<=1e-6,difference


def prior_bindings():
    from prepare_factory_partitioned_learned import verify_protocol as verify_old,model_path as old_model
    from freeze_factory_partitioned_learned import validate_terminal
    old_protocol=verify_old();frozen=json.loads(OLD_FREEZE.read_text())
    if (frozen['all_three_required_and_complete'] is not True or frozen['winner'] is not None
            or frozen['protocol_sha256']!=sha(OLD/'protocol-before-all3-transforms.json')):raise ValueError('All3 originalstrictfailures required')
    rows=[]
    for step in STEPS:
        numeric_path=OLD/f'checkpoint{step}-numeric.json';terminal_path=OLD/f'checkpoint{step}-terminal.json'
        numeric=json.loads(numeric_path.read_text());terminal=json.loads(terminal_path.read_text());validate_terminal(terminal,numeric)
        if (terminal['status']!='numeric_failed_no_CAL' or terminal['CAL_inputs_opened']!=0
                or terminal['numeric_sha256']!=sha(numeric_path) or numeric['model_sha256']!=sha(old_model(step))
                or frozen['all_three_terminal_outcomes'][str(step)]['terminal_sha256']!=sha(terminal_path)):
            raise ValueError('Exact3 prior numeric failures/source binding required')
        if any((OLD/name).exists() for name in [f'checkpoint{step}-CAL-attempt.json',f'checkpoint{step}-calibration.json',f'checkpoint{step}-actual-mixed-scoremaps.npz']):raise ValueError('Priornumericfail musthavezeroCAL')
        rows.append({'step':step,'old_model_path':str(old_model(step).relative_to(ROOT)),'model_sha256':sha(old_model(step)),
            'old_metadata_sha256':sha(old_model(step).with_suffix('.onnx.json')),'old_numeric_sha256':sha(numeric_path),'old_terminal_sha256':sha(terminal_path),
            'old_Torch_errors':[r['max_abs_error'] for r in numeric['cases']]})
    return {'old_protocol_sha256':sha(OLD/'protocol-before-all3-transforms.json'),'old_global_freeze_sha256':sha(OLD_FREEZE),'candidates':rows,
            'split_items':old_protocol['split_items'],'raw_high_grid':old_protocol['raw_high_grid'],'area_grid':old_protocol['area_grid'],
            'growth_grid':old_protocol['growth_grid'],'selection_rank':old_protocol['selection_rank'],'calibration_source_identity_sha256':CAL_SHA}


def protocol_value():
    plan=json.loads(PLAN.read_text());fixture=OUT/'preparation-fixtures.json';f=json.loads(fixture.read_text())
    if plan!=source_plan() or f['all_checks_passed'] is not True:raise ValueError('New method/source/fixture changed')
    for rel,h in f['source_pins'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Fixture source drift')
    return {'plan_sha256':sha(PLAN),'method':plan,'prior':prior_bindings(),'fixture_sha256':sha(fixture),'reference_sha256':L4_REF_SHA}


def verify_protocol():
    protocol=json.loads(PROTOCOL.read_text());decision=json.loads((OUT/'parent-approved-deployed-precision-decision.json').read_text())
    if (protocol!=protocol_value() or decision.get('approved_deployed_precision_all3') is not True
            or decision['plan_sha256']!=sha(PLAN)):raise ValueError('Exact root-approved newmethod/source required')
    return protocol


def copy_assets(protocol):
    marker=OUT/'copy-all3-started.json'
    if marker.exists():raise ValueError('Preserve first asset copy attempt')
    write_new(marker,{'protocol_sha256':sha(PROTOCOL),'NN_calls':0});copies=[]
    archive=OUT/'executed-sources-before-copy'
    for rel,h in protocol['method']['source_pins'].items():
        target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/rel).read_bytes())
        if sha(target)!=h:raise ValueError('Source archive mismatch')
    write_new(archive/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'source_pins':protocol['method']['source_pins']})
    for row in protocol['prior']['candidates']:
        step=row['step'];source=ROOT/row['old_model_path'];target=model_path(step)
        if target.exists() or target.with_suffix('.onnx.json').exists():raise ValueError('Preserve newnamedassets')
        if sha(source)!=row['model_sha256'] or sha(source.with_suffix('.onnx.json'))!=row['old_metadata_sha256']:raise ValueError('Oldsource changed')
        with target.open('xb') as stream:stream.write(source.read_bytes())
        meta=json.loads(source.with_suffix('.onnx.json').read_text())
        meta.update(profile_version=NAME,model_name='Experimental deployed-precision GPU-feature/CPU-head anomaly model',deployed_precision_protocol_sha256=sha(PROTOCOL),
                    Torch_equivalent_export=False,pixel_threshold=.05,minimum_component_pixels=4,hysteresis_growth_threshold=None)
        write_new(target.with_suffix('.onnx.json'),meta)
        copies.append({'step':step,'model_sha256':sha(target),'initial_metadata_sha256':sha(target.with_suffix('.onnx.json'))})
    write_new(OUT/'all3-assets-complete.json',{'protocol_sha256':sha(PROTOCOL),'candidates':copies,'new_NN_calls':0,'graph_bytes_unchanged':True})


def assets(step,protocol,selected=False):
    complete=json.loads((OUT/'all3-assets-complete.json').read_text());rows=complete['candidates']
    if complete['protocol_sha256']!=sha(PROTOCOL) or [r['step'] for r in rows]!=STEPS:raise ValueError('All3namedassets required')
    row=rows[STEPS.index(step)];path=model_path(step);prior=protocol['prior']['candidates'][STEPS.index(step)]
    if sha(path)!=row['model_sha256'] or sha(path)!=prior['model_sha256']:raise ValueError('Actualsamegraph identity changed')
    if not selected and sha(path.with_suffix('.onnx.json'))!=row['initial_metadata_sha256']:raise ValueError('Initialsidecar changed')
    return path,row


def backend_metadata(model):
    properties={}
    for key in ['FULL_DEVICE_NAME','DRIVER_VERSION','DEVICE_ARCHITECTURE']:
        try:properties[key]={'supported':True,'value':str(model.core.get_property('GPU',key))}
        except Exception as error:properties[key]={'supported':False,'error':str(error)}
    return {'software':software(),'actual_GPU_device_properties':properties,'configuration':configuration_audit(model.compiled,'GPU')}


def numeric(step,protocol,scope):
    target=OUT/f'checkpoint{step}-empirical-numeric.json';attempt=OUT/f'checkpoint{step}-numeric-attempt.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve first6call numeric run')
    path,asset=assets(step,protocol);write_new(attempt,{'protocol_sha256':sha(PROTOCOL),'declared_calls':6,'scope':scope})
    model=adapter_type(sha(path))(path);backend=backend_metadata(model);audit=backend['configuration']
    if not audit['actual_configuration_passed']:raise ValueError('ActualunchangedGPUrecipe failed')
    definition=OUT/'actual-backend-definition-before-first-forward.json'
    if definition.exists():
        if json.loads(definition.read_text())!=backend:raise ValueError('Deployedbackend identity drift')
    else:write_new(definition,backend)
    with np.load(L4_REF,allow_pickle=False) as reference:
        if sha(L4_REF)!=L4_REF_SHA:raise ValueError('Fixedinput reference drift')
        inputs=reference['inputs_bgr'];expected=reference['scores']
        if inputs.shape!=(3,1280,512,3) or inputs.dtype!=np.uint8:raise ValueError('Fixedinput geometry differs')
        maps=[];cases=[]
        for index in [0,1,2,0,0,0]:
            try:score=model.predict(inputs[index],.05,4)[0]
            except ValueError as error:
                if str(error) not in {'Factory neural model produced invalid scores','L9 encoder feature is nonfinite','L9 CPU head produced invalid score'}:raise
                score=None;cases.append({'input_index':index,'known_numeric_failure':str(error)})
            else:cases.append({'input_index':index,'range':[float(score.min()),float(score.max())],'known_numeric_failure':None})
            maps.append(score)
    passed,difference=empirical_acceptance(maps,audit)
    cache=OUT/f'checkpoint{step}-empirical-scoremaps.npz'
    with cache.open('xb') as stream:np.savez_compressed(stream,**{f'call{i+1}':v for i,v in enumerate(maps) if v is not None})
    report={'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'initial_metadata_sha256':asset['initial_metadata_sha256'],
        'reference_sha256':L4_REF_SHA,'fixed_calls_attempted':6,'cases':cases,'all_checks_passed':bool(passed),'repeatability_max_pairwise_error':difference,
        'backend_definition_sha256':sha(definition),'compiled_provenance':audit,'scorecache_sha256':sha(cache),
        'old_Torch_parity_errors':protocol['prior']['candidates'][STEPS.index(step)]['old_Torch_errors'],
        'current_Torch_errors_diagnostic_only':[float(np.max(np.abs(maps[i]-expected[i]))) if maps[i] is not None else None for i in range(3)],
        'Torch_equivalence_claim':False,'scope':scope+'; notspeedoraccuracy'}
    write_new(target,report)
    if not passed:write_new(OUT/f'checkpoint{step}-terminal.json',{'step':step,'status':'empirical_numeric_failed_no_CAL','numeric_sha256':sha(target),'CAL_inputs_opened':0,'protocol_sha256':sha(PROTOCOL)})
    print(json.dumps({'step':step,'empirical_numeric_pass':passed,'repeatability':difference},indent=2))


def numeric_pass(step,protocol):
    path=OUT/f'checkpoint{step}-empirical-numeric.json';r=json.loads(path.read_text());model,_=assets(step,protocol,selected=True)
    if (r['step']!=step or r['protocol_sha256']!=sha(PROTOCOL) or r['model_sha256']!=sha(model) or r['fixed_calls_attempted']!=6
            or r['reference_sha256']!=L4_REF_SHA or r['Torch_equivalence_claim'] is not False):raise ValueError('Empiricalnumeric lineage differs')
    cache=OUT/f'checkpoint{step}-empirical-scoremaps.npz'
    if sha(cache)!=r['scorecache_sha256']:raise ValueError('Empiricalscoremap evidence changed')
    with np.load(cache,allow_pickle=False) as values:maps=[values[f'call{i+1}'] if f'call{i+1}' in values else None for i in range(6)]
    passed,difference=empirical_acceptance(maps,r['compiled_provenance'])
    if r['all_checks_passed']!=passed or r['repeatability_max_pairwise_error']!=difference:raise ValueError('Empiricalgate recomputation differs')
    return passed,path


def capacity(step,protocol,scope):
    import cv2
    from edge_runtime.contracts import FrameContext
    from calibrate_factory_partitioned_learned import build_profile
    for candidate in STEPS:numeric_pass(candidate,protocol)
    passed,numeric_path=numeric_pass(step,protocol)
    if not passed:raise ValueError('Numericfailed candidate cannot timecapacity or openCAL')
    target=OUT/f'checkpoint{step}-first64-capacity.json';attempt=OUT/f'checkpoint{step}-capacity-attempt.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve first64 capacity attempt')
    path,asset=assets(step,protocol);write_new(attempt,{'protocol_sha256':sha(PROTOCOL),'measured_calls':64,'scope':scope})
    from probe_factory_stfpm_partition import verify_protocol as initial_protocol
    rows=initial_protocol()['capacity']['real_sources'];frames=[]
    for row in rows:
        source=ROOT/'datasets/factory/KolektorSDD'/row['image']
        if sha(source)!=row['image_sha256']:raise ValueError('FixednormalTRAIN source changed')
        frame=cv2.imread(str(source))
        if frame is None:raise ValueError('TRAIN source decodefailed')
        frames.append(frame)
    started=time.perf_counter();profile=build_profile(path);profile.prepare_worker();startup=time.perf_counter()-started
    for i,frame in enumerate(frames[:4]):profile.process(frame,FrameContext(i,time.perf_counter(),unit_id=f'L10warm{i}'))
    profile.reset();records=[]
    for i in range(64):
        started=time.perf_counter();result=profile.process(frames[i%16],FrameContext(i,started,unit_id=f'L10-{step}-{i}'))
        records.append({'profile_ms':1000*(time.perf_counter()-started),'source':rows[i%16]['image'],'neural_stage_ms':result.metrics.get('factory_neural_stage_ms'),
                        'partition_stage_ms':profile._defect_model.neural.request.last_timing.copy(),'status':result.status})
    times=np.asarray([r['profile_ms'] for r in records]);audit=configuration_audit(profile._defect_model.neural.compiled,'GPU')
    passed=all(r['neural_stage_ms'] is not None for r in records) and audit['actual_configuration_passed'] and 1000/times.mean()>=15
    report={'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),'numeric_sha256':sha(numeric_path),
        'records':records,'startup_seconds':startup,'compiled_provenance':audit,'scope':scope,'capacity_gate_passed':bool(passed),
        'profile_ms':{'mean':float(times.mean()),'p50':float(np.median(times)),'p95':float(np.percentile(times,95)),'p99':float(np.percentile(times,99)),
                      'max':float(times.max()),'serial_calls_per_second':1000/float(times.mean())},'policy':{'high':.05,'area':4,'growth':None}}
    write_new(target,report)
    if not passed:write_new(OUT/f'checkpoint{step}-terminal.json',{'step':step,'status':'capacity_failed_no_CAL','numeric_sha256':sha(numeric_path),'capacity_sha256':sha(target),'CAL_inputs_opened':0,'protocol_sha256':sha(PROTOCOL)})
    print(json.dumps({'step':step,'capacity_pass':passed,'profile_ms':report['profile_ms']},indent=2))


def capacity_pass(step,protocol):
    passed,n=numeric_pass(step,protocol)
    if not passed:return False,None
    path=OUT/f'checkpoint{step}-first64-capacity.json';r=json.loads(path.read_text());times=np.asarray([v['profile_ms'] for v in r['records']])
    model,_=assets(step,protocol,selected=True)
    if (len(times)!=64 or not np.isfinite(times).all() or (times<=0).any() or r['step']!=step or r['protocol_sha256']!=sha(PROTOCOL)
            or r['model_sha256']!=sha(model) or r['numeric_sha256']!=sha(n)):raise ValueError('First64 source/count/timings changed')
    accepted=bool(all(v['neural_stage_ms'] is not None for v in r['records']) and r['compiled_provenance']['actual_configuration_passed'] is True
        and gpu_configuration_passes(r['compiled_provenance']['properties']) and 1000/times.mean()>=15)
    if r['capacity_gate_passed']!=accepted or r['profile_ms']['mean']!=float(times.mean()):raise ValueError('Firstcapacity gate changed')
    return accepted,path


def calibration(step,protocol,scope):
    import cv2
    from calibrate_factory_stfpm_learn import select
    from factory_mask_metrics import original_pixel_weights
    from train_factory_surface import samples
    for candidate in STEPS:capacity_pass(candidate,protocol)
    accepted,capacity_path=capacity_pass(step,protocol)
    if not accepted:raise ValueError('Failednumeric/speed candidate forbidden from CAL')
    target=OUT/f'checkpoint{step}-calibration.json';attempt=OUT/f'checkpoint{step}-CAL-attempt.json'
    if target.exists() or attempt.exists() or (OUT/f'checkpoint{step}-terminal.json').exists():raise ValueError('Preserve firstCAL attempt')
    path,asset=assets(step,protocol);write_new(attempt,{'protocol_sha256':sha(PROTOCOL),'scope':scope})
    sources=[];rows=[];base=ROOT/'datasets/factory/KolektorSDD';policy=protocol['prior']
    for image,truth,defective in samples(base,policy['split_items']['calibration']):
        frame=cv2.imread(str(image));mask=image.with_name(image.stem+'_label.bmp')
        if frame is None:raise ValueError('ExactCAL decodefailed')
        sources.append({'image':str(image.relative_to(base)),'defective':bool(defective),'image_sha256':sha(image),'mask_sha256':sha(mask),
                        'native_image_shape':list(frame.shape),'native_mask_shape':list(truth.shape)})
        rows.append({'image':sources[-1]['image'],'defective':bool(defective),'truth':truth,'frame':frame})
    if len(sources)!=80 or sum(r['defective'] for r in sources)!=10 or identity(sources)!=CAL_SHA:raise ValueError('Onlyprotected80CAL permitted beforeNN')
    model=adapter_type(sha(path))(path);audit=configuration_audit(model.compiled,'GPU')
    if not audit['actual_configuration_passed']:raise ValueError('Actualdeployedrecipe drift')
    for row in rows:
        row['scores']=model.predict(row.pop('frame'),.05,4)[0];row['projection_weights']=original_pixel_weights(row['truth'],row['scores'].shape)
    for source in sources:
        image=base/source['image']
        if sha(image)!=source['image_sha256'] or sha(image.with_name(image.stem+'_label.bmp'))!=source['mask_sha256']:raise ValueError('CALsourcechanged')
    cache=OUT/f'checkpoint{step}-actual-deployed-scoremaps.npz'
    with cache.open('xb') as stream:np.savez_compressed(stream,scores=np.stack([r['scores'] for r in rows]),model_sha256=np.asarray(sha(path)),protocol_sha256=np.asarray(sha(PROTOCOL)),sources_json=np.asarray(json.dumps(sources)))
    raw,grown,point,selected=select(rows,policy)
    report={'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'capacity_sha256':sha(capacity_path),'scorecache_sha256':sha(cache),
        'calibration_source_identity':{'ordered_sources':sources,'sha256':identity(sources)},'raw_grid':raw,'grown_grid':grown,'selected':point,'selected_records':selected,
        'actual_deployed_precision_maps':True,'Torch_equivalence_claim':False,'compiled_provenance':audit,'scope':scope+'; no speedclaim'}
    validate_search_report(report,policy)
    if point:
        meta_path=path.with_suffix('.onnx.json');meta=json.loads(meta_path.read_text());write_new(OUT/f'checkpoint{step}-initial-metadata.json',meta)
        meta.update(pixel_threshold=point['high'],minimum_component_pixels=point['area'],hysteresis_growth_threshold=point['growth'],
                    calibration_selected_point=point,calibration_source_identity_sha256=CAL_SHA)
        temporary=meta_path.with_name(meta_path.name+'.selected-writing');write_new(temporary,meta);temporary.replace(meta_path);report['metadata_sha256']=sha(meta_path)
    write_new(target,report);write_new(OUT/f'checkpoint{step}-terminal.json',{'step':step,'status':'calibration_complete','protocol_sha256':sha(PROTOCOL),
        'calibration_sha256':sha(target),'capacity_sha256':sha(capacity_path),'CAL_inputs_opened':80})


def freeze(protocol):
    target=OUT/'winner-before-root-selected-policy-capacity.json'
    if target.exists():raise ValueError('Preserve globalfreeze')
    choices=[];outcomes={}
    for step in STEPS:
        path,asset=assets(step,protocol,selected=True);tpath=OUT/f'checkpoint{step}-terminal.json';terminal=json.loads(tpath.read_text())
        numeric,npath=numeric_pass(step,protocol);speed,spath=capacity_pass(step,protocol)
        if terminal['step']!=step or terminal['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Terminalidentity drift')
        status=terminal['status'];outcomes[str(step)]={'status':status,'terminal_sha256':sha(tpath)}
        if status in {'empirical_numeric_failed_no_CAL','capacity_failed_no_CAL'}:
            if (status=='empirical_numeric_failed_no_CAL' and numeric) or (status=='capacity_failed_no_CAL' and (not numeric or speed)) or terminal['CAL_inputs_opened']!=0:raise ValueError('Failedgate arithmeticchanged')
            if any((OUT/name).exists() for name in [f'checkpoint{step}-CAL-attempt.json',f'checkpoint{step}-calibration.json',f'checkpoint{step}-actual-deployed-scoremaps.npz']):raise ValueError('Failedgate openedCAL')
            if sha(path.with_suffix('.onnx.json'))!=asset['initial_metadata_sha256']:raise ValueError('Excludedmeta drift')
            continue
        if status!='calibration_complete' or not numeric or not speed or terminal['CAL_inputs_opened']!=80:raise ValueError('CompleteCAL requiresvalidnumericandspeed')
        rpath=OUT/f'checkpoint{step}-calibration.json';r=json.loads(rpath.read_text());sources=r['calibration_source_identity']['ordered_sources']
        if (terminal['calibration_sha256']!=sha(rpath) or r['protocol_sha256']!=sha(PROTOCOL) or r['model_sha256']!=sha(path)
                or r['capacity_sha256']!=sha(spath) or r['actual_deployed_precision_maps'] is not True or r['Torch_equivalence_claim'] is not False
                or r['scorecache_sha256']!=sha(OUT/f'checkpoint{step}-actual-deployed-scoremaps.npz')
                or len(sources)!=80 or sum(s['defective'] for s in sources)!=10 or identity(sources)!=CAL_SHA
                or not gpu_configuration_passes(r['compiled_provenance']['properties'])):raise ValueError('CALsource/recipe/maplineage changed')
        validate_search_report(r,protocol['prior']);point=r['selected']
        if point is None:
            if sha(path.with_suffix('.onnx.json'))!=asset['initial_metadata_sha256']:raise ValueError('Noeligiblepointmeta changed')
            continue
        meta_path=path.with_suffix('.onnx.json');meta=json.loads(meta_path.read_text())
        if (r['metadata_sha256']!=sha(meta_path) or sha(OUT/f'checkpoint{step}-initial-metadata.json')!=asset['initial_metadata_sha256']
                or meta['model_sha256']!=sha(path) or meta['deployed_precision_protocol_sha256']!=sha(PROTOCOL) or meta['Torch_equivalent_export'] is not False
                or meta['calibration_selected_point']!=point or meta['calibration_source_identity_sha256']!=CAL_SHA
                or (meta['pixel_threshold'],meta['minimum_component_pixels'],meta['hysteresis_growth_threshold'])!=(point['high'],point['area'],point['growth'])):raise ValueError('Selectedsidecar/policychanged')
        choices.append({'step':step,'model_path':str(path),'model_sha256':sha(path),'metadata_sha256':sha(meta_path),'point':point})
    choices.sort(key=lambda r:rank(r['point']),reverse=True)
    write_new(target,{'protocol_sha256':sha(PROTOCOL),'all_three_terminal_outcomes':outcomes,'all_three_complete':True,'winner':choices[0] if choices else None,
        'ranked_candidates':choices,'accepted_for_original_audit':False,'accepted_for_root_selected_policy_capacity':bool(choices),
        'original_prerequisite':'Freshactualselectedfittedpolicy>=15 rootcapacity first; olddefaultpolicyfirst64doesnottransfer',
        'acceptance_prerequisite':'Root16inputFPSsustainedruntime counts/tails +actualaccuracy; no productionclaim','Torch_equivalence_claim':False,
        'selection_scope':'ExposeddevelopmentCAL/original, notblind; no new fit/config/interface/ROI search','official_SDD2_test_still_sealed':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['plan','record-protocol','copy-all3','numeric','capacity','calibration','freeze'],required=True)
    parser.add_argument('--step',type=int,choices=STEPS);parser.add_argument('--root-released',action='store_true');parser.add_argument('--contention-note',default='Root coordinated execution; ordinary host services uncontrolled');args=parser.parse_args()
    if args.stage=='plan':write_new(PLAN,source_plan());return
    if args.stage=='record-protocol':
        value=protocol_value();decision=json.loads((OUT/'parent-approved-deployed-precision-decision.json').read_text())
        if decision.get('approved_deployed_precision_all3') is not True or decision['plan_sha256']!=sha(PLAN):raise ValueError('ExactROOTnewmethodapproval required')
        write_new(PROTOCOL,value);return
    if not args.root_released:raise ValueError('SeparateROOTphaseNN/quietrelease required')
    import cv2
    cv2.setNumThreads(1);protocol=verify_protocol()
    if args.stage=='copy-all3':copy_assets(protocol);return
    if args.stage=='freeze':freeze(protocol);return
    if args.step is None:raise ValueError('Fixedcandidate step required')
    try:{'numeric':numeric,'capacity':capacity,'calibration':calibration}[args.stage](args.step,protocol,args.contention_note)
    except Exception as error:
        failure=OUT/f'checkpoint{args.step}-{args.stage}-first-failure.json'
        if not failure.exists():write_new(failure,{'protocol_sha256':sha(PROTOCOL),'error':str(error),'no_retry':True})
        raise


if __name__=='__main__':main()
