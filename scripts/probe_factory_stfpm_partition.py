"""L9 separately protected six-feature GPU plus NumPy f32 complete-profile probe."""
import argparse,json,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_factory_stfpm_partition import OUT,NAME,MODEL,MANIFEST,sha,write_new,feature_rows,L4_NN,L4_NN_SHA,L4_REF,L4_REF_SHA,L4_STATE_SHA
from factory_partitioned_segmentation import FactoryPartitionedSegmentation,BaseFactorySegmentation,adapter_type
from factory_stfpm_cpu_common import L4,L4_PROTOCOL_SHA,L4_BN_SHA,CAL_SHA,NORMAL_SHA
from factory_stfpm_common import HIGH_GRID,AREA_GRID,GROWTH_GRID,RANK
from diagnose_factory_stfpm_stages import configuration_audit
PROTOCOL=OUT/'protocol-before-numeric-and-conditional-all3.json'
PINS=['scripts/factory_partitioned_segmentation.py','scripts/factory_stfpm_partition_math.py','scripts/prepare_factory_stfpm_partition.py',
      'scripts/probe_factory_stfpm_partition.py','scripts/factory_tapped_segmentation.py','scripts/diagnose_factory_stfpm_stages.py',
      'scripts/factory_stfpm_cpu_common.py','scripts/factory_stfpm_common.py','scripts/factory_stfpm_bn_common.py',
      'scripts/factory_stfpm_graph.py','scripts/factory_stfpm_math.py',
      'edge_runtime/profiles/factory.py','edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py',
      '.venv/Lib/site-packages/openvino/_ov_api.py','.venv/Lib/site-packages/openvino/_ov_api.pyi']


def learned_sources():
    folder=ROOT/'output/optimization/factory-round3l7-learned';rows=[]
    for step in (200,400,600):
        prefix=f'spatial_round3l7_learned_stfpm-step{step}'
        paths=[ROOT/f'models/factory/{prefix}.onnx',ROOT/f'models/factory/{prefix}.pt',folder/f'{prefix}-torch-reference.npz',folder/f'checkpoint{step}-training.json']
        rows.append({'step':step,'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
    return {'all_three':rows,'training_complete_sha256':sha(folder/'training-complete.json'),
            'original_single_output_freeze_sha256':sha(folder/'winner-frozen-before-comparison.json')}


def protocol_value():
    if (sha(L4_NN)!=L4_NN_SHA or sha(L4_REF)!=L4_REF_SHA or sha(L4/'protocol-before-BN-fitting.json')!=L4_PROTOCOL_SHA
            or sha(L4/'real-TRAIN-BN64-fitted-state.pt')!=L4_STATE_SHA or sha(L4/'BN-fitting-complete.json')!=L4_BN_SHA):raise ValueError('Retained L4 identity changed')
    manifest=json.loads(MANIFEST.read_text());fixture=OUT/'preparation-fixtures.json';f=json.loads(fixture.read_text())
    if not f['all_checks_passed']:raise ValueError('Pure source/math/adapter fixture PASS required')
    for path,h in f['source_pins'].items():
        if sha(ROOT/path)!=h:raise ValueError('Fixture source changed')
    if (manifest['source_model_sha256']!=L4_NN_SHA or manifest['model_sha256']!=sha(MODEL)
            or manifest['metadata_sha256']!=sha(MODEL.with_suffix('.onnx.json'))
            or manifest['ordered_outputs']!=feature_rows() or not manifest['extraction']['all_retained_node_initializer_input_opset_bytes_identical']):raise ValueError('Six-feature source proof changed')
    old=json.loads((L4/'protocol-before-BN-fitting.json').read_text())
    return {'name':NAME,'scope':'Architecture probe only; source-only conditional learned correction; root release mandatory for every NN phase',
        'source_L4_model_sha256':L4_NN_SHA,'source_BN_state_sha256':L4_STATE_SHA,'reference_sha256':L4_REF_SHA,
        'encoder_model_sha256':sha(MODEL),'encoder_sidecar_sha256':sha(MODEL.with_suffix('.onnx.json')),'extraction_manifest_sha256':sha(MANIFEST),
        'architecture':'Fixed teacher/student raw block0/3/6 interface, six outputs; deterministic ancestor graph extraction, identical retained bytes. Not an output-subset search.',
        'ordered_outputs':feature_rows(),'output_bytes':manifest['single_case_output_bytes'],
        'GPU_recipe':{'device':'GPU','precision_hint':'f16','disable_winograd_convolution':True,'streams':1,'performance':'LATENCY','share_outputs':True},
        'CPU_head':'All NumPy float32; unchanged absmaxfloor2^-8/sumsquaredfloor2^-12/unitfeatures/.25squaredL2/halfpixelbilinear_aligncornersFalse/loggeomeanfloor2^-14. No fitted score stats.',
        'inherited_contract':'Only standalone constructor/requestproxy changes; Base.predict/warmup/postprocess inherited. Process-local profile constructor binding restored immediately; shared files unchanged.',
        'numeric':{'fixed_cases':['first_normal_TRAIN','zero','255'],'same_inputs_and_Torch_scores':L4_REF_SHA,'maximum_error':.01,'finite_bounded_nonconstant':True,
                   'failure':'Only exact known invalid score/feature error is explicit failedcase; all3fixedcases attempted. Other source/compile/shape errors abort; no retry.'},
        'capacity':{'real_sources':old['preflight']['source_rows'],'initialization_calls':3,'extra_real_warmups':4,'first_measured_calls':64,'minimum_mean_calls_per_second':15,'all_calls_infer':True,'root_quiet_release':True,
                    'stage_timing':'Per same measured call GPUrequest/frameworkoutputs,featureviewaccess,CPUf32head,total; no extra NN. GPUrequest is host wall time, not GPUkernel-only event timing.'},
        'conditional_ALL3_learned':{'source_bindings':learned_sources(),'status':'BlockeduntilinitialnumericANDfirstquietcapacityPASSandseparaterootrelease',
            'checkpoints':[200,400,600],'new_prefix':'spatial_round3l9_learned_partitioned',
            'transform':'Apply same six-output ancestor extraction to ALL3 fixed L7 ONNX once; no weights/BN/optimizer/inputnormalization/node edits; no new fit. Same CPU head and GPU recipe.',
            'numeric':'Own learned Torch score refs, byteidentical3inputs, unchanged<=.01/config/finite/bounded/nonconstant; numericfail terminal/noCAL',
            'CAL_source_identity_sha256':CAL_SHA,'raw_high_grid':HIGH_GRID,'area_grid':AREA_GRID,'growth_grid':GROWTH_GRID,'ranking':RANK,
            'guards':'NormalFA<=1/70 and pixelprecision>=.65, fixed weakest-positive rank, no threshold/output/ROI search from original',
            'freeze':'All3 terminal numericfail-or-completeCAL required beforeglobalfreeze; rootONLYoriginalwinner; finalfitted>=15capacity+16inputFPSruntime required',
            'L7_single_output_and_L8_failed_results_unchanged':True},
        'normal_source_identity_sha256':NORMAL_SHA,'official_SDD2_test_sealed':True,'source_pins':{p:sha(ROOT/p) for p in PINS},'fixture_sha256':sha(fixture)}


def verify_protocol():
    value=json.loads(PROTOCOL.read_text())
    if value!=protocol_value():raise ValueError('Protected L9 protocol/source changed')
    return value


def numeric_case(model,frame,expected):
    try:score,_,_=model.predict(frame,.05,4)
    except ValueError as error:
        if str(error) not in {'Factory neural model produced invalid scores','L9 encoder feature is nonfinite','L9 CPU head produced invalid score'}:raise
        return {'max_abs_error':None,'score_range':None,'numeric_failure':str(error)}
    if score.shape!=(640,256):raise ValueError('Numeric score geometry changed')
    return {'max_abs_error':float(np.max(np.abs(score-expected))),'score_range':[float(score.min()),float(score.max())],'numeric_failure':None}


def parity(protocol):
    target=OUT/'initial-three-case-numeric.json';attempt=OUT/'numeric-attempt.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve first L9 numeric run')
    write_new(attempt,{'protocol_sha256':sha(PROTOCOL),'declared_calls':3,'no_retry':True})
    model=adapter_type()(MODEL);audit=configuration_audit(model.compiled,'GPU')
    if not audit['actual_configuration_passed']:raise ValueError('Actual GPU config failed before cases')
    cases=[]
    with np.load(L4_REF,allow_pickle=False) as data:
        if (data['inputs_bgr'].shape!=(3,1280,512,3) or data['inputs_bgr'].dtype!=np.uint8 or data['scores'].shape!=(3,640,256)
                or not np.isfinite(data['scores']).all() or data['scores'].min()<0 or data['scores'].max()>1
                or data['inputs_bgr'][1].any() or not (data['inputs_bgr'][2]==255).all()):raise ValueError('Fixed references changed')
        for frame,expected in zip(data['inputs_bgr'],data['scores']):cases.append(numeric_case(model,frame,expected))
    finite=all(not r['numeric_failure'] and np.isfinite(r['max_abs_error']) and all(np.isfinite(v) for v in r['score_range']) for r in cases)
    passed=finite and all(0<=r['score_range'][0]<=r['score_range'][1]<=1 and r['max_abs_error']<=.01 for r in cases) and cases[0]['score_range'][1]-cases[0]['score_range'][0]>=1e-7
    report={'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(MODEL),'metadata_sha256':sha(MODEL.with_suffix('.onnx.json')),'reference_sha256':sha(L4_REF),
        'fixed_cases_attempted':len(cases),'cases':cases,'compiled_provenance':audit,'all_checks_passed':bool(passed),'architecture_probe_only':True,'source_pins_unchanged_after':verify_protocol()==protocol}
    write_new(target,report);print(json.dumps({k:v for k,v in report.items() if k!='compiled_provenance'},indent=2))
    if not passed:raise ValueError('L9 numeric failed; no capacity/CAL/correction')


def capacity(protocol):
    import cv2
    from edge_runtime.profiles.factory import FactoryProfile
    from edge_runtime.contracts import FrameContext
    import edge_runtime.profiles.factory_encoder as module
    numeric=OUT/'initial-three-case-numeric.json';n=json.loads(numeric.read_text())
    if not n['all_checks_passed'] or n['protocol_sha256']!=sha(PROTOCOL) or n['model_sha256']!=sha(MODEL):raise ValueError('L9 own numeric PASS required')
    target=OUT/'initial-fullprofile-capacity.json';attempt=OUT/'capacity-attempt.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve first L9 capacity run')
    write_new(attempt,{'protocol_sha256':sha(PROTOCOL),'measured_calls':64,'no_retry':True})
    rows=protocol['capacity']['real_sources'];frames=[]
    for row in rows:
        path=ROOT/'datasets/factory/KolektorSDD'/row['image']
        if sha(path)!=row['image_sha256']:raise ValueError('Fixed normal TRAIN changed')
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError('Fixed TRAIN decode failed')
        frames.append(frame)
    started=time.perf_counter();original=module.FactorySegmentation;cls=adapter_type()
    with patch.object(module,'FactorySegmentation',cls):profile=FactoryProfile({'defect_model_path':str(MODEL),'defect_backend':'auto','confirmation_frames':1})
    if module.FactorySegmentation is not original or type(profile._defect_model.neural) is not cls:raise ValueError('Standalone constructor override restoration failed')
    profile.prepare_worker();startup=time.perf_counter()-started
    for i,frame in enumerate(frames[:4]):profile.process(frame,FrameContext(i,time.perf_counter(),unit_id=f'L9warm{i}'))
    profile.reset();records=[]
    for i in range(64):
        started=time.perf_counter();result=profile.process(frames[i%16],FrameContext(i,started,unit_id=f'L9-{i}'))
        records.append({'source':rows[i%16]['image'],'profile_ms':1000*(time.perf_counter()-started),'status':result.status,'neural_stage_ms':result.metrics.get('factory_neural_stage_ms'),
                        'partition_stage_ms':profile._defect_model.neural.request.last_timing.copy()})
    times=np.asarray([r['profile_ms'] for r in records]);audit=configuration_audit(profile._defect_model.neural.compiled,'GPU')
    inferred=all(r['neural_stage_ms'] is not None for r in records);passed=inferred and audit['actual_configuration_passed'] and 1000/times.mean()>=15
    report={'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(MODEL),'numeric_sha256':sha(numeric),'records':records,'startup_seconds':startup,
        'all64_inferred':inferred,'compiled_provenance':audit,'profile_ms':{'mean':float(times.mean()),'p50':float(np.median(times)),'p95':float(np.percentile(times,95)),
        'p99':float(np.percentile(times,99)),'max':float(times.max()),'serial_calls_per_second':1000/float(times.mean())},'capacity_gate_passed':bool(passed),
        'CPU_head_and_inherited_postprocessing_included':True,'architecture_probe_only':True,'source_pins_unchanged_after':verify_protocol()==protocol}
    write_new(target,report);print(json.dumps({k:v for k,v in report.items() if k not in ['records','compiled_provenance']},indent=2))
    if not passed:raise ValueError('L9 capacity failed; no learned correction/CAL')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['record-protocol','numeric','capacity'],required=True);parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if args.stage=='record-protocol':write_new(PROTOCOL,protocol_value());print(json.dumps({'protocol_sha256':sha(PROTOCOL),'NN_executed':False}));return
    if not args.root_released:raise ValueError('Explicit root NN/quiet release required')
    import cv2
    cv2.setNumThreads(1);protocol=verify_protocol()
    try:(parity if args.stage=='numeric' else capacity)(protocol)
    except Exception as error:
        path=OUT/f'first-failure-{args.stage}.json'
        if not path.exists():write_new(path,{'protocol_sha256':sha(PROTOCOL),'error':str(error),'no_retry':True})
        raise


if __name__=='__main__':main()
