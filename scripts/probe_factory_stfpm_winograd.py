"""ONE L3 f16/noWinograd candidate, fixed3numericcases then separately releasedspeed."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnose_factory_stfpm_numeric import verify_sources,inputs

OUT=ROOT/'output/optimization/factory-round3l-no-winograd-v3'
PROTOCOL=OUT/'protocol-before-one-candidate-NN.json'
SOURCE=ROOT/'models/factory/spatial_round3l_stfpm_mobile-step0.onnx'
CANDIDATE=ROOT/'models/factory/spatial_round3l_stfpm_mobile-L3-no-winograd-f16.onnx'
L2=ROOT/'output/optimization/factory-round3l-numeric-diagnosis-v2'
L2_SHA='ee7ce06a3f4eb3d3c3f9ceab4925ad0f59d0a22f373bbbd967fe2df4260d4af3'
D_SHA='16771a46116ea1cde4e4d10561090e3b927f3224bea716c05086f7ef262f908f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def recipe():
    original=verify_sources()
    if sha(L2/'protocol-before-diagnostic-NN.json')!=L2_SHA or sha(L2/'D-numeric.json')!=D_SHA:raise ValueError('Preserve L2failedoutcomes')
    fixtures=OUT/'mock-fixtures.json';f=json.loads(fixtures.read_text(encoding='utf-8'))
    if not f['all_checks_passed'] or f['tests']!=4:raise ValueError('Allprecision/noWinograddefaultmockfixtures required')
    for p,h in f['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Fixture source changed')
    document=OUT/'official-openvino-2026-gpu-device.html'
    if b'GPU_DISABLE_WINOGRAD_CONVOLUTION' not in document.read_bytes():raise ValueError('Primarydocsource evidence missing')
    pins={p:sha(ROOT/p) for p in ['scripts/probe_factory_stfpm_winograd.py','scripts/diagnose_factory_stfpm_numeric.py',
             'edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py','edge_runtime/profiles/factory.py']}
    return {'name':'L3_one_noWinograd_f16','maximum_candidates':1,'config':{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1',
            'INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True},
            'algorithm':'Identical failedL ONNX bytes/weights/graph/thresholds; ONE documented GPUkernel-selectioncorrection, no search/precisionfallback',
            'source_model_sha256':sha(SOURCE),'source_metadata_sha256':sha(SOURCE.with_suffix('.onnx.json')),
            'failed_L2_protocol_sha256':L2_SHA,'failed_L2_D_numeric_sha256':D_SHA,
            'numeric':{'fixed_input_cases':['samefirstnormalTRAIN','samezero','sameall255'],'max_abs_error':.01,'finite':True,'bounded_0_1':True,
                       'source':'ExistingL1TorchreferenceNPZ;noCAL/original/newmodelinputs'},
            'actual_config_audit':'Querycompiledprecision/noWinograd/executionmode/executiondevices ifsupported; preserveunsupportederrors; summarize runtimegraph runtimePrecision/execType/originallayers. Never claim requestedhint alone proveskernelprecision.',
            'primary_source':{'url':'https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html',
                              'saved_original_HTML_sha256':sha(document),'claim':'OfficialGPUlimitationsdescribepossibleinaccuracyforWinogradconvsanddocumentdisableflag;thisisahypothesis,notverifiedcause'},
            'mock_fixture_sha256':sha(fixtures),'implementation_pins':pins,'speed':original['preflight'],
            'speed_gate':'ONLYafternumericpass+separateROOTprojectquiescentreleasesame16TRAIN/4warm/64completeprofilecalls;mean>=15/allinferred,alltailsretained',
            'failure':'StopfirstL3numeric/speedfailure,preserve,noalternativeconfig/noerrortolerancerelaxation/no600fit',
            'fit_or_CAL_authorized':False,'official_sdd2_test_still_sealed':True}


def verify_recipe():
    value=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if value!=recipe():raise ValueError('L3protectedrecipe/sourcechanged')
    return value


def property_audit(model):
    def plain(value):
        if isinstance(value,(bool,int,float,str)) or value is None:return value
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        try:return value.astype(str)
        except (AttributeError,TypeError,RuntimeError):return str(value)
    result={}
    for name in ['SUPPORTED_PROPERTIES','INFERENCE_PRECISION_HINT','GPU_DISABLE_WINOGRAD_CONVOLUTION','EXECUTION_MODE_HINT','EXECUTION_DEVICES']:
        try:result[name]={'supported':True,'value':plain(model.compiled.get_property(name))}
        except Exception as error:result[name]={'supported':False,'error':str(error)}
    try:
        nodes=[]
        for op in model.compiled.get_runtime_model().get_ops():
            info=op.get_rt_info();selected={}
            for key in ['runtimePrecision','execType','layerType','originalLayersNames']:
                if key in info:selected[key]=plain(info[key])
            nodes.append({'name':op.get_friendly_name(),'type':op.get_type_name(),**selected})
        result['runtime_graph_nodes']=nodes
        from collections import Counter
        result['runtime_precision_histogram']=dict(Counter(str(n.get('runtimePrecision','unspecified')) for n in nodes))
        result['winograd_named_exec_nodes']=[n for n in nodes if 'winograd' in str(n.get('execType','')).lower()]
    except Exception as error:result['runtime_graph_query_error']=str(error)
    return result


def numeric():
    import numpy as np
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    verify_recipe();target=OUT/'three-case-numeric.json'
    if target.exists() or CANDIDATE.exists() or CANDIDATE.with_suffix('.onnx.json').exists():raise ValueError('Preserveexisting/partialONEcandidate')
    CANDIDATE.write_bytes(SOURCE.read_bytes())
    meta=json.loads(SOURCE.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    meta.update(openvino_inference_precision='f16',openvino_disable_winograd_convolution=True,
                profile_version='L3_noWinograd_f16_probe',numeric_diagnosis_protocol_sha256=sha(PROTOCOL),capacity_probe_only=True)
    write_new(CANDIDATE.with_suffix('.onnx.json'),meta)
    model=FactorySegmentation(CANDIDATE,output_logits=False,precision_hint='f16',disable_winograd_convolution=True)
    audit=property_audit(model);arrays,expected=inputs();maps=[]
    for frame in arrays:maps.append(model.predict(frame,.05,4)[0])
    values=np.stack(maps);errors=[float(np.max(np.abs(a-b))) for a,b in zip(values,expected)]
    finite=bool(np.isfinite(values).all());bounded=bool(values.min()>=0 and values.max()<=1)
    raw=OUT/'actual-scoremaps.npz'
    with raw.open('xb') as stream:np.savez_compressed(stream,scores=values)
    value={'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(CANDIDATE),'metadata_sha256':sha(CANDIDATE.with_suffix('.onnx.json')),
           'max_abs_errors':errors,'all_finite':finite,'bounded_0_1':bounded,'numeric_guard_passed':bool(finite and bounded and max(errors)<=.01),
           'scoremaps_sha256':sha(raw),'requested_config':recipe()['config'],'actual_compiled_audit':audit,
           'scope':'ONEfixedf16/noWinogradcandidate,3sameTRAIN/initializationcases;noGT/CAL/fit/guardchanges'}
    write_new(target,value);print(json.dumps({k:v for k,v in value.items() if k!='actual_compiled_audit'},indent=2))
    if not value['numeric_guard_passed']:raise ValueError('L3onecandidate numericfailed; no speed/fit')


def capacity():
    import cv2,numpy as np
    from edge_runtime.profiles.factory import FactoryProfile
    from edge_runtime.contracts import FrameContext
    value=verify_recipe();numeric_path=OUT/'three-case-numeric.json';guard=json.loads(numeric_path.read_text(encoding='utf-8'))
    if not guard['numeric_guard_passed'] or guard['protocol_sha256']!=sha(PROTOCOL):raise ValueError('L3numericguardfailed/stale')
    if sha(CANDIDATE)!=guard['model_sha256'] or sha(CANDIDATE.with_suffix('.onnx.json'))!=guard['metadata_sha256']:raise ValueError('Checkedcandidatechanged')
    target=OUT/'fullprofile-capacity.json'
    if target.exists():raise ValueError('Preservefirstcapacitytails')
    cv2.setNumThreads(1);rows=value['speed']['source_rows'];frames=[]
    for row in rows:
        p=ROOT/'datasets/factory/KolektorSDD'/row['image']
        if sha(p)!=row['image_sha256']:raise ValueError('TRAINspeedsourcechanged')
        frame=cv2.imread(str(p))
        if frame is None:raise ValueError('Cannotdecodesource')
        frames.append(frame)
    start=time.perf_counter();profile=FactoryProfile({'defect_model_path':str(CANDIDATE),'defect_backend':'auto','confirmation_frames':1})
    profile.prepare_worker();startup=time.perf_counter()-start;model=profile._defect_model.neural
    if model.precision_hint!='f16' or model.disable_winograd_convolution is not True:raise ValueError('Actualprofileconfigmismatch')
    audit=property_audit(model)
    for i,f in enumerate(frames[:4]):profile.process(f,FrameContext(i,time.perf_counter(),unit_id=f'L3warm-{i}'))
    profile.reset();records=[]
    for i in range(64):
        tick=time.perf_counter();r=profile.process(frames[i%16],FrameContext(i,tick,unit_id=f'L3probe-{i}'))
        records.append({'image':rows[i%16]['image'],'profile_ms':1000*(time.perf_counter()-tick),'status':r.status,
                        'neural_stage_ms':r.metrics.get('factory_neural_stage_ms')})
    times=np.asarray([r['profile_ms'] for r in records]);allinferred=all(r['neural_stage_ms'] is not None for r in records)
    report={'protocol_sha256':sha(PROTOCOL),'numeric_sha256':sha(numeric_path),'model_sha256':sha(CANDIDATE),
            'metadata_sha256':sha(CANDIDATE.with_suffix('.onnx.json')),'startup_seconds':startup,'records':records,
            'actual_compiled_audit':audit,'all_calls_inferred':allinferred,'capacity_gate_passed':bool(allinferred and 1000/times.mean()>=15),
            'profile_ms':{'mean':float(times.mean()),'p50':float(np.median(times)),'p95':float(np.percentile(times,95)),
                          'p99':float(np.percentile(times,99)),'max':float(times.max()),'serial_calls_per_second':1000/float(times.mean())},
            'scope':'Rootseparateprojectquiescentrelease;64fullprofilecalls/16sameTRAIN;alltailsretained;ordinaryhostnotexclusive;noaccuracy/finalfittedspeedclaim'}
    write_new(target,report);print(json.dumps({k:v for k,v in report.items() if k not in ['records','actual_compiled_audit']},indent=2))
    if not report['capacity_gate_passed']:raise ValueError('L3capacity<15; nofit')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['record-protocol','numeric','capacity'],required=True)
    parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if args.stage=='record-protocol':write_new(PROTOCOL,recipe());print(json.dumps({'protocol':str(PROTOCOL),'sha256':sha(PROTOCOL),'NN_executed':False}));return
    if not args.root_released:raise ValueError('Actualrootnumeric/quietcapacityrelease required')
    try:(numeric if args.stage=='numeric' else capacity)()
    except Exception as error:
        target=OUT/f'{args.stage}-failure.json'
        if not target.exists():write_new(target,{'error_type':type(error).__name__,'error':str(error),'protocol_sha256':sha(PROTOCOL),'automatic_retry':False})
        raise


if __name__=='__main__':main()
