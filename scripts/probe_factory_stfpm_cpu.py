"""L5 one fixed CPU/f32/4-thread numeric path and fresh full-profile preflight."""
import argparse,json,time,sys
import numpy as np
import cv2
from factory_stfpm_cpu_common import ROOT,OUT,NAME,sha,write_new,verify_protocol,protocol_path,model_path
sys.path.insert(0,str(ROOT))


def property_audit(model):
    """CPU-supported provenance only; never query GPU-specific properties."""
    def plain(value):
        if isinstance(value,(bool,int,float,str)) or value is None:return value
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        return str(value)
    result={}
    for name in ['SUPPORTED_PROPERTIES','INFERENCE_PRECISION_HINT','INFERENCE_NUM_THREADS','NUM_STREAMS','PERFORMANCE_HINT','EXECUTION_DEVICES']:
        try:result[name]={'supported':True,'value':plain(model.compiled.get_property(name))}
        except Exception as error:result[name]={'supported':False,'error':str(error)}
    try:
        from collections import Counter
        nodes=[]
        for op in model.compiled.get_runtime_model().get_ops():
            info=op.get_rt_info();row={'name':op.get_friendly_name(),'type':op.get_type_name()}
            for key in ['runtimePrecision','execType','layerType','originalLayersNames']:
                if key in info:row[key]=plain(info[key])
            nodes.append(row)
        result['runtime_graph_nodes']=nodes;result['runtime_precision_histogram']=dict(Counter(str(n.get('runtimePrecision','unspecified')) for n in nodes))
    except Exception as error:result['runtime_graph_query_error']=str(error)
    return result


def cpu_configuration_passes(audit):
    required=['INFERENCE_PRECISION_HINT','INFERENCE_NUM_THREADS','NUM_STREAMS','PERFORMANCE_HINT','EXECUTION_DEVICES']
    if any(not audit.get(key,{}).get('supported') for key in required):return False
    values={key:audit[key]['value'] for key in required}
    return (values['EXECUTION_DEVICES']==['CPU'] and type(values['INFERENCE_NUM_THREADS']) is not bool
            and str(values['INFERENCE_NUM_THREADS'])=='4' and str(values['NUM_STREAMS']) in {'1','<NumStreams: 1>'}
            and str(values['INFERENCE_PRECISION_HINT']) in {"<Type: 'float32'>",'f32','float32'}
            and str(values['PERFORMANCE_HINT']) in {'PerformanceMode.LATENCY','LATENCY'})


def parity(step,protocol):
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    from prepare_factory_stfpm_cpu import verify_bn_fit
    fit_receipt,_,fit=verify_bn_fit(protocol)
    path=model_path(step);target=OUT/f'{NAME}-step{step}-cpu-parity.json'
    if target.exists():raise ValueError('Preserve first numeric parity result')
    export=OUT/f'{NAME}-step{step}-export.json';receipt=json.loads(export.read_text(encoding='utf-8'))
    ref=OUT/f'{NAME}-step{step}-torch-reference.npz'
    if (receipt['model_sha256']!=sha(path) or receipt['reference_sha256']!=sha(ref) or receipt['protocol_sha256']!=sha(protocol_path())
            or receipt['initial_metadata_sha256']!=sha(path.with_suffix('.onnx.json'))):
        raise ValueError('Export/source/reference binding changed')
    if receipt['BN_fit_receipt_sha256']!=sha(fit_receipt) or receipt['BN_fitted_state_sha256']!=fit['state_sha256']:raise ValueError('BN fitted source changed')
    model=FactorySegmentation(path,output_logits=False,precision_hint='f32',device='CPU',cpu_threads=4)
    differences=[];allfinite=True;allbounded=True;actual_ranges=[]
    with np.load(ref,allow_pickle=False) as data:
        if data['inputs_bgr'].shape!=(3,1280,512,3) or data['scores'].shape!=(3,640,256):raise ValueError('Exactly three fixed numeric cases required')
        for frame,expected in zip(data['inputs_bgr'],data['scores']):
            actual,_,_=model.predict(frame,.05,4)
            if actual.shape!=(640,256):raise ValueError('Actual CPU geometry differs')
            allfinite=allfinite and bool(np.isfinite(actual).all())
            allbounded=allbounded and bool(actual.min()>=0 and actual.max()<=1)
            differences.append(float(np.max(np.abs(actual-expected))))
            actual_ranges.append([float(actual.min()),float(actual.max())])
    nonconstant=(actual_ranges[0][1]-actual_ranges[0][0])>=1e-7
    properties=property_audit(model);recipe_passed=cpu_configuration_passes(properties)
    value={'step':step,'model_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),'export_receipt_sha256':sha(export),'protocol_sha256':sha(protocol_path()),
           'max_abs_errors':differences,'actual_score_ranges':actual_ranges,'all_finite':allfinite,'all_bounded_0_1':allbounded,
           'preflight_nonconstant':nonconstant,'score_meaning':protocol['score'],'actual_execution_devices':model.execution_devices,
           'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'queried_configuration_and_execution':properties,'actual_CPU4_configuration_passed':recipe_passed,
           'all_checks_passed':bool(recipe_passed and allfinite and allbounded and max(differences)<=.01 and (step!=0 or nonconstant)),
           'scope':'Fixed first realTRAIN normal + zero/all255 numerical initialization cases; no accuracy claim'}
    write_new(target,value)
    if not value['all_checks_passed']:raise ValueError('Fresh actual CPU numeric guard failed; stop before fit/CAL')
    return value


def preflight(protocol):
    from edge_runtime.profiles.factory import FactoryProfile
    from edge_runtime.contracts import FrameContext
    target=OUT/'fullprofile-preflight.json'
    if target.exists():raise ValueError('Preserve first capacity result including tails')
    from prepare_factory_stfpm_cpu import verify_bn_fit
    fit_receipt,_,fit=verify_bn_fit(protocol)
    parity_path=OUT/f'{NAME}-step0-cpu-parity.json';numeric=json.loads(parity_path.read_text(encoding='utf-8'))
    if (not numeric['all_checks_passed'] or numeric['protocol_sha256']!=sha(protocol_path()) or numeric['model_sha256']!=sha(model_path(0))
            or numeric['BN_fit_receipt_sha256']!=sha(fit_receipt) or numeric['metadata_sha256']!=sha(model_path(0).with_suffix('.onnx.json'))):raise ValueError('New L5 CPU4 numeric pass required BEFORE capacity')
    path=model_path(0);rows=protocol['preflight']['source_rows'];base=ROOT/'datasets/factory/KolektorSDD'
    frames=[]
    for row in rows:
        source=base/row['image']
        if sha(source)!=row['image_sha256']:raise ValueError('Preflight source changed')
        frame=cv2.imread(str(source))
        if frame is None:raise ValueError('Cannot decode original TRAIN source')
        frames.append(frame)
    start=time.perf_counter();profile=FactoryProfile({'defect_model_path':str(path),'defect_backend':'auto','confirmation_frames':1})
    profile.prepare_worker();startup=time.perf_counter()-start
    # Sidecar must carry the one approved CPU recipe, rather than GPU defaults.
    model_meta=json.loads(path.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    if (model_meta['openvino_inference_precision']!='f32' or model_meta['openvino_device']!='CPU'
            or model_meta['openvino_cpu_threads']!=4 or 'openvino_disable_winograd_convolution' in model_meta):raise ValueError('L5 CPU capacity recipe mismatch')
    actual=profile._defect_model.neural
    if actual.device!='CPU' or actual.cpu_threads!=4 or actual.backend_name!='openvino_cpu_neural':raise ValueError('Actual full-profile adapter must use CPU4')
    for index,frame in enumerate(frames[:4]):profile.process(frame,FrameContext(index,time.perf_counter(),unit_id=f'Lwarm-{index}'))
    profile.reset();records=[]
    for index in range(64):
        tick=time.perf_counter();result=profile.process(frames[index%16],FrameContext(index,tick,unit_id=f'Lprobe-{index}'))
        records.append({'source':rows[index%16]['image'],'profile_ms':1000*(time.perf_counter()-tick),'status':result.status,
                        'neural_stage_ms':result.metrics.get('factory_neural_stage_ms')})
    values=np.asarray([r['profile_ms'] for r in records]);all_inferred=all(r['neural_stage_ms'] is not None for r in records)
    properties=property_audit(actual);recipe_passed=cpu_configuration_passes(properties)
    value={'protocol_sha256':sha(protocol_path()),'model_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),
           'initialization':profile.describe()['model_initialization'],'startup_seconds':startup,'records':records,
           'numeric_parity_sha256':sha(parity_path),'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'actual_execution_devices':actual.execution_devices,'queried_configuration_and_execution':properties,'actual_CPU4_configuration_passed':recipe_passed,
           'scope':'Root-authorized project-quiescent first nonconstant twoencoder fullFactoryProfile;16realTRAIN/64calls, ordinary host notexclusive; no GT/accuracy claim',
           'profile_ms':{'mean':float(values.mean()),'p50':float(np.median(values)),'p95':float(np.percentile(values,95)),
                         'p99':float(np.percentile(values,99)),'max':float(values.max()),'serial_calls_per_second':1000/float(values.mean())},
           'all_calls_inferred':all_inferred,'capacity_gate_passed':bool(recipe_passed and all_inferred and 1000/values.mean()>=15),
           'comparison_images_opened':0,'official_sdd2_test_still_sealed':True}
    write_new(target,value)
    if not value['capacity_gate_passed']:raise ValueError('L capacity gate failed; stop before fit')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--parity-step',type=int,choices=[0,200,400,600]);parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if not args.root_released or args.preflight==(args.parity_step is not None):raise ValueError('One explicit authorized stage required')
    cv2.setNumThreads(1);_,protocol=verify_protocol()
    result=preflight(protocol) if args.preflight else parity(args.parity_step,protocol)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
