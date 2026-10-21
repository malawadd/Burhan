"""L6 append-only ONNX stage diagnosis; preparation is non-NN, run needs root release."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from factory_stfpm_cpu_common import L4,L4_NN,L4_NN_SHA,L4_REF,L4_REF_SHA,L4_STATE_SHA,L4_BN_SHA,sha
OUT=ROOT/'output/optimization/factory-round3l6-stage-diagnosis'
MODEL=OUT/'same-score-graph-existing-stage-outputs.onnx';MANIFEST=OUT/'append-only-output-manifest.json';PROTOCOL=OUT/'protocol-before-twelve-NN-calls.json'
CASE_NAMES=['fixed_first_normal_TRAIN','zero_initialization','all255_initialization']
CPU={'device':'CPU','config':{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':4}}
GPU={'device':'GPU','config':{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True}}
PINS=['scripts/diagnose_factory_stfpm_stages.py','scripts/factory_stfpm_cpu_common.py','scripts/factory_stfpm_bn_common.py',
      'scripts/factory_stfpm_common.py','scripts/probe_factory_stfpm_cpu.py','scripts/probe_factory_stfpm_winograd.py',
      'edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py']


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def output(name):return f'{name}_output_0'


def declared_stages():
    rows=[]
    def add(label,tensor,group,floor=None):rows.append({'label':label,'tensor':tensor,'group':group,'semantic_floor':floor})
    for label,name in [('cast_BRG_u8_to_f32','/Cast'),('RGB_channel_permutation','/Gather'),('RGB_div255','/Div'),('RGB_minus_mean','/Sub'),('RGB_ImageNet_normalized','/Div_1')]:add(label,output(name),'preprocess')
    for level in range(7):
        for branch in ['teacher','student']:
            name=(f'/{branch}.0/{branch}.0.2/HardSwish' if level==0 else
                  f'/{branch}.{level}/block/block.2/block.2.0/Conv' if level in [1,2] else
                  f'/{branch}.{level}/Add' if level in [3,5,6] else
                  f'/{branch}.4/block/block.3/block.3.0/Conv')
            add(f'{branch}_block{level}',output(name),'encoder_block')
    # These exact existing tensor names follow the six norm calls in immutable source graph.
    for index,(level,branch) in enumerate([(0,'teacher'),(0,'student'),(3,'teacher'),(3,'student'),(6,'teacher'),(6,'student')]):
        suffix='' if index==0 else f'_{index}'
        div=2+2*index;clip=[0,2,5,7,10,12][index];sum_index=[0,1,3,4,6,7][index]
        prefix=f'{branch}_level{level}'
        for part,tensor,floor in [
            ('channel_absmax',output('/ReduceMax'+suffix),None),('scale_clamped',output('/Clip'+('' if clip==0 else f'_{clip}')),2**-8),
            ('scaled_features',output(f'/Div_{div}'),None),('sum_squared',output('/ReduceSum'+('' if sum_index==0 else f'_{sum_index}')),None),
            ('squared_denom_clamped',output(f'/Clip_{clip+1}'),2**-12),('sqrt_denom',output('/Sqrt'+suffix),None),
            ('unit_feature',output(f'/Div_{div+1}'),None)]:add(prefix+'_'+part,tensor,'normalization',floor)
    for level,clip in zip([0,3,6],[4,9,14]):add(f'level{level}_disagreement',output(f'/Clip_{clip}'),'disagreement')
    for level,name in [(3,'/Resize'),(6,'/Resize_1')]:add(f'level{level}_bilinear_to_fine',output(name),'interpolation')
    for level,clip,log in zip([0,3,6],[15,16,17],['/Log','/Log_1','/Log_2']):
        add(f'level{level}_map_clamped',output(f'/Clip_{clip}'),'geometric_fusion',2**-14);add(f'level{level}_log_map',output(log),'geometric_fusion')
    add('mean_three_log_maps',output('/Div_14'),'geometric_fusion');add('exp_mean',output('/Exp'),'geometric_fusion')
    return [{'label':'final_score','tensor':'anomaly_score','group':'score','semantic_floor':2**-14},*rows]


def graph_core(model):
    fields=['node','initializer','sparse_initializer','input']
    value={field:[v.SerializeToString(deterministic=True).hex() for v in getattr(model.graph,field)] for field in fields}
    value['opset']=[v.SerializeToString(deterministic=True).hex() for v in model.opset_import]
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def check_append_only(original,diagnostic,stages):
    if graph_core(original)!=graph_core(diagnostic):raise ValueError('An existing node/initializer/input/opset changed')
    if len(original.graph.output)!=1 or diagnostic.graph.output[0].SerializeToString()!=original.graph.output[0].SerializeToString():raise ValueError('Original score output changed')
    if [v.name for v in diagnostic.graph.output]!=[r['tensor'] for r in stages]:raise ValueError('Exact ordered stage inventory required')
    if len({r['tensor'] for r in stages})!=len(stages):raise ValueError('Repeated diagnostic tensor')


def prepare():
    import onnx
    if MODEL.exists() or MANIFEST.exists():raise ValueError('Preserve initial diagnostic preparation')
    if sha(L4_NN)!=L4_NN_SHA or sha(L4_REF)!=L4_REF_SHA:raise ValueError('Original graph/reference changed')
    original=onnx.load(str(L4_NN));inferred=onnx.shape_inference.infer_shapes(original);types={v.name:v for v in [*inferred.graph.value_info,*inferred.graph.input,*inferred.graph.output]}
    stages=declared_stages();diagnostic=copy.deepcopy(original)
    for row in stages[1:]:
        if row['tensor'] not in types:raise ValueError('Existing tensor shape/type unavailable')
        extra=copy.deepcopy(types[row['tensor']])
        shape=[d.dim_value for d in extra.type.tensor_type.shape.dim]
        if any(v<=0 for v in shape):
            # ONNX shape inference cannot resolve runtime Shape/Slice/Concat sizes.
            # Immutable source F.interpolate targets the known fine disagreement grid;
            # later pointwise log/mean/exp retain it. This edits ONLY new output metadata.
            expected=[1,1,640,256]
            allowed=['level3_bilinear_to_fine','level6_bilinear_to_fine','level3_map_clamped','level3_log_map',
                     'level6_map_clamped','level6_log_map','mean_three_log_maps','exp_mean']
            if row['label'] not in allowed or len(shape)!=4 or any(a>0 and a!=b for a,b in zip(shape,expected)):raise ValueError('Unexpected symbolic stage shape')
            for dim,value in zip(extra.type.tensor_type.shape.dim,expected):dim.ClearField('dim_param');dim.dim_value=value
        diagnostic.graph.output.append(extra)
    check_append_only(original,diagnostic,stages);onnx.checker.check_model(diagnostic)
    inventory=[]
    for row,value in zip(stages,diagnostic.graph.output):
        tensor=value.type.tensor_type;shape=[d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type!=onnx.TensorProto.FLOAT or not shape or any(v<=0 for v in shape):raise ValueError('Require fixed complete float32 shapes')
        inventory.append(row|{'shape':shape,'ONNX_elem_type':tensor.elem_type,'value_info_sha256':hashlib.sha256(value.SerializeToString()).hexdigest()})
    output_bytes=sum(int(np.prod(r['shape']))*4 for r in inventory)
    if len(inventory)>80 or output_bytes>250*1024**2:raise ValueError('Declared bounded output/memory envelope exceeded')
    OUT.mkdir(parents=True,exist_ok=True)
    with MODEL.open('xb') as stream:stream.write(diagnostic.SerializeToString())
    receipt={'source_model_sha256':sha(L4_NN),'diagnostic_model_sha256':sha(MODEL),'original_graph_core_sha256':graph_core(original),
             'diagnostic_graph_core_sha256':graph_core(diagnostic),'ordered_outputs':inventory,'single_case_output_bytes':output_bytes,
             'only_existing_graph_outputs_appended':True,'nodes':len(original.graph.node),'initializers':len(original.graph.initializer),
             'onnx_version':onnx.__version__,'NN_executed':False,'Torch_weight_load_or_export':False}
    write_new(MANIFEST,receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='ordered_outputs'},indent=2))


def protocol_value():
    if sha(L4_NN)!=L4_NN_SHA or sha(L4_REF)!=L4_REF_SHA or sha(L4/'real-TRAIN-BN64-fitted-state.pt')!=L4_STATE_SHA or sha(L4/'BN-fitting-complete.json')!=L4_BN_SHA:raise ValueError('Retained L4 graph/state/reference changed')
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));fixtures=OUT/'preparation-fixtures.json';f=json.loads(fixtures.read_text(encoding='utf-8'))
    if not f['all_checks_passed']:raise ValueError('Diagnostic fixtures failed')
    for p,h in f['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Fixture source drift')
    if manifest['diagnostic_model_sha256']!=sha(MODEL) or manifest['original_graph_core_sha256']!=manifest['diagnostic_graph_core_sha256']:raise ValueError('Append-only identity changed')
    return {'name':'L6_same_L4_stage_diagnosis','source_model_sha256':L4_NN_SHA,'source_BN_state_sha256':L4_STATE_SHA,
            'same_three_input_reference_sha256':L4_REF_SHA,'diagnostic_model_sha256':sha(MODEL),'append_manifest_sha256':sha(MANIFEST),
            'ordered_outputs':manifest['ordered_outputs'],'single_case_output_bytes':manifest['single_case_output_bytes'],
            'case_order':CASE_NAMES,'model_order':['served_CPU','served_GPU','diagnostic_CPU','diagnostic_GPU'],
            'actual_inference_calls':12,'each_model_calls':3,'CPU_recipe':CPU,'GPU_recipe':GPU,
            'context_guard_max_abs_error':{'CPU':1e-5,'GPU':1e-5},
            'attribution_guard':'Exact12orderedcalls; BOTHserved+diagnosticactualrecipesPASS; ALL3 scorecontextCPU+GPU within1e-5; ALLdiagnosticoutputsfinite andALLfinalscoresbounded BEFORE ANYfirstobservedstagedivergence claim',
            'serving_CPU_Torch_max_abs_error_guard':.01,'GPU_Torch_agreement':'Knownfailedsource; report,notrelaxortrain',
            'summary':'PerstageCPUvsGPU max/meanabs/RMS/relativeRMS andfloat16 tiny/floorcounts; noGT; observedfirstrelativeRMS>.01 isdescriptive,notkernelcausality',
            'relative_RMS_descriptive_flag':.01,'relative_RMS_denominator_floor':1e-12,'half_normal_boundary':2**-14,'half_min_subnormal':2**-24,
            'failure_policy':'Preservefirstcontext/numeric/nonfinitefailure andall12callaccounting; nooutputsubsetretry,flags,training,CALororiginalaccess',
            'scoremap_artifact':'four3casefinalscorestacks; intermediatesstatistics retained,notfullintermediatearrays',
            'timing_scope':'ContendeddiagnosisONLY; wallCPU1/media/ordinaryhostmayoverlap; NOcapacityclaim',
            'source_pins':{p:sha(ROOT/p) for p in PINS},'fixture_sha256':sha(fixtures),'CAL_or_comparison_inputs':0,'official_sdd2_test_still_sealed':True}


def verify_protocol():
    value=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if value!=protocol_value():raise ValueError('Protected diagnostic source/protocol changed')
    return value


def statistics(a,b,floor=None):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('All matching diagnostic tensors must be finite before summaries')
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);error=np.abs(a-b);rms=float(np.sqrt(np.mean((a-b)**2)));base=float(np.sqrt(np.mean(a*a)))
    def distribution(x):
        absolute=np.abs(x);nonzero=absolute[absolute>0]
        return {'min':float(x.min()),'max':float(x.max()),'rms':float(np.sqrt(np.mean(x*x))),
                'zero_fraction':float(np.mean(absolute==0)),'nonzero_below_half_normal_fraction':float(np.mean((absolute>0)&(absolute<2**-14))),
                'nonzero_below_half_subnormal_fraction':float(np.mean((absolute>0)&(absolute<2**-24))),
                'minimum_nonzero_abs':float(nonzero.min()) if nonzero.size else None,
                'at_or_below_semantic_floor_fraction':float(np.mean(x<=floor)) if floor is not None else None}
    return {'shape':list(a.shape),'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),'rms_error':rms,
            'relative_rms_error':rms/max(base,1e-12),'CPU':distribution(a),'GPU':distribution(b)}


def expected_calls():return [(kind,case) for kind in ['served_CPU','served_GPU','diagnostic_CPU','diagnostic_GPU'] for case in CASE_NAMES]


def context_passes(errors,all_finite,CPU_Torch_errors,all_bounded,configurations_passed,calls):
    return (all_finite and all_bounded and configurations_passed and [(r['model'],r['case']) for r in calls]==expected_calls()
            and len(errors['CPU'])==3 and len(errors['GPU'])==3 and len(CPU_Torch_errors)==3
            and all(np.isfinite(v) and v<=1e-5 for kind in ['CPU','GPU'] for v in errors[kind])
            and all(np.isfinite(v) and v<=.01 for v in CPU_Torch_errors))


def bind_outputs(outputs,rows):
    bound=[];used=set()
    for row in rows:
        candidates=[i for i,out in enumerate(outputs) if row['tensor'] in out.get_names()]
        if len(candidates)!=1 or candidates[0] in used:raise ValueError('Missing/ambiguous/aliased protected compiled tensor name')
        used.add(candidates[0]);bound.append(outputs[candidates[0]])
    if len(used)!=len(outputs):raise ValueError('Unexpected compiled diagnostic output')
    return bound


def configuration_audit(compiled,device):
    def plain(value):
        if isinstance(value,(bool,int,float,str)) or value is None:return value
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        try:return value.astype(str)
        except (AttributeError,TypeError,RuntimeError):return str(value)
    names=['EXECUTION_DEVICES','INFERENCE_PRECISION_HINT','NUM_STREAMS','PERFORMANCE_HINT']
    names+=['INFERENCE_NUM_THREADS'] if device=='CPU' else ['GPU_DISABLE_WINOGRAD_CONVOLUTION']
    props={}
    for key in names:
        try:props[key]={'supported':True,'value':plain(compiled.get_property(key))}
        except Exception as error:props[key]={'supported':False,'error':str(error)}
    passed=all(r['supported'] for r in props.values())
    if passed:
        values={k:v['value'] for k,v in props.items()};hint='float32' if device=='CPU' else 'float16'
        passed=(values['EXECUTION_DEVICES']==(['CPU'] if device=='CPU' else ['GPU.0'])
                and str(values['INFERENCE_PRECISION_HINT']) in {f"<Type: '{hint}'>",hint,'f32' if device=='CPU' else 'f16'}
                and str(values['NUM_STREAMS']) in {'1','<NumStreams: 1>'} and str(values['PERFORMANCE_HINT']) in {'LATENCY','PerformanceMode.LATENCY'}
                and (type(values['INFERENCE_NUM_THREADS']) is not bool and str(values['INFERENCE_NUM_THREADS'])=='4' if device=='CPU' else values['GPU_DISABLE_WINOGRAD_CONVOLUTION'] is True))
    nodes=[]
    try:
        for op in compiled.get_runtime_model().get_ops():
            info=op.get_rt_info();row={'name':op.get_friendly_name(),'type':op.get_type_name()}
            for key in ['runtimePrecision','execType','layerType','originalLayersNames']:
                if key in info:row[key]=plain(info[key])
            nodes.append(row)
    except Exception as error:props['runtime_graph_query_error']=str(error)
    return {'properties':props,'runtime_nodes':nodes,'actual_configuration_passed':bool(passed)}


def run():
    protocol=verify_protocol();started=OUT/'NN-started-before-compile.json';report=OUT/'diagnosis.json';journal=OUT/'actual-calls.jsonl'
    if any(p.exists() for p in [started,report,journal,OUT/'scoremaps.npz']):raise ValueError('Preserve all partial/failed first diagnosis')
    import cv2,openvino as ov
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    cv2.setNumThreads(1)
    with np.load(L4_REF,allow_pickle=False) as data:inputs=data['inputs_bgr'].copy();expected=data['scores'].copy()
    if inputs.shape!=(3,1280,512,3) or inputs.dtype!=np.uint8 or expected.shape!=(3,640,256) or not np.isfinite(expected).all():raise ValueError('Exact3fixedreferencegeometry required')
    if inputs[1].any() or not (inputs[2]==255).all():raise ValueError('Fixed numerical cases changed')
    write_new(started,{'protocol_sha256':sha(PROTOCOL),'call_order':protocol['model_order'],'calls_per_model':3,'no_warmup_or_retries':True})
    with journal.open('x',encoding='utf-8'):pass
    calls=[];attempts=[];served={};diagnostic={};properties={};all_finite=True
    def attempt(kind,index):
        if len(attempts)>=12 or (kind,CASE_NAMES[index])!=expected_calls()[len(attempts)]:raise ValueError('An unplanned/repeated NN call was requested')
        value={'attempt':len(attempts)+1,'model':kind,'case':CASE_NAMES[index]};attempts.append(value)
        with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps({'event':'call_attempted',**value})+'\n');stream.flush()
    def account(kind,index,elapsed):
        value={'call':len(calls)+1,'model':kind,'case':CASE_NAMES[index],'elapsed_diagnostic_seconds':elapsed}
        with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps({'event':'call_completed',**value})+'\n');stream.flush()
        calls.append(value)
    try:
        for device,kwargs in [('CPU',{'precision_hint':'f32','device':'CPU','cpu_threads':4}),('GPU',{'precision_hint':'f16','disable_winograd_convolution':True})]:
            model=FactorySegmentation(L4_NN,output_logits=False,**kwargs);properties['served_'+device]=configuration_audit(model.compiled,device);scores=[]
            if not properties['served_'+device]['actual_configuration_passed']:raise ValueError('Served actual backend configuration failed before data NN')
            for index,frame in enumerate(inputs):
                attempt('served_'+device,index);tick=time.perf_counter();score=model.predict(frame,.05,4)[0];account('served_'+device,index,time.perf_counter()-tick)
                scores.append(score);all_finite=all_finite and bool(np.isfinite(score).all())
            served[device]=np.stack(scores);del model
        for device,recipe in [('CPU',CPU),('GPU',GPU)]:
            core=ov.Core();compiled=core.compile_model(core.read_model(str(MODEL)),recipe['device'],recipe['config']);request=compiled.create_infer_request()
            if len(compiled.outputs)!=len(protocol['ordered_outputs']):raise ValueError('Actual diagnostic output count changed')
            bound=bind_outputs(list(compiled.outputs),protocol['ordered_outputs']);properties['diagnostic_'+device]=configuration_audit(compiled,device);values=[]
            if not properties['diagnostic_'+device]['actual_configuration_passed']:raise ValueError('Diagnostic actual backend configuration failed before data NN')
            for index,frame in enumerate(inputs):
                attempt('diagnostic_'+device,index);tick=time.perf_counter();result=request.infer({compiled.input():np.ascontiguousarray(frame.transpose(2,0,1)[None])})
                arrays=[np.array(result[out],dtype=np.float32,copy=True) for out in bound]
                account('diagnostic_'+device,index,time.perf_counter()-tick)
                if any(list(a.shape)!=row['shape'] for a,row in zip(arrays,protocol['ordered_outputs'])):raise ValueError('Actual named tensor shape changed')
                all_finite=all_finite and all(bool(np.isfinite(a).all()) for a in arrays)
                values.append(arrays)
            diagnostic[device]=values;del request,compiled,core
        def finite_error(a,b):return float(np.max(np.abs(a-b))) if np.isfinite(a).all() and np.isfinite(b).all() else None
        errors={d:[finite_error(diagnostic[d][i][0][0,0],served[d][i]) for i in range(3)] for d in ['CPU','GPU']}
        CPU_Torch=[float(np.max(np.abs(served['CPU'][i]-expected[i]))) for i in range(3)]
        GPU_Torch=[float(np.max(np.abs(served['GPU'][i]-expected[i]))) for i in range(3)]
        bounded=all(bool(np.min(served[d])>=0 and np.max(served[d])<=1) and all(bool(np.min(a[0])>=0 and np.max(a[0])<=1) for a in diagnostic[d]) for d in ['CPU','GPU'])
        configs=all(v['actual_configuration_passed'] for v in properties.values())
        eligible=context_passes(errors,all_finite,CPU_Torch,bounded,configs,calls);metrics=[]
        if all_finite:
            for index,name in enumerate(CASE_NAMES):
                rows=[row|statistics(diagnostic['CPU'][index][j],diagnostic['GPU'][index][j],row['semantic_floor']) for j,row in enumerate(protocol['ordered_outputs'])]
                flagged=[r for r in rows[1:] if r['relative_rms_error']>.01]
                metrics.append({'case':name,'stages':rows,'first_observed_relative_RMS_above_1pct':flagged[0]['label'] if eligible and flagged else None})
        with (OUT/'scoremaps.npz').open('xb') as stream:
            np.savez_compressed(stream,served_CPU=served['CPU'],served_GPU=served['GPU'],diagnostic_CPU=np.stack([r[0][0,0] for r in diagnostic['CPU']]),diagnostic_GPU=np.stack([r[0][0,0] for r in diagnostic['GPU']]))
        value={'protocol_sha256':sha(PROTOCOL),'diagnostic_model_sha256':sha(MODEL),'source_model_sha256':sha(L4_NN),
               'calls':calls,'actual_calls_attempted':len(attempts),'actual_calls_completed':len(calls),'all_outputs_finite':all_finite,'all_finalscores_bounded':bounded,'all_actual_configurations_passed':configs,'context_max_abs_errors':errors,
               'served_CPU_Torch_max_abs_errors':CPU_Torch,'served_GPU_Torch_max_abs_errors':GPU_Torch,
               'context_guard_passed_and_stage_attribution_eligible':eligible,'stage_metrics':metrics,'compiled_provenance':properties,
               'scoremaps_sha256':sha(OUT/'scoremaps.npz'),'source_pins_unchanged_after_execution':verify_protocol()==protocol,
               'scope':'Fixed3normal/initializationcases; firstobserveddivergenceNOTkernelcausality, noGT/accuracy/FPS/trainingclaim; nooutputsubsetretry'}
        write_new(report,value);print(json.dumps({k:v for k,v in value.items() if k not in ['calls','stage_metrics','compiled_provenance']},indent=2))
        if not eligible:raise ValueError('Diagnostic context/finite guard FAILED: no served-path stage attribution permitted')
    except Exception as error:
        target=OUT/'first-failure.json'
        if not target.exists():write_new(target,{'error':str(error),'calls_attempted':attempts,'calls_completed':calls,
                                    'unprocessed_calls':[list(row) for row in expected_calls()[len(calls):]],'protocol_sha256':sha(PROTOCOL),'no_automatic_retry':True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','record-protocol','run'],required=True);parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='record-protocol':write_new(PROTOCOL,protocol_value());print(json.dumps({'protocol_sha256':sha(PROTOCOL),'NN_executed':False}));return
    if not args.root_released:raise ValueError('Root must release the one fixed diagnosis')
    run()


if __name__=='__main__':main()
