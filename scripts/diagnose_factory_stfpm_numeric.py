"""Separately pinned L2 four-path numerical diagnosis; no CAL, fit or guard changes."""
import argparse,hashlib,json,time,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'output/optimization/factory-round3l-numeric-diagnosis-v2'
OLD=ROOT/'output/optimization/factory-round3l'
NAME='spatial_round3l_stfpm_mobile'
PROTOCOL=OUT/'protocol-before-diagnostic-NN.json'
PINS={
 'output/optimization/factory-round3l/protocol-v2-score-semantics-before-any-NN.json':'84e3ab65009e40a5e13eb8f0e241ef2ea766aff1431841e1c75ae5e81a0f4720',
 'output/optimization/factory-round3l/spatial_round3l_stfpm_mobile-step0-export.json':'df98576f1ef7f21c4bf5a8206afefb7224305e5bb907ed88901ca05883d51a0b',
 'output/optimization/factory-round3l/fullprofile-preflight.json':'e90c0d84e4a189352708ce616d21b53cebd17b78a3ac0b0cf68c42199b52a7e6',
 'output/optimization/factory-round3l/spatial_round3l_stfpm_mobile-step0-gpu-parity.json':'d4bd1178a2daa867fdb21ecdd6dfa156a328903015140b04d0d7d30afcd06fac',
 'output/optimization/factory-round3l/spatial_round3l_stfpm_mobile-step0-torch-reference.npz':'0ac945a1673196860f3b78510b1ac12bd6b9c495b6bfefa2d129c9c769f88592',
 'models/factory/spatial_round3l_stfpm_mobile-step0.onnx':'0383b508411a4a86c6f7d0cef17b348a6f17b27f3076e0cdb2f48bf66168566d',
 'models/factory/spatial_round3l_stfpm_mobile-step0.onnx.json':'a79d03dbf490188caeecf9c6a2d5a7ff060b06045dff24642d48859862802e2b',
}
RUNTIME=['edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py','edge_runtime/profiles/factory.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError(f'Preserve diagnostic evidence: {path}')
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def verify_sources():
    for relative,expected in PINS.items():
        if sha(ROOT/relative)!=expected:raise ValueError('Failed L source evidence changed: '+relative)
    original=json.loads((OLD/'protocol-v2-score-semantics-before-any-NN.json').read_text(encoding='utf-8'))
    for relative,expected in original['implementation_pins'].items():
        if relative not in RUNTIME and sha(ROOT/relative)!=expected:raise ValueError('Original L implementation changed')
    return original


def recipe():
    original=verify_sources()
    fixtures=OUT/'precision-metadata-fixtures.json';f=json.loads(fixtures.read_text(encoding='utf-8'))
    if not f['all_checks_passed'] or f['tests']!=2:raise ValueError('Pure runtime precision fixtures required')
    current={r:sha(ROOT/r) for r in RUNTIME+['scripts/diagnose_factory_stfpm_numeric.py','tests/test_factory_precision_metadata.py']}
    for r,h in f['source_pins'].items():
        if sha(ROOT/r)!=h:raise ValueError('Precision fixture source drift')
    return {'name':'L2_numeric_diagnosis_v2','failed_source_pins':PINS,'implementation_pins':current,
            'pure_fixture_sha256':sha(fixtures),'inputs':'EXACTexisting3Torchreference arrays: fixednormalTRAIN/zero/all255; no newimages/GT or CAL',
            'maximum_backend_precision_paths':4,'ordered_paths':[
             {'id':'A','backend':'ONNXRuntime CPUExecutionProvider','precision':'FP32','threads':1,'source':'unchanged failed L ONNX'},
             {'id':'B','backend':'OpenVINO CPU','precision_hint':'f32','threads':1,'source':'unchanged failed L ONNX'},
             {'id':'C','backend':'Torch CPU','precision':'FP16 encoder and nativehalf normalization; SAMEencoder features also normalizedFP32 for isolation','threads':1,'source':'sameactualteacher/randomstudent states; nooptimizer/BNupdates'},
             {'id':'D','backend':'OpenVINO GPU','precision_hint':'f32','streams':1,'source':'byteidentical copied failed L ONNX, separatemetadata explicitf32'},
            ],
            'guard':{'maximum_absolute_score_error':.01,'finite':True,'bounded_0_1':True,'tolerance_relaxation':False},
            'selection':'Deterministic soledeploymentcandidate D(f32GPU), not choosing masks/GT; A/B/C diagnosisonly; D mustpass numeric THENnewfullprofilemean>=15 before anyfit',
            'CPU_execution':'A/B mayrunruntimeenv; C requiresisolatedTorch existingenv; CPU1accuracydiagnosis mayoverlapotheraccuracy work, nevercapacityclaims',
            'GPU_execution':'RootseparateGPUrelease required AFTERA/B/C; speed requiresseparateprojectquiescentrelease afterallotherCPU/NN/prepquiet',
            'speed':dict(original['preflight'],policy={'high':.05,'area':4,'growth':None}),
            'model_source':'models/factory/spatial_round3l_stfpm_mobile-step0.onnx',
            'candidate_model':'models/factory/spatial_round3l_stfpm_mobile-L2-gpu-f32.onnx',
            'note':'Allinitialfailedreceipts/protocol/source snapshot preserved. Newprecisionmetadata plumbing keeps olddefaultf16/scorearrays/thresholds. Original L fitguards remainblocked; laterfit needs a new protected executionaddendum.',
            'fit_updates':0,'CAL_inputs_opened':0,'original_comparison_inputs_opened':0}


def verify_recipe():
    value=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if value!=recipe():raise ValueError('Protected L2 diagnostic recipe/source changed')
    return value


def inputs():
    import numpy as np
    path=OLD/f'{NAME}-step0-torch-reference.npz'
    with np.load(path,allow_pickle=False) as data:
        return data['inputs_bgr'].copy(),data['scores'].copy()


def report_maps(label,maps,expected,extra=None):
    import numpy as np
    target=OUT/f'{label}-numeric.json';raw=OUT/f'{label}-actual-scoremaps.npz'
    if target.exists() or raw.exists():raise ValueError('Never rerun or overwrite a diagnostic path')
    values=np.asarray(maps,dtype=np.float32)
    if values.shape!=(3,640,256):raise ValueError('Diagnostic geometry differs')
    errors=[float(np.max(np.abs(a-e))) for a,e in zip(values,expected)]
    finite=bool(np.isfinite(values).all());bounded=bool(values.min()>=0 and values.max()<=1)
    with raw.open('xb') as stream:np.savez_compressed(stream,scores=values)
    value={'path':label,'protocol_sha256':sha(PROTOCOL),'model_sha256':PINS['models/factory/spatial_round3l_stfpm_mobile-step0.onnx'],
           'scoremaps_sha256':sha(raw),'max_abs_errors':errors,'all_finite':finite,'bounded_0_1':bounded,
           'numeric_guard_passed':bool(finite and bounded and max(errors)<=.01),'fixed_error_limit':.01,
           'scope':'Samefixed3numericalcases,noGT/CAL/fit/tolerancechange',**(extra or {})}
    write_new(target,value);print(json.dumps(value,indent=2),flush=True);return value


def cpu_backends():
    import numpy as np,onnxruntime as ort,openvino as ov
    verify_recipe();arrays,expected=inputs();path=ROOT/'models/factory'/f'{NAME}-step0.onnx'
    if (OUT/'A-numeric.json').exists() or (OUT/'B-numeric.json').exists():raise ValueError('Preserve allpriorCPU paths; no automaticretries')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);maps=[]
    for frame in arrays:maps.append(session.run(None,{session.get_inputs()[0].name:np.ascontiguousarray(frame.transpose(2,0,1)[None])})[0][0,0])
    report_maps('A',maps,expected,{'backend':'ORTCPUFP32','runtime_version':ort.__version__});del session
    core=ov.Core();compiled=core.compile_model(core.read_model(str(path)),'CPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_NUM_THREADS':1,'INFERENCE_PRECISION_HINT':'f32'})
    request=compiled.create_infer_request();maps=[]
    for frame in arrays:
        result=request.infer({0:np.ascontiguousarray(frame.transpose(2,0,1)[None])})
        maps.append(np.array(result[compiled.output()][0,0],dtype=np.float32,copy=True))
    report_maps('B',maps,expected,{'backend':'OpenVINOCPUFP32hint','runtime_version':ov.get_version()})


def torch_half():
    import numpy as np,torch
    from torch.nn import functional as F
    from factory_stfpm_graph import PyramidTeacherStudent,disagreement
    from prepare_factory_stfpm import module_digest
    from factory_stfpm_common import MAP_FLOOR
    verify_recipe()
    if (OUT/'C-numeric.json').exists() or (OUT/'C-f32-normalization-numeric.json').exists():raise ValueError('Preserve priorTorchhalf path')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original=json.loads((OLD/'protocol-v2-score-semantics-before-any-NN.json').read_text(encoding='utf-8'))
    export=json.loads((OLD/f'{NAME}-step0-export.json').read_text(encoding='utf-8'))
    model=PyramidTeacherStudent(original['teacher'])
    if module_digest(model.teacher)!=export['teacher_state_sha256'] or module_digest(model.student)!=export['student_state_sha256']:
        raise ValueError('Torchdiagnosis muststart exactfailedteacher/student states')
    model.half().eval();arrays,expected=inputs();native=[];mixed=[];features=[]
    with torch.inference_mode():
        for frame in arrays:
            x=torch.from_numpy(frame.transpose(2,0,1)[None].copy()).to(torch.float16)
            x=(x[:,[2,1,0]]/255.-model.mean)/model.std;t=s=x;ts=[];ss=[]
            for i,(tb,sb) in enumerate(zip(model.teacher,model.student)):
                t=tb(t);s=sb(s)
                if i in (0,3,6):ts.append(t);ss.append(s)
            half_maps=[disagreement(t,s) for t,s in zip(ts,ss)]
            float_maps=[disagreement(t.float(),s.float()) for t,s in zip(ts,ss)]
            for maps,bucket in [(half_maps,native),(float_maps,mixed)]:
                size=maps[0].shape[-2:]
                projected=[maps[0]]+[F.interpolate(m,size=size,mode='bilinear',align_corners=False) for m in maps[1:]]
                score=torch.exp(sum(torch.log(m.clamp(min=MAP_FLOOR,max=1)) for m in projected)/3.).clamp(MAP_FLOOR,1)
                bucket.append(score[0,0].float().numpy().copy())
            stats=[]
            for label,values in [('teacher',ts),('student',ss)]:
                for level,value in enumerate(values):
                    a=value.abs().float()
                    stats.append({'branch':label,'level':level,'max_abs':float(a.max()),'zero_fraction':float((a==0).float().mean()),
                                  'nonzero_below_halfnormal_fraction':float(((a>0)&(a<2**-14)).float().mean()),
                                  'nonzero_min':float(a[a>0].min()) if torch.any(a>0) else None})
            features.append(stats)
    report_maps('C',native,expected,{'backend':'TorchCPUhalf approximation ofFP16path','runtime_version':torch.__version__,
                'feature_statistics':features,'note':'FP16encoder+FP16preprocess+FP16normalization;CPUhalf FTZ maydifferfromGPUkernels; diagnostic only'})
    report_maps('C-f32-normalization',mixed,expected,{'backend':'SameCforwardfeatures;onlynormalization/maparithmetic promotedFP32',
                'same_encoder_forward_as':'C','extra_encoder_passes':0,'note':'isolatesnormalization arithmetic without newweights or newinput; not adeployedcorrection'})


def gpu_numeric():
    import numpy as np
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    verify_recipe()
    for label in ['A','B','C','C-f32-normalization']:
        if not (OUT/f'{label}-numeric.json').exists():raise ValueError('AllfixedCPUdiagnostics must finish beforeD')
    if (OUT/'D-numeric.json').exists():raise ValueError('Preserve first GPUFP32 result')
    candidate=ROOT/'models/factory'/f'{NAME}-L2-gpu-f32.onnx'
    if candidate.exists() or candidate.with_suffix('.onnx.json').exists():raise ValueError('Preserve existing/partial FP32candidate')
    source=ROOT/'models/factory'/f'{NAME}-step0.onnx';candidate.write_bytes(source.read_bytes())
    meta=json.loads(source.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    meta.update(openvino_inference_precision='f32',profile_version=NAME+'_L2_f32_probe',
                numeric_diagnosis_protocol_sha256=sha(PROTOCOL),capacity_probe_only=True)
    write_new(candidate.with_suffix('.onnx.json'),meta)
    arrays,expected=inputs();model=FactorySegmentation(candidate,output_logits=False,precision_hint='f32');maps=[]
    for frame in arrays:maps.append(model.predict(frame,.05,4)[0])
    value=report_maps('D',maps,expected,{'backend':'OpenVINOGPUFP32hint','execution_devices':model.execution_devices,
                'candidate_model_sha256':sha(candidate),'candidate_metadata_sha256':sha(candidate.with_suffix('.onnx.json')),
                'requested_precision_hint':model.precision_hint,'source_NN_bytes_identical':sha(candidate)==sha(source)})
    if not value['numeric_guard_passed']:raise ValueError('D FP32 numeric guard failed; no speed/fit')


def gpu_capacity():
    import cv2,numpy as np
    from edge_runtime.profiles.factory import FactoryProfile
    from edge_runtime.contracts import FrameContext
    recipe_value=verify_recipe();numeric=OUT/'D-numeric.json';guard=json.loads(numeric.read_text(encoding='utf-8'))
    if not guard['numeric_guard_passed'] or guard['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Dnumeric<=.01 prerequisite failed')
    candidate=ROOT/'models/factory'/f'{NAME}-L2-gpu-f32.onnx';target=OUT/'D-fullprofile-capacity.json'
    if target.exists():raise ValueError('Preserve first FP32 capacity including all tails')
    if sha(candidate)!=guard['candidate_model_sha256'] or sha(candidate.with_suffix('.onnx.json'))!=guard['candidate_metadata_sha256']:
        raise ValueError('Numerically checkedFP32candidate changed')
    cv2.setNumThreads(1);rows=recipe_value['speed']['source_rows'];frames=[]
    for row in rows:
        source=ROOT/'datasets/factory/KolektorSDD'/row['image']
        if sha(source)!=row['image_sha256']:raise ValueError('Preflight source changed')
        frame=cv2.imread(str(source))
        if frame is None:raise ValueError('Preflight decode failed')
        frames.append(frame)
    start=time.perf_counter();profile=FactoryProfile({'defect_model_path':str(candidate),'defect_backend':'auto','confirmation_frames':1})
    profile.prepare_worker();startup=time.perf_counter()-start
    if profile._defect_model.neural.precision_hint!='f32':raise ValueError('Actual fullprofile precision differs')
    for i,f in enumerate(frames[:4]):profile.process(f,FrameContext(i,time.perf_counter(),unit_id=f'L2warm-{i}'))
    profile.reset();records=[]
    for i in range(64):
        tick=time.perf_counter();r=profile.process(frames[i%16],FrameContext(i,tick,unit_id=f'L2probe-{i}'))
        records.append({'source':rows[i%16]['image'],'profile_ms':1000*(time.perf_counter()-tick),'status':r.status,
                        'neural_stage_ms':r.metrics.get('factory_neural_stage_ms'),'score_semantics':r.metrics.get('model_score_semantics')})
    values=np.asarray([r['profile_ms'] for r in records]);allinferred=all(r['neural_stage_ms'] is not None for r in records)
    value={'protocol_sha256':sha(PROTOCOL),'numeric_report_sha256':sha(numeric),'model_sha256':sha(candidate),
           'metadata_sha256':sha(candidate.with_suffix('.onnx.json')),'precision_hint':'f32','startup_seconds':startup,'records':records,
           'profile_ms':{'mean':float(values.mean()),'p50':float(np.median(values)),'p95':float(np.percentile(values,95)),
                         'p99':float(np.percentile(values,99)),'max':float(values.max()),'serial_calls_per_second':1000/float(values.mean())},
           'capacity_gate_passed':bool(allinferred and 1000/values.mean()>=15),'all_calls_inferred':allinferred,
           'scope':'Newrootprojectquiescent64fullprofilecalls/16sameTRAINviews,actualf32metadata path; ordinaryhostnotexclusive;noaccuracyclaim'}
    write_new(target,value);print(json.dumps({k:v for k,v in value.items() if k!='records'},indent=2))
    if not value['capacity_gate_passed']:raise ValueError('D FP32 mean15 capacity guard failed; nofit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['record-protocol','cpu-backends','torch-half','gpu-numeric','gpu-capacity'])
    parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if args.stage=='record-protocol':write_new(PROTOCOL,recipe());print(json.dumps({'protocol':str(PROTOCOL),'sha256':sha(PROTOCOL),'NN_executed':False}));return
    if not args.root_released:raise ValueError('Actualcoordinatorstage/GPU/quiescent release required')
    function={'cpu-backends':cpu_backends,'torch-half':torch_half,'gpu-numeric':gpu_numeric,'gpu-capacity':gpu_capacity}[args.stage]
    try:function()
    except Exception as error:
        failure=OUT/f'{args.stage}-failure.json'
        if not failure.exists():write_new(failure,{'stage':args.stage,'exception_type':type(error).__name__,'error':str(error),
                                                'protocol_sha256':sha(PROTOCOL),'scope':'preservedpartialdiagnostic;noautomaticretry/alternativeinput'})
        raise


if __name__=='__main__':main()
