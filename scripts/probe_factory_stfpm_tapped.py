"""L8 standalone fixed75-output architecture proposal/preflight; no shared edits."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from factory_tapped_segmentation import FactoryTappedSegmentation,BaseFactorySegmentation,SCHEMA,SCHEMA_SHA,INITIAL_GRAPH_SHA,protected_rows,adapter_type
from factory_stfpm_cpu_common import L4,L4_NN_SHA,L4_STATE_SHA,L4_REF,L4_REF_SHA,CAL_SHA,NORMAL_SHA
from diagnose_factory_stfpm_stages import configuration_audit
OUT=ROOT/'output/optimization/factory-round3l8-tapped75';NAME='spatial_round3l8_tapped75_bn64'
SOURCE=ROOT/'output/optimization/factory-round3l6-stage-diagnosis/same-score-graph-existing-stage-outputs.onnx'
MODEL=ROOT/f'models/factory/{NAME}-step0.onnx';PROTOCOL=OUT/'protocol-before-probe-and-conditional-learned75-correction.json'
PINS=['scripts/factory_tapped_segmentation.py','scripts/probe_factory_stfpm_tapped.py','scripts/diagnose_factory_stfpm_stages.py',
      'scripts/factory_stfpm_cpu_common.py','scripts/factory_stfpm_common.py','scripts/factory_stfpm_bn_common.py',
      'edge_runtime/profiles/factory.py','edge_runtime/profiles/factory_defect.py','edge_runtime/profiles/factory_encoder.py',
      '.venv/Lib/site-packages/openvino/_ov_api.py','.venv/Lib/site-packages/openvino/_ov_api.pyi']
VARIANTS=['host_copy','shared_views']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def metadata():
    return {'feature_version':'surface_mobilenet_seg_v1','profile_version':NAME,'model_sha256':INITIAL_GRAPH_SHA,
            'model_name':'ARCHITECTURE PROBE ONLY exact75-output BN64 teacher-student','output_logits':False,
            'openvino_device':'GPU','openvino_inference_precision':'f16','openvino_disable_winograd_convolution':True,
            'pixel_threshold':.05,'minimum_component_pixels':4,'hysteresis_growth_threshold':None,
            'score_semantics':'Bounded_multiscale_teacher_student_anomaly_disagreement_not_probability',
            'score_meaning':'ExactL4mathematicalscore; all75outputsretained to preserve changedcompilercontext; no defect-probability claim',
            'supported_height_width_ratio':[2.3,2.7],'architecture_probe_only':True,'all_outputs_retained':75,
            'tapped_schema_sha256':SCHEMA_SHA,'BN_fitted_state_sha256':L4_STATE_SHA,
            'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False,'field_validated':False,
            'app_default_unchanged':True,'license':'DerivedKolektor noncommercial CC-BY-NC-SA-4.0 research prototype; officialImageNetteacher terms'}


def prepare():
    if MODEL.exists() or MODEL.with_suffix('.onnx.json').exists():raise ValueError('Preserve initial prototype graph copy')
    if sha(SOURCE)!=INITIAL_GRAPH_SHA or sha(L4_REF)!=L4_REF_SHA:raise ValueError('Exact retained graph/reference changed')
    protected_rows();MODEL.parent.mkdir(parents=True,exist_ok=True)
    with MODEL.open('xb') as stream:stream.write(SOURCE.read_bytes())
    write_new(MODEL.with_suffix('.onnx.json'),metadata())
    write_new(OUT/'graph-copy-no-NN.json',{'source_sha256':sha(SOURCE),'new_model_sha256':sha(MODEL),'sidecar_sha256':sha(MODEL.with_suffix('.onnx.json')),
               'same_reference_sha256':sha(L4_REF),'all_outputs_retained':75,'NN_executed':False,'new_BN_or_optimizer_updates':0})


def protocol_value():
    if sha(SOURCE)!=INITIAL_GRAPH_SHA or sha(MODEL)!=INITIAL_GRAPH_SHA or sha(SCHEMA)!=SCHEMA_SHA or sha(L4_REF)!=L4_REF_SHA:raise ValueError('Prototype/schema/reference changed')
    old=json.loads((L4/'protocol-before-BN-fitting.json').read_text(encoding='utf-8'))
    if sha(L4/'protocol-before-BN-fitting.json')!='6ff093a5c51dc8e799bf316bf80e0a6799e6cf78210b1a7316ea7087af63ac6c':raise ValueError('Exact normal-source protocol changed')
    fixture=OUT/'preparation-fixtures.json';f=json.loads(fixture.read_text(encoding='utf-8'))
    if not f['all_checks_passed']:raise ValueError('Architecture/source/mock fixtures failed')
    for p,h in f['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Fixture source drift')
    return {'name':NAME,'scope':'ArchitectureprobeONLY; no training/CAL/original/appintegration authorized',
            'initial_model_sha256':INITIAL_GRAPH_SHA,'initial_sidecar_sha256':sha(MODEL.with_suffix('.onnx.json')),'exact75_schema_sha256':SCHEMA_SHA,
            'same_three_reference_sha256':L4_REF_SHA,'retained_BN_state_sha256':L4_STATE_SHA,'all_ordered_outputs':protected_rows(),
            'GPU_recipe':{'device':'GPU','precision_hint':'f16','disable_winograd_convolution':True,'streams':1},
            'runtime_injection':'StandaloneCLI process-local variant-bound subclass patch ONLY during FactoryProfile construction; restoremodulebinding; sharedfiles/defaultsunchanged',
            'prediction_policy':'EXACTinheritedBase.predict/warmup; onlyexplicitlycopyfinalscore, retainall75compiledoutputs; runtimeinternaloutputcostincluded',
            'runtime_variants':VARIANTS,'variant_study':'Fixedhost_copy defaultnativeinferbaseline vs shared_views nativeinfer(share_outputs=True); noRemoteTensor/deviceallocation; samegraph/config/buffers/all75outputs',
            'variant_order':'Both3case numeric reports before anycapacity; eachnumericpassingvariant first64calls inhostthenviewsorder; nofallback or silent substitution',
            'future_runtime_selection':'Afterfixedcomparison choosehighestmeanthroughputamongnumericANDcapacitypassingvariants; exacttiehost_copy; freezebeforelearnedCAL/original; noqualificationifnone',
            'numeric':{'cases':['firstfixednormalTRAIN','zero_initialization','all255_initialization'],'max_abs_error':.01,'finite_bounded_nonconstant':True,'firstreceiptpreserved':True,
                       'nonfinite_model_score_policy':'Onlyexactadapterinvalidscoreserrorbecomesexplicitfailedcase/null; attemptall3fixedcasesandretainfailedreport; unrelatedlineage/shape/compileerrorsstop'},
            'capacity':{'real_sources':old['preflight']['source_rows'],'profile_initialization_calls':3,'extra_real_warmup_calls':4,'measured_calls':64,'mean_min_calls_per_second':15,
                        'all_calls_must_infer':True,'rootquietrelease_required':True,'retain_first_tails':True},
            'pre_L7_numeric_ALL3_correction_proposal_sha256':sha(OUT/'conditional-learned75-proposal-before-L7-numeric.json'),
            'source_only_readback_decision_sha256':sha(OUT/'memory-placement-source-only-decision.json'),
            'conditional_learned75_correction':{
                'status':'SOURCEONLYproposal; requiresinitialL8numericandquietcapacityPASSthenrootseparaterelease',
                'ALL3_graph_plan_predeclared_in_prior_proposal_before_L7_numeric':True,'current_runtime_protocol_recorded_after_L7_singleoutput_outcomes':True,'checkpoints':[200,400,600],
                'source_prefix':'spatial_round3l7_learned_stfpm','new_deployment_prefix':'spatial_round3l8_learned_tapped75',
                'transform':'ApplyEXACTsame75existingoutputinventorytoALL3L7ONNXonce; everyexistingnode/initializer/input/opset/scoreunchanged; no weights/BN/optimizer change, nooutputsubsetsearch',
                'identities':'BeforeanynewNNbindcompleteL7training/source/checkpointONNX+PT/TorchrefSHA; separatetransformedgraph/metadata+appendproofpercheckpoint',
                'reference':'Samebyte3L4inputs withcorrespondingL7learnedTorchscores; no oldBN-only score targets',
                'numeric':'UnchangedGPU<=.01/config/finite/bounded/nonconstant forALL3; failednumericexplicitexcludedandNO_CAL',
                'CAL':'PassingnumericONLY; same80identity '+CAL_SHA+'; samefixedraw/growthgrid/weakestpositive-rank/FP<=1of70/pixelprecision>=.65',
                'freeze':'Completeall3transformation/numericfailorfullCALoutcomes beforeglobalwinner; preserveL7singleoutputoutcomesunchanged',
                'original':'RootONLYfrozenwinner; no original/CALselectionfeedback maychoosegraph/outputsubset orwhichcheckpointstransform',
                'speed':'Freshactualfittedwinner>=15fullprofileand16inputFPSruntime mandatory; initialprototypecapacitydoesnottransfer'},
            'official_sdd2_test_still_sealed':True,'normal_source_identity_sha256':NORMAL_SHA,'source_pins':{p:sha(ROOT/p) for p in PINS},'fixture_sha256':sha(fixture)}


def verify_protocol():
    value=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if value!=protocol_value():raise ValueError('Protected architecture-probe proposal/source changed')
    return value


def numeric_case(model,frame,expected):
    try:score,_,_=model.predict(frame,.05,4)
    except ValueError as error:
        if str(error)!='Factory neural model produced invalid scores':raise
        return {'max_abs_error':None,'score_range':None,'error':str(error),'numeric_output_failed':True}
    return {'max_abs_error':float(np.max(np.abs(score-expected))),
            'score_range':[float(score.min()),float(score.max())],'numeric_output_failed':False}


def parity(protocol,variant):
    target=OUT/f'initial-three-case-numeric-{variant}.json'
    attempt=OUT/f'numeric-attempt-{variant}.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve first prototype numeric receipt/partialattempt')
    if variant=='shared_views' and not (OUT/'initial-three-case-numeric-host_copy.json').exists():raise ValueError('Fixedhostnumericbaseline goesfirst')
    write_new(attempt,{'protocol_sha256':sha(PROTOCOL),'runtime_variant':variant,'new_numeric_calls_declared':3,'no_automatic_retry':True})
    model=adapter_type(variant)(MODEL);audit=configuration_audit(model.compiled,'GPU');cases=[]
    if not audit['actual_configuration_passed']:raise ValueError('Actual GPU recipe differs before numeric calls')
    with np.load(L4_REF,allow_pickle=False) as data:
        if (data['inputs_bgr'].shape!=(3,1280,512,3) or data['inputs_bgr'].dtype!=np.uint8
                or data['scores'].shape!=(3,640,256) or not np.isfinite(data['scores']).all()
                or data['scores'].min()<0 or data['scores'].max()>1
                or data['inputs_bgr'][1].any() or not (data['inputs_bgr'][2]==255).all()):raise ValueError('Exactly3fixedreferences required')
        for frame,expected in zip(data['inputs_bgr'],data['scores']):
            cases.append(numeric_case(model,frame,expected))
    errors=[row['max_abs_error'] for row in cases];ranges=[row['score_range'] for row in cases]
    finite=(len(cases)==3 and all(not row['numeric_output_failed'] for row in cases)
            and all(v is not None and np.isfinite(v) for v in errors)
            and all(r is not None and all(np.isfinite(v) for v in r) for r in ranges))
    passed=finite and all(0<=r[0]<=r[1]<=1 for r in ranges) and ranges[0][1]-ranges[0][0]>=1e-7 and max(errors)<=.01
    value={'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(MODEL),'metadata_sha256':sha(MODEL.with_suffix('.onnx.json')),
           'reference_sha256':sha(L4_REF),'max_abs_errors':errors,'score_ranges':ranges,'numeric_cases':cases,'fixed_cases_attempted':len(cases),'compiled_provenance':audit,
           'all_outputs_retained':75,'Base_predict_is_inherited':FactoryTappedSegmentation.predict is BaseFactorySegmentation.predict,
           'runtime_variant':variant,'native_share_outputs':variant=='shared_views','architecture_probe_only':True,'all_checks_passed':bool(passed),'defect_accuracy_or_speed_claim':False}
    write_new(target,value)
    if not passed:raise ValueError('Prototype numeric failed; no capacity/learning/CAL')
    print(json.dumps({k:v for k,v in value.items() if k!='compiled_provenance'},indent=2))


def preflight(protocol,variant):
    import cv2
    from edge_runtime.profiles.factory import FactoryProfile
    from edge_runtime.contracts import FrameContext
    import edge_runtime.profiles.factory_encoder as encoder_module
    for required in VARIANTS:
        if not (OUT/f'initial-three-case-numeric-{required}.json').exists():raise ValueError('Bothdeclarednumericoutcomes required beforeanycapacity')
    numeric=OUT/f'initial-three-case-numeric-{variant}.json';n=json.loads(numeric.read_text(encoding='utf-8'))
    if not n['all_checks_passed'] or n['protocol_sha256']!=sha(PROTOCOL) or n['model_sha256']!=sha(MODEL) or n['metadata_sha256']!=sha(MODEL.with_suffix('.onnx.json')):raise ValueError('Newprototype numericPASS requiredbeforecapacity')
    if variant=='shared_views':
        baseline=json.loads((OUT/'initial-three-case-numeric-host_copy.json').read_text(encoding='utf-8'))
        if baseline['all_checks_passed'] and not (OUT/'initial-fullprofile-capacity-host_copy.json').exists():raise ValueError('Fixedhostcapacitybaseline goesfirst eveniffailed')
    target=OUT/f'initial-fullprofile-capacity-{variant}.json'
    attempt=OUT/f'capacity-attempt-{variant}.json'
    if target.exists() or attempt.exists():raise ValueError('Preserve first fullprofile prototype run/partialattempt')
    write_new(attempt,{'protocol_sha256':sha(PROTOCOL),'runtime_variant':variant,'measured_calls_declared':64,'no_automatic_retry':True})
    rows=protocol['capacity']['real_sources'];frames=[];base=ROOT/'datasets/factory/KolektorSDD'
    for row in rows:
        path=base/row['image']
        if sha(path)!=row['image_sha256']:raise ValueError('Fixed realTRAIN source changed')
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError('OriginalTRAIN decode failed')
        frames.append(frame)
    started=time.perf_counter();original_class=encoder_module.FactorySegmentation;variant_class=adapter_type(variant)
    with patch.object(encoder_module,'FactorySegmentation',variant_class):
        profile=FactoryProfile({'defect_model_path':str(MODEL),'defect_backend':'auto','confirmation_frames':1})
    if encoder_module.FactorySegmentation is not original_class or type(profile._defect_model.neural) is not variant_class:raise ValueError('Isolated override/restore failed')
    profile.prepare_worker();startup=time.perf_counter()-started
    for i,frame in enumerate(frames[:4]):profile.process(frame,FrameContext(i,time.perf_counter(),unit_id=f'L8warm-{i}'))
    profile.reset();records=[]
    for i in range(64):
        tick=time.perf_counter();result=profile.process(frames[i%16],FrameContext(i,tick,unit_id=f'L8-{i}'))
        records.append({'source':rows[i%16]['image'],'profile_ms':1000*(time.perf_counter()-tick),'status':result.status,'neural_stage_ms':result.metrics.get('factory_neural_stage_ms')})
    values=np.asarray([r['profile_ms'] for r in records]);inferred=all(r['neural_stage_ms'] is not None for r in records)
    audit=configuration_audit(profile._defect_model.neural.compiled,'GPU');passed=inferred and audit['actual_configuration_passed'] and 1000/values.mean()>=15
    value={'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(MODEL),'metadata_sha256':sha(MODEL.with_suffix('.onnx.json')),'numeric_sha256':sha(numeric),
           'startup_seconds':startup,'initialization':profile.describe()['model_initialization'],'records':records,'all64_inferred':inferred,'all_outputs_retained':75,
           'profile_ms':{'mean':float(values.mean()),'p50':float(np.median(values)),'p95':float(np.percentile(values,95)),'p99':float(np.percentile(values,99)),'max':float(values.max()),'serial_calls_per_second':1000/float(values.mean())},
           'runtime_variant':variant,'native_share_outputs':variant=='shared_views','compiled_provenance':audit,'process_local_override_restored':True,'scope':'RootquietcompleteprofilearchitectureprobeONLY,16realTRAIN/64calls; ordinaryhostuncontrolled; no accuracy/productionclaim',
           'capacity_gate_passed':bool(passed),'source_pins_unchanged_after':verify_protocol()==protocol}
    write_new(target,value)
    if not passed:raise ValueError('Prototype capacityfailed; nolearning/CAL')
    print(json.dumps({k:v for k,v in value.items() if k not in ['records','compiled_provenance']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','record-protocol','numeric','capacity'],required=True);parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='record-protocol':write_new(PROTOCOL,protocol_value());print(json.dumps({'protocol_sha256':sha(PROTOCOL),'NN_executed':False}));return
    if not args.root_released or args.variant is None:raise ValueError('Root must explicitly release numeric/quietcapacity andexactnamedvariant')
    import cv2
    cv2.setNumThreads(1);protocol=verify_protocol()
    try:(parity if args.stage=='numeric' else preflight)(protocol,args.variant)
    except Exception as error:
        failure=OUT/f'first-failure-{args.stage}-{args.variant}.json'
        if not failure.exists():write_new(failure,{'protocol_sha256':sha(PROTOCOL),'stage':args.stage,'runtime_variant':args.variant,'error':str(error),'no_automatic_retry':True})
        raise


if __name__=='__main__':main()
