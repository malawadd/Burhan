"""L7 distinct fixed-fit experiment: exact L4 state, unchanged post-fit GPU gates."""
from pathlib import Path
import hashlib,json,math
from factory_stfpm_cpu_common import (ROOT,L4,L4_NN,L4_NN_SHA,L4_REF,L4_REF_SHA,L4_STATE_SHA,L4_BN_SHA,
    l4_metadata,sha,identity,write_new,validate_bn_rows,rank,validate_point,validate_search_report,CAL_SHA)

OUT=ROOT/'output/optimization/factory-round3l7-learned';MODELS=ROOT/'models/factory'
NAME='spatial_round3l7_learned_stfpm';SAVE_AT=[200,400,600]
from factory_stfpm_common import SEED
CPU_NUMERIC=ROOT/'output/optimization/factory-round3l5-cpu4/spatial_round3l5_stfpm_bn64_cpu4-step0-cpu-parity.json'
CPU_CAPACITY=ROOT/'output/optimization/factory-round3l5-cpu4/fullprofile-preflight.json'
L4_EXPORT=L4/'spatial_round3l4_stfpm_bn64-step0-export.json'
L4_EXPORT_SHA='bfc53b4cc52a58531bfabf780160987528ca0b82c7cfc14d1a5b109a3338a536'
IMPLEMENTATIONS=['factory_stfpm_learn_common.py','prepare_factory_stfpm_learn.py','train_factory_stfpm_learn.py',
 'probe_factory_stfpm_learn.py','calibrate_factory_stfpm_learn.py','freeze_factory_stfpm_learn.py']
DEPENDENCIES=['scripts/factory_stfpm_graph.py','scripts/factory_stfpm_common.py','scripts/factory_stfpm_math.py',
 'scripts/factory_stfpm_cpu_common.py','scripts/prepare_factory_stfpm_cpu.py','scripts/factory_stfpm_bn_common.py',
 'scripts/prepare_factory_stfpm_bn.py','scripts/prepare_factory_stfpm.py','scripts/probe_factory_stfpm_winograd.py',
 'scripts/factory_mask_metrics.py','scripts/train_factory_surface.py','edge_runtime/profiles/factory.py',
 'edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py']

def protocol_path():return OUT/'protocol-before-fixed600-fit.json'
def model_path(step):return MODELS/f'{NAME}-step{step}.onnx'

def initialization_evidence():
    old=l4_metadata();cpu=json.loads(CPU_NUMERIC.read_text(encoding='utf-8'));capacity=json.loads(CPU_CAPACITY.read_text(encoding='utf-8'))
    if sha(L4_EXPORT)!=L4_EXPORT_SHA:raise ValueError('Original L4 seeded/fitted export receipt changed')
    if (not cpu['all_checks_passed'] or not cpu['actual_CPU4_configuration_passed'] or cpu['model_sha256']!=L4_NN_SHA
            or len(cpu['max_abs_errors'])!=3 or max(cpu['max_abs_errors'])>.01):raise ValueError('Exact L4 CPU FP32 numerical initialization evidence required')
    if capacity['capacity_gate_passed'] or capacity['model_sha256']!=L4_NN_SHA:raise ValueError('Preserve actual failed CPU4 capacity; do not turn it into qualification')
    return old,{'L4_model_sha256':L4_NN_SHA,'L4_reference_sha256':L4_REF_SHA,'L4_export_receipt_sha256':L4_EXPORT_SHA,'BN_fit_receipt_sha256':L4_BN_SHA,
        'BN_fitted_state_sha256':L4_STATE_SHA,'CPU_numeric_receipt_sha256':sha(CPU_NUMERIC),
        'failed_CPU_capacity_sha256':sha(CPU_CAPACITY),'CPU_capacity_is_qualification':False}

def gpu_configuration_passes(audit):
    keys=['EXECUTION_DEVICES','INFERENCE_PRECISION_HINT','GPU_DISABLE_WINOGRAD_CONVOLUTION','NUM_STREAMS','PERFORMANCE_HINT']
    if any(not audit.get(k,{}).get('supported') for k in keys):return False
    v={k:audit[k]['value'] for k in keys}
    return (v['EXECUTION_DEVICES']==['GPU.0'] and str(v['INFERENCE_PRECISION_HINT']) in {"<Type: 'float16'>",'float16','f16'}
        and v['GPU_DISABLE_WINOGRAD_CONVOLUTION'] is True and str(v['NUM_STREAMS']) in {'1','<NumStreams: 1>'}
        and str(v['PERFORMANCE_HINT']) in {'LATENCY','PerformanceMode.LATENCY'})

def validate_numeric_receipt(value):
    if value['fixed_cases_attempted']!=3:raise ValueError('Exactly three numeric cases must be attempted')
    errors=value['max_abs_errors'];ranges=value['actual_score_ranges']
    geometry=(len(errors)==3 and len(ranges)==3 and all(len(r)==2 for r in ranges))
    finite_number=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    finite=geometry and all(finite_number(e) for e in errors) and all(finite_number(x) for r in ranges for x in r)
    bounded=finite and all(0<=r[0]<=r[1]<=1 for r in ranges)
    nonconstant=bounded and ranges[0][1]-ranges[0][0]>=1e-7
    config=gpu_configuration_passes(value['queried_configuration_and_execution'])
    passed=bool(finite and bounded and nonconstant and value['all_finite'] and value['all_bounded_0_1']
        and value['reference_inputs_byte_identical_to_L4'] and value['reference_scores_finite_bounded'] and config and max(errors)<=.01)
    if value['all_checks_passed']!=passed:raise ValueError('Numeric receipt acceptance arithmetic/configuration differs')
    return passed

def protocol_value():
    old,evidence=initialization_evidence();fixture=OUT/'preparation-fixtures.json';f=json.loads(fixture.read_text(encoding='utf-8'))
    if not f['all_checks_passed']:raise ValueError('Pure preparation guards required')
    for p,h in f['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Fixture source drift')
    keys=['teacher','input_shape','output_shape','architecture','score','normalization','norm_scale_floor','norm_squared_floor','map_floor',
        'split_items','normal_sources','normal_source_identity_sha256','BN_fit_sources','BN_fit_source_identity_sha256',
        'actual_BN_factory','BN_fit','fit','raw_high_grid','area_grid','growth_grid','selection_rank','seed_search','calibration_source_identity_sha256']
    p={k:old[k] for k in keys}
    p.update(name=NAME,seed=SEED,save_at=SAVE_AT,initialization=evidence,
        hypothesis='Actual learned student may have different GPU numerical behavior from random/BN-only state; hypothesis not guaranteed and no prior failure rewritten',
        unfitted_preflight_waiver='NEW experiment explicitly permits fixed CPU FP32 training despite failed unfitted GPU parity and CPU capacity. Neither failure is acceptance; prior GPU speed is architecture context only.',
        GPU_config={'device':'GPU','precision_hint':'f16','disable_winograd_convolution':True,'num_streams':1},
        fixed_execution='Complete600/save200400600 FIRST, no numeric/CAL feedback until completion; no extension, early stop, BN refit or graph/normalization change',
        checkpoint_outcomes='Each learned checkpoint gets same3 GPU numeric <=.01 finite/bounded gate. Numeric-failed checkpoint excluded WITHOUT any CAL pixels. All3 terminal outcomes required before globalfreeze.',
        calibration='Passing-checkpoint only: same80 ordered CAL/GT, exact prior grid/search/rank, normalVIEW FP<=1/70 and pixelprecision>=.65; no comparison selection',
        comparison_gate='Root ONLY globalwinner after complete600 and all3 terminal outcomes; exposed SDD1 comparison, not blind test',
        final_performance='Fitted winner policy must pass fresh>=15 fullprofile and sustained16FPS end-to-end capture/processing/drops/late/queue gates before acceptance; no prefit speed transfer',
        source_pins={p:sha(ROOT/p) for p in [*[f'scripts/{f}' for f in IMPLEMENTATIONS],*DEPENDENCIES,'scripts/check_factory_stfpm_learn_preparation.py','tests/test_factory_stfpm_learn_preparation.py']},
        fixture_sha256=sha(fixture),official_sdd2_test_still_sealed=True,CAL_or_comparison_inputs_used_for_training=0)
    return p

def verify_protocol():
    value=json.loads(protocol_path().read_text(encoding='utf-8'))
    if value!=protocol_value() or value!=json.loads((OUT/'proposed-protocol-before-fit.json').read_text(encoding='utf-8')):raise ValueError('Protected L7 protocol/source/proposal changed')
    approval=OUT/'parent-approved-fit-decision.json';a=json.loads(approval.read_text(encoding='utf-8'))
    if not a.get('approved_exact_600') or a['proposal_sha256']!=sha(OUT/'proposed-protocol-before-fit.json'):raise ValueError('Parent exact600 decision changed')
    return protocol_path(),value

def validate_initialization(protocol):
    _,evidence=initialization_evidence()
    if protocol['initialization']!=evidence:raise ValueError('Exact initialization lineage changed')
    return evidence

def archive_executed_sources(protocol,execution):
    folder=OUT/'executed-sources-before-optimizer'
    if folder.exists():raise ValueError('Preserve prior source archive/attempt')
    records=[]
    for rel,h in protocol['source_pins'].items():
        source=ROOT/rel
        if sha(source)!=h:raise ValueError('Source changed before optimizer')
        target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        records.append({'source':rel,'sha256':h})
    write_new(folder/'manifest.json',{'protocol_sha256':sha(protocol_path()),'execution_receipt_sha256':sha(execution),'sources':records,'before_optimizer':True})

def metadata(path,protocol,point=None):
    point=point or {'high':.05,'area':4,'growth':None}
    return {'feature_version':'surface_mobilenet_seg_v1','profile_version':NAME,'model_name':'EXPERIMENTAL learned normal-only teacher-student GPU',
        'model_sha256':sha(path),'output_logits':False,'openvino_device':'GPU','openvino_inference_precision':'f16',
        'openvino_disable_winograd_convolution':True,'input_shape':protocol['input_shape'],'output_shape':protocol['output_shape'],
        'input_width':512,'input_height':1280,'model_grid_width':256,'model_grid_height':640,
        'pixel_threshold':point['high'],'minimum_component_pixels':point['area'],'hysteresis_growth_threshold':point['growth'],
        'score_meaning':protocol['score'],'score_semantics':'Bounded_multiscale_teacher_student_anomaly_disagreement_not_probability',
        'architecture':protocol['architecture'],'split_items':protocol['split_items'],'supported_height_width_ratio':[2.3,2.7],
        'training_protocol':protocol,'BN_fitted_from_real_TRAIN':True,'BN_fit_forward_passes':64,
        'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False,'field_validated':False,
        'license':'OfficialtorchvisionImageNetteacher terms; derivedKolektor CC-BY-NC-SA-4.0 noncommercial research prototype',
        'official_sdd2_test_still_sealed':True}

def require_complete_training(protocol):
    complete=OUT/'training-complete.json';c=json.loads(complete.read_text(encoding='utf-8'))
    execution=OUT/'training-execution-before-optimizer.json'
    if (c['fixed_steps_completed']!=600 or c['save_at']!=SAVE_AT or [r['step'] for r in c['history']]!=list(range(1,601))
        or c['protocol_sha256']!=sha(protocol_path()) or c['execution_receipt_sha256']!=sha(execution) or not c['teacher_unchanged']):raise ValueError('Exact completed600 must precede any GPU numeric/CAL')
    for step in SAVE_AT:
        if c['checkpoint_reports'][str(step)]!=sha(OUT/f'checkpoint{step}-training.json'):raise ValueError('All3 checkpoint reports changed')
    return complete,c
