"""L5 single CPU4 backend, exact retained L4 BN-fitted graph/normal sources."""
from pathlib import Path
import hashlib,json
from factory_stfpm_common import HIGH_GRID,AREA_GRID,GROWTH_GRID,RANK,SEED,CAL_SHA,NORMAL_SHA,rank,validate_point,validate_search_report
from factory_stfpm_bn_common import allowed_bn_keys,check_only_bn_changed,validate_bn_rows
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/optimization/factory-round3l5-cpu4';MODELS=ROOT/'models/factory';NAME='spatial_round3l5_stfpm_bn64_cpu4';SAVE_AT=[200,400,600]
L4=ROOT/'output/optimization/factory-round3l4-bn64'
L4_PROTOCOL_SHA='6ff093a5c51dc8e799bf316bf80e0a6799e6cf78210b1a7316ea7087af63ac6c'
L4_FAILED_SHA='2349463e4c38a29e7d47a71fd29ce6af7fc5b7d6517484655b697c08273cbc9b'
L4_SNAPSHOT_SHA='01a12c2181f95bfb0be9e666e514f1d3c5d897bc8fb1ada5e821e5100ecf9e13'
L4_BN_SHA='dba5c934745c030417006d58927f531112e0250ccd0c31fe228bc553e02bcf00'
L4_STATE_SHA='3a8030a7e50d0e2d1c7795bc6cad1139d5d9427aa6020ca09347cbaf096a4207'
L4_NN=MODELS/'spatial_round3l4_stfpm_bn64-step0.onnx';L4_NN_SHA='3eb5f1cd0936342098658e4a808176e4b6bdb8bc1e2103dc3873e7542ed07ace'
L4_REF=L4/'spatial_round3l4_stfpm_bn64-step0-torch-reference.npz';L4_REF_SHA='5e870bccd9243f1835f2bc2d68e666110bfe4d4903ddcbc605bd577b19e08409'
IMPLEMENTATIONS=['factory_stfpm_cpu_common.py','prepare_factory_stfpm_cpu.py','probe_factory_stfpm_cpu.py','train_factory_stfpm_cpu.py','calibrate_factory_stfpm_cpu.py','freeze_factory_stfpm_cpu.py']
DEPENDENCIES=['scripts/factory_stfpm_graph.py','scripts/factory_stfpm_common.py','scripts/factory_stfpm_math.py','scripts/factory_stfpm_bn_common.py',
              'scripts/prepare_factory_stfpm_bn.py','scripts/prepare_factory_stfpm.py','edge_runtime/profiles/factory.py',
              'edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py','scripts/factory_mask_metrics.py','scripts/train_factory_surface.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
def protocol_path():return OUT/'protocol-before-CPU4-NN.json'
def model_path(step):return MODELS/f'{NAME}-step{step}.onnx'


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Preserve existing/partial L5 artifact')
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def l4_metadata():
    pins={L4/'protocol-before-BN-fitting.json':L4_PROTOCOL_SHA,L4/'spatial_round3l4_stfpm_bn64-step0-gpu-parity.json':L4_FAILED_SHA,
          L4/'failed-L4-executed-source-snapshot/snapshot-manifest.json':L4_SNAPSHOT_SHA,L4/'BN-fitting-complete.json':L4_BN_SHA,
          L4/'real-TRAIN-BN64-fitted-state.pt':L4_STATE_SHA,L4_NN:L4_NN_SHA,L4_REF:L4_REF_SHA}
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Exact retained L4 source/state/reference changed')
    old=json.loads((L4/'protocol-before-BN-fitting.json').read_text(encoding='utf-8'))
    if len(old['normal_sources'])!=208 or identity(old['normal_sources'])!=NORMAL_SHA:raise ValueError('Protected208normalidentitychanged')
    return old


def protocol_value():
    old=l4_metadata();fixture=OUT/'preparation-fixtures.json';f=json.loads(fixture.read_text(encoding='utf-8'))
    if not f['all_checks_passed']:raise ValueError('CPU opt-in/default fixtures failed')
    for p,h in f['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('L5 fixture source changed')
    pins={f'scripts/{p}':sha(ROOT/'scripts'/p) for p in IMPLEMENTATIONS};pins.update({p:sha(ROOT/p) for p in DEPENDENCIES})
    keys=['seed','teacher','input_shape','output_shape','architecture','score','normalization','norm_scale_floor','norm_squared_floor','map_floor',
          'split_items','supported_height_width_ratio','normal_sources','normal_source_identity_sha256','BN_fit_sources','BN_fit_source_identity_sha256',
          'actual_BN_factory','BN_fit','fit','raw_high_grid','area_grid','growth_grid','selection_rank','seed_search','calibration_source_identity_sha256',
          'calibration','comparison_gate','final_performance']
    value={key:old[key] for key in keys}
    value.update(name=NAME,stage='ONECPU4backend onEXACTsavedL4BN64graph; NOnewBNfit',save_at=SAVE_AT,
                 parent_L4_protocol_sha256=L4_PROTOCOL_SHA,parent_L4_failed_numeric_sha256=L4_FAILED_SHA,parent_L4_failed_source_snapshot_sha256=L4_SNAPSHOT_SHA,
                 retained_BN_fit_receipt_sha256=L4_BN_SHA,retained_BN_fitted_state_sha256=L4_STATE_SHA,
                 initial_model_sha256=L4_NN_SHA,initial_same_three_input_reference_sha256=L4_REF_SHA,
                 CPU_config={'device':'CPU','PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':4},
                 calibration='All3 actualCPU/f32/exactly4threads/1stream single-outputmaps +fixedTRAIN/zero/255 numericguards BEFOREsameordered80CAL; scorecacheboundmodel/protocol/source/nativeGTgeometrySHA; nooriginalscoringbyowner',
                 preflight=old['preflight'],numeric_gate='EXACTL4BNfittedONNXand3references; actualCPUf32threads4maxabs<=.01,finiteboundednonconstant',
                 capacity_gate='AfterCPU4numericpass+rootquietrelease:NEW16realTRAIN/64fullprofilecalls allinferred/mean>=15;noGPUoroldweightsspeedtransfer',
                 single_backend_only='NoAUTO/fallback,threadsearch,resizing,BNrepeat,precisionsearch,newnormalstats orweightchanges beforepreflight',
                 implementation_pins=pins,fixture_sha256=sha(fixture),official_sdd2_test_still_sealed=True)
    return value


def verify_protocol():
    path=protocol_path();p=json.loads(path.read_text(encoding='utf-8'))
    if p!=protocol_value():raise ValueError('Protected L5 protocol/source changed')
    return path,p


def metadata(path,protocol,point=None):
    point=point or protocol['preflight']['policy']
    return {'feature_version':'surface_mobilenet_seg_v1','profile_version':NAME,'model_name':'EXPERIMENTAL BN64 teacher-student CPU4',
            'model_sha256':sha(path),'output_logits':False,'openvino_device':'CPU','openvino_cpu_threads':4,'openvino_inference_precision':'f32',
            'input_shape':protocol['input_shape'],'output_shape':protocol['output_shape'],'input_width':512,'input_height':1280,
            'model_grid_width':256,'model_grid_height':640,'pixel_threshold':point['high'],'minimum_component_pixels':point['area'],
            'hysteresis_growth_threshold':point['growth'],'score_meaning':protocol['score'],
            'score_semantics':'Bounded_multiscale_teacher_student_anomaly_disagreement_not_probability',
            'architecture':protocol['architecture'],'split_items':protocol['split_items'],'supported_height_width_ratio':[2.3,2.7],
            'training_protocol':protocol,'BN_fitted_from_real_TRAIN':True,'BN_fit_forward_passes':64,
            'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False,'field_validated':False,
            'license':'OfficialtorchvisionImageNetteacher terms; derivedKolektor model/dataCC-BY-NC-SA-4.0 research/non-commercial prototype',
            'official_sdd2_test_still_sealed':True}
