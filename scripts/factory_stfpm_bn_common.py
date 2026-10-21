"""L4 independent real-TRAIN BN-fit protocol, preserving all L1/L2/L3 artifacts."""
from pathlib import Path
import hashlib,json,math
from factory_stfpm_common import HIGH_GRID,AREA_GRID,GROWTH_GRID,RANK,SEED,CAL_SHA,NORMAL_SHA,rank,validate_point,validate_search_report

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/optimization/factory-round3l4-bn64'
MODELS=ROOT/'models/factory'
NAME='spatial_round3l4_stfpm_bn64'
SAVE_AT=[200,400,600]
ORIGINAL=ROOT/'output/optimization/factory-round3l/protocol-v2-score-semantics-before-any-NN.json'
ORIGINAL_SHA='84e3ab65009e40a5e13eb8f0e241ef2ea766aff1431841e1c75ae5e81a0f4720'
FAILED_L3=ROOT/'output/optimization/factory-round3l-no-winograd-v3/three-case-numeric.json'
FAILED_L3_SHA='845ef8fe5aaa370eca12c241ac9fd018b5b10511d73fb27f4a0a8ba13a72788d'
IMPLEMENTATIONS=['factory_stfpm_bn_common.py','prepare_factory_stfpm_bn.py','probe_factory_stfpm_bn.py',
                'train_factory_stfpm_bn.py','calibrate_factory_stfpm_bn.py','freeze_factory_stfpm_bn.py']
DEPENDENCIES=['scripts/factory_stfpm_graph.py','scripts/factory_stfpm_common.py','scripts/factory_stfpm_math.py',
              'scripts/probe_factory_stfpm_winograd.py','scripts/prepare_factory_stfpm.py','scripts/train_factory_stfpm.py',
              'edge_runtime/profiles/factory.py','edge_runtime/profiles/factory_encoder.py','edge_runtime/profiles/factory_defect.py',
              'scripts/factory_mask_metrics.py','scripts/train_factory_surface.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
def protocol_path():return OUT/'protocol-before-BN-fitting.json'
def model_path(step):return MODELS/f'{NAME}-step{step}.onnx'


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Preserve existing/partial L4 artifact')
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def source_metadata():
    if sha(ORIGINAL)!=ORIGINAL_SHA or sha(FAILED_L3)!=FAILED_L3_SHA:raise ValueError('Preserve failed original L versions')
    old=json.loads(ORIGINAL.read_text(encoding='utf-8'))
    if len(old['normal_sources'])!=208 or identity(old['normal_sources'])!=NORMAL_SHA:raise ValueError('Established normal source identity changed')
    return old


def select64(normals,items):
    if len(items)!=30 or len(set(items))!=30:raise ValueError('Exactly30 distinct TRAIN items required')
    groups={item:sorted([r for r in normals if Path(r['image']).parts[0]==item],key=lambda r:r['image']) for item in items}
    if any(len(groups[item])<2 for item in items) or any(len(groups[item])<3 for item in items[:4]):
        raise ValueError('Declared64 source counts unavailable; no substitution')
    selected=[groups[item][view] for view in [0,1] for item in items]+[groups[item][2] for item in items[:4]]
    if len(selected)!=64 or len({r['image'] for r in selected})!=64 or any(r['defective'] for r in selected):raise ValueError('64 unique known normal sources required')
    return selected


def allowed_bn_keys(spec):
    return {f'student.{r["name"]}.{suffix}' for r in spec for suffix in ['running_mean','running_var','num_batches_tracked']}


def check_only_bn_changed(before,after,allowed):
    if set(before)!=set(after):raise ValueError('State inventory changed')
    changed=[key for key in before if before[key]!=after[key]]
    if any(key not in allowed for key in changed):raise ValueError('A parameter/teacher/normalization/non-BN state changed')
    return changed


def validate_bn_rows(rows,spec,counter):
    if len(rows)!=len(spec) or [r['name'] for r in rows]!=[r['name'] for r in spec]:raise ValueError('Exact BN inventory required')
    for row,expected in zip(rows,spec):
        if (row['counter']!=counter or not row['finite'] or row['restored_momentum']!=.01
                or row['preserved_eps']!=expected['eps'] or not math.isfinite(row['variance_min'])
                or not math.isfinite(row['variance_max']) or row['variance_min']<0 or row['variance_max']<row['variance_min']):
            raise ValueError('Exact BN counters/epsilon/momentum/finite variance required')


def protocol_value():
    old=source_metadata();meta_path=OUT/'actual-BN-factory-metadata.json';bn=json.loads(meta_path.read_text(encoding='utf-8'))
    if bn['forward_calls']!=0 or len(bn['student_BN'])!=18 or any(r['momentum']!=.01 or r['eps']!=.001 or r['initial_counter']!=0 for r in bn['student_BN']):raise ValueError('Verifyactual MobileNet BN factory metadata')
    if sha(Path(bn['torchvision_source_path']))!=bn['torchvision_source_sha256']:raise ValueError('ActualinstalledMobileNetfactorysource changed')
    selection_path=OUT/'selected64-before-any-forward.json';selection=json.loads(selection_path.read_text(encoding='utf-8'))
    selected=select64(old['normal_sources'],old['split_items']['train'])
    if selection['sources']!=selected or selection['source_identity_sha256']!=identity(selected):raise ValueError('64 TRAIN-only selection changed')
    fixtures=OUT/'preparation-fixtures.json';f=json.loads(fixtures.read_text(encoding='utf-8'))
    if not f['all_checks_passed']:raise ValueError('L4 pure preparation fixtures failed')
    for p,h in f['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('L4 preparation fixture source changed')
    pins={f'scripts/{p}':sha(ROOT/'scripts'/p) for p in IMPLEMENTATIONS}
    pins.update({p:sha(ROOT/p) for p in DEPENDENCIES})
    return {'name':NAME,'seed':SEED,'stage':'TRAIN-BN fitting thenfreshnumeric/capacity BEFOREany600gradientfit',
            'original_L_protocol_sha256':ORIGINAL_SHA,'failed_L3_numeric_sha256':FAILED_L3_SHA,
            'teacher':old['teacher'],'input_shape':old['input_shape'],'output_shape':old['output_shape'],
            'architecture':old['architecture'],'score':old['score'],'normalization':old['normalization'],
            'norm_scale_floor':old['norm_scale_floor'],'norm_squared_floor':old['norm_squared_floor'],'map_floor':old['map_floor'],
            'split_items':old['split_items'],'supported_height_width_ratio':[2.3,2.7],
            'normal_sources':old['normal_sources'],'normal_source_identity_sha256':NORMAL_SHA,
            'BN_fit_sources':selected,'BN_fit_source_identity_sha256':identity(selected),'BN_fit_source_selection_sha256':sha(selection_path),
            'actual_BN_factory_metadata_sha256':sha(meta_path),'actual_BN_factory':bn,
            'BN_fit':{'forward_passes':64,'cpu_threads':1,'optimizer':None,'gradients':False,
                      'reset_student_running_stats':True,'statistics_momentum':None,'preserve_actual_eps':True,
                      'subsequent_training_momentum':.01,'normal_preprocessing':'Sameactualfull512×1280INTER_AREAresize/RGB255ImageNetnorm; no crops,syntheticimages,zero/whitecasesinBNfit',
                      'source_selection':'2normalviewsper30TRAINitems,then3rdnormalviewfirst4items; exactmetadataorderbeforepixels',
                      'allowed_state_changes':'ONLYstudentBNrunning_mean/running_var/num_batches_tracked; everycounter64,finite/nonnegativevariance; allparameters+teacher+mean/std invariant',
                      'artifact':'RealTRAINBN-fitted fullstate savedonce; nevercalledunfitted; graphweightswillchangeafterBNfolding',
                      'future_gradient_initialization':'LoadEXACTsavedBN-fittedfullstatebeforeoptimizer andprobe-statehashguard; no rerun/reset ofBNfit'},
            'GPU_config':{'precision_hint':'f16','disable_winograd_convolution':True,'num_streams':1},
            'preflight':old['preflight'],'save_at':SAVE_AT,'fit':dict(old['fit'],student_policy='Randomconvolutionweights andBNaffineones/zeros preservedthroughBNfit; explicitstudentBNmomentum.01through600; teacheralwaysfrozeneval; BNfittedinitialstate loadedonce',student_BN_training_momentum=.01),
            'raw_high_grid':HIGH_GRID,'area_grid':AREA_GRID,'growth_grid':GROWTH_GRID,'selection_rank':RANK,
            'seed_search':old['seed_search'],'calibration_source_identity_sha256':CAL_SHA,'calibration':old['calibration'],
            'comparison_gate':old['comparison_gate'],'final_performance':old['final_performance'],
            'numeric_gate':'SamefixedTRAIN/zero/all255INPUTS,newTorchreferencesfromBNfittedstate;actualf16/noWinogradGPUmaxabs<=.01,finiteboundednonconstant',
            'capacity_gate':'NEWrootquiescent64completeprofilecalls/same16TRAIN afternumericpass;mean>=15/allinferred;noL1orKspeedtransfer',
            'implementation_pins':pins,'fixture_sha256':sha(fixtures),
            'original_L_momentum_documentation_correction':'L1claimed.1butactualMobileNetfactory.01; NOstudentBNfit/600ranunderL1,soeval-onlyfailednumericunchanged. L4explicit.01.',
            'official_sdd2_test_still_sealed':True,'CAL_or_comparison_used_for_BN_fit':0}


def verify_protocol():
    path=protocol_path();value=json.loads(path.read_text(encoding='utf-8'))
    if value!=protocol_value():raise ValueError('Protected L4 protocol/source changed')
    return path,value


def metadata(path,protocol,point=None):
    point=point or protocol['preflight']['policy']
    return {'feature_version':'surface_mobilenet_seg_v1','profile_version':NAME,'model_name':'EXPERIMENTAL TRAIN-BN-fitted multiscale student-teacher',
            'model_sha256':sha(path),'output_logits':False,'openvino_inference_precision':'f16','openvino_disable_winograd_convolution':True,
            'input_shape':protocol['input_shape'],'output_shape':protocol['output_shape'],'input_width':512,'input_height':1280,
            'model_grid_width':256,'model_grid_height':640,'pixel_threshold':point['high'],'minimum_component_pixels':point['area'],
            'hysteresis_growth_threshold':point['growth'],'score_meaning':protocol['score'],
            'score_semantics':'Bounded_multiscale_teacher_student_anomaly_disagreement_not_probability',
            'architecture':protocol['architecture'],'split_items':protocol['split_items'],'supported_height_width_ratio':[2.3,2.7],
            'training_protocol':protocol,'BN_fitted_from_real_TRAIN':True,'BN_fit_forward_passes':64,
            'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False,'field_validated':False,
            'license':'OfficialtorchvisionImageNetteacher terms; derivedKolektor model/dataCC-BY-NC-SA-4.0 research/non-commercial prototype',
            'official_sdd2_test_still_sealed':True}
