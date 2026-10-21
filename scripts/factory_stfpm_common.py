"""Independent L identities and exact normal-only protocol; stdlib only."""
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/optimization/factory-round3l'
MODELS=ROOT/'models/factory'
NAME='spatial_round3l_stfpm_mobile'
INITIAL_PROTOCOL_SHA='9980d288dbb2754735d8c77c0bcfb3b763ec9290942851973f614041ab916f2e'
SCORE_SEMANTICS='Bounded_multiscale_teacher_student_anomaly_disagreement_not_probability'
PROPOSAL=ROOT/'output/optimization/factory-round3/spatial_round3l_stfpm_mobile_preflight-metadata-proposal-only.json'
PROPOSAL_SHA='f84b55a47d12dfe748903ea2e6310fec6f8ea06172b01b345375cad7d883ef3f'
NORMAL_SHA='f7ec8222431065047de889ff2b03f24172bc59257d6962c51103377566f733b4'
CAL_SHA='7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07'
OLD=ROOT/'output/optimization/factory-round3/spatial_round3k_c200_density_f16-protocol-before-fit.json'
OLD_SHA='69433138b7a8a8c8d18ac68ae901487647d5e103db8330c2a7f8ece16bf0662a'
TEMPLATE=MODELS/'spatial_round3e_phase_max_refine-step200.onnx.json'
TEMPLATE_SHA='6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb'
SEED=20261030
SAVE_AT=[200,400,600]
NORM_SCALE_FLOOR=2**-8
NORM_SQUARED_FLOOR=2**-12
MAP_FLOOR=2**-14
HIGH_GRID=[2**-14, .0001,.0003,.001,.003,.01,.03,.05,.1,.2,.3,.5,.7,.9,.95,.99]
AREA_GRID=[1,2,4,8,16,32]
GROWTH_GRID=[None,2**-14,.0001,.0003,.001,.003,.01,.03,.05,.1,.2,.3,.5,.7,.9]
RANK=['meaningfulIoU>=.1viewrecall','worst_positive_IoU','coverage50pct_viewrecall','worst_positive_GTcoverage','microIoU','fewerNormalFP','foregroundrecall']
IMPLEMENTATIONS=['factory_stfpm_common.py','factory_stfpm_graph.py','factory_stfpm_math.py',
                'prepare_factory_stfpm.py','probe_factory_stfpm.py','train_factory_stfpm.py',
                'calibrate_factory_stfpm.py','freeze_factory_stfpm.py']
DEPENDENCIES=['edge_runtime/profiles/factory.py','edge_runtime/profiles/factory_defect.py',
              'edge_runtime/profiles/factory_encoder.py','scripts/factory_mask_metrics.py',
              'scripts/train_factory_surface.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError(f'Preserve existing/partial artifact: {path}')
    temporary=path.with_name(path.name+'.writing')
    if temporary.exists():raise ValueError('Preserve existing interrupted write')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def identity(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def validate_normal_sources(normals,split):
    train=set(split['train']);other=set(split['calibration']+split['test'])
    if len(normals)!=208 or identity(normals)!=NORMAL_SHA or len(train)!=30 or train&other:
        raise ValueError('Protected normal source identity/split changed')
    if any(r['defective'] or Path(r['image']).parts[0] not in train for r in normals):
        raise ValueError('Only original TRAIN normal images permitted')


def validate_point(point,protocol):
    if any(not math.isfinite(float(point[key])) for key in ['normal_fp','foreground_pixel_precision',
            'meaningful_iou_0_1_recall','worst_positive_iou','coverage_50pct_recall',
            'worst_positive_coverage','micro_pixel_iou','foreground_pixel_recall']):raise ValueError('Nonfinite selected metric')
    if point['normal_fp']>1 or point['foreground_pixel_precision']<.65:raise ValueError('Fixed guard violated')
    if point['high'] not in protocol['raw_high_grid'] or point['area'] not in protocol['area_grid'] or point['growth'] not in protocol['growth_grid']:
        raise ValueError('Point outside protected grid')


def validate_search_report(report,protocol):
    raw=report['raw_grid'];grown=report['grown_grid']
    if [(r['high'],r['area']) for r in raw]!=[(h,a) for h in protocol['raw_high_grid'] for a in protocol['area_grid']]:
        raise ValueError('Raw grid must contain the exact ordered Cartesian search once')
    metric_keys=['meaningful_iou_0_1_recall','worst_positive_iou','coverage_50pct_recall',
                 'worst_positive_coverage','micro_pixel_iou','foreground_pixel_recall','foreground_pixel_precision']
    for point in raw+grown:
        if type(point['normal_fp']) is not int or not 0<=point['normal_fp']<=70:raise ValueError('Invalid normal FP count')
        if any(not math.isfinite(float(point[k])) or not 0<=point[k]<=1 for k in metric_keys):raise ValueError('Invalid search metric')
    seeds=[]
    for high in protocol['raw_high_grid']:
        eligible=[r for r in raw if r['high']==high and r['normal_fp']<=1]
        if eligible:seeds.append(max(eligible,key=rank))
    seeds=sorted(seeds,key=rank,reverse=True)[:6]
    expected=[(s['high'],s['area'],low) for s in seeds for low in protocol['growth_grid'] if low is None or low<=s['high']]
    if [(r['high'],r['area'],r['growth']) for r in grown]!=expected:raise ValueError('Missing/duplicate/reordered fixed seed/growth search')
    for p in grown:
        seed=next(s for s in seeds if s['high']==p['high'] and s['area']==p['area'])
        if p['normal_fp']!=seed['normal_fp']:raise ValueError('Growth changed flagged normal count')
    eligible=[p for p in grown if p['normal_fp']<=1 and p['foreground_pixel_precision']>=.65]
    selected=max(eligible,key=rank) if eligible else None
    if report['selected']!=selected:raise ValueError('Selected policy is not the exact stable protected ranking winner')


def rank(point):
    return (point['meaningful_iou_0_1_recall'],point['worst_positive_iou'],point['coverage_50pct_recall'],
            point['worst_positive_coverage'],point['micro_pixel_iou'],-point['normal_fp'],point['foreground_pixel_recall'])


def protocol_value():
    if sha(OUT/'protocol-before-any-NN.json')!=INITIAL_PROTOCOL_SHA:raise ValueError('Preserve original pre-NN L protocol')
    if sha(PROPOSAL)!=PROPOSAL_SHA or sha(OLD)!=OLD_SHA or sha(TEMPLATE)!=TEMPLATE_SHA:
        raise ValueError('Prior metadata-only source identities changed')
    proposal=json.loads(PROPOSAL.read_text(encoding='utf-8'))
    old=json.loads(OLD.read_text(encoding='utf-8'));normals=proposal['normal_sources']
    if len(normals)!=208 or identity(normals)!=NORMAL_SHA or normals!=old['normal_sources']:
        raise ValueError('Exact208 original TRAIN normal source identities required')
    validate_normal_sources(normals,old['split_items'])
    teacher=proposal['teacher']
    if sha(ROOT/teacher['file'])!=teacher['sha256']:raise ValueError('Official ImageNet source changed')
    fixture=OUT/'preparation-fixtures-v2-score-semantics.json'
    receipt=json.loads(fixture.read_text(encoding='utf-8'))
    if not receipt['all_checks_passed']:raise ValueError('Preparation fixtures failed')
    for relative,digest in receipt['implementation_pins'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Preparation fixture source changed')
    pins={f'scripts/{name}':sha(ROOT/'scripts'/name) for name in IMPLEMENTATIONS}
    pins.update({name:sha(ROOT/name) for name in DEPENDENCIES})
    return {'name':NAME,'seed':SEED,'preparation_only_until_root_release':True,
            'protocol_version':2,'original_protocol_sha256':INITIAL_PROTOCOL_SHA,
            'pre_NN_addendum':'Metadata-only runtime score_semantics passthrough added byroot; graph/training/weights/thresholds unchanged; original9980protocol/fixtures/source snapshot preserved',
            'proposal_sha256':PROPOSAL_SHA,'supersedes_proposal_architecture_details':'Concrete max-scaled feature normalization and representable power-of-two floors below; first proposal preserved',
            'teacher':teacher,'student_initialization':'Separate random MobileNetV3Small with seed20261030; never copied teacher',
            'input_shape':[1,3,1280,512],'output_shape':[1,1,640,256],
            'architecture':'Two independent MobileNetV3Small first7blocks; taps0/3/6 channels16/24/40; fixed bounded unit-feature L2 disagreements and equal log-geometric mean; no C model/density/OR/GT/ROI',
            'norm_scale_floor':NORM_SCALE_FLOOR,'norm_squared_floor':NORM_SQUARED_FLOOR,'map_floor':MAP_FLOOR,
            'score':'.25*sum((normalized_teacher-normalized_student)^2),clamp[0,1] perlevel; bilinear middle/coarse maps tostride2fine; exp(mean(log(clamp(each,map_floor,1)))); uncalibrated anomaly disagreement, NOT defect probability',
            'normalization':'BGRuint8→RGBfloat/255→ImageNetmean/std; vector unit uses maxabs scaling floor2^-8 then sumsquare floor2^-12 before sqrt, keeping half reductions finite',
            'split_items':old['split_items'],'supported_height_width_ratio':[2.3,2.7],
            'normal_sources':normals,'normal_source_identity_sha256':NORMAL_SHA,
            'normal_input_verification':'All208 image+maskSHA beforefit andafterfit, decodeGT-empty; no31positive sources/noSDD2/noCAL/nooriginaltraining',
            'fit':{'steps':600,'save_at':SAVE_AT,'cpu_threads':1,'batch':1,'optimizer':'AdamW','lr':.001,'weight_decay':.00001,
                   'sampling':'Shuffle all208 without replacement eachcycle with protectedNumPyseed;600steps cover every source twice and184 third visits',
                   'crop':'Resize actual source to512×1280 INTER_AREA once; identical teacher/student native640H×512W fullwidth crop;20% fixededge coverage (.1topY0,.1bottomY640),otherwiseuniform integerY0..640; no warp/flip/90/inserted defects',
                   'loss':'Mean overthree levels of .5*mean_spatial(sum_channels((normalizedT-normalizedS)^2)); actualnormal correspondence only',
                   'teacher_policy':'Allweights frozen; fullteacher eval includingBN always; strict officialsource loading',
                   'student_policy':'Randomconv,torchdefaultBNaffineones/zeros; studenttrain with normal-onlyBatchNorm runningstatistics defaultmomentum.1; exporteval frozenactualBN atcheckpoint; featurepath has no dropout',
                   'schedule':'Fixed600,noearlystop,save200/400/600,all3mandatory; finalcomplete receipt andteacherunchanged guard beforefreeze'},
            'preflight':{'source_rows':normals[:16],'warmup_calls':4,'measured_calls':64,'minimum_serial_calls_per_second':15,
                         'policy':{'high':.05,'area':4,'growth':None},'shape':[1,3,1280,512],
                         'numeric_max_abs_error':.01,'numeric_inputs':'firstpinnedTRAINnormal,zero,all255 initialization; noaccuracyclaim',
                         'timing':'Rootrelease+allprojectNNquiet; actualnonconstanttwoencoderfullFactoryProfile; first64tails preserved; allcallsgraph/noabstain; no published/previousspeedtransfer',
                         'failure':'Stopbeforefit oncapacity/parity/nonfinite/geometry/sourcefailure; preservefirstreceipt; noautomaticresize/substitution'},
            'save_at':SAVE_AT,'raw_high_grid':HIGH_GRID,'area_grid':AREA_GRID,'growth_grid':GROWTH_GRID,
            'selection_rank':RANK,'seed_search':'For eachhigh FP<=1 seed-best-area byweakestpositive rank,stablearea ties; top6 distincthighs bystablefixedorder; unchangedreferencegrowth; finalFP<=1/70 ANDpixelprecision>=.65',
            'calibration_source_identity_sha256':CAL_SHA,'calibration':'All3 actualsingle-outputGPU+fixedTRAIN/zero/255 numericguards BEFOREsameordered80CAL; scorecache boundmodel/protocol/source/nativeGTgeometrySHA; no originalscoring byowner',
            'comparison_gate':'All3CAL + complete600 + unchangedteacher/source/BN + globalwinnerfreeze required; rootONLYfrozenwinneroriginalaudit; exposeddevelopment notblindtest',
            'final_performance':'Fresh≥15 fittedwinner fullprofile+pairedsustainedruntime afterallfits; preflight is necessary notfinalspeedqualification',
            'template_metadata_sha256':TEMPLATE_SHA,'fixture_sha256':sha(fixture),'implementation_pins':pins,
            'official_sdd2_test_still_sealed':True,'original_pixels_used_for_fit_or_selection':0}


def protocol_path():return OUT/'protocol-v2-score-semantics-before-any-NN.json'


def verify_protocol():
    path=protocol_path();value=json.loads(path.read_text(encoding='utf-8'))
    if value!=protocol_value():raise ValueError('Protected L protocol/source changed')
    return path,value


def model_path(step):return MODELS/f'{NAME}-step{step}.onnx'


def metadata(path,protocol,point=None):
    policy=point or protocol['preflight']['policy']
    return {'feature_version':'surface_mobilenet_seg_v1','profile_version':NAME,
            'model_name':'EXPERIMENTAL normal-only multiscale student-teacher surface review',
            'model_sha256':sha(path),'output_logits':False,'input_shape':protocol['input_shape'],'output_shape':protocol['output_shape'],
            'input_width':512,'input_height':1280,'model_grid_width':256,'model_grid_height':640,
            'pixel_threshold':policy['high'],'minimum_component_pixels':policy['area'],'hysteresis_growth_threshold':policy['growth'],
            'minimum_component_area_units':'256×640 gridcells; unchanged exactsource projection',
            'score_meaning':protocol['score'],'score_semantics':SCORE_SEMANTICS,'architecture':protocol['architecture'],'split_items':protocol['split_items'],
            'supported_height_width_ratio':protocol['supported_height_width_ratio'],'training_protocol':protocol,
            'teacher_source_sha256':protocol['teacher']['sha256'],'dataset':'KolektorSDD',
            'dataset_url':'https://www.vicos.si/resources/kolektorsdd/',
            'license':'Official torchvision teacher code BSD-style; ImageNet pretrained terms apply; derived Kolektor model/data CC-BY-NC-SA-4.0 research/non-commercial prototype',
            'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False,'field_validated':False,
            'official_sdd2_test_still_sealed':True}
