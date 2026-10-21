"""Protected C200-density identities/protocol. Stdlib only; no model/data decode."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'
NAME = 'spatial_round3k_c200_density_f16'
SOURCE = ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.pt'
SOURCE_ONNX = SOURCE.with_suffix('.onnx')
SOURCE_FREEZE = OUT / 'winner-attempt3-phase-max-freeze-before-comparison.json'
SOURCE_PINS = {
    SOURCE: 'ff3b5571295e80978b4be873b6cce24b7cb947ae25415959a4121337dfe22112',
    SOURCE_ONNX: 'ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318',
    SOURCE_ONNX.with_suffix('.onnx.json'): '6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb',
    SOURCE_FREEZE: 'c3c2a6d86f86d625f9f6499c4d7477a25c2b9610b26a870be994a29ac969db40',
}
CANDIDATES = [(1., 0.)] + [(a, b) for a in (.5, 1.) for b in (1., 2., 4., 8.)]
IMPLEMENTATIONS = [
    'factory_c_density_common.py', 'factory_c_density_graph.py', 'prepare_factory_c_density.py',
    'fit_factory_c_density.py', 'factory_c_density_identity.py', 'factory_c_density_math.py',
    'check_factory_c_density_parity.py', 'probe_factory_c_density.py',
    'audit_factory_c_density_clipping.py', 'calibrate_factory_c_density.py', 'freeze_factory_c_density.py',
    'check_factory_c_density_preparation.py',
    'train_factory_round3d.py', 'train_factory_round3c.py', 'train_factory_round3.py', 'train_factory_spatial_unet.py',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    for path, expected in SOURCE_PINS.items():
        if sha(path) != expected:
            raise ValueError(f'Immutable C200 source identity changed: {path}')


def validate_fusion_plan(value):
    expected=[{'id':i,'alpha':a,'beta':b} for i,(a,b) in enumerate(CANDIDATES)]
    if value['candidates']!=expected or value['save_at']!=list(range(9)):
        raise ValueError('Exactly ordered control plus eight fixed signed fusions required')
    if value['input_shape']!=[1,3,1280,512] or value['output_shape']!=[1,1,640,256]:
        raise ValueError('C semanticpool/fine alignment changed')


def protocol():
    verify_source()
    fixture_path=OUT/(NAME+'-preparation-fixtures.json')
    fixture=json.loads(fixture_path.read_text(encoding='utf-8'))
    if not fixture['all_checks_passed']:
        raise ValueError('Pure numeric/AST preparation fixtures must pass before protocol')
    for relative,expected in fixture['source_pins'].items():
        if sha(ROOT/relative)!=expected:
            raise ValueError('Preparation fixtures are stale after a source change')
    meta = json.loads(SOURCE_ONNX.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    # Only this already established TRAIN identity list is reused. No feature
    # moments, bank, q values, graph, threshold, or speed comes from F/F2.
    receipt = OUT / 'spatial_round3i_normal_density_f16-normal-fit.json'
    established = json.loads(receipt.read_text(encoding='utf-8'))
    normals = established['normal_sources']
    identity = hashlib.sha256(json.dumps(normals, sort_keys=True).encode()).hexdigest()
    if len(normals) != 208 or identity != established['normal_source_identity_sha256']:
        raise ValueError('Established208-normal TRAIN identity receipt changed')
    inventory_path=OUT/'spatial_round3j_native_curvature-pairing-fixtures.json'
    if sha(inventory_path)!='9805e7f858c465e4c081853953a1c203f8ce5ba297900dbfc68e3c96882396c2':
        raise ValueError('Independent original239 TRAIN image/mask inventory changed')
    inventory=json.loads(inventory_path.read_text())
    empty={row['image']:row for row in inventory['train_sources'] if row['components']==0}
    if len(empty)!=208 or set(empty)!={row['image'] for row in normals}:
        raise ValueError('Normal list differs from independent GT-empty TRAIN inventory')
    if any(row['image_sha256']!=empty[row['image']]['image_sha256'] or row['mask_sha256']!=empty[row['image']]['mask_sha256'] for row in normals):
        raise ValueError('Normal data hashes differ from independent original source inventory')
    train = set(meta['split_items']['train'])
    if len(train) != 30 or any(row['defective'] or Path(row['image']).parts[0] not in train for row in normals):
        raise ValueError('Normal source manifest is not exclusively original TRAIN')
    value = {
        'name': NAME, 'seed': 20261026, 'preparation_only': True,
        'preparation_fixtures_sha256':sha(fixture_path),
        'execution_authorization': 'Protocol preparation does not authorize export/preflight/bankfit/CAL; coordinator must separately schedule execution after current benchmark and constructionGPU windows',
        'architecture': 'Frozen C200 context+packed-phase refiner, originalfullreslogits thenMaxPool2d2; same-forward stride2fine16ch signed normal-density fusion AFTER semantic pool; single-serving-output',
        'warmstart': str(SOURCE), 'warmstart_sha256': SOURCE_PINS[SOURCE],
        'source_onnx': str(SOURCE_ONNX), 'source_onnx_sha256': SOURCE_PINS[SOURCE_ONNX],
        'source_metadata_sha256': SOURCE_PINS[SOURCE_ONNX.with_suffix('.onnx.json')],
        'source_freeze': str(SOURCE_FREEZE), 'source_freeze_sha256': SOURCE_PINS[SOURCE_FREEZE],
        'source_policy': {'high': meta['pixel_threshold'], 'area': meta['minimum_component_pixels'], 'growth': meta.get('hysteresis_growth_threshold')},
        'split_items': meta['split_items'], 'input_shape': [1,3,1280,512], 'output_shape': [1,1,640,256],
        'fine_steps': 0, 'save_at': list(range(9)), 'candidate_id_is_not_sgd_step': True,
        'candidates': [{'id': i, 'alpha': a, 'beta': b} for i,(a,b) in enumerate(CANDIDATES)],
        'normal_sources': normals, 'normal_source_identity_sha256': identity,
        'normal_identity_receipt_only': str(receipt), 'normal_identity_receipt_sha256': sha(receipt),
        'independent_train_source_inventory':str(inventory_path),'independent_train_source_inventory_sha256':sha(inventory_path),
        'normal_fit': 'Exactly208 actual GT-empty originalSDD1 TRAIN views/30items; verifyall239/31positive liveTRAIN then exact208sourceSHA identity; no SDD2/CAL/comparison fit',
        'feature_fit': 'Only newC200 full semantic+density3-output pre-fit diagnostic fine features; extractor is byte-identical copy retaining ALL outputs; initial exportmanifest binds both hashes equal. No firstblock-only/pruned extraction or F2bank/feature reuse.',
        'numeric_variant': 'C200 with F2 exact-half constants and full compiler-context statistics; freshly fitted bank only',
        'moments': 'Float64 mu64/EffT; deployedmuq nearesthalf; residualcov=EffT-mean_x(mu64muqT+muqmu64T-muqmuqT)',
        'covariance': '.9cov+.1diag(maxdiag0);symmetrize;eigenfloor max1e-6,.001mediandiag;globalWrowL1<=8;Wmagnitudes roundedtowardzero half BEFOREquantilepass',
        'numeric_guards': 'centered+-32;white+-32;mean-square16;logmaxD1e-6;normalize exactstoredhalfq95/independentqspan>=.001;signedevidence+-8;finiteguards;fixednotretuned',
        'normal_quantiles': 'Same208normal-onlysecondpass;2048pixels/viewwithoutreplacement;seed20261026;425984samples;rawq50/q95 thenexplicitq95/qspan nearesthalf;graphreferenceusedeployedmuq/Wq/q95/qspan',
        'fusion': 'R=alpha*MaxPool2d2(originalCfull-resolutionsemantic+phase-residual logits)+beta*signeddensity(fine);oneencoderpass;noOR/maskunion/fallback/ROI;sigmoidonlyafterfusion',
        'selection_gate': 'NormalcalFP<=1/70 and aggregateforegroundprecision>=.65',
        'selection_rank': ['meaningfulIoU>=.1viewrecall','worst_positive_IoU','coverage50pct_viewrecall','worst_positive_GTcoverage','microIoU','fewerNormalFP','foregroundrecall'],
        'raw_high_grid': [.05,.1,.25,.5,.7,.9,.95,.98,.99,.995,.999,.9999,.99999,.999999],
        'area_grid': [1,2,4,8,16,32], 'growth_grid': [None,.05,.1,.25,.5,.7,.8,.9,.95,.98,.99],
        'seed_search': 'FP-onlyeligiblebestarea/high;weakestpositive rank;top6distincthighs;finalFP+precisionguard;heuristicnotexhaustive',
        'calibration': 'ALL9 actualsingle-outputGPUgraphs andexactordered80CALsourcepixel/maskgeometrySHA;noCPUfusionsubstitute;knownsourceidentity7b09d8cbf39129dad144fcad6ecbad0ade0bb19e39de6e8fbdfc634eb6ed6d07',
        'comparison_gate': 'All9CALmandatory;globalfreeze;eligiblecontrolwinner or equalrankREJECT;noncontrolmuststrictlyimproverankovereligiblecontrol;controlineligible+fusioneligible counts stricteligibilitygain;no losing originalscore',
        'capacity_gate': 'NEW nonconstantdensehead full-profile>=15calls/s BEFOREbankfit inprojectquietauthorizedslot AFTERconstructionfit/currentbenchmark;NOF/F2/Cspeedreceipttransfer. Finalnewfittedwinnerpairedcapacity+realruntime>=15mandatoryafterallfits/probesfinish.',
        'parity_gate': 'ExactCpooledTorchrawpath;actualGPUcontrolsourceallclose1e-3 andfrozenCpolicy maskidentity onfixedTRAIN;fullcontextfine/evidence NumPyexactstoredconstantsmaxerror<=.1;candidate7singlevsdiagallclose1e-3;finiteTRAIN/zero;failbeforeCAL',
        'control': 'id0alpha1beta0;Ccontext/refiner/BNcompleteimmutable;nooptimizer/BNupdates',
        'clipping_diagnostics': 'FixedTRAIN/CALguardobservationsonlyAFTERfittedparitypass;bank/guardsunchanged;nohyperparamselection',
        'implementation_pins': {f'scripts/{p}': sha(ROOT/'scripts'/p) for p in IMPLEMENTATIONS},
        'comparison_images_used_for_fit_selection': 0, 'official_sdd2_test_still_sealed': True,
        'hypothesis_scope': 'Iterative exposed-development strategy;notblindtest;no originalpixels/IDs/ROIs/masks or filename-conditioned rules;TRAINstats only',
    }
    validate_fusion_plan(value)
    return value


def record_protocol():
    value = protocol()
    path = OUT / (NAME + '-protocol-before-fit.json')
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Refusing to overwrite protected C-density protocol')
    if not path.exists():
        path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    return path, value


def verify_protocol():
    path = OUT / (NAME + '-protocol-before-fit.json')
    value = json.loads(path.read_text(encoding='utf-8'))
    verify_source()
    validate_fusion_plan(value)
    if (Path(value['warmstart']).resolve()!=SOURCE.resolve() or value['warmstart_sha256']!=SOURCE_PINS[SOURCE]
            or Path(value['source_onnx']).resolve()!=SOURCE_ONNX.resolve() or value['source_metadata_sha256']!=SOURCE_PINS[SOURCE_ONNX.with_suffix('.onnx.json')]
            or Path(value['source_freeze']).resolve()!=SOURCE_FREEZE.resolve() or value['source_freeze_sha256']!=SOURCE_PINS[SOURCE_FREEZE]):
        raise ValueError('Protected C PT/ONNX/metadata/freeze binding changed')
    for relative, expected in value['implementation_pins'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Protected implementation changed: {relative}')
    if value['source_onnx_sha256'] != SOURCE_PINS[SOURCE_ONNX] or value['candidates'] != [{'id':i,'alpha':a,'beta':b} for i,(a,b) in enumerate(CANDIDATES)]:
        raise ValueError('Source/fixed fusion declaration changed')
    return path, value
