"""Source-only L9 learned plan; transforming all3 needs separate root approval."""
import argparse,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_factory_stfpm_partition import extract,feature_rows,write_new,sha
from record_factory_partitioned_method_amendment import verify_amendment,AMENDMENT,PROTOCOL as INITIAL_PROTOCOL
from probe_factory_stfpm_partition import learned_sources,verify_protocol as verify_initial_protocol
from factory_stfpm_learn_common import require_complete_training as require_L7_training,verify_protocol as verify_L7_protocol,gpu_configuration_passes

OUT=ROOT/'output/optimization/factory-round3l9-learned-partitioned'
NAME='spatial_round3l9_learned_partitioned';STEPS=[200,400,600]
L7=ROOT/'output/optimization/factory-round3l7-learned'
PROTOCOL=OUT/'protocol-before-all3-transforms.json'
SCRIPTS=['scripts/prepare_factory_partitioned_learned.py','scripts/calibrate_factory_partitioned_learned.py',
 'scripts/freeze_factory_partitioned_learned.py','tests/test_factory_partitioned_learned_preparation.py']

def source_path(step):return ROOT/f'models/factory/spatial_round3l7_learned_stfpm-step{step}.onnx'
def model_path(step):return ROOT/f'models/factory/{NAME}-step{step}.onnx'

def source_metadata():
    rows={}
    for step in STEPS:
        source=source_path(step);meta=source.with_suffix('.onnx.json')
        export=L7/f'spatial_round3l7_learned_stfpm-step{step}-export.json';e=json.loads(export.read_text())
        if e['model_sha256']!=sha(source) or e['initial_metadata_sha256']!=sha(meta):raise ValueError('Original L7 export/metadata drift')
        rows[str(step)]={'metadata_sha256':sha(meta),'export_receipt_sha256':sha(export)}
    return rows


def proposal_value():
    initial=verify_initial_protocol();amendment=verify_amendment();_,training=verify_L7_protocol();require_L7_training(training)
    bindings=learned_sources()
    if amendment['source_bindings']!=bindings or amendment['checkpoints']!=STEPS:raise ValueError('Exact predeclared all3 learned source identities required')
    pins={**initial['source_pins'],**amendment['source_pins']}
    for path in SCRIPTS+['scripts/calibrate_factory_stfpm_learn.py','scripts/factory_stfpm_learn_common.py',
                       'scripts/factory_mask_metrics.py','scripts/train_factory_surface.py']:
        pins[path]=sha(ROOT/path)
    value={'name':NAME,'initial_protocol_sha256':sha(INITIAL_PROTOCOL),'pre_NN_amendment_sha256':sha(AMENDMENT),
        'source_bindings':bindings,'source_metadata':source_metadata(), 'checkpoints':STEPS,'initial_pass_required':False,'separate_root_release_required':True,
        'graph_transform':'Same6rawteacher/student0/3/6 deterministicancestors, retainallnode/initializer/input/opsetbytes; sameNumPyf32head; nofit/normalstats/outputsubset/configsearch',
        'GPU_recipe':initial['GPU_recipe'],'ordered_outputs':feature_rows(),'CPU_head':initial['CPU_head'],
        'numeric':'ALL3 learnedcheckpoints, ownL7Torchscores withbyteidenticalL4inputs; actualGPUrecipe finiteboundednonconstant maxabs<=.01. Numericfail terminalwithzeroCAL; unrelatederrorshardstop',
        'CAL_policy':amendment['CAL_policy'],'split_items':training['split_items'],
        'raw_high_grid':training['raw_high_grid'],'area_grid':training['area_grid'],'growth_grid':training['growth_grid'],
        'selection_rank':training['selection_rank'],'calibration_source_identity_sha256':training['calibration_source_identity_sha256'],'selection':'Samefixed80CAL/grids/rank, ALL3 terminaloutcomes beforeglobalfreeze. RootONLYwinneroriginalaudit, exposeddevelopmentnotblind',
        'maps_backend':'ActualGPUrawfeatures plus NumPyCPUfloat32head, mixedexecution; never labeled GPU-only scoremaps',
        'final_performance':'Rootfreshfittedpolicy>=15completeprofile REQUIRED BEFORE originalaudit; sustained16FPSendtoend+allcounters mandatory beforeacceptance; initialspeednotqualification',
        'source_pins':pins,'new_training_or_BN_updates':0,'official_SDD2_test_still_sealed':True}
    fixture=OUT/'preparation-fixtures.json'
    if fixture.exists():
        check_fixture(json.loads(fixture.read_text()))
        value['fixture_sha256']=sha(fixture)
    return value

def check_fixture(fixture):
    if fixture.get('all_checks_passed') is not True or fixture.get('tests')!=7:raise ValueError('Preparation fixtures failed or incomplete')
    for rel,h in fixture['source_pins'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Preparation fixture source drift')

def verify_protocol():
    fixture=json.loads((OUT/'preparation-fixtures.json').read_text())
    check_fixture(fixture)
    value=json.loads(PROTOCOL.read_text(encoding='utf-8'));proposal=OUT/'proposed-protocol-before-all3-transforms.json'
    decision=OUT/'parent-approved-transform-decision.json';approved=json.loads(decision.read_text(encoding='utf-8'))
    if value!=proposal_value() or value!=json.loads(proposal.read_text(encoding='utf-8')):raise ValueError('Protected learned partition source/proposal changed')
    if approved.get('approved_all3_transform_numeric_CAL') is not True or approved['proposal_sha256']!=sha(proposal):raise ValueError('Separate exactall3 root approval required')
    return value

def numeric_acceptance(cases,audit):
    finite=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    if (not isinstance(cases,list) or len(cases)!=3 or not isinstance(audit,dict)
            or audit.get('actual_configuration_passed') is not True or not gpu_configuration_passes(audit.get('properties',{}))):return False
    for row in cases:
        if not isinstance(row,dict) or row.get('numeric_failure') is not None:return False
        error=row.get('max_abs_error');bounds=row.get('score_range')
        if not finite(error) or not 0<=error<=.01 or not isinstance(bounds,list) or len(bounds)!=2:return False
        if not all(finite(x) for x in bounds) or not 0<=bounds[0]<=bounds[1]<=1:return False
    return cases[0]['score_range'][1]-cases[0]['score_range'][0]>=1e-7


def transform_all(protocol):
    import onnx
    started=OUT/'transform-all3-started.json'
    if started.exists():raise ValueError('Preserve any partial all3 transformation')
    write_new(started,{'protocol_sha256':sha(PROTOCOL),'checkpoints':STEPS,'new_NN_calls':0})
    archive=OUT/'executed-source-snapshot-before-transform';records=[]
    for rel,h in protocol['source_pins'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Source drift before transform')
        target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/rel).read_bytes());records.append({'path':rel,'sha256':h})
    write_new(archive/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'sources':records})
    transformed=[]
    for step in STEPS:
        if source_metadata()!=protocol['source_metadata']:raise ValueError('Source metadata changed immediately before transform')
        path=model_path(step);receipt=OUT/f'checkpoint{step}-transform.json'
        if any(p.exists() for p in [path,path.with_suffix('.onnx.json'),receipt]):raise ValueError('Preserve partial transformed model')
        source=source_path(step);graph,proof=extract(onnx.load(str(source)),feature_rows())
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(graph.SerializeToString())
        meta=json.loads(source.with_suffix('.onnx.json').read_text(encoding='utf-8'))
        meta.update(model_sha256=sha(path),profile_version=NAME,architecture_probe_only=True,all_outputs_retained=6,
            partitioned_score_head='GPUfeatures_plus_CPU_Numpy_float32',partitioned_protocol_sha256=sha(PROTOCOL))
        write_new(path.with_suffix('.onnx.json'),meta)
        ref=L7/f'spatial_round3l7_learned_stfpm-step{step}-torch-reference.npz'
        write_new(receipt,{'step':step,'protocol_sha256':sha(PROTOCOL),'source_model_sha256':sha(source),
            'model_sha256':sha(path),'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),
            'source_PT_sha256':sha(source.with_suffix('.pt')),'reference_sha256':sha(ref),'extraction':proof,'NN_executed':False})
        transformed.append({'step':step,'receipt_sha256':sha(receipt)})
    write_new(OUT/'all3-transforms-complete.json',{'protocol_sha256':sha(PROTOCOL),'checkpoints':transformed,'new_training_or_BN_updates':0,'NN_executed':False})

def record_fixtures():
    import unittest,io,time,ast
    target=OUT/'preparation-fixtures.json'
    if target.exists():raise ValueError('Preserve first preparation receipt')
    for rel in SCRIPTS:ast.parse((ROOT/rel).read_text())
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_factory_partitioned_learned_preparation.py')
    log=io.StringIO();tick=time.perf_counter();result=unittest.TextTestRunner(log,verbosity=2).run(suite)
    write_new(target,{'all_checks_passed':result.wasSuccessful() and result.testsRun==7,'tests':result.testsRun,
        'elapsed_seconds':time.perf_counter()-tick,'test_log':log.getvalue(),'source_pins':proposal_value()['source_pins'],
        'NN_executed':False,'model_transformations':0,'source_image_decodes':0})
    if not result.wasSuccessful():raise ValueError('Pure learned preparation failed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['fixtures','proposal','record-protocol','transform-all3'],required=True)
    parser.add_argument('--root-released',action='store_true');a=parser.parse_args()
    if a.stage=='fixtures':record_fixtures();return
    if a.stage in ['proposal','record-protocol'] and json.loads((OUT/'preparation-fixtures.json').read_text())['all_checks_passed'] is not True:raise ValueError('Passing pure fixtures required')
    if a.stage=='proposal':write_new(OUT/'proposed-protocol-before-all3-transforms.json',proposal_value());return
    if a.stage=='record-protocol':
        proposal=OUT/'proposed-protocol-before-all3-transforms.json';decision=json.loads((OUT/'parent-approved-transform-decision.json').read_text())
        value=proposal_value()
        if decision.get('approved_all3_transform_numeric_CAL') is not True or decision['proposal_sha256']!=sha(proposal) or value!=json.loads(proposal.read_text()):raise ValueError('Root exactproposal approval required')
        write_new(PROTOCOL,value);return
    if not a.root_released:raise ValueError('Root must separately release actual transformations')
    transform_all(verify_protocol())

if __name__=='__main__':main()
