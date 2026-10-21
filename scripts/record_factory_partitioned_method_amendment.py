"""L9 additive PRE-NN method amendment; preserve initial protocol/source history."""
import copy,hashlib,json,sys,unittest,io,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from probe_factory_stfpm_partition import OUT,PROTOCOL,verify_protocol,sha,write_new,learned_sources
BASE_SHA='522a1f549ca08e964fa46c2787f32370da0b4787dba3fe1ab3809efc1f87b32c'
SNAPSHOT=OUT/'pre-NN-source-snapshot-v1/snapshot-manifest.json'
SNAPSHOT_SHA='bf4c90466de80cb083144f2a338155b7e41bc3ceefe407de7d5e18c3180a8fe9'
AMENDMENT=OUT/'pre-NN-learned-investigation-amendment-v2.json'
FIXTURE=OUT/'method-amendment-v2-fixtures.json'


def source_pins():
    paths=['scripts/record_factory_partitioned_method_amendment.py','tests/test_factory_partitioned_method_amendment.py']
    return {p:sha(ROOT/p) for p in paths}


def value():
    if sha(PROTOCOL)!=BASE_SHA or sha(SNAPSHOT)!=SNAPSHOT_SHA:raise ValueError('Original initial source history changed')
    base=verify_protocol()
    return {'version':2,'parent_initial_protocol_sha256':BASE_SHA,'parent_source_snapshot_sha256':SNAPSHOT_SHA,
        'method_decision':'Root declared before ANY L9 NN/result; separately investigate all3 already-trained weights even if initial BN-only prototype numeric/capacity fails.',
        'scope_overridden':'ONLY conditional_ALL3_learned.status requiring initial prototype pass; initial prototype recipe/results/gates remain unchanged.',
        'initial_probe_unchanged':True,'initial_failures_preserved':True,'initial_pass_required_for_learned_investigation':False,
        'learned_requires_separate_root_release':True,'checkpoints':[200,400,600],'source_bindings':learned_sources(),
        'fixed_transform':'Same fixed six raw feature interface/ancestor extraction/all retained weights/nodes/inputnorm bytes, CPUf32 head and GPUrecipe for ALL3 once. No subset/config/weight/ROI search; no new fit.',
        'complete_all3_numeric_before_selection':True,'numeric_failed_checkpoint_CAL_inputs':0,'numeric_max_abs_error':.01,
        'passing_only_CAL':True,'CAL_policy':{k:base['conditional_ALL3_learned'][k] for k in ['CAL_source_identity_sha256','raw_high_grid','area_grid','growth_grid','ranking','guards']},
        'freeze':'All3 terminal numericfailure-or-fullfixedCAL outcomes mandatory; rootONLY one frozenwinner original; no outcome-driven model/interface/threshold changes.',
        'final_fitted_mean_calls_per_second_minimum':15,'final_sustained_input_FPS':16,
        'fresh_fitted_capacity_before_original_if_declared':True,'production_acceptance_requires_numeric_accuracy_and_fitted_speed':True,
        'source_pins':source_pins(),'recorded_before_ANY_L9_NN':True,'new_NN_calls':0}


def validate(amendment):
    expected=value()
    if amendment!=expected:raise ValueError('Method amendment/protected guards/source differs')
    return True


def main():
    if AMENDMENT.exists() or FIXTURE.exists():raise ValueError('Preserve additive amendment/fixture history')
    # This assertion is performed only at record time. Later numeric receipts may
    # exist; validating the immutable amendment must not erase that chronology.
    if any((OUT/name).exists() for name in ['numeric-attempt.json','initial-three-case-numeric.json','capacity-attempt.json','initial-fullprofile-capacity.json']):raise ValueError('Amendment must precede any L9 NN')
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_factory_partitioned_method_amendment.py')
    log=io.StringIO();start=time.perf_counter();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    receipt={'all_checks_passed':result.wasSuccessful() and result.testsRun==3,'tests':result.testsRun,'elapsed_seconds':time.perf_counter()-start,
        'source_pins':source_pins(),'test_log':log.getvalue(),'NN_executed':False}
    write_new(FIXTURE,receipt)
    if not receipt['all_checks_passed']:raise ValueError('Amendment pure fixtures failed')
    amendment=value();validate(amendment);amendment['fixture_sha256']=sha(FIXTURE)
    write_new(AMENDMENT,amendment)
    print(json.dumps({'initial_protocol_sha256':BASE_SHA,'amendment_sha256':sha(AMENDMENT),'fixture_sha256':sha(FIXTURE),'NN_executed':False},indent=2))


def verify_amendment():
    amendment=json.loads(AMENDMENT.read_text());fixture=amendment.pop('fixture_sha256')
    if fixture!=sha(FIXTURE):raise ValueError('Method fixture changed')
    validate(amendment)
    return amendment


if __name__=='__main__':main()
