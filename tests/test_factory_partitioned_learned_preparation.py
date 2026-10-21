"""Pure learned partitioned control/identity tests; no transforms or NN."""
import ast,copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import prepare_factory_partitioned_learned as p
from freeze_factory_partitioned_learned import validate_terminal,validate_CAL_config
from unittest.mock import patch

def audit():
    values={'EXECUTION_DEVICES':['GPU.0'],'INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True,'NUM_STREAMS':'1','PERFORMANCE_HINT':'LATENCY'}
    return {'actual_configuration_passed':True,'properties':{k:{'supported':True,'value':v} for k,v in values.items()}}
def report(step=200,passed=True):
    return {'step':step,'fixed_cases_attempted':3,'cases':[{'numeric_failure':None,'max_abs_error':.001 if passed else .02,'score_range':[.1,.5]} for _ in range(3)],
        'compiled_provenance':audit(),'all_checks_passed':passed}

class LearnedPartitionPreparation(unittest.TestCase):
    def test_receipts_cannot_inherit_stale_fixture_or_CAL_property_PASS(self):
        fixture={'all_checks_passed':True,'tests':7,'source_pins':{'some.py':'old'}}
        with patch.object(p,'sha',return_value='old'):p.check_fixture(fixture)
        with patch.object(p,'sha',return_value='new'):
            with self.assertRaises(ValueError):p.check_fixture(fixture)
        validate_CAL_config(audit())
        corrupt=audit();corrupt['properties']['INFERENCE_PRECISION_HINT']['value']='f32'
        with self.assertRaises(ValueError):validate_CAL_config(corrupt)

    def test_numeric_checker_rejects_negative_bool_malformed_nonfinite_and_recipe(self):
        valid=report();self.assertTrue(p.numeric_acceptance(valid['cases'],valid['compiled_provenance']))
        for change in [{'max_abs_error':-1},{'max_abs_error':True},{'max_abs_error':float('nan')},{'score_range':[.1]},
                       {'score_range':[False,.5]},{'score_range':None},{'numeric_failure':'L9 encoder feature is nonfinite'}]:
            cases=copy.deepcopy(valid['cases']);cases[0].update(change);self.assertFalse(p.numeric_acceptance(cases,audit()))
        bad=audit();bad['properties']['GPU_DISABLE_WINOGRAD_CONVOLUTION']['value']=False
        self.assertFalse(p.numeric_acceptance(valid['cases'],bad))

    def test_all3_terminal_numericfail_zero_CAL_and_pass_requires80(self):
        validate_terminal({'step':200,'status':'numeric_failed_no_CAL','CAL_inputs_opened':0},report(passed=False))
        validate_terminal({'step':200,'status':'calibration_complete','CAL_inputs_opened':80,'calibration_sha256':'x'},report())
        for terminal,numeric in [({'step':200,'status':'numeric_failed_no_CAL','CAL_inputs_opened':1},report(passed=False)),
            ({'step':200,'status':'numeric_failed_no_CAL','CAL_inputs_opened':0},report()),
            ({'step':200,'status':'calibration_complete','CAL_inputs_opened':80,'calibration_sha256':'x'},report(passed=False))]:
            with self.assertRaises(ValueError):validate_terminal(terminal,numeric)

    def test_pre_NN_amendment_fixed_all3_ownrefs_metadata_bound(self):
        value=p.proposal_value()
        self.assertEqual(value['checkpoints'],[200,400,600]);self.assertFalse(value['initial_pass_required'])
        self.assertEqual([r['step'] for r in value['source_bindings']['all_three']],[200,400,600])
        self.assertEqual(set(value['source_metadata']),{'200','400','600'})
        self.assertEqual(len(value['split_items']['calibration']),10)
        self.assertEqual(value['calibration_source_identity_sha256'],value['CAL_policy']['CAL_source_identity_sha256'])
        self.assertEqual(value['raw_high_grid'],value['CAL_policy']['raw_high_grid'])
        self.assertEqual(value['area_grid'],value['CAL_policy']['area_grid']);self.assertEqual(value['growth_grid'],value['CAL_policy']['growth_grid'])

    def test_transform_reuses_exact_fixed_extractor_no_optimizer_or_reference_reexport(self):
        source=(ROOT/'scripts/prepare_factory_partitioned_learned.py').read_text();tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transform_all');text=ast.unparse(fn)
        self.assertIn('for step in STEPS:',text);self.assertIn('extract(onnx.load(str(source)), feature_rows())',text)
        identifiers={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
        imports={a.name for n in ast.walk(fn) if isinstance(n,ast.Import) for a in n.names}
        self.assertNotIn('torch',identifiers|imports);self.assertNotIn('optimizer',identifiers);self.assertNotIn('.infer(',text)
        self.assertIn("source_metadata() != protocol['source_metadata']",text)

    def test_all3_numeric_before_CAL_and_mixedbackend_truthful(self):
        text=(ROOT/'scripts/calibrate_factory_partitioned_learned.py').read_text();body=text[text.index('def calibration('):text.index('def build_profile(')]
        self.assertLess(body.index('for required in STEPS:'),body.index('for image,truth,defective in samples'))
        self.assertLess(body.index('Only exact CAL80'),body.index('model=adapter_type'))
        self.assertIn('GPU raw features + CPU NumPyfloat32 score head',body);self.assertNotIn('actual_GPU_maps',body)

    def test_global_freezer_requires_all3_and_capacity_before_original(self):
        text=(ROOT/'scripts/freeze_factory_partitioned_learned.py').read_text()
        self.assertIn('for step in STEPS:',text);self.assertIn('validate_search_report(r,protocol)',text)
        self.assertIn("'accepted_for_independent_audit':False",text);self.assertIn('BEFORE originalaudit',p.proposal_value()['final_performance'])
        self.assertIn("numeric['reference_sha256']!=sha(ref)",text)
        self.assertIn("numeric['initial_metadata_sha256']!=t['initial_metadata_sha256']",text)

if __name__=='__main__':unittest.main()
