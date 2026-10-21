"""Pure L7 outcome/source contracts; no Torch import, model or pixel decode."""
import ast,copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import factory_stfpm_learn_common as c
from freeze_factory_stfpm_learn import validate_terminal

class LearnPreparation(unittest.TestCase):
    def numeric(self,step,passed=True):
        props={'EXECUTION_DEVICES':['GPU.0'],'INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True,'NUM_STREAMS':'1','PERFORMANCE_HINT':'LATENCY'}
        return {'step':step,'all_checks_passed':passed,'max_abs_errors':[.001 if passed else .02]*3,'fixed_cases_attempted':3,
            'actual_score_ranges':[[.1,.5]]*3,'all_finite':True,'all_bounded_0_1':True,'reference_inputs_byte_identical_to_L4':True,
            'reference_scores_finite_bounded':True,'queried_configuration_and_execution':{k:{'supported':True,'value':v} for k,v in props.items()}}
    def test_exact_208_normals_and_original_seed_scope(self):
        old,evidence=c.initialization_evidence()
        self.assertEqual(c.SEED,old['seed']);self.assertEqual(len(old['normal_sources']),208)
        train=set(old['split_items']['train']);others=set(old['split_items']['calibration']+old['split_items']['test'])
        self.assertFalse(train&others)
        self.assertTrue(all(not r['defective'] and Path(r['image']).parts[0] in train for r in old['normal_sources']))
        self.assertFalse(evidence['CPU_capacity_is_qualification']);self.assertEqual(evidence['BN_fitted_state_sha256'],c.L4_STATE_SHA)

    def test_numeric_fail_opens_no_CAL_and_cannot_represent_pass(self):
        value={'step':200,'status':'numeric_failed_no_CAL','CAL_inputs_opened':0}
        validate_terminal(value,self.numeric(200,False))
        for update in [{'CAL_inputs_opened':80},{'calibration_report_sha256':'fake'}]:
            with self.assertRaises(ValueError):validate_terminal(value|update,self.numeric(200,False))
        with self.assertRaises(ValueError):validate_terminal(value,self.numeric(200,True))

    def test_completed_CAL_requires_numeric_pass_and_full80(self):
        value={'step':400,'status':'calibration_complete','CAL_inputs_opened':80,'calibration_report_sha256':'receipt'}
        validate_terminal(value,self.numeric(400,True))
        for parity in [self.numeric(200,True),self.numeric(400,False)]:
            with self.assertRaises(ValueError):validate_terminal(value,parity)
        with self.assertRaises(ValueError):validate_terminal(value|{'CAL_inputs_opened':79},self.numeric(400,True))

    def test_numeric_acceptance_recomputed_not_declared(self):
        self.assertTrue(c.validate_numeric_receipt(self.numeric(200)))
        for update in [{'max_abs_errors':[.02]*3},{'actual_score_ranges':[[.4,.4]]*3},{'reference_inputs_byte_identical_to_L4':False}]:
            with self.assertRaises(ValueError):c.validate_numeric_receipt(self.numeric(200)|update)
        value=self.numeric(200);value['queried_configuration_and_execution']['GPU_DISABLE_WINOGRAD_CONVOLUTION']['value']=False
        with self.assertRaises(ValueError):c.validate_numeric_receipt(value)

    def test_nonfinite_numeric_output_is_terminal_failure_not_lineage_swallow(self):
        from probe_factory_stfpm_learn import numeric_case
        class Failing:
            def __init__(self,message):self.message=message
            def predict(self,*args):raise ValueError(self.message)
        actual=numeric_case(Failing('Factory neural model produced invalid scores'),None,None)
        self.assertFalse(actual['finite']);self.assertIsNone(actual['max_abs_error'])
        value=self.numeric(200,False)|{'max_abs_errors':[None,.001,.001],'actual_score_ranges':[[None,None],[.1,.5],[.1,.5]],'all_finite':False}
        self.assertFalse(c.validate_numeric_receipt(value))
        validate_terminal({'step':200,'status':'numeric_failed_no_CAL','CAL_inputs_opened':0},value)
        json.dumps(value,allow_nan=False)
        with self.assertRaises(ValueError):numeric_case(Failing('Export checksum changed'),None,None)
        with self.assertRaises(ValueError):c.validate_numeric_receipt(value|{'fixed_cases_attempted':2})

    def test_same_sampler_and_loss_no_feedback_during_fixed_fit(self):
        old=ast.parse((ROOT/'scripts/train_factory_stfpm_cpu.py').read_text());new=ast.parse((ROOT/'scripts/train_factory_stfpm_learn.py').read_text())
        for name in ['sample_plan','verify_normal_files','teacher_digest','bn_rows']:
            fn=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(fn(old),include_attributes=False),ast.dump(fn(new),include_attributes=False))
        text=ast.unparse(new)
        self.assertIn('plan = sample_plan(208)',text);self.assertIn('loss = model.normal_loss(',text)
        self.assertNotIn('parity(',text);self.assertNotIn('calibrate(',text)
        self.assertLess(text.index('model.load_state_dict('),text.index('optimizer = torch.optim.AdamW('))
        self.assertLess(text.index('archive_executed_sources('),text.index('optimizer = torch.optim.AdamW('))
        self.assertNotIn('reset_running_stats(',text)

    def test_GPU_recipe_all_branches_and_complete_before_any_forward(self):
        for name in ['probe_factory_stfpm_learn.py','calibrate_factory_stfpm_learn.py']:
            tree=ast.parse((ROOT/'scripts'/name).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FactorySegmentation']
            self.assertEqual(len(calls),1)
            self.assertEqual({k.arg:ast.literal_eval(k.value) for k in calls[0].keywords},{'output_logits':False,'precision_hint':'f16','disable_winograd_convolution':True})
        text=(ROOT/'scripts/calibrate_factory_stfpm_learn.py').read_text()
        body=text[text.index('def main():'):]
        self.assertLess(body.index('require_complete_training(protocol)'),body.index('parity(step,protocol)'))
        self.assertLess(body.index('parity(step,protocol)'),body.index('cached_rows(step,protocol,path)'))
        self.assertIn("'numeric_failed_no_CAL'",body)

    def test_metadata_deploys_GPU_no_CPU_and_preserves_score(self):
        old=c.l4_metadata();protocol=dict(old,name=c.NAME)
        meta=c.metadata(c.L4_NN,protocol)
        self.assertEqual(meta['openvino_device'],'GPU');self.assertEqual(meta['openvino_inference_precision'],'f16')
        self.assertIs(meta['openvino_disable_winograd_convolution'],True);self.assertNotIn('openvino_cpu_threads',meta)
        self.assertFalse(meta['output_logits']);self.assertEqual(meta['score_meaning'],old['score'])

    def test_freezer_accounts_all3_and_fixed_no_test_ranking(self):
        source=(ROOT/'scripts/freeze_factory_stfpm_learn.py').read_text()
        self.assertIn('for step in SAVE_AT:',source);self.assertIn('validate_terminal(t,parity)',source)
        self.assertIn('validate_search_report(r,protocol)',source);self.assertIn('require_complete_training(protocol)',source)
        self.assertIn("'all_three_terminal_outcomes'",source);self.assertEqual(c.SAVE_AT,[200,400,600])
        self.assertNotIn('factory_localization_audit',source);self.assertNotIn('cv2.imread',source)

if __name__=='__main__':unittest.main()
