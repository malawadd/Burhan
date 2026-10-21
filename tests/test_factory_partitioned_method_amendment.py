"""L9 additive method decision invariants; no NN/model/data decode."""
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import record_factory_partitioned_method_amendment as a


class MethodAmendment(unittest.TestCase):
    def test_all3_can_be_investigated_without_initial_pass_but_need_root(self):
        value=a.value();self.assertTrue(a.validate(value));self.assertFalse(value['initial_pass_required_for_learned_investigation'])
        self.assertTrue(value['initial_probe_unchanged']);self.assertTrue(value['initial_failures_preserved']);self.assertTrue(value['learned_requires_separate_root_release'])
        self.assertEqual(value['checkpoints'],[200,400,600]);self.assertEqual([r['step'] for r in value['source_bindings']['all_three']],[200,400,600])

    def test_acceptance_numeric_speed_and_noCAL_for_failed_guards_cannot_relax(self):
        original=a.value()
        for key,new in [('numeric_max_abs_error',.02),('final_fitted_mean_calls_per_second_minimum',14),('final_sustained_input_FPS',15),
                        ('passing_only_CAL',False),('numeric_failed_checkpoint_CAL_inputs',1),('complete_all3_numeric_before_selection',False),('learned_requires_separate_root_release',False)]:
            value=copy.deepcopy(original);value[key]=new
            with self.assertRaises(ValueError):a.validate(value)

    def test_source_and_all3_identity_binding_rejects_single_checkpoint(self):
        original=a.value()
        for mutate in [lambda v:v.update(parent_initial_protocol_sha256='0'*64),lambda v:v['source_bindings']['all_three'].pop(),lambda v:v['checkpoints'].pop()]:
            value=copy.deepcopy(original);mutate(value)
            with self.assertRaises(ValueError):a.validate(value)


if __name__=='__main__':unittest.main()
