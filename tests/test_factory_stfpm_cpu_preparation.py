"""L5 source/lineage/decoder contract checks without model execution."""
import ast,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import factory_stfpm_cpu_common as c


class CPUPreparation(unittest.TestCase):
    def test_exact_L4_graph_references_saved_BN_state_are_pinned(self):
        old=c.l4_metadata();self.assertEqual(old['normal_source_identity_sha256'],c.NORMAL_SHA)
        self.assertEqual(len(old['BN_fit_sources']),64);self.assertEqual(c.sha(c.L4_NN),c.L4_NN_SHA);self.assertEqual(c.sha(c.L4_REF),c.L4_REF_SHA)
        from prepare_factory_stfpm_cpu import verify_bn_fit
        _,_,fit=verify_bn_fit(old);self.assertEqual(fit['state_sha256'],c.L4_STATE_SHA);self.assertEqual(fit['forward_passes_completed'],64)

    def test_no_BN_repeat_stage_or_initial_ONNX_reexport(self):
        tree=ast.parse((ROOT/'scripts/prepare_factory_stfpm_cpu.py').read_text());names=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)]
        self.assertNotIn('fit_bn',names)
        copy=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='copy_preflight');text=ast.unparse(copy)
        self.assertIn('L4_NN.read_bytes()',text);self.assertIn('L4_REF.read_bytes()',text)
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='torch' for n in ast.walk(copy)));self.assertNotIn('.infer(',text)

    def test_future_trainer_loads_exact_retained_state_before_optimizer(self):
        text=ast.unparse(ast.parse((ROOT/'scripts/train_factory_stfpm_cpu.py').read_text()))
        self.assertLess(text.index('model.load_state_dict('),text.index('optimizer = torch.optim.AdamW('))
        self.assertIn('weights_only=True',text);self.assertIn('strict=True',text);self.assertNotIn('reset_running_stats(',text)
        self.assertIn('bn.momentum = 0.01',text);self.assertIn('64 + step',text)

    def test_ALL_future_numeric_and_CAL_calls_use_exact_CPU_recipe(self):
        for file in ['probe_factory_stfpm_cpu.py','calibrate_factory_stfpm_cpu.py']:
            tree=ast.parse((ROOT/'scripts'/file).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FactorySegmentation']
            self.assertEqual(len(calls),1)
            self.assertEqual({k.arg:ast.literal_eval(k.value) for k in calls[0].keywords},{'output_logits':False,'precision_hint':'f32','device':'CPU','cpu_threads':4})
        cal=(ROOT/'scripts/calibrate_factory_stfpm_cpu.py').read_text();freeze=(ROOT/'scripts/freeze_factory_stfpm_cpu.py').read_text()
        self.assertIn('actual_CPU_maps',cal);self.assertIn('actual_CPU_maps',freeze)
        self.assertIn('validate_search_report(r,protocol)',freeze);self.assertIn('for step in SAVE_AT:',freeze)
        self.assertEqual(c.SAVE_AT,[200,400,600])


if __name__=='__main__':unittest.main()
