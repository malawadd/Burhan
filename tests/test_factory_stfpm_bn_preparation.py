"""L4 metadata-only/pure state-contract fixtures: no Torch, image decode or NN."""
import ast,copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import factory_stfpm_bn_common as c


class BNPreparation(unittest.TestCase):
    def test_actual64_selection_exact_source_identity_and_coverage(self):
        old=c.source_metadata();selected=c.select64(old['normal_sources'],old['split_items']['train'])
        saved=json.loads((c.OUT/'selected64-before-any-forward.json').read_text(encoding='utf-8'))
        self.assertEqual(selected,saved['sources']);self.assertEqual(len({r['image'] for r in selected}),64)
        self.assertEqual(c.identity(selected),'bdd7458ceede545360d640664b0009ab41e7558c21c9311f6611177584842d12')
        items=old['split_items']['train']
        self.assertEqual([Path(r['image']).parts[0] for r in selected[:30]],items)
        self.assertEqual([Path(r['image']).parts[0] for r in selected[30:60]],items)
        self.assertEqual([Path(r['image']).parts[0] for r in selected[60:]],items[:4])
        self.assertFalse(any(r['defective'] for r in selected));self.assertEqual(saved['NN_forward_calls'],0)

    def test_selection_fails_without_declared_sources_no_substitution(self):
        old=c.source_metadata();items=old['split_items']['train'];rows=old['normal_sources']
        with self.assertRaises(ValueError):c.select64(rows,[*items[:-1],items[0]])
        kept=[r for r in rows if Path(r['image']).parts[0]!=items[0]]
        with self.assertRaises(ValueError):c.select64(kept,items)
        bad=copy.deepcopy(rows);bad[0]['defective']=True
        with self.assertRaises(ValueError):c.select64(bad,items)

    def test_only_student_running_statistics_may_change(self):
        spec=[{'name':'0.1'}];allowed=c.allowed_bn_keys(spec)
        before={'teacher.0.weight':'a','student.0.weight':'b','student.0.1.weight':'c','mean':'d','std':'e',**{key:'old' for key in allowed}}
        after=before|{key:'fitted' for key in allowed}
        self.assertEqual(set(c.check_only_bn_changed(before,after,allowed)),allowed)
        for key in ['teacher.0.weight','student.0.weight','student.0.1.weight','mean','std']:
            with self.assertRaises(ValueError):c.check_only_bn_changed(before,after|{key:'changed'},allowed)
        with self.assertRaises(ValueError):c.check_only_bn_changed(before,after|{'new':'x'},allowed)

    def test_counters_epsilon_variance_and_momentum_guards(self):
        spec=[{'name':'0.1','eps':.001}]
        rows=[{'name':'0.1','counter':64,'finite':True,'variance_min':0.,'variance_max':1.,'preserved_eps':.001,'restored_momentum':.01}]
        c.validate_bn_rows(rows,spec,64)
        for key,value in [('counter',63),('finite',False),('variance_min',-.001),('variance_max',float('inf')),('preserved_eps',1e-5),('restored_momentum',.1),('name','other')]:
            with self.assertRaises(ValueError):c.validate_bn_rows([rows[0]|{key:value}],spec,64)
        with self.assertRaises(ValueError):c.validate_bn_rows([],spec,64)
        with self.assertRaises(ValueError):c.validate_bn_rows(rows,spec,264)
        c.validate_bn_rows([rows[0]|{'counter':264}],spec,264)

    def test_actual_factory_metadata_epsilon_and_momentum_not_generic_defaults(self):
        metadata=json.loads((c.OUT/'actual-BN-factory-metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['forward_calls'],0);self.assertFalse(metadata['weights_loaded'])
        self.assertEqual(len(metadata['student_BN']),18)
        self.assertTrue(all(r['eps']==.001 and r['momentum']==.01 and r['initial_counter']==0 for r in metadata['student_BN']))
        self.assertEqual(c.sha(metadata['torchvision_source_path']),metadata['torchvision_source_sha256'])

    def test_BN_fitting_no_optimizer_and_only_real64_inputs(self):
        source=(ROOT/'scripts/prepare_factory_stfpm_bn.py').read_text();tree=ast.parse(source)
        fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_bn');text=ast.unparse(fit)
        self.assertIn("protocol['BN_fit_sources']",text);self.assertIn('bn.reset_running_stats()',text)
        self.assertIn('bn.momentum = None',text);self.assertIn('bn.momentum = 0.01',text)
        self.assertIn('with torch.inference_mode()',text);self.assertIn('parameter.requires_grad_(False)',text)
        self.assertNotIn('torch.optim',text);self.assertNotIn('.backward(',text);self.assertNotIn('reference_inputs(',text)
        self.assertIn('check_only_bn_changed(',text);self.assertIn('validate_bn_rows(',text)

    def test_future_trainer_loads_exact_fitted_state_before_optimizer(self):
        source=(ROOT/'scripts/train_factory_stfpm_bn.py').read_text();tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');text=ast.unparse(main)
        load=text.index('model.load_state_dict(');optimizer=text.index('optimizer = torch.optim.AdamW(')
        self.assertLess(load,optimizer);self.assertLess(text.index('initial_bn = bn_rows('),optimizer)
        self.assertIn('torch.load(fitted_state, map_location=\'cpu\', weights_only=True)',text)
        self.assertIn('strict=True',text);self.assertIn('bn.momentum = 0.01',text);self.assertNotIn('reset_running_stats(',text)
        self.assertIn('64 + step',text);self.assertIn('model.student.state_dict().values()',text)

    def test_GPU_recipe_is_explicit_for_parity_and_actual_CAL(self):
        for filename in ['probe_factory_stfpm_bn.py','calibrate_factory_stfpm_bn.py']:
            tree=ast.parse((ROOT/'scripts'/filename).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FactorySegmentation']
            self.assertEqual(len(calls),1)
            args={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
            self.assertEqual(args,{'output_logits':False,'precision_hint':'f16','disable_winograd_convolution':True})
        text=(ROOT/'scripts/probe_factory_stfpm_bn.py').read_text()
        self.assertIn('numeric_parity_sha256',text);self.assertIn('property_audit(model)',text)


if __name__=='__main__':unittest.main()
