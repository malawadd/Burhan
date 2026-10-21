"""L6 pure ONNX/array/mock contracts: no model forward or source pixels."""
import copy,json,sys,types,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import diagnose_factory_stfpm_stages as d


class StageDiagnosis(unittest.TestCase):
    def test_prepared_graph_retains_every_existing_node_weight_input_and_score(self):
        import onnx
        original=onnx.load(str(d.L4_NN));diagnostic=onnx.load(str(d.MODEL));stages=d.declared_stages()
        d.check_append_only(original,diagnostic,stages)
        self.assertEqual(len(stages),75);self.assertEqual(len(original.graph.node),253)
        manifest=json.loads(d.MANIFEST.read_text(encoding='utf-8'))
        self.assertEqual(manifest['source_model_sha256'],d.L4_NN_SHA);self.assertEqual(d.sha(d.MODEL),manifest['diagnostic_model_sha256'])
        self.assertEqual(manifest['single_case_output_bytes'],134420480)
        self.assertTrue(all(row['ONNX_elem_type']==1 and all(v>0 for v in row['shape']) for row in manifest['ordered_outputs']))
        self.assertFalse(manifest['NN_executed']);self.assertFalse(manifest['Torch_weight_load_or_export'])

    def test_graph_guard_rejects_node_weight_input_original_score_and_inventory_edits(self):
        import onnx
        original=onnx.load(str(d.L4_NN));diagnostic=onnx.load(str(d.MODEL));stages=d.declared_stages()
        bad=copy.deepcopy(diagnostic);bad.graph.node[0].name+='changed'
        with self.assertRaises(ValueError):d.check_append_only(original,bad,stages)
        bad=copy.deepcopy(diagnostic);bad.graph.initializer[0].name+='changed'
        with self.assertRaises(ValueError):d.check_append_only(original,bad,stages)
        bad=copy.deepcopy(diagnostic);bad.graph.input[0].name+='changed'
        with self.assertRaises(ValueError):d.check_append_only(original,bad,stages)
        bad=copy.deepcopy(diagnostic);bad.graph.output[0].name+='changed'
        with self.assertRaises(ValueError):d.check_append_only(original,bad,stages)
        bad=copy.deepcopy(diagnostic);bad.graph.output[1].name=bad.graph.output[2].name
        with self.assertRaises(ValueError):d.check_append_only(original,bad,stages)

    def test_protected_compiled_names_handle_reordering_but_reject_aliases(self):
        def out(*names):return types.SimpleNamespace(get_names=lambda:set(names))
        a,b=out('a'),out('b');rows=[{'tensor':'a'},{'tensor':'b'}]
        self.assertEqual(d.bind_outputs([b,a],rows),[a,b])
        for outputs in [[out('a'),out('missing')],[out('a','b'),out('other')],[out('a'),out('a','b')]]:
            with self.assertRaises(ValueError):d.bind_outputs(outputs,rows)

    def test_both_backend_actual_configs_gate_without_cross_device_property_queries(self):
        def audit(device,changed=None,unsupported=None):
            values={'EXECUTION_DEVICES':['CPU'] if device=='CPU' else ['GPU.0'],'INFERENCE_PRECISION_HINT':"<Type: 'float32'>" if device=='CPU' else "<Type: 'float16'>",
                    'NUM_STREAMS':1,'PERFORMANCE_HINT':'LATENCY'}
            values.update({'INFERENCE_NUM_THREADS':4} if device=='CPU' else {'GPU_DISABLE_WINOGRAD_CONVOLUTION':True})
            values.update(changed or {});queries=[]
            class Compiled:
                def get_property(self,key):
                    queries.append(key)
                    if key==unsupported:raise RuntimeError('unsupported')
                    return values[key]
                def get_runtime_model(self):return types.SimpleNamespace(get_ops=lambda:[])
            return d.configuration_audit(Compiled(),device),queries
        cpu,queries=audit('CPU');self.assertTrue(cpu['actual_configuration_passed']);self.assertNotIn('GPU_DISABLE_WINOGRAD_CONVOLUTION',queries)
        gpu,queries=audit('GPU');self.assertTrue(gpu['actual_configuration_passed']);self.assertNotIn('INFERENCE_NUM_THREADS',queries)
        for device,changed in [('CPU',{'INFERENCE_NUM_THREADS':2}),('CPU',{'INFERENCE_PRECISION_HINT':"<Type: 'float16'>"}),('GPU',{'GPU_DISABLE_WINOGRAD_CONVOLUTION':False}),('GPU',{'NUM_STREAMS':2})]:
            self.assertFalse(audit(device,changed)[0]['actual_configuration_passed'])
        self.assertFalse(audit('CPU',unsupported='NUM_STREAMS')[0]['actual_configuration_passed'])

    def test_context_requires_all12_exact_order_bound_finite_config_and_score_agreement(self):
        errors={'CPU':[0.,1e-6,0.],'GPU':[0.,0.,1e-6]};calls=[{'model':model,'case':case} for model,case in d.expected_calls()]
        self.assertTrue(d.context_passes(errors,True,[1e-6]*3,True,True,calls))
        self.assertFalse(d.context_passes(errors,False,[1e-6]*3,True,True,calls));self.assertFalse(d.context_passes(errors,True,[1e-6]*3,False,True,calls))
        self.assertFalse(d.context_passes(errors,True,[1e-6]*3,True,False,calls));self.assertFalse(d.context_passes(errors,True,[1e-6]*3,True,True,calls[:-1]))
        self.assertFalse(d.context_passes(errors,True,[1e-6]*3,True,True,[calls[1],calls[0],*calls[2:]]))
        self.assertFalse(d.context_passes(errors|{'GPU':[0.,.00002,0.]},True,[1e-6]*3,True,True,calls))
        self.assertFalse(d.context_passes(errors,True,[.02,0.,0.],True,True,calls))

    def test_stage_metrics_tiny_floor_and_nonfinite_scope(self):
        a=np.array([0.,2**-15,2**-25,1.]);b=np.zeros_like(a);row=d.statistics(a,b,2**-14)
        self.assertEqual(row['max_abs_error'],1.);self.assertEqual(row['CPU']['zero_fraction'],.25)
        self.assertEqual(row['CPU']['nonzero_below_half_normal_fraction'],.5)
        self.assertEqual(row['CPU']['nonzero_below_half_subnormal_fraction'],.25)
        self.assertEqual(row['CPU']['at_or_below_semantic_floor_fraction'],.75)
        self.assertEqual(row['relative_rms_error'],1.)
        with self.assertRaises(ValueError):d.statistics(np.array([np.nan]),np.array([0.]))
        with self.assertRaises(ValueError):d.statistics(np.ones(3),np.ones(4))


if __name__=='__main__':unittest.main()
