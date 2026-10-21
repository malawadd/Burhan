"""L8 fake compilation/view-lifetime/source fixtures; no NN/device execution."""
import ast,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import factory_tapped_segmentation as a
import probe_factory_stfpm_tapped as p


class Port:
    def __init__(self,row):self.names={row['tensor']};self.shape=row['shape'];self.dtype='f32'
    def get_names(self):return self.names
    def get_element_type(self):return types.SimpleNamespace(get_type_name=lambda:self.dtype)


class FakeRequest:
    def __init__(self,outputs):self.outputs=outputs;self.score=np.full((1,1,640,256),.25,dtype=np.float32);self.flags=[];self.read=[]
    def infer(self,inputs,share_outputs=False):
        self.flags.append(share_outputs);owner=self
        class Result:
            def __getitem__(self,key):
                owner.read.append(key)
                if key is not owner.outputs[0]:raise AssertionError('Auxiliary output explicitly read')
                return owner.score if share_outputs else owner.score.copy()
        return Result()


def fake_openvino(calls,wrong_compiled_order=False):
    rows=a.protected_rows()
    class Graph:
        def __init__(self):self.outputs=[Port(row) for row in rows]
        def input(self):return types.SimpleNamespace(shape=[1,3,1280,512])
    class Compiled:
        def __init__(self):
            self.outputs=[Port(row) for row in rows]
            if wrong_compiled_order:self.outputs[1],self.outputs[2]=self.outputs[2],self.outputs[1]
            self.request=FakeRequest(self.outputs)
        def input(self):return 'input'
        def get_property(self,key):return ['GPU.0']
        def create_infer_request(self):return self.request
    class Core:
        available_devices=['GPU.0']
        def read_model(self,path):return Graph()
        def compile_model(self,graph,device,config):calls.append((device,config.copy()));return Compiled()
    return types.SimpleNamespace(Core=Core)


class TappedPreparation(unittest.TestCase):
    def test_byte_exact75_graph_and_source_references_unchanged(self):
        self.assertEqual(p.sha(p.MODEL),a.INITIAL_GRAPH_SHA);self.assertEqual(p.sha(p.SOURCE),a.INITIAL_GRAPH_SHA)
        self.assertEqual(p.sha(p.L4_REF),p.L4_REF_SHA);self.assertEqual(len(a.protected_rows()),75)
        meta=json.loads(p.MODEL.with_suffix('.onnx.json').read_text());self.assertEqual(meta['all_outputs_retained'],75)
        self.assertTrue(meta['architecture_probe_only']);self.assertFalse(meta['output_logits'])

    def test_both_paths_retain75_and_identical_compile_config_with_exact_Base_methods(self):
        calls=[]
        with patch.dict(sys.modules,{'openvino':fake_openvino(calls)}):
            for variant in p.VARIANTS:
                model=a.adapter_type(variant)(p.MODEL)
                self.assertEqual(len(model.compiled.outputs),75);self.assertEqual(model.output.get_names(),{'anomaly_score'})
                self.assertIs(type(model).predict,a.BaseFactorySegmentation.predict);self.assertIs(type(model).warmup,a.BaseFactorySegmentation.warmup)
        self.assertEqual(calls,[('GPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True})]*2)

    def test_shared_views_copy_owned_final_before_request_can_change_buffers(self):
        calls=[]
        with patch.dict(sys.modules,{'openvino':fake_openvino(calls)}):
            for variant in p.VARIANTS:
                model=a.adapter_type(variant)(p.MODEL);native=model.request.inner if variant=='shared_views' else model.request
                scores,_,metrics=model.predict(np.zeros((48,20,3),dtype=np.uint8),.9,4)
                self.assertEqual(native.flags,[variant=='shared_views']);self.assertEqual(native.read,[model.output])
                native.score.fill(.75);self.assertTrue((scores==.25).all());self.assertTrue(scores.flags.owndata)
                self.assertIn(variant,metrics['model_backend']);self.assertEqual(model.initialization_report['all_graph_outputs_retained'],75)

    def test_output_subset_order_alias_shape_and_wrong_recipe_fail_closed(self):
        rows=a.protected_rows();ports=[Port(row) for row in rows]
        with self.assertRaises(ValueError):a.validate_ports(ports[:-1],rows)
        bad=[Port(row) for row in rows];bad[1],bad[2]=bad[2],bad[1]
        with self.assertRaises(ValueError):a.validate_ports(bad,rows)
        bad=[Port(row) for row in rows];bad[0].names.add(rows[1]['tensor'])
        with self.assertRaises(ValueError):a.validate_ports(bad,rows)
        bad=[Port(row) for row in rows];bad[0].shape=[1,1,1280,512]
        with self.assertRaises(ValueError):a.validate_ports(bad,rows)
        bad=[Port(row) for row in rows];bad[0].dtype='f16'
        with self.assertRaises(ValueError):a.validate_ports(bad,rows)
        for kwargs in [{'precision_hint':'f32'},{'disable_winograd_convolution':False},{'device':'CPU'},{'output_logits':True},{'cpu_threads':4}]:
            with self.assertRaises(ValueError):a.FactoryTappedSegmentation(p.MODEL,**kwargs)
        with self.assertRaises(ValueError):a.adapter_type('AUTO')
        with self.assertRaises(ValueError):a.adapter_type('host_copy','0'*64)(p.MODEL)

    def test_constructor_validates_compiled_order_before_request_creation(self):
        with patch.dict(sys.modules,{'openvino':fake_openvino([],wrong_compiled_order=True)}):
            with self.assertRaisesRegex(ValueError,'names/order/shapes'):a.FactoryTappedSegmentation(p.MODEL)

    def test_injection_is_process_local_and_restores_default_constructor(self):
        from edge_runtime.profiles.factory import FactoryProfile
        import edge_runtime.profiles.factory_encoder as module
        original=module.FactorySegmentation;variant=a.adapter_type('shared_views')
        before=p.sha(ROOT/'edge_runtime/profiles/factory_encoder.py')
        with patch.dict(sys.modules,{'openvino':fake_openvino([])}):
            with patch.object(module,'FactorySegmentation',variant):profile=FactoryProfile({'defect_model_path':str(p.MODEL),'defect_backend':'auto'})
        self.assertIs(module.FactorySegmentation,original);self.assertIs(type(profile._defect_model.neural),variant)
        self.assertEqual(p.sha(ROOT/'edge_runtime/profiles/factory_encoder.py'),before)

    def test_predeclared_ALL3_correction_is_before_numeric_and_no_subset_or_retrain(self):
        proposal=json.loads((p.OUT/'conditional-learned75-proposal-before-L7-numeric.json').read_text())
        self.assertTrue(proposal['declared_before_any_L7_GPU_numeric_or_CAL']);self.assertEqual([r['step'] for r in proposal['source_models']],[200,400,600])
        self.assertEqual(proposal['ordered75_output_schema_sha256'],a.SCHEMA_SHA);self.assertEqual(proposal['new_NN_calls'],0)
        self.assertIn('ALL200/400/600',proposal['checkpoint_selection']);self.assertIn('no learnedweight/BN/trainingchange',proposal['graph_transform'])
        source=(ROOT/'scripts/factory_tapped_segmentation.py').read_text();self.assertNotIn('create_tensor(',source);self.assertNotIn('get_default_context(',source)
        self.assertIn('share_outputs=True',source)

    def test_numeric_output_failure_is_explicit_but_unrelated_error_stops(self):
        class Invalid:
            def predict(self,*args):raise ValueError('Factory neural model produced invalid scores')
        row=p.numeric_case(Invalid(),None,None)
        self.assertTrue(row['numeric_output_failed']);self.assertIsNone(row['max_abs_error']);self.assertIsNone(row['score_range'])
        json.dumps(row,allow_nan=False)
        class Changed:
            def predict(self,*args):raise ValueError('Source checksum changed')
        with self.assertRaisesRegex(ValueError,'checksum'):p.numeric_case(Changed(),None,None)


if __name__=='__main__':unittest.main()
