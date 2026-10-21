"""L9 graph-metadata, pure arithmetic and fake-request fixtures; no NN."""
import json,math,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import factory_stfpm_partition_math as m
import factory_partitioned_segmentation as a
import prepare_factory_stfpm_partition as p
import probe_factory_stfpm_partition as b


class Port:
    def __init__(self,row):self.names={row['tensor']};self.shape=row['shape'];self.dtype='f32'
    def get_names(self):return self.names
    def get_element_type(self):return types.SimpleNamespace(get_type_name=lambda:self.dtype)


def fake_openvino(calls):
    rows=p.feature_rows()
    class Graph:
        outputs=[Port(row) for row in rows]
        def input(self):return types.SimpleNamespace(shape=[1,3,1280,512])
    class Compiled:
        outputs=[Port(row) for row in rows]
        def input(self):return 'native_input'
        def get_property(self,key):return ['GPU.0']
        def create_infer_request(self):return types.SimpleNamespace(infer=lambda *args,**kwargs:None)
    class Core:
        available_devices=['GPU.0']
        def read_model(self,path):return Graph()
        def compile_model(self,graph,device,config):calls.append((device,config));return Compiled()
    return types.SimpleNamespace(Core=Core)


class PartitionPreparation(unittest.TestCase):
    def test_deterministic_ancestor_extraction_retains_exact_nodes_weights_inputs(self):
        import onnx
        source=onnx.load(str(p.L4_NN));derived,proof=p.extract(source,p.feature_rows())
        self.assertEqual(derived.SerializeToString(),p.MODEL.read_bytes())
        self.assertTrue(proof['all_retained_node_initializer_input_opset_bytes_identical'])
        self.assertLess(proof['retained_nodes'],proof['original_nodes'])
        self.assertEqual([v.name for v in derived.graph.output],[r['tensor'] for r in p.feature_rows()])
        self.assertEqual([v.SerializeToString() for v in source.graph.input],[v.SerializeToString() for v in derived.graph.input])
        self.assertEqual(p.sha(p.L4_NN),p.L4_NN_SHA);self.assertEqual(p.sha(p.L4_REF),p.L4_REF_SHA)

    def test_extraction_does_not_include_score_head_nodes(self):
        import onnx
        model=onnx.load(str(p.MODEL));names={node.op_type for node in model.graph.node}
        self.assertNotIn('Log',names);self.assertNotIn('Exp',names);self.assertNotIn('Resize',names)
        self.assertIn('Gather',names);self.assertIn('Div',names);self.assertEqual(len(model.graph.output),6)

    def test_unit_feature_zero_tiny_halfmax_orthogonal_cases(self):
        for value in [np.zeros((1,3,2,2),np.float32),np.full((1,3,2,2),2**-24,np.float32),np.full((1,3,2,2),65504,np.float32)]:
            out=m.unit_feature(value);self.assertTrue(np.isfinite(out).all());self.assertEqual(out.dtype,np.float32)
            self.assertTrue((np.sum(out*out,axis=1)<=1.000001).all())
        x=np.array([[[[1]],[[0]]]],np.float32);y=x[:,::-1].copy()
        np.testing.assert_allclose(m.disagreement(x,x),0,atol=0)
        np.testing.assert_allclose(m.disagreement(x,y),.5,atol=1e-7)
        np.testing.assert_allclose(m.disagreement(x,-x),1,atol=1e-7)

    def test_bilinear_halfpixel_matches_independent_scalar_double_oracle(self):
        array=np.array([[[[0.,1.,3.],[2.,7.,4.]]]],np.float32)
        for height,width in [(8,12),(3,5),(1,2),(2,3)]:
            expected=np.empty((1,1,height,width),np.float64)
            for y in range(height):
                yy=max((y+.5)*2/height-.5,0);y0=int(math.floor(yy));y1=min(y0+1,1);fy=yy-y0
                for x in range(width):
                    xx=max((x+.5)*3/width-.5,0);x0=int(math.floor(xx));x1=min(x0+1,2);fx=xx-x0
                    expected[0,0,y,x]=(1-fy)*((1-fx)*array[0,0,y0,x0]+fx*array[0,0,y0,x1])+fy*((1-fx)*array[0,0,y1,x0]+fx*array[0,0,y1,x1])
            np.testing.assert_allclose(m.bilinear_half_pixel(array,height,width),expected,rtol=1e-6,atol=1e-6)

    def test_fixed_head_matches_declared_numpy_math_and_floor(self):
        from factory_stfpm_math import disagreement,geometric_mean
        rng=np.random.default_rng(9);features=[rng.normal(size=(1,channels,h,w)).astype(np.float32) for channels,h,w in [(3,8,4),(3,8,4),(4,2,1),(4,2,1),(5,1,1),(5,1,1)]]
        maps=[disagreement(features[i],features[i+1]) for i in (0,2,4)]
        expected=geometric_mean([maps[0],m.bilinear_half_pixel(maps[1],8,4),m.bilinear_half_pixel(maps[2],8,4)])
        np.testing.assert_allclose(m.score_from_features(features),expected,rtol=1e-6,atol=1e-7)
        self.assertTrue((m.score_from_features([np.zeros_like(f) for f in features])==np.float32(2**-14)).all())
        with self.assertRaises(ValueError):m.score_from_features(features[:-1])
        bad=features.copy();bad[0]=features[0]*np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):m.score_from_features(bad)

    def test_proxy_requests_shared_features_but_returns_owned_score(self):
        arrays=[np.ones((1,c,h,w),np.float32) for c,h,w in [(2,8,4),(2,8,4),(3,2,1),(3,2,1),(4,1,1),(4,1,1)]];flags=[];ports=list(range(6))
        class Request:
            def infer(self,inputs,share_outputs=False):flags.append(share_outputs);return dict(zip(ports,arrays))
        proxy=a.PartitionRequest(Request(),ports,'score');score=proxy.infer({})['score'];before=score.copy()
        for array in arrays:array.fill(3)
        self.assertEqual(flags,[True]);self.assertTrue(score.flags.owndata);np.testing.assert_array_equal(score,before)
        self.assertTrue(all(np.isfinite(v) and v>=0 for v in proxy.last_timing.values()))
        self.assertAlmostEqual(sum(v for k,v in proxy.last_timing.items() if k!='partition_total_ms'),proxy.last_timing['partition_total_ms'],places=6)

    def test_constructor_inherits_base_math_and_fails_wrong_schema_recipe(self):
        calls=[]
        with patch.dict(sys.modules,{'openvino':fake_openvino(calls)}):model=a.adapter_type()(p.MODEL)
        self.assertIs(type(model).predict,a.BaseFactorySegmentation.predict);self.assertIs(type(model).warmup,a.BaseFactorySegmentation.warmup)
        self.assertEqual(calls,[('GPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True})])
        for kwargs in [{'precision_hint':'f32'},{'output_logits':True},{'device':'CPU'},{'disable_winograd_convolution':False}]:
            with self.assertRaises(ValueError):a.FactoryPartitionedSegmentation(p.MODEL,**kwargs)
        rows=p.feature_rows();ports=[Port(r) for r in rows];ports[0].dtype='f16'
        with self.assertRaises(ValueError):a.validate_ports(ports,rows)
        with self.assertRaises(ValueError):a.validate_ports(ports[:-1],rows)

    def test_profile_override_restores_shared_constructor_without_file_change(self):
        from edge_runtime.profiles.factory import FactoryProfile
        import edge_runtime.profiles.factory_encoder as module
        original=module.FactorySegmentation;cls=a.adapter_type();before=p.sha(ROOT/'edge_runtime/profiles/factory_encoder.py')
        with patch.dict(sys.modules,{'openvino':fake_openvino([])}):
            with patch.object(module,'FactorySegmentation',cls):profile=FactoryProfile({'defect_model_path':str(p.MODEL),'defect_backend':'auto'})
        self.assertIs(module.FactorySegmentation,original);self.assertIs(type(profile._defect_model.neural),cls)
        self.assertEqual(before,p.sha(ROOT/'edge_runtime/profiles/factory_encoder.py'))

    def test_numeric_nonfinite_failure_is_jsonsafe_unrelated_error_propagates(self):
        class Bad:
            def predict(self,*args):raise ValueError('L9 encoder feature is nonfinite')
        value=b.numeric_case(Bad(),None,None);self.assertIsNone(value['max_abs_error']);json.dumps(value,allow_nan=False)
        class Changed:
            def predict(self,*args):raise ValueError('Hash changed')
        with self.assertRaisesRegex(ValueError,'Hash'):b.numeric_case(Changed(),None,None)

    def test_all_three_learned_sources_are_prebound_and_original_failures_untouched(self):
        evidence=b.learned_sources();self.assertEqual([row['step'] for row in evidence['all_three']],[200,400,600])
        self.assertTrue(all(len(row['sources'])==4 for row in evidence['all_three']))
        self.assertFalse((p.OUT/'initial-three-case-numeric.json').exists());self.assertFalse((p.OUT/'initial-fullprofile-capacity.json').exists())


if __name__=='__main__':unittest.main()
