"""Optional precision metadata plumbing with fake compilation, no NN/device use."""
import hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class Precision(unittest.TestCase):
    def test_precision_constructor_default_and_opt_in_config(self):
        from edge_runtime.profiles.factory_encoder import FactorySegmentation
        calls=[]
        class Graph:
            def input(self):return types.SimpleNamespace(shape=[1,3,1280,512])
            def output(self):return types.SimpleNamespace(shape=[1,1,640,256])
        class Compiled:
            def input(self):return 0
            def output(self):return 1
            def create_infer_request(self):return object()
            def get_property(self,key):return ['GPU.0']
        class Core:
            available_devices=['GPU.0']
            def read_model(self,path):return Graph()
            def compile_model(self,graph,device,config):calls.append((device,config.copy()));return Compiled()
        with patch.dict(sys.modules,{'openvino':types.SimpleNamespace(Core=Core)}):
            default=FactorySegmentation('unused');explicit=FactorySegmentation('unused',precision_hint='f32')
            self.assertEqual(default.precision_hint,'f16');self.assertEqual(explicit.precision_hint,'f32')
            self.assertEqual(calls[0],('GPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16'}))
            self.assertEqual(calls[1][1],calls[0][1]|{'INFERENCE_PRECISION_HINT':'f32'})
            for invalid in ['fp64',None,16,{}]:
                with self.assertRaises(ValueError):FactorySegmentation('unused',precision_hint=invalid)
        self.assertEqual(len(calls),2)

    def test_surface_metadata_defaults_f16_and_passes_exact_f32(self):
        from edge_runtime.profiles.factory_defect import SurfaceDefectModel
        calls=[]
        class FakeSegmentation:
            grid_width=256
            def __init__(self,path,**kwargs):calls.append(kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fake.onnx';path.write_bytes(b'mock-graph-no-inference')
            meta={'feature_version':'surface_mobilenet_seg_v1','model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'pixel_threshold':.05,'minimum_component_pixels':4,'output_logits':False}
            with patch('edge_runtime.profiles.factory_encoder.FactorySegmentation',FakeSegmentation):
                path.with_suffix('.onnx.json').write_text(json.dumps(meta))
                SurfaceDefectModel(path)
                path.with_suffix('.onnx.json').write_text(json.dumps(meta|{'openvino_inference_precision':'f32'}))
                SurfaceDefectModel(path)
        self.assertEqual(calls,[{'output_logits':False,'growth_threshold':None,'precision_hint':'f16'},
                                {'output_logits':False,'growth_threshold':None,'precision_hint':'f32'}])


if __name__=='__main__':unittest.main()
