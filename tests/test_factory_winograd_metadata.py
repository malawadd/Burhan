"""NoWinograd strict opt-in/default equivalence using only fake compilation."""
import hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class Winograd(unittest.TestCase):
    def test_default_has_no_new_compile_key_and_true_is_exact_bool(self):
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
            def compile_model(self,graph,device,config):calls.append(config.copy());return Compiled()
        with patch.dict(sys.modules,{'openvino':types.SimpleNamespace(Core=Core)}):
            FactorySegmentation('mock');FactorySegmentation('mock',disable_winograd_convolution=False)
            model=FactorySegmentation('mock',disable_winograd_convolution=True)
            self.assertEqual(calls[0],calls[1]);self.assertNotIn('GPU_DISABLE_WINOGRAD_CONVOLUTION',calls[0])
            self.assertEqual(calls[2],calls[0]|{'GPU_DISABLE_WINOGRAD_CONVOLUTION':True})
            self.assertTrue(model.disable_winograd_convolution);self.assertEqual(model.precision_hint,'f16')
            for invalid in [0,1,'true',None,{}]:
                with self.assertRaises(ValueError):FactorySegmentation('mock',disable_winograd_convolution=invalid)
        self.assertEqual(len(calls),3)

    def test_only_declared_metadata_passes_no_winograd_flag(self):
        from edge_runtime.profiles.factory_defect import SurfaceDefectModel
        calls=[]
        class Fake:
            grid_width=256
            def __init__(self,path,**kwargs):calls.append(kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mock.onnx';path.write_bytes(b'no NN')
            meta={'feature_version':'surface_mobilenet_seg_v1','model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'pixel_threshold':.05,'minimum_component_pixels':4,'output_logits':False}
            with patch('edge_runtime.profiles.factory_encoder.FactorySegmentation',Fake):
                path.with_suffix('.onnx.json').write_text(json.dumps(meta));SurfaceDefectModel(path)
                path.with_suffix('.onnx.json').write_text(json.dumps(meta|{'openvino_disable_winograd_convolution':True}));SurfaceDefectModel(path)
        self.assertNotIn('disable_winograd_convolution',calls[0])
        self.assertEqual(calls[1],calls[0]|{'disable_winograd_convolution':True})


if __name__=='__main__':unittest.main()
