"""Strict CPU4 opt-in/default GPU equivalence using fake compilation only."""
import hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))


class CPUBackend(unittest.TestCase):
    def test_one_CPU_recipe_and_exact_legacy_GPU_config(self):
        from edge_runtime.profiles.factory_encoder import FactorySegmentation
        calls=[]
        class Graph:
            def input(self):return types.SimpleNamespace(shape=[1,3,1280,512])
            def output(self):return types.SimpleNamespace(shape=[1,1,640,256])
        class Compiled:
            def input(self):return 0
            def output(self):return 1
            def create_infer_request(self):return object()
            def get_property(self,key):return ['CPU'] if calls[-1][0]=='CPU' else ['GPU.0']
        class Core:
            available_devices=['CPU','GPU.0']
            def read_model(self,path):return Graph()
            def compile_model(self,graph,device,config):calls.append((device,config.copy()));return Compiled()
        with patch.dict(sys.modules,{'openvino':types.SimpleNamespace(Core=Core)}):
            old=FactorySegmentation('mock');cpu=FactorySegmentation('mock',precision_hint='f32',device='CPU',cpu_threads=4)
            self.assertEqual(calls[0],('GPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16'}))
            self.assertEqual(calls[1],('CPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':4}))
            self.assertEqual(old.backend_name,'openvino_gpu_neural');self.assertEqual(cpu.backend_name,'openvino_cpu_neural')
            self.assertNotIn('GPU_DISABLE_WINOGRAD_CONVOLUTION',calls[1][1]);self.assertEqual(cpu.initialization_report['inference_threads'],4)
            self.assertNotIn('device',old.initialization_report)
            for options in [{'device':'CPU'},{'device':'CPU','precision_hint':'f32','cpu_threads':True},{'device':'CPU','precision_hint':'f32','cpu_threads':2},
                            {'device':'CPU','precision_hint':'f32','cpu_threads':4,'disable_winograd_convolution':True},{'cpu_threads':4},{'device':'AUTO'},{'device':None}]:
                with self.assertRaises(ValueError):FactorySegmentation('mock',**options)
        self.assertEqual(len(calls),2)

    def test_CPU_needs_no_GPU_and_legacy_has_no_automatic_fallback(self):
        from edge_runtime.profiles.factory_encoder import FactorySegmentation
        class Graph:
            def input(self):return types.SimpleNamespace(shape=[1,3,1280,512])
            def output(self):return types.SimpleNamespace(shape=[1,1,640,256])
        class Compiled:
            def input(self):return 0
            def output(self):return 1
            def create_infer_request(self):return object()
            def get_property(self,key):return ['CPU']
        class Core:
            available_devices=['CPU']
            def read_model(self,path):return Graph()
            def compile_model(self,graph,device,config):self.device=device;return Compiled()
        with patch.dict(sys.modules,{'openvino':types.SimpleNamespace(Core=Core)}):
            self.assertEqual(FactorySegmentation('mock',precision_hint='f32',device='CPU',cpu_threads=4).device,'CPU')
            with self.assertRaisesRegex(ValueError,'explicitly requires the installed Intel GPU'):FactorySegmentation('mock')

    def test_surface_metadata_CPU_plumbing_and_legacy_default_reporting(self):
        from edge_runtime.profiles.factory_defect import SurfaceDefectModel
        calls=[]
        class Fake:
            grid_width=256
            def __init__(self,path,**kwargs):calls.append(kwargs);self.backend_name='openvino_cpu_neural' if kwargs.get('device')=='CPU' else 'openvino_gpu_neural'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fake.onnx';path.write_bytes(b'no NN')
            meta={'feature_version':'surface_mobilenet_seg_v1','model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pixel_threshold':.05,'minimum_component_pixels':4,'output_logits':False}
            with patch('edge_runtime.profiles.factory_encoder.FactorySegmentation',Fake):
                path.with_suffix('.onnx.json').write_text(json.dumps(meta));old=SurfaceDefectModel(path)
                path.with_suffix('.onnx.json').write_text(json.dumps(meta|{'openvino_device':'CPU','openvino_cpu_threads':4,'openvino_inference_precision':'f32'}));cpu=SurfaceDefectModel(path)
        self.assertEqual(calls,[{'output_logits':False,'growth_threshold':None,'precision_hint':'f16'},
                                {'output_logits':False,'growth_threshold':None,'precision_hint':'f32','device':'CPU','cpu_threads':4}])
        self.assertEqual(old.backend,'openvino_gpu_neural');self.assertEqual(cpu.backend,'openvino_cpu_neural')

    def test_predict_and_warmup_report_actual_CPU_backend(self):
        from edge_runtime.profiles.factory_encoder import FactorySegmentation
        model=object.__new__(FactorySegmentation);model.device='CPU';model.backend_name='openvino_cpu_neural';model.execution_devices=['CPU']
        model.input=0;model.output=1;model.output_logits=False;model.growth_threshold=None
        model.request=types.SimpleNamespace(infer=lambda values:{1:np.zeros((1,1,640,256),dtype=np.float32)})
        _,candidates,metrics=model.predict(np.zeros((64,24,3),dtype=np.uint8),.5,4)
        self.assertEqual(candidates,[]);self.assertEqual(metrics['model_backend'],'openvino_cpu_neural');self.assertEqual(metrics['encoder_execution_devices'],['CPU'])
        model.initialization_report={};model.predict=lambda *args:None
        warm=model.warmup(.5,4,iterations=1)
        self.assertIn('OS/CPU scheduling',warm['residual_latency_caveat']);self.assertNotIn('OS/GPU',warm['residual_latency_caveat'])

    def test_CPU_property_audit_never_queries_GPU_keys(self):
        from probe_factory_stfpm_cpu import property_audit
        queried=[]
        class Compiled:
            def get_property(self,key):queried.append(key);return 'fake CPU property'
            def get_runtime_model(self):return types.SimpleNamespace(get_ops=lambda:[])
        property_audit(types.SimpleNamespace(compiled=Compiled()))
        self.assertIn('INFERENCE_NUM_THREADS',queried);self.assertFalse(any('GPU' in key for key in queried))

    def test_CPU_metadata_cannot_silently_select_old_GPU_encoder_or_forest(self):
        from edge_runtime.profiles.factory_defect import SurfaceDefectModel
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fake.joblib';path.write_bytes(b'not loaded')
            for version in ['surface_mobilenet_early_v1','surface_gray_multiscale_v1']:
                meta={'feature_version':version,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'openvino_device':'CPU','openvino_cpu_threads':4}
                path.with_suffix('.joblib.json').write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError,'only for fused neural segmentation'):SurfaceDefectModel(path)

    def test_actual_compiled_configuration_guard_rejects_wrong_or_unsupported_recipe(self):
        from probe_factory_stfpm_cpu import cpu_configuration_passes
        values={'INFERENCE_PRECISION_HINT':"<Type: 'float32'>",'INFERENCE_NUM_THREADS':4,'NUM_STREAMS':1,
                'PERFORMANCE_HINT':'PerformanceMode.LATENCY','EXECUTION_DEVICES':['CPU']}
        audit={key:{'supported':True,'value':value} for key,value in values.items()}
        self.assertTrue(cpu_configuration_passes(audit))
        for key,value in [('INFERENCE_PRECISION_HINT',"<Type: 'float16'>"),('INFERENCE_NUM_THREADS',2),('NUM_STREAMS',2),
                          ('PERFORMANCE_HINT','PerformanceMode.THROUGHPUT'),('EXECUTION_DEVICES',['GPU.0'])]:
            self.assertFalse(cpu_configuration_passes(audit|{key:{'supported':True,'value':value}}))
        self.assertFalse(cpu_configuration_passes(audit|{'INFERENCE_NUM_THREADS':{'supported':False,'error':'unsupported'}}))


if __name__=='__main__':unittest.main()
