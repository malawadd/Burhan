"""L9 GPU feature pyramid/CPU float32 head; inherited deployed postprocessing."""
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from edge_runtime.profiles.factory_encoder import FactorySegmentation as BaseFactorySegmentation
from prepare_factory_stfpm_partition import MODEL,MANIFEST,sha,feature_rows
from factory_stfpm_partition_math import score_from_features


def validate_ports(ports,rows):
    if len(ports)!=6:raise ValueError('Exactly six raw encoder outputs required')
    expected={r['tensor'] for r in rows}
    for port,row in zip(ports,rows):
        if (set(port.get_names())&expected!={row['tensor']} or list(port.shape)!=row['shape']
                or port.get_element_type().get_type_name()!='f32'):raise ValueError('L9 fixed feature names/order/shapes/types changed')


class PartitionRequest:
    def __init__(self,request,ports,output):self.inner=request;self.ports=ports;self.output=output;self.last_timing=None
    def infer(self,inputs):
        start=time.perf_counter();native=self.inner.infer(inputs,share_outputs=True);ready=time.perf_counter()
        # Views stay valid throughout the synchronous CPU head. No new request or
        # in-place write occurs; final score is independently owned before return.
        arrays=[native[port] for port in self.ports];views=time.perf_counter()
        score=score_from_features(arrays);done=time.perf_counter()
        self.last_timing={'gpu_request_and_framework_output_ms':1000*(ready-start),'feature_view_access_ms':1000*(views-ready),
            'CPU_float32_head_ms':1000*(done-views),'partition_total_ms':1000*(done-start)}
        return {self.output:score}


class FactoryPartitionedSegmentation(BaseFactorySegmentation):
    expected_model_sha256='b8ecf60fa5ac104857d9c0713d44ea94c4c5c253be261f8d8df6e990da07018c'
    def __init__(self,path,output_logits=False,growth_threshold=None,precision_hint='f16',disable_winograd_convolution=True,device='GPU',cpu_threads=None):
        started=time.perf_counter()
        if (output_logits is not False or precision_hint!='f16' or disable_winograd_convolution is not True
                or device!='GPU' or cpu_threads is not None):raise ValueError('L9 requires fixed GPU f16/noWinograd recipe')
        if sha(path)!=self.expected_model_sha256:raise ValueError('Exact protocol-bound extracted encoder required')
        rows=feature_rows()
        import openvino as ov
        self.core=ov.Core()
        if not any(d.startswith('GPU') for d in self.core.available_devices):raise ValueError('L9 explicitly requires Intel GPU')
        graph=self.core.read_model(str(path))
        if list(graph.input().shape)!=[1,3,1280,512]:raise ValueError('Native full input changed')
        validate_ports(list(graph.outputs),rows)
        self.compiled=self.core.compile_model(graph,'GPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True})
        ports=list(self.compiled.outputs);validate_ports(ports,rows)
        self.input=self.compiled.input();self.output='anomaly_score'
        self.request=PartitionRequest(self.compiled.create_infer_request(),ports,self.output)
        self.execution_devices=list(self.compiled.get_property('EXECUTION_DEVICES'))
        self.output_logits=False;self.growth_threshold=growth_threshold;self.grid_height=640;self.grid_width=256
        self.precision_hint='f16';self.disable_winograd_convolution=True;self.device='GPU';self.cpu_threads=None
        self.backend_name='openvino_gpu_features_numpy_f32_head_neural'
        self.initialization_report={'compile_and_request_ms':round((time.perf_counter()-started)*1000,2),'warmup_completed':False,
            'model_backend':self.backend_name,'feature_outputs':6,'native_share_outputs':True,
            'CPU_score_head':'NumPy f32 exact floors/unit-L2/quarter squareddisagreement/halfpixelbilinear/loggeomean',
            'shared_runtime_files_modified':False,'architecture_probe_only':True}


def adapter_type(model_sha256=None):
    return type('BoundFactoryPartitionedSegmentation',(FactoryPartitionedSegmentation,),
                {'expected_model_sha256':model_sha256 or FactoryPartitionedSegmentation.expected_model_sha256})
