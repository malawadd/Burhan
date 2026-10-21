"""L8 isolated architecture-probe adapter; shared runtime/defaults are untouched."""
import hashlib,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.factory_encoder import FactorySegmentation as BaseFactorySegmentation
SCHEMA=ROOT/'output/optimization/factory-round3l6-stage-diagnosis/append-only-output-manifest.json'
SCHEMA_SHA='4df46f2bde8bb83a7a8475332a0c7686c3322bf3993e5ccbd07c5055ded54e6a'
INITIAL_GRAPH_SHA='b8538781f53b1e58cc6c45a19fac98326e014e2f80ccaaa66975e5cc88bca6c5'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_rows():
    if sha(SCHEMA)!=SCHEMA_SHA:raise ValueError('Exact75-output schema changed')
    value=json.loads(SCHEMA.read_text(encoding='utf-8'));rows=value['ordered_outputs']
    if len(rows)!=75 or rows[0]['tensor']!='anomaly_score':raise ValueError('Fixed75-output inventory required')
    return rows


def validate_ports(ports,rows):
    if len(ports)!=75 or len(rows)!=75:raise ValueError('All75 outputs must remain declared and compiled')
    expected={r['tensor'] for r in rows}
    for port,row in zip(ports,rows):
        names=set(port.get_names())
        if (names&expected!={row['tensor']} or list(port.shape)!=row['shape']
                or row['ONNX_elem_type']!=1 or port.get_element_type().get_type_name()!='f32'):
            raise ValueError('Protected output names/order/shapes/types differ')
    return ports[0]


class SharedOutputRequest:
    """Return native views; inherited prediction immediately owns the final copy."""
    def __init__(self,request):self.inner=request
    def infer(self,inputs):
        return self.inner.infer(inputs,share_outputs=True)


class FactoryTappedSegmentation(BaseFactorySegmentation):
    """Constructor only: inherit the exact deployed predict/warmup implementation."""
    runtime_variant='host_copy'
    expected_model_sha256=INITIAL_GRAPH_SHA
    def __init__(self,path,output_logits=False,growth_threshold=None,precision_hint='f16',disable_winograd_convolution=True,device='GPU',cpu_threads=None):
        started=time.perf_counter()
        if (output_logits is not False or precision_hint!='f16' or disable_winograd_convolution is not True
                or device!='GPU' or cpu_threads is not None):raise ValueError('L8 requires the one GPU/f16/no-Winograd recipe and bounded score output')
        if self.runtime_variant not in {'host_copy','shared_views'}:raise ValueError('Only the two separately declared output-copy variants are allowed')
        if sha(path)!=self.expected_model_sha256:raise ValueError('Exact protocol-bound75-output graph identity required')
        rows=protected_rows()
        import openvino as ov
        self.core=ov.Core()
        if not any(d.startswith('GPU') for d in self.core.available_devices):raise ValueError('L8 explicitly requires the installed Intel GPU')
        graph=self.core.read_model(str(path))
        if list(graph.input().shape)!=[1,3,1280,512]:raise ValueError('L8 native input geometry changed')
        validate_ports(list(graph.outputs),rows)
        self.compiled=self.core.compile_model(graph,'GPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16','GPU_DISABLE_WINOGRAD_CONVOLUTION':True})
        self.output=validate_ports(list(self.compiled.outputs),rows)
        self.input=self.compiled.input();request=self.compiled.create_infer_request()
        self.request=SharedOutputRequest(request) if self.runtime_variant=='shared_views' else request
        self.execution_devices=list(self.compiled.get_property('EXECUTION_DEVICES'))
        self.output_logits=False;self.growth_threshold=growth_threshold;self.grid_height=640;self.grid_width=256
        self.precision_hint='f16';self.disable_winograd_convolution=True;self.device='GPU';self.cpu_threads=None
        self.backend_name=f'openvino_gpu_tapped75_{self.runtime_variant}_neural'
        self.initialization_report={'compile_and_request_ms':round((time.perf_counter()-started)*1000,2),'warmup_completed':False,
            'all_graph_outputs_retained':75,'schema_sha256':SCHEMA_SHA,'model_backend':self.backend_name,
            'readback_policy':'Inherited Base.predict explicitly copies anomaly_score only; framework execution/materialization of all75 remains and must be timed',
            'runtime_variant':self.runtime_variant,'native_share_outputs':self.runtime_variant=='shared_views',
            'shared_runtime_files_modified':False,'architecture_probe_only':True}


def adapter_type(variant,expected_model_sha256=INITIAL_GRAPH_SHA):
    if variant not in {'host_copy','shared_views'}:raise ValueError('Unknown fixed output-copy variant')
    return type(f'FactoryTapped_{variant}',(FactoryTappedSegmentation,),{'runtime_variant':variant,'expected_model_sha256':expected_model_sha256})
