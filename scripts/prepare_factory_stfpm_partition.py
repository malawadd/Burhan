"""L9 deterministic source-ancestor extraction only; no weights load or NN."""
import copy,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from factory_stfpm_cpu_common import L4_NN,L4_NN_SHA,L4_REF,L4_REF_SHA,L4_STATE_SHA
from factory_tapped_segmentation import SCHEMA,SCHEMA_SHA,protected_rows
OUT=ROOT/'output/optimization/factory-round3l9-partitioned';NAME='spatial_round3l9_partitioned_bn64'
MODEL=ROOT/f'models/factory/{NAME}-step0.onnx';MANIFEST=OUT/'six-feature-extraction-manifest.json'
LABELS=['teacher_block0','student_block0','teacher_block3','student_block3','teacher_block6','student_block6']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def feature_rows():
    bylabel={r['label']:r for r in protected_rows()}
    return [bylabel[label] for label in LABELS]


def extract(source,rows):
    """Walk fixed named output ancestors; retain original node/weight/input bytes."""
    import onnx
    names=[r['tensor'] for r in rows];producers={}
    for i,node in enumerate(source.graph.node):
        if any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for a in node.attribute):raise ValueError('Unexpected subgraph')
        for output in node.output:
            if output in producers:raise ValueError('Duplicate graph producer')
            producers[output]=i
    needed=set(names);keep=set();stack=list(names)
    while stack:
        name=stack.pop()
        if name not in producers:continue
        index=producers[name]
        if index in keep:continue
        keep.add(index)
        for value in source.graph.node[index].input:
            if value and value not in needed:needed.add(value);stack.append(value)
    result=copy.deepcopy(source)
    del result.graph.node[:];result.graph.node.extend(copy.deepcopy(n) for i,n in enumerate(source.graph.node) if i in keep)
    del result.graph.initializer[:];result.graph.initializer.extend(copy.deepcopy(v) for v in source.graph.initializer if v.name in needed)
    if source.graph.sparse_initializer or source.functions:raise ValueError('Unexpected sparse weights/functions')
    del result.graph.output[:]
    for row in rows:
        result.graph.output.append(onnx.helper.make_tensor_value_info(row['tensor'],onnx.TensorProto.FLOAT,row['shape']))
    inputs={v.name for v in source.graph.input};weights={v.name for v in source.graph.initializer}
    unknown=needed-set(producers)-inputs-weights
    if unknown:raise ValueError('Unresolved ancestor input: '+repr(unknown))
    if [v.SerializeToString() for v in result.graph.input]!=[v.SerializeToString() for v in source.graph.input]:raise ValueError('Input bytes changed')
    if [v.SerializeToString() for v in result.opset_import]!=[v.SerializeToString() for v in source.opset_import]:raise ValueError('Opset changed')
    for node,index in zip(result.graph.node,sorted(keep)):
        if node.SerializeToString()!=source.graph.node[index].SerializeToString():raise ValueError('Retained node bytes changed')
    sourceweights={v.name:v.SerializeToString() for v in source.graph.initializer}
    if any(v.SerializeToString()!=sourceweights[v.name] for v in result.graph.initializer):raise ValueError('Weight bytes changed')
    onnx.checker.check_model(result)
    proof={'retained_node_indices':sorted(keep),'original_nodes':len(source.graph.node),'retained_nodes':len(keep),
           'retained_node_bytes_sha256':hashlib.sha256(b''.join(v.SerializeToString() for v in result.graph.node)).hexdigest(),
           'retained_initializer_names':[v.name for v in result.graph.initializer],
           'retained_initializer_bytes_sha256':hashlib.sha256(b''.join(v.SerializeToString() for v in result.graph.initializer)).hexdigest(),
           'all_retained_node_initializer_input_opset_bytes_identical':True}
    return result,proof


def metadata(model_sha):
    return {'feature_version':'surface_mobilenet_seg_v1','profile_version':NAME,'model_sha256':model_sha,
            'model_name':'ARCHITECTURE PROBE ONLY GPU six raw features plus NumPy float32 head','output_logits':False,
            'openvino_device':'GPU','openvino_inference_precision':'f16','openvino_disable_winograd_convolution':True,
            'pixel_threshold':.05,'minimum_component_pixels':4,'hysteresis_growth_threshold':None,
            'score_semantics':'Bounded_multiscale_teacher_student_anomaly_disagreement_not_probability',
            'supported_height_width_ratio':[2.3,2.7],'architecture_probe_only':True,'all_outputs_retained':6,
            'retained_BN_state_sha256':L4_STATE_SHA,'manual_review_only':True,'automatic_acceptance':False,
            'automatic_rejection':False,'field_validated':False,'app_default_unchanged':True,
            'license':'DerivedKolektor noncommercial CC-BY-NC-SA-4.0 research prototype; officialImageNetteacher terms'}


def prepare():
    import onnx
    if any(p.exists() for p in [MODEL,MODEL.with_suffix('.onnx.json'),MANIFEST]):raise ValueError('Preserve first L9 extraction')
    if sha(L4_NN)!=L4_NN_SHA or sha(L4_REF)!=L4_REF_SHA or sha(SCHEMA)!=SCHEMA_SHA:raise ValueError('Retained source identity changed')
    rows=feature_rows();model,proof=extract(onnx.load(str(L4_NN)),rows)
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    with MODEL.open('xb') as stream:stream.write(model.SerializeToString())
    write_new(MODEL.with_suffix('.onnx.json'),metadata(sha(MODEL)))
    receipt={'source_model_sha256':L4_NN_SHA,'source_schema_sha256':SCHEMA_SHA,'model_sha256':sha(MODEL),
             'metadata_sha256':sha(MODEL.with_suffix('.onnx.json')),'reference_sha256':L4_REF_SHA,
             'ordered_outputs':rows,'single_case_output_bytes':sum(__import__('math').prod(r['shape'])*4 for r in rows),
             'extraction':proof,'NN_executed':False,'weights_loaded':False}
    write_new(MANIFEST,receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['extraction','ordered_outputs']},indent=2))


if __name__=='__main__':prepare()
