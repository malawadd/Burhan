"""Hash/integrity and native-Torch versus CPU-ONNX parity on fixed TRAIN pixels."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,functools,hashlib,json
import cv2,numpy as np,onnx,onnxruntime as ort,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)


def verify_yolo(manifest_path=None,checkpoint_path=None,output_path=None):
    from ultralytics import YOLO
    manifest_path=Path(manifest_path) if manifest_path else ROOT/'models/construction/expansion-sdnet-v1/yolov8n/manifest.json'
    directory=manifest_path.parent
    path=directory/'model.onnx';manifest=json.loads((directory/'manifest.json').read_text())
    size=manifest['input_size']
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=manifest['model_sha256']:raise ValueError('ONNX hash mismatch')
    graph=onnx.load(str(path),load_external_data=False);onnx.checker.check_model(graph)
    if any(x.data_location==onnx.TensorProto.EXTERNAL or x.external_data for x in graph.graph.initializer):
        raise ValueError('External-data ONNX not bound by model SHA')
    dataset=json.loads((ROOT/'datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json').read_text())
    row=next(r for r in dataset['records']if r['split']=='train')
    image=cv2.imread(str(ROOT/row['image']));rgb=cv2.cvtColor(cv2.resize(image,(size,size)),cv2.COLOR_BGR2RGB)
    tensor=np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255
    checkpoint_path=Path(checkpoint_path) if checkpoint_path else ROOT/'output/training/construction-expansion-sdnet-v1/yolo-fit/weights/last.pt'
    model=YOLO(str(checkpoint_path)).model.float().eval()
    with torch.no_grad():reference=model(torch.from_numpy(tensor))[0].numpy()
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    actual_output=session.run(None,{session.get_inputs()[0].name:tensor})[0]
    error=np.abs(reference-actual_output)
    passed=bool(np.allclose(reference,actual_output,atol=.002,rtol=.0001))
    result={'model_sha256':actual,'external_data':False,'onnx_checker_pass':True,'training_input':row['image'],
            'training_input_sha256':row['source_sha256'],'preprocessing':f'{size}linearresizeRGBNCHWfloat32div255; source640square',
            'torch_threads':torch.get_num_threads(),'ort_intra_threads':1,'ort_inter_threads':1,'providers':session.get_providers(),
            'reference_shape':list(reference.shape),'onnx_shape':list(actual_output.shape),'max_absolute_error':float(error.max()),
            'mean_absolute_error':float(error.mean()),'tolerance_absolute':.002,'tolerance_relative':.0001,'passed':passed}
    output=Path(output_path) if output_path else ROOT/'output/training/construction-expansion-sdnet-v1/yolov8n-cpu-onnx-parity.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not passed:raise ValueError('CPU ONNX parity failed')


def verify_ssd():
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
    from torch import nn
    directory=ROOT/'models/construction/expansion-sdnet-v1/ssdlite320'
    path=directory/'model.onnx';manifest=json.loads((directory/'manifest.json').read_text())
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=manifest['model_sha256']:raise ValueError('ONNX hash mismatch')
    graph=onnx.load(str(path),load_external_data=False);onnx.checker.check_model(graph)
    if any(x.data_location==onnx.TensorProto.EXTERNAL or x.external_data for x in graph.graph.initializer):raise ValueError('External ONNX data')
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=91)
    channels=[branch[-1].in_channels for branch in model.head.classification_head.module_list]
    model.head.classification_head=SSDLiteClassificationHead(channels,model.anchor_generator.num_anchors_per_location(),3,
        functools.partial(nn.BatchNorm2d,eps=.001,momentum=.03))
    model.load_state_dict(torch.load(directory/'model-state.pt',map_location='cpu',weights_only=True));model.eval()
    data=json.loads((ROOT/'datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json').read_text())
    row=next(r for r in data['records']if r['split']=='train')
    image=cv2.imread(str(ROOT/row['image']));rgb=cv2.cvtColor(cv2.resize(image,(320,320),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    tensor=np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255
    with torch.no_grad():reference=model([torch.from_numpy(tensor[0])])[0]
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    outputs=dict(zip([v.name for v in session.get_outputs()],session.run(None,{session.get_inputs()[0].name:tensor})))
    errors={};passed=True
    for name in ['boxes','labels','scores']:
        expected=reference[name].numpy();out=outputs[name]
        same_shape=expected.shape==out.shape
        maximum=float(np.abs(expected-out).max()) if same_shape and expected.size else 0.
        close=same_shape and bool(np.allclose(expected,out,atol=.002,rtol=.0001))
        passed &= close;errors[name]={'shape':list(out.shape),'same_shape':same_shape,'max_absolute_error':maximum,'passed':close}
    result={'model_sha256':actual,'external_data':False,'onnx_checker_pass':True,'training_input':row['image'],
            'training_input_sha256':row['source_sha256'],'torch_threads':torch.get_num_threads(),'ort_threads':1,
            'providers':session.get_providers(),'outputs':errors,'passed':bool(passed),'internal_nms_iou':.55}
    (ROOT/'output/training/construction-expansion-sdnet-v1/ssdlite320-cpu-onnx-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not passed:raise ValueError('CPU ONNX parity failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['yolo','ssd'],default='yolo')
    parser.add_argument('--manifest',type=Path);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.model=='ssd':verify_ssd()
    else:verify_yolo(args.manifest,args.checkpoint,args.output)
