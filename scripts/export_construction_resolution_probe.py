"""Protected higher-resolution export of the actual failed fit; calibration only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
NAME='evcivil-yolov8n-r3-adapted-640-calprobe'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=ROOT/'output/training/construction-round3/evcivil-yolov8n-r3-adapted/weights/best.pt'
    parent=ROOT/'models/construction/evcivil-yolov8n-r3-adapted.provenance.json'
    prior=json.loads(parent.read_text(encoding='utf8'))
    folder=ROOT/'output/training/construction-round3/resolution-probe640';folder.mkdir(parents=True,exist_ok=True)
    protocol={'schema_version':1,'candidate':NAME,'frozen_before_resolution_probe_forward':True,
        'source_local_checkpoint':str(source.relative_to(ROOT)),'source_checkpoint_sha256':sha(source),
        'source_416_onnx_sha256':prior['onnx_sha256'],'source_provenance_sha256':sha(parent),
        'export_input_size':640,'actual_training_input_size':416,'new_training':False,
        'scope':'Unchanged122calibrationimages only; no heldout or exposed forward',
        'reason':'Higher-detail object grid for small chips/thin cracks; resolution change selected before new fit or any heldout result',
        'calibration_protocol':'Existing raw-output exactruntime cutoffs/NMS/globalcap calibration, unchanged thresholdgrid andgates',
        'negative_manifest_sha256':prior['negative_manifest_sha256'],
        'calibration_exposures_sha256':sha(ROOT/'datasets/construction/evcivil-round3/exposures.json'),
        'capacity_scope':'No real-timeclaim from export/calibration; final isolated speed separate'}
    frozen=folder/'preregistered-protocol.json'
    if frozen.exists()and json.loads(frozen.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Resolution-probe protocol/source changed')
    frozen.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    # Export from a separate trusted LOCAL checkpoint copy so best.onnx beside
    # the original fit is never overwritten by a different input geometry.
    local=folder/'source-local.pt';shutil.copyfile(source,local)
    model=YOLO(str(local),task='detect')
    if model.model.names!={0:'spalling',1:'crack'}:raise RuntimeError('Actual learned classmap changed')
    output=Path(model.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,batch=1))
    target=ROOT/f'models/construction/{NAME}.onnx';shutil.copyfile(output,target)
    metadata=prior|{'experiment':NAME,'onnx_sha256':sha(target),'model_path':str(target),'input_size':640,
                    'resolution_probe':protocol,'resolution_protocol_sha256':sha(frozen),
                    'selection':'Same learned weights, higher-resolution export; actual training remains416. Calibration-only probe, no new fit or heldout selection.'}
    target.with_suffix('.provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
