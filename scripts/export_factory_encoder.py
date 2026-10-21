"""Official torchvision weights -> fixed local early-feature ONNX encoder.

Run only in the separate torch export environment; runtime uses OpenVINO.
"""
from pathlib import Path
import argparse,hashlib,json,time
import torch,torchvision
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download-only',action='store_true')
a=p.parse_args()
directory=ROOT/'models/factory/encoder_sources';directory.mkdir(parents=True,exist_ok=True)
weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
state=torch.hub.load_state_dict_from_url(weights.url,model_dir=str(directory),check_hash=True,weights_only=True,progress=False)
path=directory/Path(weights.url).name
provenance={'weights_url':weights.url,'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'weights_bytes':path.stat().st_size,'torch':torch.__version__,'torchvision':torchvision.__version__,
            'weights_loading':'torch.hub officialURL check_hash=True weights_only=True',
            'feature_layers':[0,3],'input_shape':[1,3,1280,512],'input_dtype':'uint8 BGR',
            'normalization':'BGR->RGB, /255, ImageNet mean/std inside graph',
            'purpose':'Frozen pretrained early edges and mid-level textures; head trained separately on training masks'}
(directory/'weights-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
if not a.download_only:
    torch.set_num_threads(2)
    network=mobilenet_v3_small(weights=None);network.load_state_dict(state);network.eval()
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.blocks=torch.nn.Sequential(*list(network.features.children())[:4])
            self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        def forward(self,value):
            value=value.to(torch.float32)[:,[2,1,0]]/255.
            value=(value-self.mean)/self.std
            first=None
            for index,block in enumerate(self.blocks):
                value=block(value)
                if index==0:first=value
            return first,value
    model=Encoder().eval();artifact=ROOT/'models/factory/mobilenetv3_early_512.onnx'
    with torch.inference_mode():
        torch.onnx.export(model,(torch.zeros((1,3,1280,512),dtype=torch.uint8),),str(artifact),
                          input_names=['bgr'],output_names=['stride2','stride8'],opset_version=17,
                          dynamo=False,do_constant_folding=True)
    provenance.update(onnx_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),onnx_bytes=artifact.stat().st_size)
    artifact.with_suffix('.onnx.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2))
