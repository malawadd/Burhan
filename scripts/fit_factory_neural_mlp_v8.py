"""Fit sampled train-only pointwise head, export fused local segmentation ONNX."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch,torchvision
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2);torch.manual_seed(20261025)
directory=ROOT/'output/optimization/factory-accuracy'
data=np.load(directory/'neural-hard-train-samples.npz');x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y'])[:,None]
mean=x.mean(0);std=x.std(0).clamp_min(.05);x=(x-mean)/std
head=torch.nn.Sequential(torch.nn.Linear(40,32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
loss_function=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.]))
started=time.perf_counter();losses=[]
for epoch in range(22):
 order=torch.randperm(len(x));total=0.
 for start in range(0,len(x),8192):
  batch=order[start:start+8192];optimizer.zero_grad();loss=loss_function(head(x[batch]),y[batch]);loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
 losses.append(total/len(x));print(f'Head epoch{epoch+1}/22loss{losses[-1]:.5f}',flush=True)
head.eval()
weights=ROOT/'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
network=mobilenet_v3_small(weights=None);network.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));network.eval()
class Segmentation(torch.nn.Module):
 def __init__(self):
  super().__init__();self.blocks=torch.nn.Sequential(*list(network.features.children())[:4])
  self.first=torch.nn.Conv2d(40,32,1);self.middle=torch.nn.Conv2d(32,16,1);self.last=torch.nn.Conv2d(16,1,1)
  with torch.no_grad():
   self.first.weight.copy_((head[0].weight/std[None]).reshape(32,40,1,1));self.first.bias.copy_(head[0].bias-(head[0].weight*mean[None]/std[None]).sum(1))
   self.middle.weight.copy_(head[2].weight.reshape(16,32,1,1));self.middle.bias.copy_(head[2].bias)
   self.last.weight.copy_(head[4].weight.reshape(1,16,1,1));self.last.bias.copy_(head[4].bias)
  self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
 def forward(self,value):
  value=(value.float()[:,[2,1,0]]/255.-self.mean)/self.std;early=None
  for index,block in enumerate(self.blocks):
   value=block(value)
   if index==0:early=value
  value=torch.nn.functional.interpolate(value,size=(640,256),mode='bilinear',align_corners=False)
  value=torch.cat([early,value],dim=1)
  return self.last(torch.relu(self.middle(torch.relu(self.first(value)))))
model=Segmentation().eval();artifact=ROOT/'models/factory/kolektor_mobilenet_seg_v8.onnx'
with torch.inference_mode():
 torch.onnx.export(model,(torch.zeros((1,3,1280,512),dtype=torch.uint8),),str(artifact),opset_version=17,
                   input_names=['bgr'],output_names=['defect_scores'],dynamo=False,do_constant_folding=True)
report={'epochs':22,'losses':losses,'fit_seconds':time.perf_counter()-started,'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'model_bytes':artifact.stat().st_size,'pretrained_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'torch':torch.__version__,'torchvision':torchvision.__version__}
(directory/'neural-head-fit-v8.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
