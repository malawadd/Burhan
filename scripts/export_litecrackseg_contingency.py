"""Export the reviewed author LiteCrackSeg TUT checkpoint; compute handoff required.

No installed app model is replaced. Author preprocessing is RGB FLOAT 0..255:
the source's ImageNet constants are unused by BOTH actual train/test pipelines.
The model/repository license is undeclared; this remains a research contingency.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import types
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'models/construction/thin-contingency'
CHECKPOINT_SHA='c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497'
REVISION='082d33111f1a3b9ac1d78e984b06fd3530bf3533'


def reviewed_model():
    source=BASE/'lite-source/model/LiteCrackSeg.py'
    text=source.read_text(encoding='utf8')
    # No trainer/config/Visdom imports, no downloaded pickle objects, no network.
    # Instantiate the architecture from the pinned local JSON, then load every
    # actual trained tensor strictly. Unused ImageNet weights are unnecessary.
    old='MobileViTModel.from_pretrained("model/mobilevit-xx-small", local_files_only=True).base_model'
    replacement=f'MobileViTModel(MobileViTConfig.from_json_file({str(BASE/"lite-source/model/mobilevit-xx-small/config.json")!r})).base_model'
    if text.count(old)!=1:raise RuntimeError('Reviewed encoder-constructor source changed')
    text=text.replace(old,replacement).replace('from transformers import MobileViTModel','from transformers import MobileViTModel, MobileViTConfig')
    # DropPath has no parameters and is exactly identity in eval; this avoids
    # adding a timm dependency merely for its inference-inactive utility.
    text=text.replace('from timm.models.layers import DropPath, trunc_normal_',
        'from torch.nn.init import trunc_normal_\nclass DropPath(nn.Identity):\n    def __init__(self, drop_prob=0.): super().__init__()')
    text=text.replace('from thop import profile','')
    module=types.ModuleType('reviewed_litecrackseg_export');module.__file__=str(source)
    exec(compile(text,str(source),'exec'),module.__dict__)
    model=module.LiteCrackSeg().eval()
    checkpoint=BASE/'litecrackseg-TUT.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise RuntimeError('Checkpoint hash mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):raise RuntimeError('Expected a tensor-only state dictionary')
    model.load_state_dict(state,strict=True)
    return model,source,len(state)


class Export(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,bgr):
        rgb=bgr[:,[2,1,0],:,:].float()  # Author actual raw RGB float path; no /255 or ImageNet normalization.
        return torch.sigmoid(self.model(rgb)[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--size',type=int,default=384);args=parser.parse_args()
    if args.size not in [256,384,512]:raise ValueError('Frozen MobileViT-compatible square sizes only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,source,tensors=reviewed_model();wrapper=Export(model).eval()
    target=BASE/f'litecrackseg-tut-{args.size}.onnx'
    with torch.inference_mode():
        example=torch.zeros(1,3,args.size,args.size,dtype=torch.uint8)
        torch.onnx.export(wrapper,example,str(target),input_names=['bgr_uint8'],output_names=['crack_probability'],
                          opset_version=17,dynamo=False,do_constant_folding=True)
    import onnx
    graph=onnx.load(target);onnx.checker.check_model(graph)
    result={'file':str(target.relative_to(ROOT/'models/construction')),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source':f'https://github.com/1Kaleb/LiteCrackSeg/tree/{REVISION}','source_checkpoint_sha256':CHECKPOINT_SHA,
            'source_code_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'classes':['crack'],'input_size':args.size,
            'input':'uint8 BGR NCHW; RGB float0..255 conversion in graph, following actual author train/test code',
            'output':'One native-input-resolution sigmoid crack probability map; components are segmentation-derived, not detector instances',
            'loaded_state_tensors':tensors,'parameters':sum(p.numel() for p in model.parameters()),
            'export_adaptations':['Local configuration constructor instead of loading unused ImageNet tensor file','DropPath eval identity; source thop profiling import omitted','Main logits sigmoid only; auxiliary heads do not become separate detections'],
            'license':'Whole-model/source license undeclared; MobileViT subcomponent license does not clear whole-model use. Experimental evaluation only; commercial/redistribution clearance unresolved.',
            'accuracy_scope':'No accuracy claim from export. No heldout/exposed input used; evaluate on unchanged calibration only before final selection.'}
    target.with_suffix('.provenance.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
