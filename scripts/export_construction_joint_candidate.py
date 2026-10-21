"""Export the pinned AGPL joint-aware nano checkpoint with restricted weights-only loading.

This is an experimental research candidate, never an automatic runtime replacement.
"""
from __future__ import annotations
import argparse
import builtins
import collections
import hashlib
import importlib
import json
from pathlib import Path
import pickletools
import zipfile
from urllib.request import urlopen
import torch

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'models/construction/yolov8n-seg-cracks-joints.pt'
SHA='4aee1057161d3ce0130f438d118a3920997462f1ce187f24767c7ea5e1905e00'
SOURCE='https://huggingface.co/levanell/yolov8n-seg-cracks-joints/tree/44e73b6b015247bde629e9e7f12efef2ba4c7f3b'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--download',action='store_true',help='Explicitly fetch the pinned 6.8 MB checkpoint before export');args=parser.parse_args()
    if args.download and not CHECKPOINT.exists():
        try:
            from pip._vendor import truststore
            truststore.inject_into_ssl()
        except ImportError:pass
        url='https://huggingface.co/levanell/yolov8n-seg-cracks-joints/resolve/44e73b6b015247bde629e9e7f12efef2ba4c7f3b/yolov8n-seg-cracks-joints.pt'
        with urlopen(url,timeout=90) as response:blob=response.read()
        if hashlib.sha256(blob).hexdigest()!=SHA:raise RuntimeError('Downloaded checkpoint hash mismatch')
        CHECKPOINT.parent.mkdir(exist_ok=True,parents=True);CHECKPOINT.write_bytes(blob)
    torch.set_num_threads(1)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()!=SHA:raise RuntimeError('Candidate checkpoint hash mismatch')
    with zipfile.ZipFile(CHECKPOINT) as archive:
        raw=archive.read(next(n for n in archive.namelist() if n.endswith('data.pkl')))
    allowed=[builtins.set,collections.OrderedDict]
    for operation,argument,_ in pickletools.genops(raw):
        if operation.name=='GLOBAL':
            module,name=argument.split()
            if module=='__builtin__' and name=='set':continue
            if module=='collections' and name=='OrderedDict':continue
            if not (module.startswith('torch.') or module=='torch' or module.startswith('ultralytics.nn.')):
                raise RuntimeError(f'Unreviewed pickle global: {module}.{name}')
            allowed.append(getattr(importlib.import_module(module),name))
    with torch.serialization.safe_globals(allowed):
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    model=checkpoint['ema'] if checkpoint.get('ema') is not None else checkpoint['model']
    model=model.float().eval().fuse();model.requires_grad_(False)
    names=[str(model.names[i]).lower() for i in sorted(model.names)]
    if len(names)!=2 or 'joint' not in names[0] or 'crack' not in names[1]:
        raise RuntimeError(f'Unexpected joint/crack class mapping: {model.names}')
    head=model.model[-1];head.export=True;head.format='onnx';head.dynamic=False
    output=ROOT/'models/construction/crack-joints-yolov8n-seg.onnx'
    tensor=torch.zeros((1,3,640,640),dtype=torch.float32)
    with torch.inference_mode():
        result=model(tensor)
        print('Output shapes:',[list(x.shape) for x in result])
        torch.onnx.export(model,tensor,str(output),input_names=['images'],output_names=['output0','output1'],
                          opset_version=17,dynamo=False,do_constant_folding=True)
    metadata={'source':SOURCE,'checkpoint_sha256':SHA,'onnx_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'classes':model.names,'license':'AGPL-3.0; merged Roboflow training-source licenses not cleared for commercial redistribution',
              'loading':'torch.load weights_only=True with explicitly reviewed standard Torch/Ultralytics globals',
              'status':'Experimental candidate, not runtime default; no validation claim from author metrics'}
    output.with_suffix('.provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf8');print(json.dumps(metadata,indent=2))
