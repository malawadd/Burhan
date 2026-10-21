"""Restricted weights-only restoration; no architecture rebuild/forward/export.

Only audited official Torch/Ultralytics classes are allowed. Generic legacy
getattr/dill operations are replaced by exact restricted shims, not imported.
"""
from pathlib import Path
import hashlib,importlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/construction/cubit-author-release-v1'
EXPECTED='e463069e06daf66fd99c21f5d9a3d88fe7a48c71d43263ed4d338a56e6a86d30'
REVIEWED={
 'builtins.getattr','dill._dill._load_type','numpy.dtype','numpy._core.multiarray.scalar',
 'torch.nn.modules.activation.SiLU','torch.nn.modules.batchnorm.BatchNorm2d','torch.nn.modules.container.ModuleList',
 'torch.nn.modules.container.Sequential','torch.nn.modules.conv.Conv2d','torch.nn.modules.conv.ConvTranspose2d',
 'torch.nn.modules.loss.BCEWithLogitsLoss','torch.nn.modules.pooling.MaxPool2d','torch.nn.modules.upsampling.Upsample',
 'ultralytics.nn.modules.block.Bottleneck','ultralytics.nn.modules.block.C2f','ultralytics.nn.modules.block.DFL',
 'ultralytics.nn.modules.block.Proto','ultralytics.nn.modules.block.SPPF','ultralytics.nn.modules.conv.Concat',
 'ultralytics.nn.modules.conv.Conv','ultralytics.nn.modules.head.Detect','ultralytics.nn.modules.head.Segment',
 'ultralytics.nn.tasks.SegmentationModel','ultralytics.utils.IterableSimpleNamespace',
 'ultralytics.utils.loss.BboxLoss','ultralytics.utils.loss.v8SegmentationLoss','ultralytics.utils.tal.TaskAlignedAssigner'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def restricted_load_type(name):
    # Original pickle requests dill._load_type('set') for module hook state.
    # No generic dill loader, code/function/type synthesis or module lookup.
    if name!='set':raise RuntimeError('Unreviewed legacy dill type request: '+repr(name))
    return set


def restricted_getattr(owner,name):
    from ultralytics.nn.modules.head import Detect
    # Original Segment stores its known official Detect.forward reference.
    if owner is not Detect or name!='forward':raise RuntimeError('Unreviewed legacy getattr request')
    return Detect.forward


def main():
    import torch,numpy as np
    from torch import nn
    from ultralytics.nn.tasks import SegmentationModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=OUT/'author-best.pt'
    if sha(source)!=EXPECTED:raise RuntimeError('Primary author checkpoint bytes differ')
    unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(source))
    if unsafe-REVIEWED:raise RuntimeError('Reject unreviewed globals: '+repr(sorted(unsafe-REVIEWED)))
    protocol=OUT/'extraction-protocol-before-weights-only-load.json'
    if protocol.exists():raise RuntimeError('Preserve previous extraction attempt; no silent retry')
    write(protocol,{'source_sha256':EXPECTED,'extractor_sha256':sha(Path(__file__)),'reviewed_globals':sorted(unsafe),
                    'safe_legacy_shims':{'dill._dill._load_type':'only exact set request; returns builtin set, never generic dill',
                                         'builtins.getattr':'only exact official Detect class + forward attribute; no generic attribute lookup'},
                    'scope':'torch.load(weights_only=True) plus explicit known types; audited object restoration only. No weights_only=False, arbitrary unpickle, architecture rebuild, network forward or export. Plain state_dict+YAML only.',
                    'class_mapping':{'0':'crack','1':'spalling'},'license':'Standalone model license and exact train-source overlap unknown; research only.'})
    allowed=[set,np.dtypes.Float16DType,np.dtypes.Float32DType,np.dtypes.Float64DType]
    for symbol in sorted(unsafe):
        if symbol=='builtins.getattr':allowed.append((restricted_getattr,symbol))
        elif symbol=='dill._dill._load_type':allowed.append((restricted_load_type,symbol))
        else:
            module,name=symbol.rsplit('.',1);allowed.append(getattr(importlib.import_module(module),name))
    with torch.serialization.safe_globals(allowed):checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict):raise RuntimeError('Unexpected author checkpoint root')
    model=checkpoint.get('ema')
    if not isinstance(model,nn.Module):model=checkpoint.get('model')
    if not isinstance(model,SegmentationModel) or model.names!={0:'crack',1:'spalling'}:raise RuntimeError('Author model type/class order differs')
    if int(model.model[-1].nc)!=2 or int(model.model[-1].nm)!=32:raise RuntimeError('Author segment-head contract differs')
    state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):raise RuntimeError('Nonfinite source tensor')
    target=OUT/'author-state-dict.pt'
    if target.exists():raise RuntimeError('Preserve extracted source state')
    torch.save(state,target)
    write(OUT/'author-model-yaml.json',model.yaml)
    report={'source_sha256':EXPECTED,'extraction_protocol_sha256':sha(protocol),'extractor_sha256':sha(Path(__file__)),
            'state_dict_sha256':sha(target),'state_tensor_count':len(state),'state_total_elements':sum(v.numel() for v in state.values()),
            'author_names':model.names,'author_task':'segment','head_nc':2,'head_mask_coefficients':32,'model_yaml_sha256':sha(OUT/'author-model-yaml.json'),
            'weights_only_true':True,'all_source_tensors_finite':True,'audited_object_restoration_only':True,'architecture_rebuilt':False,'network_forward_performed':False,
            'torch_version':torch.__version__,'license_status':'Standalone checkpoint rights and training overlap unknown; research only.'}
    write(OUT/'safe-extraction.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
