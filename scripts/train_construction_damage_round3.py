"""Supervised crack/spalling detection fine-tuning on new real video exposures.

Coordinate training with the parent; capacity measurements require idle compute.
Exposed recordings cannot select checkpoints or thresholds. New fitting from
their capture folders must explicitly downgrade those old cases to development.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import pickletools
import time
import zipfile
import torch
from ultralytics import YOLO
from ultralytics.utils import callbacks
from ultralytics.models.yolo.detect.train import DetectionTrainer
import yaml

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/construction/yolov8n-seg-cracks-joints.pt'
SOURCE_SHA='4aee1057161d3ce0130f438d118a3920997462f1ce187f24767c7ea5e1905e00'
BASE=ROOT/'datasets/construction/evcivil-round3'


def source_state():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Source checkpoint hash mismatch')
    with zipfile.ZipFile(SOURCE) as source:
        data=source.read(next(n for n in source.namelist() if n.endswith('data.pkl')))
    allowed=[set]
    for op,arg,_ in pickletools.genops(data):
        if op.name!='GLOBAL':continue
        module,name=arg.split()
        if module=='__builtin__' and name=='set':continue
        if not(module=='torch' or module.startswith('torch.') or module.startswith('ultralytics.nn.') or module=='collections'):
            raise RuntimeError(f'Unreviewed pickle global {module}.{name}')
        allowed.append(getattr(importlib.import_module(module),name))
    with torch.serialization.safe_globals(allowed):checkpoint=torch.load(SOURCE,map_location='cpu',weights_only=True)
    network=checkpoint['ema'] if checkpoint.get('ema') is not None else checkpoint['model']
    if network.names!={0:'joint',1:'crack'}:raise RuntimeError('Source semantic class mapping changed')
    return network.float().state_dict()


def main():
    executed_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--epochs',type=int,default=32);a.add_argument('--image-size',type=int,default=416)
    a.add_argument('--batch',type=int,default=4);a.add_argument('--nominal-batch',type=int,default=16);a.add_argument('--threads',type=int,default=4);a.add_argument('--freeze',type=int,default=10)
    a.add_argument('--memory-format',choices=['contiguous','channels_last'],default='contiguous')
    a.add_argument('--exposure-policy',choices=['off','low_light_gain'],default='off')
    a.add_argument('--data-root',type=Path,default=BASE);a.add_argument('--continue-from',type=Path);a.add_argument('--learning-rate',type=float,default=.0015);a.add_argument('--max-minutes',type=float,default=12);a.add_argument('--fraction',type=float,default=1.0);a.add_argument('--name',default='evcivil-yolov8n-r3');args=a.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    data_root=args.data_root.resolve()
    if not data_root.is_relative_to((ROOT/'datasets/construction').resolve()):raise RuntimeError('Training data root must stay within construction datasets')
    if not (data_root/'background-import.json').exists():raise RuntimeError('True verified background negatives must be imported before training')
    protocol=json.loads((data_root/'protocol.json').read_text(encoding='utf8'))
    if not protocol.get('frozen_before_candidate_training',protocol.get('frozen_before_round3_model_training',False)):raise RuntimeError('Missing frozen candidate split protocol')
    if protocol.get('training_script_sha256') and protocol['training_script_sha256']!=executed_script_sha256:
        raise RuntimeError('Trainer source differs from the frozen future-fit protocol')
    if protocol.get('strict_configuration'):
        actual={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
        for name,value in protocol['configuration'].items():
            if name in actual and actual[name]!=value:raise RuntimeError(f'Command differs from frozen configuration: {name}')
    if args.exposure_policy!='off':
        prepared=json.loads((data_root/'data-protocol.json').read_text(encoding='utf8'))
        shared=ROOT/'edge_runtime/profiles/construction_preprocessing.py'
        if prepared['policy']!=args.exposure_policy or prepared['preprocessing_source_sha256']!=hashlib.sha256(shared.read_bytes()).hexdigest():
            raise RuntimeError('Prepared image policy differs from exact shared serving policy')
        native=data_root/'native-input-manifest.json'
        if hashlib.sha256(native.read_bytes()).hexdigest()!=prepared['native_input_manifest_sha256']:
            raise RuntimeError('Prepared image-input manifest changed')
        for row in json.loads(native.read_text(encoding='utf8')):
            image=(ROOT/row['image']).resolve()
            label=(ROOT/row['label']).resolve()
            if not image.is_relative_to(data_root) or not label.is_relative_to(data_root) or hashlib.sha256(image.read_bytes()).hexdigest()!=row['image_sha256'] or hashlib.sha256(label.read_bytes()).hexdigest()!=row['label_sha256']:
                raise RuntimeError('Prepared native model input changed')
        if protocol.get('prepared_data_protocol_sha256')!=hashlib.sha256((data_root/'data-protocol.json').read_bytes()).hexdigest():
            raise RuntimeError('Fit protocol does not pin prepared inputs')
    callbacks.add_integration_callbacks=lambda instance:None  # Per-process: no cloud experiment tracking or uploads.
    seed=32026;torch.manual_seed(seed)
    template=YOLO('yolov8n.yaml',task='detect').model.yaml;template['nc']=2
    model_yaml=ROOT/'models/construction/evcivil-yolov8n-round3.yaml';model_yaml.write_text(yaml.safe_dump(template),encoding='utf8')
    continuation=None
    if args.continue_from:
        local=args.continue_from.resolve()
        if not local.is_relative_to((ROOT/'output/training/construction-round3').resolve()) or not local.is_file():raise RuntimeError('Continuation must use this experiment locally generated checkpoint')
        wrapper=YOLO(str(local),task='detect')
        if wrapper.model.names!={0:'spalling',1:'crack'}:raise RuntimeError('Continuation class semantics changed')
        continuation={'checkpoint':str(local),'sha256':hashlib.sha256(local.read_bytes()).hexdigest(),'optimizer':'Fresh AdamW state; all learned classifier parameters retained, no class reset'}
        if protocol.get('starting_checkpoint_sha256') and continuation['sha256']!=protocol['starting_checkpoint_sha256']:
            raise RuntimeError('Actual retained checkpoint differs from frozen start identity')
        transferred={};head=wrapper.model.model[-1]
    else:
        wrapper=YOLO(str(model_yaml),task='detect');state=source_state();target=wrapper.model.state_dict()
        transferred={k:v for k,v in state.items() if k in target and v.shape==target[k].shape}
        wrapper.model.load_state_dict(transferred,strict=False)
        # Newly supervised spalling gets a new final classifier, not renamed joint outputs.
        head=wrapper.model.model[-1]
        with torch.no_grad():
            for branch in head.cv3:
                layer=branch[-1];layer.weight[0].normal_(0,.01);layer.bias[0].fill_(-4.0)
    wrapper.model.names={0:'spalling',1:'crack'}
    initial_spall=torch.cat([torch.cat([branch[-1].weight[0].detach().flatten(),branch[-1].bias[0].detach().reshape(1)]) for branch in head.cv3]).clone()
    initial_spall_sha=hashlib.sha256(initial_spall.numpy().tobytes()).hexdigest()
    initial_backbone={i:next(wrapper.model.model[i].parameters()).detach().float().cpu().clone() for i in [4,6,9]}
    memory_format=torch.channels_last if args.memory_format=='channels_last' else torch.contiguous_format
    initialized=wrapper.model.to(memory_format=memory_format)
    class SeededDetectionTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            return initialized  # Preserve the explicitly transferred/reset parameters; no YAML reseeding.
        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch)
            # Physical storage layout changes, not image values/channels/geometry.
            # The bounded TRAIN-only probe chooses this before the future fit.
            batch['img']=batch['img'].contiguous(memory_format=memory_format)
            return batch
    started=time.perf_counter()
    result=wrapper.train(trainer=SeededDetectionTrainer,data=str(data_root/'dataset.yaml'),epochs=args.epochs,imgsz=args.image_size,batch=args.batch,
        device='cpu',workers=0,time=None if args.epochs==1 or args.max_minutes<=0 else args.max_minutes/60,fraction=args.fraction,optimizer='AdamW',nbs=args.nominal_batch,cls_pw=.5,lr0=args.learning_rate,lrf=.1,cos_lr=True,freeze=args.freeze,
        amp=False,plots=False,patience=8,seed=seed,deterministic=True,cache='ram',save=True,warmup_epochs=0 if continuation else 3,
        project=str(ROOT/'output/training/construction-round3'),name=args.name,exist_ok=False,
        mosaic=0,mixup=0,copy_paste=0,hsv_h=0,hsv_s=0,hsv_v=.65,degrees=7,translate=.05,scale=.15,
        fliplr=.5,flipud=0,erasing=0,close_mosaic=0,pretrained=False,verbose=False)
    fitting_seconds=time.perf_counter()-started
    save_dir=Path(wrapper.trainer.save_dir);best=save_dir/'weights/best.pt'
    completed_epochs=max(0,len((save_dir/'results.csv').read_text(encoding='utf8').splitlines())-1)
    # This checkpoint is locally produced by trusted installed code, not downloaded pickle.
    fitted=YOLO(str(best),task='detect')
    if fitted.model.names!={0:'spalling',1:'crack'}:raise RuntimeError('Trained class mapping changed')
    final_head=fitted.model.model[-1]
    backbone_changes={str(i):float(torch.linalg.vector_norm(next(fitted.model.model[i].parameters()).detach().float().cpu()-initial)) for i,initial in initial_backbone.items()}
    if continuation and args.freeze<=4 and not any(v>1e-8 for v in backbone_changes.values()):raise RuntimeError('Requested backbone adaptation did not update monitored backbone weights')
    learned_spall=torch.cat([torch.cat([branch[-1].weight[0].detach().flatten(),branch[-1].bias[0].detach().reshape(1)]) for branch in final_head.cv3])
    spall_delta=float(torch.linalg.vector_norm(learned_spall-initial_spall))
    if not spall_delta>1e-8:raise RuntimeError('New spalling classifier parameters did not learn')
    fitted.export(format='onnx',imgsz=args.image_size,opset=17,simplify=False,dynamic=False,batch=1)
    onnx=best.with_suffix('.onnx');target_model=ROOT/'models/construction'/f'{args.name}.onnx';target_model.write_bytes(onnx.read_bytes())
    metadata={'experiment':args.name,'source_checkpoint_sha256':SOURCE_SHA,'onnx_sha256':hashlib.sha256(target_model.read_bytes()).hexdigest(),
        'training_script_sha256':executed_script_sha256,
        'model_path':str(target_model),'input_size':args.image_size,'classes':['spalling','crack'],'architecture':'yolo_detect',
        'exposure_policy':args.exposure_policy,
        'prepared_data_protocol_sha256':hashlib.sha256((data_root/'data-protocol.json').read_bytes()).hexdigest() if (data_root/'data-protocol.json').exists() else None,
        'capabilities':['surface_spalling','surface_crack'],'source':'https://doi.org/10.11583/DTU.25924228',
        'license':'AGPL-3.0 Ultralytics/initial weights; ev-CIVIL CC BY4.0; DACL10k data CC BY-NC4.0 (non-commercial restriction applies); initial merged crack/joint source-data licenses not independently cleared',
        'validation_scope':protocol.get('validation_scope','Known current-fit source-folder/global-defect-ID exclusions; physical project independence and initial pretrained-checkpoint data overlap unverified; no Saudi-site qualification'),
        'training_data_root':str(data_root),'split_protocol_sha256':hashlib.sha256((data_root/'protocol.json').read_bytes()).hexdigest(),
        'negative_manifest_sha256':hashlib.sha256((data_root/'background-import.json').read_bytes()).hexdigest(),
        'night_holdout_protocol_sha256':hashlib.sha256((data_root/'night-holdout-protocol.json').read_bytes()).hexdigest(),
        'exposures_sha256':hashlib.sha256((data_root/'exposures.json').read_bytes()).hexdigest(),
        'continuation':continuation,'training':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}|{'seed':seed,'train_seconds':fitting_seconds,'fit_and_export_seconds':time.perf_counter()-started,'completed_epochs':completed_epochs,'transferred_tensors':len(transferred),
                             'nominal_batch_size':args.nominal_batch,'warmup_epochs':0 if continuation else 3,'monitored_backbone_weight_l2_change':backbone_changes,'spalling_class_initialized_from_joint':False,'class0_reset_before_training':continuation is None,'initial_spall_classifier_sha256':initial_spall_sha,'learned_spall_classifier_sha256':hashlib.sha256(learned_spall.numpy().tobytes()).hexdigest(),'spall_classifier_parameter_l2_change':spall_delta},
        'selection':'Best checkpoint selected using new calibration split only; thresholds still require a separate frozen calibration before challenge comparison.',
        'save_dir':str(save_dir),'results':{k:float(v) for k,v in result.results_dict.items()}}
    target_model.with_suffix('.provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf8');print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
