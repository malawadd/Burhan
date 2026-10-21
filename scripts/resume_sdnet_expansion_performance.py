"""Performance-only, hash-bound continuation of the SDNET eight-epoch512 fit.

Use only after a complete trusted local epoch checkpoint has been copied and
verified, its original process stopped, and TRAIN-only timing chooses a layout.
Data, resolution, optimizer state, eight total epochs and CAL policy are retained.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,shutil,time
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import torch_utils
import train_sdnet_expansion_v2 as experiment

ROOT=experiment.ROOT
OUT=ROOT/'output/training/construction-expansion-sdnet-v2-performance'
DEST=ROOT/'models/construction/expansion-sdnet-v2-performance/yolov8n'
SELECTED_LAYOUT='contiguous'
SELECTED_THREADS=1


def value_digest(value):
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            digest.update(b'tensor');digest.update(str(tuple(item.shape)).encode())
            array=item.detach().cpu().double().contiguous().numpy()
            digest.update(array.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):digest.update(str(key).encode());visit(item[key])
        elif isinstance(item,(list,tuple)):
            for sub in item:visit(sub)
        else:digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


class PerformanceTrainer(DetectionTrainer):
    def setup_model(self):
        checkpoint=super().setup_model()
        memory=torch.channels_last if SELECTED_LAYOUT=='channels_last' else torch.contiguous_format
        self.model.to(memory_format=memory)
        return checkpoint

    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        memory=torch.channels_last if SELECTED_LAYOUT=='channels_last' else torch.contiguous_format
        batch['img']=batch['img'].contiguous(memory_format=memory)
        if not hasattr(self,'_layout_receipt_written'):
            actual={'selected_layout':SELECTED_LAYOUT,'input_channels_last':batch['img'].is_contiguous(memory_format=torch.channels_last),
                    'input_contiguous':batch['img'].is_contiguous(),'shape':list(batch['img'].shape),'torch_threads':torch.get_num_threads()}
            experiment.write(OUT/'actual-first-batch-layout.json',actual);self._layout_receipt_written=True
        return batch


def main(args):
    global SELECTED_LAYOUT,SELECTED_THREADS
    SELECTED_LAYOUT=args.layout;SELECTED_THREADS=args.threads
    if (OUT/'protocol-before-resume.json').exists():raise ValueError('Preserve previous continuation; no silent rerun')
    checkpoint=args.checkpoint.resolve()
    if experiment.sha(checkpoint)!=args.checkpoint_sha256:raise ValueError('Copied epoch checkpoint hash changed')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)  # Own hash-pinned local checkpoint only.
    if ckpt.get('optimizer') is None or not ckpt['optimizer'].get('state'):raise ValueError('Complete optimizer state is required')
    completed=int(ckpt['epoch'])+1
    if completed!=1:raise ValueError('This continuation expects the first complete epoch of the declared eight-epoch fit')
    parameters=ckpt['train_args']
    for key,expected in [('epochs',8),('imgsz',512),('batch',4),('freeze',4)]:
        if parameters.get(key)!=expected:raise ValueError('Original fit configuration changed: '+key)
    source_model=ckpt.get('ema') or ckpt.get('model')
    if source_model.names!={i:x for i,x in enumerate(experiment.CLASSES)}:raise ValueError('Checkpoint semantics changed')
    source_model_digest=value_digest(source_model.state_dict())
    optimizer_digest=value_digest(ckpt['optimizer']['state'])
    data=experiment.DATA
    manifest=json.loads((data/'manifest.json').read_text())
    if experiment.sha(data/'dataset.yaml')!=manifest['dataset_yaml_sha256']:raise ValueError('Frozen YAML changed')
    for row in manifest['records']:
        if experiment.sha(ROOT/row['image'])!=row['source_sha256']:raise ValueError('Frozen source photograph changed')
    OUT.mkdir(parents=True,exist_ok=True)
    experiment.write(OUT/'protocol-before-resume.json',{'source_checkpoint':str(checkpoint),'source_checkpoint_sha256':args.checkpoint_sha256,
        'completed_epochs':completed,'remaining_epochs':7,'total_epochs':8,'source_model_value_sha256':source_model_digest,
        'source_optimizer_state_value_sha256':optimizer_digest,'selected_torch_threads':SELECTED_THREADS,'selected_layout':SELECTED_LAYOUT,
        'timing_probe_report':str(args.probe_report.resolve()),'timing_probe_report_sha256':experiment.sha(args.probe_report),
        'original_protocol_sha256':experiment.sha(ROOT/'output/training/construction-expansion-sdnet-v2/protocol-before-training.json'),
        'dataset_manifest_sha256':experiment.sha(data/'manifest.json'),'script_sha256':experiment.sha(Path(__file__)),
        'numerical_trajectory':'Standard checkpoint resume restores saved EMA/model and optimizer. Thread/layout changes may change floating-point trajectory; not claimed bitwise identical.',
        'unchanged':'TRAIN/CAL sources,512px,batch4,freeze4,8totalepochs,optimizer/scheduler,CAL-onlythresholdpolicy. Exposed98TEST remains reuseddevelopment.',
        'private_videos_used':False,'quality_selection_from_probe':False})
    torch.set_num_threads(SELECTED_THREADS);torch_utils.NUM_THREADS=SELECTED_THREADS;cv2.setNumThreads(1)
    wrapper=YOLO(str(checkpoint));observations=[]
    def verify_resume(trainer):
        if trainer.start_epoch!=1 or trainer.epochs!=8:raise ValueError('Resume must run epochs2through8, not eight extra epochs')
        actual_optimizer=value_digest(trainer.optimizer.state_dict()['state'])
        actual_model=value_digest(trainer.model.state_dict())
        receipt={'start_epoch_zero_based':trainer.start_epoch,'total_epochs':trainer.epochs,'torch_threads':torch.get_num_threads(),
                 'optimizer_value_digest_restored':actual_optimizer,'source_optimizer_value_digest':optimizer_digest,
                 'optimizer_restored':actual_optimizer==optimizer_digest,'model_value_digest_restored':actual_model,
                 'source_model_value_digest':source_model_digest,'model_restored':actual_model==source_model_digest}
        experiment.write(OUT/'restored-state-before-optimizer-step.json',receipt)
        if not receipt['optimizer_restored'] or not receipt['model_restored']:raise ValueError('Saved checkpoint values not restored before resumed optimizer step')
    def record_threads(trainer):
        actual={'epoch':int(trainer.epoch),'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}
        observations.append(actual);experiment.write(OUT/'actual-training-threads.json',observations)
        if actual['torch_threads']!=SELECTED_THREADS:raise ValueError('Actual CPU thread budget changed')
    wrapper.add_callback('on_train_start',verify_resume);wrapper.add_callback('on_train_epoch_start',record_threads)
    started=time.perf_counter()
    wrapper.train(trainer=PerformanceTrainer,resume=str(checkpoint),device='cpu',workers=0,save_dir=str(OUT/'fit'),channels_last=False)
    resume_seconds=time.perf_counter()-started
    wrapper=YOLO(str(OUT/'fit/weights/last.pt'))
    calrows=[r for r in manifest['records']if r['split']=='calibration'];devrows=[r for r in manifest['records']if r['split']=='test']
    cal,speed=experiment.predictions(wrapper,calrows)
    cuts={};grid={str(t):experiment.metrics(calrows,cal,t)for t in experiment.CONFIG['thresholds']}
    for label in experiment.CLASSES:
        choices=[]
        for threshold in experiment.CONFIG['thresholds']:
            m=grid[str(threshold)]['classes'][label];p,r=m['precision'],m['recall'];qualified=m['tp']>0 and p>=.95
            choices.append((qualified,r if qualified else 2*p*r/max(1e-9,p+r),r,p,threshold))
        cuts[label]=max(choices)[-1]
    experiment.write(OUT/'thresholds-before-reused-development.json',{'class_confidence':cuts,'calibration_grid':grid,'private_videos_used':False})
    development,_=experiment.predictions(wrapper,devrows)
    results={'architecture':'yolov8n','config':experiment.CONFIG,'performance_continuation':True,'resume_seconds':resume_seconds,
             'threads':observations,'layout':SELECTED_LAYOUT,'after_calibration':experiment.perclass_summary(calrows,cal,cuts),
             'reused_development':experiment.perclass_summary(devrows,development,cuts),'reused_development_fixed_0_5':experiment.metrics(devrows,development,.5),
             'speed':speed,'release_pass':False,'validation_status':'research_only','fp_scope':'Annotation-unmatched, not proven physical falsealarms; oppositeclass completenessunknown.'}
    experiment.write(OUT/'results.json',results)
    exported=Path(wrapper.export(format='onnx',imgsz=512,opset=17,simplify=False,dynamic=False,batch=1,device='cpu'))
    DEST.mkdir(parents=True,exist_ok=True);shutil.copyfile(exported,DEST/'model.onnx')
    experiment.write(DEST/'manifest.json',{'architecture':'yolov8n','input_size':512,'model_file':'model.onnx','model_sha256':experiment.sha(DEST/'model.onnx'),
        'inspection_task':'brackets_railings','title':'SDNET2025 extended fastening fit (research, CPU continuation)','classes':experiment.CLASSES,
        'confidence':min(cuts.values()),'class_confidence':cuts,'preprocessing':'letterbox_rgb_0_1','source':manifest['source'],
        'license':'SDNET data CC BY 4.0; Ultralytics AGPL-3.0 conditions apply.','validation_status':'research_only',
        'evaluation_report':str((OUT/'results.json').resolve()),'validation_summary':'Annotation-relative development only; physical-site independence unverified; no release.',
        'private_videos_used':False,'continuation_protocol':str((OUT/'protocol-before-resume.json').resolve())})
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True);parser.add_argument('--probe-report',type=Path,required=True)
    parser.add_argument('--threads',type=int,choices=[1,2,4],required=True);parser.add_argument('--layout',choices=['contiguous','channels_last'],required=True)
    main(parser.parse_args())
