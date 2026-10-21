"""Separately versioned five-class real-wall initializer, TRAIN probe and fit.

Nothing runs without reviewed prepared data. Fit additionally requires a parent
approved, hash-pinned plan created after the bounded TRAIN-only probe. No video,
old diagnostic or heldout loader is imported.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd-wall-five-class-v1/prepared-yolo'
OUT=ROOT/'output/training/construction-wall-five-class-v2'
SOURCE=ROOT/'output/training/construction-round3/evcivil-yolov8n-r4-head-640-gain/weights/best.pt'
SOURCE_SHA='5e48033ee0122d39a3a6301aa926dad6bd18b141e89468185a66d1871a5eef6c'
CLASSES=['crack','leakage','abscission','corrosion','bulge']
FIT_CONFIGURATION={'epochs':4,'image_size':640,'batch':4,'nominal_batch':4,'cpu_threads':1,'freeze':10,
                   'memory_format':'channels_last','optimizer':'AdamW','learning_rate':.0015,'final_lr_fraction':.1,
                   'warmup_epochs':0,'seed':20261020,'checkpoint':'exact final epoch4 last.pt','wall_time_cap':None}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def checked_data():
    manifest_path=DATA/'input-manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['classes']!=CLASSES:raise RuntimeError('New wall semantic class order changed')
    for row in manifest['records']:
        if sha(ROOT/row['prepared_image'])!=row['prepared_image_sha256'] or sha(ROOT/row['prepared_label'])!=row['prepared_label_sha256']:
            raise RuntimeError('Reviewed prepared input bytes changed')
    if sha(DATA/'dataset.yaml')!=manifest['dataset_yaml_sha256']:raise RuntimeError('Prepared wall YAML changed')
    return manifest


def initialize():
    import torch,yaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load
    if sha(SOURCE)!=SOURCE_SHA:raise RuntimeError('Trusted local retained backbone source changed')
    torch.manual_seed(20261020)
    source=YOLO(str(SOURCE),task='detect').model.float()
    if source.names!={0:'spalling',1:'crack'}:raise RuntimeError('Source model semantics changed')
    template=copy.deepcopy(source.yaml);template['nc']=5
    OUT.mkdir(parents=True,exist_ok=True)
    model_yaml=OUT/'yolov8n-wall-five-class.yaml'
    model_yaml.write_text(yaml.safe_dump(template),encoding='utf-8')
    resolved=yaml_model_load(model_yaml)
    if resolved.get('scale')!='n' or source.yaml.get('scale')!='n':raise RuntimeError('Actual framework YAML loader must resolve explicit nano scale before constructor initialization forward')
    source_parameters=sum(p.numel() for p in source.parameters())
    if not 2500000<source_parameters<4000000:raise RuntimeError('Trusted source is not the expected nano parameter range')
    wrapper=YOLO(str(model_yaml),task='detect');target=wrapper.model
    original=source.state_dict();destination=target.state_dict()
    if target.yaml.get('scale')!='n':raise RuntimeError('Constructed target is not nano')
    prefix='model.'+str(len(source.model)-1)+'.cv3.'
    semantic_keys={prefix+f'{i}.{len(branch)-1}.'+suffix for i,branch in enumerate(source.model[-1].cv3) for suffix in ['weight','bias']}
    expected_target_parameters=source_parameters+sum((5-2)*(branch[-1].weight.shape[1]+1) for branch in source.model[-1].cv3)
    if sum(p.numel() for p in target.parameters())!=expected_target_parameters:raise RuntimeError('Exact expected five-class nano parameter count differs')
    for key,value in original.items():
        if key not in semantic_keys and (key not in destination or destination[key].shape!=value.shape):raise RuntimeError('Source nonsemantic backbone/neck/regression tensor shape differs: '+key)
    transferred={key:value for key,value in original.items() if key in destination and value.shape==destination[key].shape}
    target.load_state_dict(transferred,strict=False)
    if any(not torch.equal(target.state_dict()[key],value) for key,value in original.items() if key not in semantic_keys):raise RuntimeError('An exact source nonsemantic tensor was not preserved')
    for index in range(10):
        if any(not torch.equal(value,target.model[index].state_dict()[key]) for key,value in source.model[index].state_dict().items()):raise RuntimeError('First10 source backbone tensors differ before TRAIN forward')
    # No crack/spall/joint channel is relabelled as moisture, paint or bulge.
    # ALL five terminal semantic channels are newly initialized before fitting.
    head=target.model[-1]
    with torch.no_grad():
        for branch in head.cv3:
            branch[-1].weight.normal_(0,.01);branch[-1].bias.fill_(-4.)
    target.names={i:name for i,name in enumerate(CLASSES)}
    reset=torch.cat([torch.cat([b[-1].weight.detach().flatten(),b[-1].bias.detach().flatten()]) for b in head.cv3])
    classifier_shapes=[list(b[-1].weight.shape) for b in head.cv3]
    if any(shape[0]!=5 for shape in classifier_shapes):raise RuntimeError('Five semantic output channels were not created')
    copied_reg=[key for key in transferred if '.cv2.' in key]
    if not copied_reg:raise RuntimeError('Source box-regression tensors were not transferred')
    if any(not torch.equal(target.state_dict()[key],original[key]) for key in copied_reg):raise RuntimeError('Box-regression initialization differs')
    identity={'source_pt':SOURCE.relative_to(ROOT).as_posix(),'source_pt_sha256':SOURCE_SHA,'classes':CLASSES,
              'semantic_channels_all_reset':True,'terminal_classifier_shapes':classifier_shapes,
              'reset_classifier_sha256':hashlib.sha256(reset.cpu().numpy().tobytes()).hexdigest(),
              'resolved_scale':'n','source_parameter_count':source_parameters,'target_parameter_count':sum(p.numel() for p in target.parameters()),
              'expected_target_parameter_count':expected_target_parameters,'all_source_nonsemantic_tensors_equal':True,'first10_backbone_tensors_equal_before_train_forward':True,
              'constructor_forward_scope':'Framework standard architecture stride-initialization dummy forward occurs only after explicit YAML-loader nano-scale verification; all tensor/count guards precede real TRAIN forward',
              'transferred_tensors':len(transferred),'copied_box_regression_tensors':len(copied_reg),
              'semantics':'Source0spalling/1crack not renamed. New0crack/1leakage/2abscission/3corrosion/4bulge is supervised from complete publisherXML.',
              'torch_load_scope':'Checkpoint produced locally by existing trusted fitter, not arbitrary downloaded pickle'}
    return wrapper,identity


def probe(manifest):
    import cv2,numpy as np,torch,psutil
    from types import SimpleNamespace
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    target=OUT/'train-probe.json'
    if target.exists():raise RuntimeError('Preserve original bounded wall TRAIN probe')
    records=[r for r in manifest['records'] if r['split']=='train']
    selected=[]
    for i in range(16):
        label=CLASSES[i%5];cohort=[r for r in records if any(b['label']==label for b in r['boxes'])]
        selected.append(cohort[(i//5)%len(cohort)])
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=OUT/'train-probe-protocol-before-forward.json'
    if protocol_path.exists():raise RuntimeError('Preserve attempted wall probe; no silent retry after its before-forward receipt')
    write(protocol_path,{'input_manifest_sha256':sha(DATA/'input-manifest.json'),'trainer_sha256':sha(Path(__file__)),
                         'source_pt_sha256':SOURCE_SHA,'classes':CLASSES,'fixed_steps':4,'batch':4,'threads':1,'image_size':640,
                         'memory_format':'channels_last','seed':20261020,'selected_training_sources':[r['source_name'] for r in selected],
                         'scope':'Written before model initialization or forward. TRAIN-only numerical/memory probe; optimizer updates discarded, no saved probe weights, no CAL/diagnostic pixels.'})
    wrapper,identity=initialize();model=wrapper.model.train()
    model.args=SimpleNamespace(box=7.5,cls=.5,dfl=1.5)
    for index in range(10):
        for parameter in model.model[index].parameters():parameter.requires_grad=False
        model.model[index].eval()
    frozen_before={f'{index}:{key}':value.detach().clone() for index in range(10) for key,value in model.model[index].state_dict().items()}
    model.to(memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0015,weight_decay=.0005)
    started=time.perf_counter();observations=[]
    for step in range(4):
        images=[];boxes=[];indices=[];classes=[]
        for batch_index,row in enumerate(selected[step*4:(step+1)*4]):
            raw=cv2.imread(str(ROOT/row['prepared_image']));h,w=raw.shape[:2];scale=min(640/w,640/h);nw,nh=round(w*scale),round(h*scale)
            pad_x,pad_y=(640-nw)//2,(640-nh)//2
            image=np.full((640,640,3),114,np.uint8);image[pad_y:pad_y+nh,pad_x:pad_x+nw]=cv2.resize(raw,(nw,nh))
            images.append(image[:,:,::-1].transpose(2,0,1).copy())
            for b in row['boxes']:
                x1,y1,x2,y2=b['source_xyxy'];boxes.append([(scale*(x1+x2)/2+pad_x)/640,(scale*(y1+y2)/2+pad_y)/640,scale*(x2-x1)/640,scale*(y2-y1)/640])
                classes.append([b['class_id']]);indices.append(batch_index)
        batch={'img':torch.from_numpy(np.stack(images)).float().div(255).contiguous(memory_format=torch.channels_last),
               'batch_idx':torch.tensor(indices),'cls':torch.tensor(classes,dtype=torch.float32),'bboxes':torch.tensor(boxes,dtype=torch.float32)}
        optimizer.zero_grad(set_to_none=True);loss,items=model(batch)
        if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite TRAIN wall probe loss')
        loss.sum().backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite TRAIN wall probe gradient')
        gradients=[float(branch[-1].weight.grad.norm()) for branch in model.model[-1].cv3]
        per_class=torch.stack([branch[-1].weight.grad.reshape(5,-1).norm(dim=1) for branch in model.model[-1].cv3]).sum(0).tolist()
        if not all(value>0 for value in gradients+per_class):raise RuntimeError('A new semantic output channel has no gradient')
        optimizer.step()
        observations.append({'step':step+1,'loss':float(loss.sum()),'head_gradient_norms':gradients,'new_class_gradient_norms':dict(zip(CLASSES,per_class)),'rss_bytes':psutil.Process().memory_info().rss})
    frozen_after={f'{index}:{key}':value.detach() for index in range(10) for key,value in model.model[index].state_dict().items()}
    if any(not torch.equal(value,frozen_after[key]) for key,value in frozen_before.items()):raise RuntimeError('Frozen backbone parameter or normalization state changed in probe')
    write(target,{'before_forward_protocol_sha256':sha(protocol_path),'source_input_manifest_sha256':sha(DATA/'input-manifest.json'),'trainer_sha256':sha(Path(__file__)),'initialization':identity,
                  'fixed_steps':4,'batch':4,'threads':1,'memory_format':'channels_last','image_size':640,'observations':observations,
                  'optimizer_seconds':time.perf_counter()-started,'probe_weights_saved':False,'frozen_backbone_and_BN_state_unchanged':True,
                  'scope':'TRAIN-only numerical/memory/capacity feasibility; any elapsed rate is contended TRAIN work, not inference FPS or accuracy. No CAL/diagnostic pixels used.'})
    print(json.dumps(json.loads(target.read_text()),indent=2),flush=True)


def fit(manifest):
    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics import YOLO
    from ultralytics.utils import callbacks
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    plan_path=OUT/'approved-fit-plan.json'
    if not plan_path.exists():raise RuntimeError('Parent-approved frozen fit plan is required after TRAIN probe')
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if not plan.get('parent_approved_fixed_four_epochs',False) or plan.get('configuration')!=FIT_CONFIGURATION or plan['trainer_sha256']!=sha(Path(__file__)) or plan['input_manifest_sha256']!=sha(DATA/'input-manifest.json'):
        raise RuntimeError('Fixed parent-approved fit/data/source identity differs')
    if plan['train_probe_sha256']!=sha(OUT/'train-probe.json') or plan['source_pt_sha256']!=SOURCE_SHA:raise RuntimeError('Frozen training probe/source differs')
    if not hasattr(DetectionTrainer,'_model_train'):raise RuntimeError('Review installed trainer train-mode hook before fitting; frozen BN must be enforced after model.train()')
    if (OUT/'fit-execution-before-optimizer.json').exists():raise RuntimeError('No silent restart of wall fit; preserve its execution receipt')
    wrapper,identity=initialize();initialized=wrapper.model.to(memory_format=torch.channels_last)
    probe_identity=json.loads((OUT/'train-probe.json').read_text(encoding='utf-8'))['initialization']
    if identity['reset_classifier_sha256']!=probe_identity['reset_classifier_sha256'] or identity['terminal_classifier_shapes']!=probe_identity['terminal_classifier_shapes']:
        raise RuntimeError('Actual five-class initializer differs from the reviewed TRAIN probe')
    write(OUT/'fit-execution-before-optimizer.json',{'approved_plan_sha256':sha(plan_path),'initialization':identity,'trainer_sha256':sha(Path(__file__))})
    class SeededTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):return initialized
        def _model_train(self):
            super()._model_train()
            # Enforce frozen BN AFTER the framework's model.train() transition,
            # rather than relying on an epoch callback that may run before it.
            for index in range(10):self.model.model[index].eval()
        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch);batch['img']=batch['img'].contiguous(memory_format=torch.channels_last);return batch
    callbacks.add_integration_callbacks=lambda instance:None
    started=time.perf_counter()
    wrapper.train(trainer=SeededTrainer,data=str(DATA/'dataset.yaml'),epochs=4,imgsz=640,batch=4,nbs=4,
                  device='cpu',workers=0,optimizer='AdamW',lr0=.0015,lrf=.1,warmup_epochs=0,freeze=10,
                  amp=False,plots=False,patience=8,seed=20261020,deterministic=True,cache='ram',save=True,
                  project=str(OUT),name='fixed-four-epoch-fit',exist_ok=False,pretrained=False,verbose=False,
                  mosaic=0,mixup=0,copy_paste=0,erasing=0,hsv_h=.015,hsv_s=.4,hsv_v=.3,
                  degrees=3,translate=.03,scale=.1,fliplr=.5,flipud=0,close_mosaic=0)
    last=Path(wrapper.trainer.save_dir)/'weights/last.pt'
    epochs=len((Path(wrapper.trainer.save_dir)/'results.csv').read_text().splitlines())-1
    if epochs!=4:raise RuntimeError('Exact fixed4epoch wall fit did not complete')
    fitted=YOLO(str(last),task='detect')
    if fitted.model.names!={i:name for i,name in enumerate(CLASSES)}:raise RuntimeError('Fitted wall semantics changed')
    fitted.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,batch=1)
    model_dir=ROOT/'models/construction/mbdd-wall-five-class-v2';model_dir.mkdir(parents=True,exist_ok=True)
    onnx=model_dir/'wall-five-class-last4-640.onnx'
    if onnx.exists():raise RuntimeError('Preserve previous wall fitted asset')
    onnx.write_bytes(last.with_suffix('.onnx').read_bytes())
    write(onnx.with_suffix('.provenance.json'),{'model_path':str(onnx),'onnx_sha256':sha(onnx),'source_pt_sha256':SOURCE_SHA,'classes':CLASSES,
          'input_size':640,'architecture':'yolo_detect','exposure_policy':'off','source':'https://zenodo.org/records/15622584',
          'license':'MBDD2025 CCBY4.0; Ultralytics/sourceweight AGPL3; inherited warmstart source-data restrictions remain, including DACLNC4.0 and source-rights uncertainty.',
          'selection':'Exact final4epoch checkpoint predeclared beforefit; no bestepoch substitution or diagnostic selection',
          'initialization':identity,'training_input_manifest_sha256':sha(DATA/'input-manifest.json'),'approved_plan_sha256':sha(plan_path),
          'training_seconds':time.perf_counter()-started,'completed_epochs':epochs,'calibration_status':'Operating point/actualserving parity pending; no heldout/diagnostic forward allowed beforefreeze',
          'scope':'Visiblecrack/leakagemark/materialdetachment/corrosion/bulge appearance only. No moisturequantity/paint-only/depth/SBC claim; healthy-wall FPR not measured.'})
    print('EXPORTED',onnx,sha(onnx),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['probe','fit'],required=True);args=parser.parse_args()
    manifest=checked_data()
    if args.stage=='probe':probe(manifest)
    else:fit(manifest)


if __name__=='__main__':main()
