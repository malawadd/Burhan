"""Conditional retained-five-class wall continuation; no action without approval.

Exactly12 additional epochs from the locally trained exactlast4 checkpoint.
No channel reset, new data, threshold rescue, heldout or best-epoch substitution.
"""
from pathlib import Path
import argparse,hashlib,json,os,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/training/construction-wall-five-class-v3'
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd-wall-five-class-v1/prepared-yolo'
SOURCE=ROOT/'output/training/construction-wall-five-class-v2/fixed-four-epoch-fit/weights/last.pt'
SOURCE_ONNX=ROOT/'models/construction/mbdd-wall-five-class-v2/wall-five-class-last4-640.onnx'
DIAG=ROOT/'output/evaluation/construction-wall-five-class-v2/train-export-diagnosis/report.json'
POINT=ROOT/'output/evaluation/construction-wall-five-class-v2/operating-point.json'
CLASSES=['crack','leakage','abscission','corrosion','bulge']
CAPABILITIES=['surface_crack','surface_leakage_appearance','surface_detachment','surface_corrosion_appearance','surface_bulge_appearance']
CONFIG={'additional_epochs':12,'total_lineage_epochs':16,'image_size':640,'batch':4,'nominal_batch':4,'cpu_threads':1,
        'freeze':4,'memory_format':'channels_last','optimizer':'fresh AdamW','learning_rate':.0005,'final_lr_fraction':.1,
        'warmup_epochs':0,'seed':20261020,'checkpoint':'exact additionalepoch12 last.pt (total16)','wall_time_cap':None}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def tensor_state_sha(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(str(tuple(value.shape)).encode());digest.update(str(value.dtype).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_proposal():
    diagnosis=json.loads(DIAG.read_text());point=json.loads(POINT.read_text())
    if not diagnosis['numeric_guards_pass'] or point['all_minimum_review_guards_pass']:raise RuntimeError('Proposed continuation requires actual numericPASS and prioraccuracyFAIL')
    evaluator=ROOT/'scripts/evaluate_mbdd_wall_detector_v3.py'
    plan={'parent_approved':False,'status':'PROPOSAL ONLY; independent static review and explicit root fit decision required',
          'configuration':CONFIG,'classes':CLASSES,'class_capabilities':dict(zip(CLASSES,CAPABILITIES)),
          'source_pt':SOURCE.relative_to(ROOT).as_posix(),'source_pt_sha256':sha(SOURCE),'source_onnx_sha256':sha(SOURCE_ONNX),
          'source_model_metadata_sha256':sha(SOURCE_ONNX.with_suffix('.provenance.json')),'prior_failed_point_sha256':sha(POINT),
          'train_numeric_diagnosis_sha256':sha(DIAG),'input_manifest_sha256':sha(DATA/'input-manifest.json'),
          'dataset_yaml_sha256':sha(DATA/'dataset.yaml'),'trainer_sha256':sha(Path(__file__)),'future_evaluator_sha256':sha(evaluator),
          'base_evaluator_sha256':sha(ROOT/'scripts/evaluate_mbdd_wall_detector.py'),
          'evaluation_grid':[.10,.15,.20,.25,.35,.45,.55,.65,.75,.85],
          'minimum_review':{'each_class_precision':.8,'each_class_box_recall':.5,'serving_parity':True,'healthy_wall_fpr':None,'production_qualification':False},
          'initialization':'Retain exact entirefiveclass last4 checkpoint, including everylearnedclassifierchannel. Freshoptimizer/schedule; no probe/updatecarry and no classreset/remap.',
          'intervention':'Longer fitting and backboneadaptation: freeze4 versus oldfreeze10, reducedlr.0005. Source/data/classmap/evalgrid unchanged; no threshold rescue.',
          'scope':'Same181TRAIN/48CAL;18previousdiagnostics exposed developmentonly afterfinalweights+pointfreeze. No protectedcameraheldout/independent-site/paint-only/hiddenmoisture/depth claim.'}
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'proposed-fit-plan-revision2.json'
    if target.exists() and json.loads(target.read_text())!=plan:raise RuntimeError('Preserve proposed continuation identity')
    write(target,plan);print(sha(target),flush=True)


def fit():
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils import callbacks
    plan_path=OUT/'approved-fit-plan.json';plan=json.loads(plan_path.read_text())
    if not plan.get('parent_approved') or plan['configuration']!=CONFIG or plan['classes']!=CLASSES:raise RuntimeError('Explicit exact12epoch parent-approved plan required')
    for path,key in [(Path(__file__),'trainer_sha256'),(SOURCE,'source_pt_sha256'),(SOURCE_ONNX,'source_onnx_sha256'),(SOURCE_ONNX.with_suffix('.provenance.json'),'source_model_metadata_sha256'),(DATA/'input-manifest.json','input_manifest_sha256'),(DATA/'dataset.yaml','dataset_yaml_sha256'),(DIAG,'train_numeric_diagnosis_sha256'),(POINT,'prior_failed_point_sha256'),(ROOT/'scripts/evaluate_mbdd_wall_detector_v3.py','future_evaluator_sha256'),(ROOT/'scripts/evaluate_mbdd_wall_detector.py','base_evaluator_sha256')]:
        if sha(path)!=plan[key]:raise RuntimeError('Protected continuation source/data/diagnosis differs: '+str(path))
    if (OUT/'execution-before-optimizer.json').exists():raise RuntimeError('Preserve attempted continuation; no silent restart')
    manifest=json.loads((DATA/'input-manifest.json').read_text())
    for row in manifest['records']:
        if sha(ROOT/row['prepared_image'])!=row['prepared_image_sha256'] or sha(ROOT/row['prepared_label'])!=row['prepared_label_sha256']:raise RuntimeError('Original prepared source or label changed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(CONFIG['seed'])
    wrapper=YOLO(str(SOURCE),task='detect');initialized=wrapper.model.float().to(memory_format=torch.channels_last)
    if initialized.names!={i:name for i,name in enumerate(CLASSES)} or sum(p.numel() for p in initialized.parameters())!=3011823:raise RuntimeError('Retained fiveclass nano identity differs')
    if any(branch[-1].weight.shape!=(5,64,1,1) for branch in initialized.model[-1].cv3):raise RuntimeError('Retained fiveclass channel shapes differ')
    initial={key:value.detach().clone() for key,value in initialized.state_dict().items()}
    head_before=torch.cat([torch.cat([branch[-1].weight.detach().flatten(),branch[-1].bias.detach().flatten()]) for branch in initialized.model[-1].cv3])
    identity={'source_pt_sha256':sha(SOURCE),'retained_full_model':True,'semantic_channels_reset':False,'classes':CLASSES,'parameter_count':3011823,
              'retained_terminal_classifier_sha256':hashlib.sha256(head_before.cpu().numpy().tobytes()).hexdigest(),
              'retained_entire_tensor_state_sha256':tensor_state_sha(initial),
              'first4_frozen_tensor_state_sha256':tensor_state_sha({key:value for key,value in initial.items() if any(key.startswith(f'model.{i}.') for i in range(4))}),
              'fresh_optimizer':'AdamW newstate, resumeFalse; no sourceoptimizer restored','source_lineage_epochs':4,'additional_fixed_epochs':12}
    write(OUT/'execution-before-optimizer.json',{'approved_plan_sha256':sha(plan_path),'configuration':CONFIG,'initialization':identity})
    if not hasattr(DetectionTrainer,'_model_train'):raise RuntimeError('Installed train-mode hook changed')
    class RetainedTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):return initialized
        def _model_train(self):
            super()._model_train()
            for i in range(4):self.model.model[i].eval()
        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch);batch['img']=batch['img'].contiguous(memory_format=torch.channels_last);return batch
    callbacks.add_integration_callbacks=lambda instance:None
    started=time.perf_counter()
    wrapper.train(trainer=RetainedTrainer,data=str(DATA/'dataset.yaml'),epochs=12,imgsz=640,batch=4,nbs=4,device='cpu',workers=0,
                  optimizer='AdamW',lr0=.0005,lrf=.1,warmup_epochs=0,freeze=4,amp=False,plots=False,patience=20,
                  seed=20261020,deterministic=True,cache='ram',save=True,project=str(OUT),name='fixed-twelve-additional',exist_ok=False,
                  pretrained=False,resume=False,verbose=False,mosaic=0,mixup=0,copy_paste=0,erasing=0,hsv_h=.015,hsv_s=.4,hsv_v=.3,
                  degrees=3,translate=.03,scale=.1,fliplr=.5,flipud=0,close_mosaic=0)
    folder=Path(wrapper.trainer.save_dir);epochs=len((folder/'results.csv').read_text().splitlines())-1
    if epochs!=12:raise RuntimeError('Exact12additionalepochs did not complete')
    last=folder/'weights/last.pt';fitted=YOLO(str(last),task='detect')
    if fitted.model.names!={i:name for i,name in enumerate(CLASSES)}:raise RuntimeError('Retained source semantics changed')
    # Check the actual finalcheckpoint, not the framework's best-loaded wrapper.
    final=fitted.model.float().state_dict()
    if any(not torch.isfinite(value).all() for value in final.values()):raise RuntimeError('Nonfinite actual finalcheckpoint tensor state')
    frozen_keys=[key for key in initial if any(key.startswith(f'model.{i}.') for i in range(4))]
    if any(not torch.equal(initial[key],final[key]) for key in frozen_keys):raise RuntimeError('Frozen first4 parameter/normalization state changed')
    trained_keys=[key for key in initial if key.endswith('weight') and not any(key.startswith(f'model.{i}.') for i in range(4))]
    delta=sum(float(torch.sum((final[key]-initial[key])**2)) for key in trained_keys)**.5
    if not delta>0:raise RuntimeError('No trainable source parameter changed')
    fitted.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,batch=1)
    model_dir=ROOT/'models/construction/mbdd-wall-five-class-v3';model_dir.mkdir(parents=True,exist_ok=True);target=model_dir/'wall-five-class-total16-last12-640.onnx'
    if target.exists():raise RuntimeError('Preserve exported continuation')
    target.write_bytes(last.with_suffix('.onnx').read_bytes())
    meta={'model_path':str(target),'onnx_sha256':sha(target),'source_pt_sha256':sha(SOURCE),'final_pt_sha256':sha(last),'classes':CLASSES,'input_size':640,'architecture':'yolo_detect','exposure_policy':'off','source':'https://zenodo.org/records/15622584',
          'license':json.loads(SOURCE_ONNX.with_suffix('.provenance.json').read_text())['license'],'initialization':identity,'approved_plan_sha256':sha(plan_path),'training_input_manifest_sha256':sha(DATA/'input-manifest.json'),
          'selection':'Exact additionalepoch12 last.pt(total16); no bestepoch substitution/18diagnostic selection/thresholdrescue','completed_additional_epochs':epochs,'total_lineage_epochs':16,'training_seconds':time.perf_counter()-started,
          'first4_parameter_and_bn_state_unchanged':True,'trainable_parameter_l2_change':delta,'calibration_status':'Fullfixed48CAL/grid/actualparity/frozenpoint required beforeall18diagnostics; no cameraheldout authorization',
          'scope':plan['scope']}
    write(target.with_suffix('.provenance.json'),meta);print('EXPORTED',target,sha(target),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['proposal','fit'],required=True);args=parser.parse_args()
    if args.stage=='proposal':freeze_proposal()
    else:fit()
