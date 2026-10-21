"""Stronger fixed-budget SDNET research candidate; never uses owner footage.

Eight 512px epochs, freeze4, verified one CPU thread. The exposed v1 TEST is
reused-development only. Per-class thresholds use CAL only, never TEST outputs.
"""
import os
import ctypes
EXECUTION={'allowed_concurrent_single_thread_training_jobs':3,'isolated_speed_qualification':False,'priority_requested':'BelowNormal'}
if os.name=='nt':
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.GetPriorityClass.argtypes=[ctypes.c_void_p]
    process_handle=kernel.GetCurrentProcess()
    EXECUTION['priority_applied']=bool(kernel.SetPriorityClass(process_handle,0x4000))
    EXECUTION['actual_priority_class']=int(kernel.GetPriorityClass(process_handle))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['YOLO_OFFLINE']='true'
from pathlib import Path
import copy,hashlib,json,random,shutil,time
import cv2,numpy as np,torch,yaml
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
from train_sdnet_expansion import CLASSES,DATA,ROOT,INITIAL,metrics,sha,write

OUT=ROOT/'output/training/construction-expansion-sdnet-v2'
DEST=ROOT/'models/construction/expansion-sdnet-v2/yolov8n'
CONFIG={'epochs':8,'image_size':512,'batch':4,'freeze_first_layers':4,'torch_threads':1,'seed':20261021,
        'learning_rate':.001,'optimizer':'AdamW','warmup_epochs':1,'checkpoint':'exact final epoch8last.pt',
        'thresholds':[.025,.05,.075,.1,.15,.2,.25,.35,.5,.65,.8,.9],
        'augmentation':'real-image horizontalflip0.5 only; no artificial damage, mosaic, mixup, spatial crop or photometric augmentation',
        'selection':'Final fixed8epochs. Perclass CAL: among thresholds withTP>0 andP>=.95 maximizeR, thenP; otherwise maximizeF1,R,P. TESTneverselects.',
        'test_role':'Previously exposed98v1testimages are reuseddevelopment only; no independent qualification.',
        'release':'research_only; unknownphysicalsites and targets unmet preventpromotion regardlessofobservedscore.'}


def predictions(wrapper,rows):
    started=time.perf_counter()
    outputs=wrapper.predict([str(ROOT/r['image'])for r in rows],imgsz=512,conf=.01,iou=.45,device='cpu',batch=4,verbose=False)
    result=[]
    for output in outputs:
        result.append([{'class_id':int(c),'score':s,'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]]}
                       for b,s,c in zip(output.boxes.xyxy.tolist(),output.boxes.conf.tolist(),output.boxes.cls.tolist())])
    elapsed=time.perf_counter()-started
    return result,{'images':len(rows),'seconds':elapsed,'observed_batch4_fps':len(rows)/elapsed,'torch_threads':torch.get_num_threads(),
                   'scope':'Observed research throughput; concurrenttrainingmayexist; notisolatedruntimepreviewbenchmark.'}


def perclass_summary(rows,predictions,cuts):
    result={'iou_threshold':.5,'class_confidence':cuts,'classes':{}}
    for label in CLASSES:
        result['classes'][label]=metrics(rows,predictions,cuts[label])['classes'][label]
    result['macro_precision']=float(np.mean([x['precision']for x in result['classes'].values()]))
    result['macro_recall']=float(np.mean([x['recall']for x in result['classes'].values()]))
    return result


def main():
    if (OUT/'protocol-before-training.json').exists():raise ValueError('Preserve v2 experiment; do not silently rerun')
    manifest=json.loads((DATA/'manifest.json').read_text());OUT.mkdir(parents=True,exist_ok=True)
    source=INITIAL/'yolov8n.pt'
    if sha(source)!='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36':raise ValueError('Official initializer changed')
    if sha(DATA/'dataset.yaml')!=manifest['dataset_yaml_sha256']:raise ValueError('Frozen dataset YAML changed')
    label_receipts=[]
    for row in manifest['records']:
        if sha(ROOT/row['image'])!=row['source_sha256']:raise ValueError('Frozen input bytes changed')
        label_path=DATA/'labels'/row['split']/(row['id']+'.txt')
        expected=''.join(f"{b['class_id']} {(b['bbox'][0]+b['bbox'][2]/2)/640:.8f} {(b['bbox'][1]+b['bbox'][3]/2)/640:.8f} {b['bbox'][2]/640:.8f} {b['bbox'][3]/640:.8f}\n" for b in row['boxes'])
        if label_path.read_text()!=expected:raise ValueError('Prepared label no longer matches frozen source polygons/boxes')
        label_receipts.append({'path':label_path.relative_to(ROOT).as_posix(),'sha256':sha(label_path)})
    write(OUT/'protocol-before-training.json',{'config':CONFIG,'dataset_manifest_sha256':sha(DATA/'manifest.json'),
         'source_pt_sha256':sha(source),'script_sha256':sha(Path(__file__)),'shared_helper_sha256':sha(Path(__file__).with_name('train_sdnet_expansion.py')),
         'verified_label_assets':label_receipts,'execution':EXECUTION,'private_videos_used':False,
         'initial_metrics_trigger':'Prior calibration severely underfit; current8epochbudget setbeforev2forward. Existingv1TESTis exposeddevelopment.'})
    torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);random.seed(CONFIG['seed']);torch.set_num_threads(1);cv2.setNumThreads(1)
    torch_utils.NUM_THREADS=1
    original=YOLO(str(source));config=copy.deepcopy(original.model.yaml);config['nc']=2
    architecture=OUT/'architecture.yaml';architecture.write_text(yaml.safe_dump(config))
    wrapper=YOLO(str(architecture)).load(str(source));wrapper.model.names={i:x for i,x in enumerate(CLASSES)}
    observations=[]
    def record_threads(trainer):
        value={'epoch':int(trainer.epoch),'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}
        observations.append(value);write(OUT/'actual-training-threads.json',observations)
        if value['torch_threads']!=1:raise ValueError('Actual Torch thread budget changed')
    wrapper.add_callback('on_train_epoch_start',record_threads)
    started=time.perf_counter()
    wrapper.train(data=str(DATA/'dataset.yaml'),epochs=8,imgsz=512,batch=4,device='cpu',workers=0,freeze=4,
        optimizer='AdamW',lr0=.001,lrf=.1,warmup_epochs=1,nbs=4,seed=CONFIG['seed'],deterministic=True,
        plots=False,save=True,cache=False,val=False,amp=False,mosaic=0,mixup=0,fliplr=.5,
        hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
        project=str(OUT),name='fit',exist_ok=False,verbose=False)
    train_seconds=time.perf_counter()-started
    wrapper=YOLO(str(OUT/'fit/weights/last.pt'))
    calrows=[r for r in manifest['records']if r['split']=='calibration'];devrows=[r for r in manifest['records']if r['split']=='test']
    cal,speed=predictions(wrapper,calrows)
    cuts={}
    grid={str(t):metrics(calrows,cal,t)for t in CONFIG['thresholds']}
    for label in CLASSES:
        options=[]
        for threshold in CONFIG['thresholds']:
            m=grid[str(threshold)]['classes'][label];p,r=m['precision'],m['recall'];qualified=m['tp']>0 and p>=.95
            options.append((qualified,r if qualified else 2*p*r/max(1e-9,p+r),r,p,threshold))
        cuts[label]=max(options)[-1]
    # Thresholds are now frozen before reusing previously exposed development pixels.
    write(OUT/'thresholds-before-reused-development.json',{'class_confidence':cuts,'calibration_grid':grid,'config_sha256':sha(OUT/'protocol-before-training.json')})
    development,_=predictions(wrapper,devrows)
    results={'architecture':'yolov8n','config':CONFIG,'train_seconds':train_seconds,'actual_threads':observations,
             'after_calibration':perclass_summary(calrows,cal,cuts),'reused_development':perclass_summary(devrows,development,cuts),
             'reused_development_fixed_0_5':metrics(devrows,development,.5),'speed':speed,'release_pass':False,'validation_status':'research_only',
             'release_blocker':'Unknownphysical-siteidentity; v1TESTalreadyexposed; all50+50independenceclaimsunavailable.'}
    write(OUT/'results.json',results)
    exported=Path(wrapper.export(format='onnx',imgsz=512,opset=17,simplify=False,dynamic=False,batch=1,device='cpu'))
    DEST.mkdir(parents=True,exist_ok=True);shutil.copyfile(exported,DEST/'model.onnx')
    write(DEST/'manifest.json',{'architecture':'yolov8n','input_size':512,'model_file':'model.onnx','model_sha256':sha(DEST/'model.onnx'),
        'inspection_task':'brackets_railings','title':'SDNET2025 512px extended fastening fit (research)','classes':CLASSES,
        'confidence':min(cuts.values()),'class_confidence':cuts,'preprocessing':'letterbox_rgb_0_1',
        'source':manifest['source'],'license':'SDNETCCBY4;UltralyticsAGPLconditionsapply','validation_status':'research_only',
        'evaluation_report':str((OUT/'results.json').resolve()),'private_videos_used':False,'validation_summary':'ExploratoryCAL/reuseddevelopment; unknownphysicalsites; noqualification'})
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
