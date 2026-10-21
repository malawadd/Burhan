"""Bounded local SDNET detector comparison. Frozen data; calibration-only thresholds.

Runs one CPU thread, two YOLO epochs and 200 SSDLite updates, sequentially.
Results remain research because physical-connection independence is unverified.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key] = '1'
os.environ['YOLO_OFFLINE'] = 'true'
from pathlib import Path
from collections import Counter
import argparse, copy, functools, hashlib, json, random, time
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'datasets/construction/expansion-public-v1/sdnet2025/prepared'
OUT = ROOT/'output/training/construction-expansion-sdnet-v1'
MODELS = ROOT/'models/construction/expansion-sdnet-v1'
INITIAL = ROOT/'models/construction/expansion-initializers'
CLASSES = ['visibly_unseated_connection','fastener_absence_appearance']
torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')


def image_tensor(row, size=320):
    image = cv2.imread(str(ROOT/row['image']))
    image = cv2.cvtColor(cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.transpose(2,0,1).copy()).float()/255


def target(row, size=320):
    boxes = []
    for b in row['boxes']:
        x,y,w,h = b['bbox']; boxes.append([x/640*size,y/640*size,(x+w)/640*size,(y+h)/640*size])
    return {'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            'labels':torch.tensor([b['class_id']+1 for b in row['boxes']],dtype=torch.int64)}


def iou(a,b):
    x,y=max(a[0],b[0]),max(a[1],b[1]); xx,yy=min(a[0]+a[2],b[0]+b[2]),min(a[1]+a[3],b[1]+b[3])
    intersection=max(0,xx-x)*max(0,yy-y)
    union=a[2]*a[3]+b[2]*b[3]-intersection
    return intersection/union if union else 0.


def metrics(rows,predictions,threshold):
    stats={c:{'tp':0,'fp':0,'fn':0,'positive_images':0,'negative_images':0,'flagged_negative_images':0,'matched_ious':[]} for c in CLASSES}
    for row,preds in zip(rows,predictions):
        for cls,label in enumerate(CLASSES):
            truth=[b['bbox'] for b in row['boxes'] if b['class_id']==cls]
            chosen=sorted([p for p in preds if p['class_id']==cls and p['score']>=threshold],key=lambda p:-p['score'])
            stats[label]['positive_images' if truth else 'negative_images']+=1
            if not truth and chosen: stats[label]['flagged_negative_images']+=1
            matched=set()
            for p in chosen:
                candidates=[(iou(p['bbox'],b),i)for i,b in enumerate(truth)if i not in matched]
                best=max(candidates,default=(0,None))
                if best[0]>=.5:
                    matched.add(best[1]); stats[label]['tp']+=1; stats[label]['matched_ious'].append(best[0])
                else: stats[label]['fp']+=1
            stats[label]['fn']+=len(truth)-len(matched)
    for s in stats.values():
        s['precision']=s['tp']/max(1,s['tp']+s['fp']); s['recall']=s['tp']/max(1,s['tp']+s['fn'])
        s['mean_matched_iou']=float(np.mean(s.pop('matched_ious'))) if s.get('matched_ious') else 0.
    tp=sum(s['tp']for s in stats.values());fp=sum(s['fp']for s in stats.values());fn=sum(s['fn']for s in stats.values())
    return {'threshold':threshold,'iou_threshold':.5,'classes':stats,'micro_precision':tp/max(1,tp+fp),'micro_recall':tp/max(1,tp+fn),'tp':tp,'fp':fp,'fn':fn}


def yolo_predictions(wrapper,rows):
    paths=[str(ROOT/r['image'])for r in rows]
    started=time.perf_counter()
    result=wrapper.predict(paths,imgsz=320,conf=.01,iou=.45,device='cpu',batch=8,verbose=False)
    elapsed=time.perf_counter()-started
    predictions=[]
    for out in result:
        boxes=[]
        for b,s,c in zip(out.boxes.xyxy.tolist(),out.boxes.conf.tolist(),out.boxes.cls.tolist()):
            x,y,xx,yy=b; boxes.append({'class_id':int(c),'score':s,'bbox':[x,y,xx-x,yy-y]})
        predictions.append(boxes)
    return predictions,{'images':len(rows),'seconds':elapsed,'batch':8,'throughput_fps':len(rows)/elapsed,'includes_decode_resize_nms':True}


def ssd_predictions(model,rows):
    model.eval(); predictions=[]; times=[]
    with torch.no_grad():
        for row in rows:
            tensor=image_tensor(row)
            started=time.perf_counter();out=model([tensor])[0];times.append(time.perf_counter()-started)
            boxes=[]
            for b,s,c in zip(out['boxes'].tolist(),out['scores'].tolist(),out['labels'].tolist()):
                if s<.01:continue
                x,y,xx,yy=b;boxes.append({'class_id':int(c)-1,'score':s,'bbox':[x*2,y*2,(xx-x)*2,(yy-y)*2]})
            predictions.append(boxes)
    return predictions,{'images':len(rows),'batch':1,'seconds':sum(times),'throughput_fps':len(rows)/sum(times),
                        'inference_ms_p50':float(np.percentile(times,50)*1000),'inference_ms_p95':float(np.percentile(times,95)*1000),'includes_decode_resize_nms':False}


def summarize(name,before,after,cal,test,cal_rows,test_rows,speed):
    choices=[]
    for threshold in [.05,.10,.15,.20,.25,.35,.50,.65,.80]:
        m=metrics(cal_rows,cal,threshold);p=m['micro_precision'];r=m['micro_recall']
        choices.append((p>=.95,r if p>=.95 else 2*p*r/max(1e-9,p+r),p,threshold))
    threshold=max(choices)[-1]
    result={'architecture':name,'validation_status':'research_only','selection':'Final fixed-budget checkpoint; threshold grid ranked on CAL only, prefer>=.95 precision then recall elseF1.',
            'before_test_fixed_point_0_5':metrics(test_rows,before,.5),'after_test_fixed_point_0_5':metrics(test_rows,test,.5),
            'after_calibration_selected':metrics(cal_rows,cal,threshold),'after_test_calibration_selected':metrics(test_rows,test,threshold),
            'speed':speed,'release_pass':False,'release_blocker':'No verified physical-connection/site grouping; no independent50+50validation; bounded fit is exploratory.'}
    write(OUT/f'{name}-results.json',result)
    return result,threshold


def train_yolo(manifest):
    if (OUT/'yolo-fit').exists():
        raise ValueError('Preserve existing YOLO fit; use a separately versioned experiment directory')
    from ultralytics import YOLO
    import yaml
    source=INITIAL/'yolov8n.pt'
    if sha(source)!='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36':raise ValueError('Official initializer changed')
    wrapper=YOLO(str(source));config=copy.deepcopy(wrapper.model.yaml);config['nc']=2
    architecture=OUT/'yolov8n-two-class.yaml';architecture.write_text(yaml.safe_dump(config))
    model=YOLO(str(architecture)).load(str(source));model.model.names={i:x for i,x in enumerate(CLASSES)}
    test_rows=[r for r in manifest['records']if r['split']=='test'];cal_rows=[r for r in manifest['records']if r['split']=='calibration']
    before,_=yolo_predictions(model,test_rows)
    started=time.perf_counter()
    model.train(data=str(DATA/'dataset.yaml'),epochs=2,imgsz=320,batch=8,device='cpu',workers=0,freeze=10,
                optimizer='AdamW',lr0=.002,lrf=.1,warmup_epochs=0,nbs=8,seed=20261021,deterministic=True,
                plots=False,save=True,cache=False,val=False,amp=False,mosaic=0,mixup=0,fliplr=0,
                hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
                project=str(OUT),name='yolo-fit',exist_ok=False,verbose=False)
    train_seconds=time.perf_counter()-started
    # Fixed final checkpoint; no best-by-test or best-by-owner-video selection.
    model=YOLO(str(OUT/'yolo-fit/weights/last.pt'))
    cal,speed=yolo_predictions(model,cal_rows);test,_=yolo_predictions(model,test_rows)
    result,threshold=summarize('yolov8n',before,None,cal,test,cal_rows,test_rows,speed);result['train_seconds']=train_seconds
    dest=MODELS/'yolov8n';dest.mkdir(parents=True,exist_ok=True)
    exported=Path(model.export(format='onnx',imgsz=320,opset=17,simplify=False,dynamic=False,batch=1,device='cpu'))
    import shutil;shutil.copyfile(exported,dest/'model.onnx')
    asset={'architecture':'yolov8n','input_size':320,'model_file':'model.onnx','model_sha256':sha(dest/'model.onnx'),
           'classes':CLASSES,'confidence':threshold,'preprocessing':'letterbox_rgb_0_1','source':manifest['source'],'inspection_task':'brackets_railings',
           'license':'SDNETdataCCBY4;UltralyticsAGPL3code/modelconditionsapply','validation_status':'research_only',
           'evaluation_report':str((OUT/'yolov8n-results.json').resolve()),'private_videos_used':False}
    write(dest/'manifest.json',asset);write(OUT/'yolov8n-results.json',result)
    return result


class SSDExport(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,images):
        result=self.model([images[0]])[0]
        return result['boxes'],result['labels'],result['scores']


def train_ssd(manifest):
    if (MODELS/'ssdlite320'/'model-state.pt').exists():
        raise ValueError('Preserve existing SSD weights; use a separately versioned experiment directory')
    source=INITIAL/'ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    if sha(source)!='a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2':raise ValueError('Official initializer changed')
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=91)
    model.load_state_dict(torch.load(source,map_location='cpu',weights_only=True))
    in_channels=[branch[-1].in_channels for branch in model.head.classification_head.module_list]
    model.head.classification_head=SSDLiteClassificationHead(in_channels,model.anchor_generator.num_anchors_per_location(),3,
        functools.partial(nn.BatchNorm2d,eps=.001,momentum=.03))
    for parameter in model.backbone.parameters():parameter.requires_grad=False
    train_rows=[r for r in manifest['records']if r['split']=='train'];cal_rows=[r for r in manifest['records']if r['split']=='calibration'];test_rows=[r for r in manifest['records']if r['split']=='test']
    before,_=ssd_predictions(model,test_rows)
    cached=[(image_tensor(r),target(r))for r in train_rows]
    generator=random.Random(20261021);optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001,weight_decay=.0005)
    losses=[];started=time.perf_counter();model.train();model.backbone.eval()
    for step in range(200):
        selected=generator.sample(cached,8);loss_dict=model([r[0]for r in selected],[r[1]for r in selected]);loss=sum(loss_dict.values())
        if not torch.isfinite(loss):raise ValueError('Nonfinite SSDLite loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();losses.append(float(loss.detach()))
        if (step+1)%25==0:print('SSD_STEP',step+1,'LOSS',float(loss.detach()),'SECONDS',time.perf_counter()-started,flush=True)
    train_seconds=time.perf_counter()-started;model.eval()
    dest=MODELS/'ssdlite320';dest.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),dest/'model-state.pt')
    cal,speed=ssd_predictions(model,cal_rows);test,_=ssd_predictions(model,test_rows)
    result,threshold=summarize('ssdlite320',before,None,cal,test,cal_rows,test_rows,speed)
    result.update({'train_seconds':train_seconds,'steps':200,'batch':8,'loss_first_10':float(np.mean(losses[:10])),'loss_last_10':float(np.mean(losses[-10:]))})
    torch.onnx.export(SSDExport(model),torch.zeros(1,3,320,320),str(dest/'model.onnx'),input_names=['images'],output_names=['boxes','labels','scores'],
                      opset_version=17,dynamo=False,dynamic_axes={'boxes':{0:'detections'},'labels':{0:'detections'},'scores':{0:'detections'}})
    asset={'architecture':'ssdlite320','input_size':320,'model_file':'model.onnx','model_sha256':sha(dest/'model.onnx'),
           'classes':CLASSES,'confidence':threshold,'preprocessing':'letterbox_rgb_0_1','source':manifest['source'],'inspection_task':'brackets_railings',
           'license':'SDNETdataCCBY4;TorchVisionBSD3Clause;COCOpretrainingconditionsapply','validation_status':'research_only',
           'evaluation_report':str((OUT/'ssdlite320-results.json').resolve()),'private_videos_used':False}
    write(dest/'manifest.json',asset);write(OUT/'ssdlite320-results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=['yolo','ssd','both'],default='both');args=parser.parse_args()
    if args.model in ['yolo','both'] and (OUT/'yolo-fit').exists():
        raise ValueError('Preserve existing YOLO fit; no replay or protocol overwrite permitted')
    if args.model in ['ssd','both'] and (MODELS/'ssdlite320'/'model-state.pt').exists():
        raise ValueError('Preserve existing SSD fit; no replay or protocol overwrite permitted')
    OUT.mkdir(parents=True,exist_ok=True);torch.manual_seed(20261021);np.random.seed(20261021);random.seed(20261021)
    manifest=json.loads((DATA/'manifest.json').read_text());
    if manifest['classes']!=CLASSES or manifest['private_videos_used']:raise ValueError('Dataset contract changed')
    for row in manifest['records']:
        if sha(ROOT/row['image'])!=row['source_sha256']:raise ValueError('Frozen image bytes changed')
    protocol=OUT/('protocol-before-training.json' if not (OUT/'protocol-before-training.json').exists() else f'protocol-before-{args.model}-retry.json')
    if protocol.exists():raise ValueError('Preserve prior protocol; do not silently repeat an experiment')
    write(protocol,{'dataset_manifest_sha256':sha(DATA/'manifest.json'),'script_sha256':sha(Path(__file__)),
        'thread_count':1,'seed':20261021,'yolo_epochs':2,'yolo_image_size':320,'yolo_freeze':10,'ssd_steps':200,'ssd_image_size':320,
        'ssd_frozen_backbone':True,'batch':8,'selection':'Fixed final checkpoint; CALthresholdgridonly. TESTnotusedforselection.',
        'physical_scene_independence':'unverified; researchonly','private_videos_used':False})
    if args.model in ['yolo','both']:train_yolo(manifest)
    if args.model in ['ssd','both']:train_ssd(manifest)
