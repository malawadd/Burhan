"""Bounded five-source TRAIN-only numerical/score diagnosis, no fitting or CAL.

First lexical TRAIN positive per class is pinned before pixels/NN. GT is used
only after full-frame inference to explain raw anchors, never as input ROI.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'output/evaluation/construction-wall-five-class-v2/train-export-diagnosis'
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd-wall-five-class-v1/prepared-yolo/input-manifest.json'
PT=ROOT/'output/training/construction-wall-five-class-v2/fixed-four-epoch-fit/weights/last.pt'
MODEL=ROOT/'models/construction/mbdd-wall-five-class-v2/wall-five-class-last4-640.onnx'
CLASSES=['crack','leakage','abscission','corrosion','bulge']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def freeze():
    rows=json.loads(DATA.read_text())['records'];selected=[]
    for label in CLASSES:
        row=sorted((row for row in rows if row['split']=='train' and any(b['label']==label for b in row['boxes'])),key=lambda row:row['source_name'])[0]
        if row['source_name'] not in [r['source_name'] for r in selected]:selected.append(row)
    paths=[Path(__file__),ROOT/'scripts/evaluate_mbdd_wall_detector.py',ROOT/'edge_runtime/profiles/construction_model.py']
    p={'scope':'Five first lexical TRAIN positives perclass, deduplicated; TRAIN-only numerical/source-score diagnosis, not accuracy or policy tuning. No CAL/18diagnostic/heldout pixels.',
       'selection':selected,'input_manifest_sha256':sha(DATA),'pt_sha256':sha(PT),'onnx_sha256':sha(MODEL),'classes':CLASSES,
       'executed_sources':{path.relative_to(ROOT).as_posix():sha(path) for path in paths},
       'numeric_guards':{'raw_shape':[1,9,8400],'finite':True,'score_range':[0,1],'coordinate_max_abs_error_input_pixels':1.,'class_score_max_abs_error':.01},
       'method':'Exact same full-frame serving letterbox RGB/255 input to fusedTorch, ONNXRuntimeCPU and actualOpenVINOGPU. GT is inspected after inference; rawanchor scores/boxIoU never become served predictions or threshold changes.',
       'policy':'Frozen all.85 selectedpoint and original.10-.85 grid unchanged; no new operatingpoint or fitting authorized by this diagnosis'}
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'protocol-before-train-diagnostic-forward.json'
    if target.exists() and json.loads(target.read_text())!=p:raise RuntimeError('Preserve TRAIN diagnostic protocol')
    write(target,p);return p


def diagnose():
    import cv2,numpy as np,torch,onnxruntime as ort
    from ultralytics import YOLO
    from evaluate_mbdd_wall_detector import load_model
    p=freeze();attempt=OUT/'execution-started.json'
    if attempt.exists():raise RuntimeError('Preserve attempted TRAIN diagnosis')
    write(attempt,{'state':'started','protocol_sha256':sha(OUT/'protocol-before-train-diagnostic-forward.json')})
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    wrapper=YOLO(str(PT),task='detect');net=wrapper.model.float().eval().fuse(verbose=False)
    if net.names!={i:name for i,name in enumerate(CLASSES)}:raise RuntimeError('Five-class Torch order changed')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    cpu=ort.InferenceSession(str(MODEL),sess_options=options,providers=['CPUExecutionProvider']);gpu=load_model();results=[]
    for row in p['selection']:
        image=ROOT/row['prepared_image']
        if sha(image)!=row['prepared_image_sha256']:raise RuntimeError('Pinned TRAIN image changed')
        raw=cv2.imread(str(image));tensor,transform=gpu.prepare_input(raw)
        with torch.inference_mode():a=net(torch.from_numpy(tensor))[0].cpu().numpy()
        b=cpu.run(None,{cpu.get_inputs()[0].name:tensor})[0]
        response=gpu.request.infer({0:tensor});c=np.array(response[gpu.compiled.output(0)],copy=True)
        outputs={'torch':a,'onnx_cpu':b,'serving_gpu':c}
        for name,output in outputs.items():
            if output.shape!=(1,9,8400) or not np.isfinite(output).all() or output[:,4:].min()<0 or output[:,4:].max()>1:raise RuntimeError('Raw shape/finite/classscore contract failed: '+name)
        parity={name:{'coordinate_max_abs_error_input_pixels':float(np.abs(value[:,:4]-a[:,:4]).max()),'class_score_max_abs_error':float(np.abs(value[:,4:]-a[:,4:]).max())} for name,value in [('onnx_cpu',b),('serving_gpu',c)]}
        for name,errors in parity.items():
            if errors['coordinate_max_abs_error_input_pixels']>1 or errors['class_score_max_abs_error']>.01:
                write(OUT/'numeric-failure.json',{'source':row['source_name'],'parity':parity});raise RuntimeError('Torch/export/serving numeric guard failed')
        xywh=c[0,:4].T;boxes=np.concatenate([xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2],axis=1);scores=c[0,4:]
        gt=[];sx,sy,px,py=transform
        for item in row['boxes']:
            x1,y1,x2,y2=item['source_xyxy'];target=np.array([x1*sx+px,y1*sy+py,x2*sx+px,y2*sy+py]);lo=np.maximum(boxes[:,:2],target[:2]);hi=np.minimum(boxes[:,2:],target[2:]);inter=np.prod(np.maximum(0,hi-lo),axis=1);areas=np.prod(np.maximum(0,boxes[:,2:]-boxes[:,:2]),axis=1);iou=inter/np.maximum(areas+np.prod(target[2:]-target[:2])-inter,1e-12)
            class_scores=scores[item['class_id']];anchor=int(np.argmax(class_scores));best=int(np.argmax(iou));localized=class_scores[iou>=.5]
            gt.append({'publisher_class':item['label'],'source_xyxy':item['source_xyxy'],'max_raw_class_score':float(class_scores[anchor]),'top_class_score_anchor_iou':float(iou[anchor]),'best_any_anchor_iou':float(iou[best]),'own_class_score_at_best_iou':float(class_scores[best]),'best_own_class_score_among_iou_ge_0_5':float(localized.max()) if len(localized) else None,'scope':'Raw-anchor TRAIN diagnostic, not a served detection or accuracy credit'})
        native=raw.copy()
        for item in row['boxes']:
            x1,y1,x2,y2=map(round,item['source_xyxy']);cv2.rectangle(native,(x1,y1),(x2,y2),(0,170,255),2);cv2.putText(native,item['label'],(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,170,255),2)
        preview=OUT/(Path(row['source_name']).stem+'-train-raw-publisher-gt.jpg');canvas=np.concatenate([raw,native],axis=1);cv2.imwrite(str(preview),canvas)
        raw_path=OUT/(Path(row['source_name']).stem+'-three-engine-raw-outputs.npz');np.savez_compressed(raw_path,**outputs)
        results.append({'source_name':row['source_name'],'raw_image_sha256':row['prepared_image_sha256'],'parity':parity,'max_class_scores':dict(zip(CLASSES,map(float,scores.max(axis=1)))),'publisher_gt_raw_anchor_diagnosis':gt,'raw_gt_preview':preview.relative_to(ROOT).as_posix(),'raw_gt_preview_sha256':sha(preview),'raw_outputs':raw_path.relative_to(ROOT).as_posix(),'raw_outputs_sha256':sha(raw_path)})
    write(OUT/'report.json',{'protocol_sha256':sha(OUT/'protocol-before-train-diagnostic-forward.json'),'numeric_guards_pass':True,'results':results,'scope':p['scope'],'no_policy_or_model_changed':True})
    write(attempt,{'state':'completed','protocol_sha256':sha(OUT/'protocol-before-train-diagnostic-forward.json'),'report_sha256':sha(OUT/'report.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['freeze','diagnose'],required=True);args=parser.parse_args()
    if args.stage=='freeze':freeze();print(sha(OUT/'protocol-before-train-diagnostic-forward.json'))
    else:diagnose()
