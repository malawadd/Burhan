"""Real labelled-image diagnostic and genuine-video execution; no synthetic proof."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.construction import ConstructionProfile
from edge_runtime.profiles.construction_model import ConstructionModel


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, d = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, r-x)*max(0, d-y)
    return inter/max(1, a[2]*a[3]+b[2]*b[3]-inter)


def ground_truth(path, shape, mode):
    h, w = shape
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = list(map(float, line.split()))
        if not values: continue
        label = int(values[0])
        if mode == "ppe":
            label = {0:5,1:0,2:7}[label]
            cx,cy,bw,bh=values[1:];box=((cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h)
        else:
            pts=np.asarray(values[1:]).reshape(-1,2)*[w,h]
            lo,hi=pts.min(0),pts.max(0);box=(*lo,*(hi-lo))
        row={"class_id":label,"bbox":box}
        if mode=="crack":
            mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.round(pts).astype(np.int32)],1);row["mask"]=mask>0
        rows.append(row)
    return rows


def measure(mode, engine, device, limit, ppe_preprocess="stretch", split="test", mask_rendering="masks"):
    model=ConstructionModel({"mode":mode,"engine":engine,"openvino_device":device,"ppe_preprocess":ppe_preprocess,"mask_rendering":mask_rendering})
    if mode=="ppe":
        base=ROOT/"datasets/construction/atharion-v1.0-public"/split;images=base/"images";labels=base/"labels"
    else:
        base=ROOT/"datasets/construction";images=base/"images/test";labels=base/"labels/test"
    paths=sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    if limit:paths=paths[:limit]
    sample=cv2.imread(str(paths[0]));cold=time.perf_counter();model.infer(sample);cold=(time.perf_counter()-cold)*1000
    for _ in range(3):model.infer(sample)
    counts={i:{"tp":0,"fp":0,"fn":0} for i in ([0,5,7] if mode=="ppe" else [0])}
    timings=[];cases=[];pixel={"tp":0,"fp":0,"fn":0};dice=[]
    for path in paths:
        frame=cv2.imread(str(path));truth=ground_truth(labels/(path.stem+".txt"),frame.shape[:2],mode)
        started=time.perf_counter();pred=model.infer(frame);timings.append((time.perf_counter()-started)*1000)
        if mode=="crack" and mask_rendering=="masks":
            true_mask=np.zeros(frame.shape[:2],bool);pred_mask=true_mask.copy()
            for g in truth:true_mask|=g["mask"]
            for p in pred:pred_mask|=p["mask"]
            intersection=int(np.count_nonzero(true_mask&pred_mask));fp=int(np.count_nonzero(pred_mask&~true_mask));fn=int(np.count_nonzero(true_mask&~pred_mask))
            pixel["tp"]+=intersection;pixel["fp"]+=fp;pixel["fn"]+=fn
            dice.append(2*intersection/max(1,2*intersection+fp+fn))
        # The PPE diagnostic has labels only for the three declared classes.
        # No-Mask and no-PPE channels are not scored as false positives here.
        pred=[p for p in pred if p["class_id"] in counts]
        unmatched=list(truth);tp=0
        for p in pred:
            candidates=[g for g in unmatched if g["class_id"]==p["class_id"]]
            best=max(candidates,key=lambda g:iou(p["bbox"],g["bbox"]),default=None)
            if best is not None and iou(p["bbox"],best["bbox"])>=.5:
                unmatched=[g for g in unmatched if g is not best];counts[p["class_id"]]["tp"]+=1;tp+=1
            else:counts[p["class_id"]]["fp"]+=1
        for g in unmatched:counts[g["class_id"]]["fn"]+=1
        cases.append({"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"ground_truth":len(truth),"predictions":len(pred),"tp":tp})
    for c in counts.values():
        c["precision"]=c["tp"]/max(1,c["tp"]+c["fp"]);c["recall"]=c["tp"]/max(1,c["tp"]+c["fn"])
    total={k:sum(c[k] for c in counts.values()) for k in ("tp","fp","fn")}
    total["precision"]=total["tp"]/max(1,total["tp"]+total["fp"]);total["recall"]=total["tp"]/max(1,total["tp"]+total["fn"])
    return {"model":model.describe(),"images":len(paths),"scope":"Public-labelled compatibility diagnostic; unknown training overlap, not independent Saudi accuracy. Crack polygons scored as bounding boxes; foreground mask metrics only when mask_rendering=masks, with no background true-negative inflation.",
            "threshold":.25,"iou_match":.5,"counts":total,"per_class":counts,"cold_ms":cold,
            "warm_ms":{k:float(np.percentile(timings,p)) for k,p in [("p50",50),("p95",95),("p99",99)]},
            "serial_fps":1000/np.mean(timings),"pixel_segmentation":({**pixel,"iou":pixel["tp"]/max(1,sum(pixel.values())),"mean_image_dice":float(np.mean(dice))} if mode=="crack" and mask_rendering=="masks" else None),"cases":cases}


def video(engine,device,ppe_preprocess="stretch"):
    path=ROOT/"datasets/construction/men-at-work-cc0.webm"
    profile=ConstructionProfile({"engine":engine,"openvino_device":device,"mode":"ppe","stage":"site_safety","ppe_preprocess":ppe_preprocess})
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);size=(int(cap.get(3)),int(cap.get(4)))
    suffix="optimized" if engine=="openvino" else "baseline"
    if ppe_preprocess!="stretch":suffix+="-"+ppe_preprocess
    out=ROOT/f"output/construction-observed-real-video-{suffix}.mp4";writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,size)
    rows=[];timings=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            i=len(rows);started=time.perf_counter();result=profile.process(frame,FrameContext(i+1,i/fps,"file:"+str(path),"construction-video-evaluation",i/fps*1000));timings.append((time.perf_counter()-started)*1000)
            writer.write(result.annotated_frame)
            rows.append({"frame":i,"source_time_ms":i/fps*1000,"status":result.status,"code_compliance":result.domain_state["code_compliance"],"observations":result.domain_state["observations"]})
            if i==25:cv2.imwrite(str(ROOT/f"output/construction-observed-real-frame-{suffix}.jpg"),result.annotated_frame)
    finally:cap.release();writer.release()
    labels_path=ROOT/"datasets/construction/video-primary-worker-labels.json"
    labels=json.loads(labels_path.read_text(encoding="utf-8"));counts={2:{"tp":0,"fp":0,"fn":0},5:{"tp":0,"fp":0,"fn":0}}
    labelled_cases=[]
    for case in labels["frames"]:
        truth=list(case["objects"]);pred=[]
        for p in rows[case["frame"]]["observations"]:
            x,y,w,h=p["bbox"];minimum={2:8000,5:100000}.get(p["class_id"])
            if minimum and x+w/2<600 and w*h>=minimum:pred.append(p)
        for p in pred:
            best=max((g for g in truth if g["class_id"]==p["class_id"]),key=lambda g:iou(p["bbox"],g["bbox"]),default=None)
            if best is not None and iou(p["bbox"],best["bbox"])>=.5:truth.remove(best);counts[p["class_id"]]["tp"]+=1
            else:counts[p["class_id"]]["fp"]+=1
        for g in truth:counts[g["class_id"]]["fn"]+=1
        labelled_cases.append({"frame":case["frame"],"predictions":pred,"unmatched_labels":truth})
    return {"source":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_fps":fps,"frames":len(rows),"source_seconds":len(rows)/fps,
            "output":str(out),"serial_profile_fps":1000/np.mean(timings),"p95_ms":float(np.percentile(timings,95)),"includes":"model, decode, tracking, annotation; excludes video decode/encode",
            "scope":"Genuine CC0 construction recording, every source frame processed. Not a Saudi/code-compliance evaluation. Five manually annotated primary-worker frames are a tiny correlated diagnostic only.",
            "manual_label_diagnostic":{"protocol":labels["annotation_protocol"],"labels_sha256":hashlib.sha256(labels_path.read_bytes()).hexdigest(),"counts_by_class":counts,"cases":labelled_cases},"frames_observed":rows}


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--limit",type=int,default=0);a.add_argument("--output",default="output/evaluation/construction-benchmark.json");a.add_argument("--video-only",action="store_true")
    a.add_argument("--mode",choices=["all","ppe","crack"],default="all");a.add_argument("--mask-rendering",choices=["masks","boxes"],default="masks")
    a.add_argument("--engine",choices=["both","cpu","gpu"],default="both");a.add_argument("--skip-video",action="store_true");args=a.parse_args()
    result={"measured_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"evaluations":[]}
    if not args.video_only:
        for mode in (("ppe","crack") if args.mode=="all" else (args.mode,)):
            engines=[("onnxruntime","CPU"),("openvino","GPU")] if args.engine=="both" else [("onnxruntime","CPU")] if args.engine=="cpu" else [("openvino","GPU")]
            for engine,device in engines:
                print("Starting",mode,engine,device,flush=True);row=measure(mode,engine,device,args.limit,mask_rendering=args.mask_rendering);result["evaluations"].append(row);print({k:row[k] for k in ("images","counts","serial_fps","warm_ms")},flush=True)
    result["videos"]=[] if args.skip_video else [video("onnxruntime","CPU"),video("openvino","GPU")]
    output=ROOT/args.output;output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8");print(output,flush=True)
