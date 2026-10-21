"""Run original recorded video in source order and save actual observations."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.construction import ConstructionProfile

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--source",required=True);a.add_argument("--mode",choices=["ppe","crack"],required=True)
    a.add_argument("--seconds",type=float,default=10);a.add_argument("--start",type=float,default=0)
    a.add_argument("--output",default="output/construction-video-run");a.add_argument("--engine",default="openvino",choices=["openvino","onnxruntime"]);a.add_argument("--device",default="GPU",choices=["CPU","GPU"])
    a.add_argument("--ppe-preprocess",default="letterbox",choices=["stretch","letterbox"])
    a.add_argument("--max-edge",type=int,default=1920,help="Analysis long-edge cap matching runtime; 0 preserves original dimensions")
    a.add_argument("--mask-rendering",choices=["masks","boxes"],default="masks")
    a.add_argument("--roi",type=float,nargs=4,metavar=("X1","Y1","X2","Y2"),help="Explicit operator material ROI, normalized; outside is unassessed")
    args=a.parse_args();source=Path(args.source).resolve();out=Path(args.output).resolve();out.parent.mkdir(exist_ok=True,parents=True)
    if not 0<args.seconds<=60 or not 0<=args.start or args.max_edge<0:raise ValueError("seconds must be >0 and <=60; start and max-edge must be >=0")
    profile=ConstructionProfile({"mode":args.mode,"engine":args.engine,"openvino_device":args.device,"ppe_preprocess":args.ppe_preprocess,"mask_rendering":args.mask_rendering,"roi":args.roi})
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps<=0:raise ValueError("Real video did not open with a valid frame rate")
    cap.set(cv2.CAP_PROP_POS_MSEC,args.start*1000);writer=None;times=[];observations=[];started=time.perf_counter();source_shape=None;analysis_shape=None
    try:
        for index in range(round(args.seconds*fps)):
            ok,frame=cap.read()
            if not ok:break
            source_shape=list(frame.shape[:2]);analysis_scale=min(1,args.max_edge/max(frame.shape[:2])) if args.max_edge else 1
            if analysis_scale<1:frame=cv2.resize(frame,(round(frame.shape[1]*analysis_scale),round(frame.shape[0]*analysis_scale)),interpolation=cv2.INTER_AREA)
            analysis_shape=list(frame.shape[:2])
            if writer is None:
                h,w=frame.shape[:2];ratio=min(1,1280/max(h,w));size=(round(w*ratio),round(h*ratio))
                writer=cv2.VideoWriter(str(out.with_suffix('.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
                if not writer.isOpened():raise ValueError("Annotated video encoder failed")
            began=time.perf_counter();result=profile.process(frame,FrameContext(index+1,args.start+index/fps,"file:"+str(source),"real-construction-video",(args.start+index/fps)*1000));times.append((time.perf_counter()-began)*1000)
            writer.write(cv2.resize(result.annotated_frame,size))
            if index in {0,round(fps*args.seconds/2)}:cv2.imwrite(str(out.with_name(out.name+f'-frame-{index}').with_suffix('.jpg')),result.annotated_frame)
            observations.append({"frame_offset":index,"source_time_ms":(args.start+index/fps)*1000,"status":result.status,"domain_state":result.domain_state})
    finally:
        cap.release()
        if writer is not None:writer.release()
    report={"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"source_fps":fps,"start_seconds":args.start,
            "frames":len(times),"clip_seconds":len(times)/fps,"full_pipeline_seconds":time.perf_counter()-started,
            "original_source_shape_hw":source_shape,"analysis_shape_hw":analysis_shape,"analysis_max_edge":args.max_edge,
            "coordinate_scope":"Observation boxes are in analysis pixels; multiply by original/analysis width and height to map back. Original source file remains unmodified.",
            "model":profile.describe(),"warm_serial_profile_fps":1000/np.mean(times[1:]) if len(times)>1 else None,
            "profile_p95_ms":float(np.percentile(times,95)) if times else None,"observations":observations,
            "scope":"Original real recording, each selected source frame processed once, no generated/duplicated frames. Annotated output is downscaled only for export. Unlabelled video is execution evidence, not accuracy or code compliance. Timed profile includes preprocessing/inference/decode/tracking/annotation, excludes video decode/encode."}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='observations'},indent=2))
