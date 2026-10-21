"""Measure the laptop camera transport only; never encode or save camera pixels."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from edge_runtime.sources import LocalSource

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seconds',type=float,default=5)
p.add_argument('--fps',type=float,default=30)
p.add_argument('--width',type=int,default=960)
p.add_argument('--height',type=int,default=540)
p.add_argument('--index',type=int,default=0)
p.add_argument('--output',type=Path,default=ROOT/'output/optimization/camera-capture.json')
a=p.parse_args()
source=LocalSource(f'camera:{a.index}','factory',a.fps,a.width,a.height)
started=time.perf_counter()
try:
    source.open()
    fps_request_accepted=source.capture_properties.get('fps_request_accepted')
    reported_fps=source.capture.get(cv2.CAP_PROP_FPS)
    frame,_=source.read()
    if frame is None:raise RuntimeError('Camera returned no frame')
    startup_ms=(time.perf_counter()-started)*1000
    dimensions=list(frame.shape[1::-1]);del frame
    timestamps=[];read_ms=[]
    end=time.perf_counter()+a.seconds
    while time.perf_counter()<end:
        read_start=time.perf_counter()
        frame,_=source.read()
        read_end=time.perf_counter()
        if frame is None:raise RuntimeError('Camera capture failed during measurement')
        timestamps.append(read_end);read_ms.append((read_end-read_start)*1000)
        del frame
    report={'scope':'Camera transport only. No factory model accuracy, exposure latency or physical line coverage measured.',
        'backend':source.backend,'requested_fps':a.fps,'fps_request_accepted':fps_request_accepted,
        'driver_reported_fps':reported_fps,'actual_dimensions':dimensions,'frames_read':len(timestamps),
        'startup_ms':round(startup_ms,2),'observed_receipt_fps':round((len(timestamps)-1)/(timestamps[-1]-timestamps[0]),2),
        'read_ms':{f'p{x}':round(float(np.percentile(read_ms,x)),2) for x in (50,95,99)},
        'capture_properties':source.capture_properties,
        'saved_media':False,'encoded_media':False}
finally:source.close()
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
