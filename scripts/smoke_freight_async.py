"""Exercise the actual async profile with public pixels and blocked network connections."""
import json
from pathlib import Path
import socket
import time
import cv2
import numpy as np
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.freight import FreightProfile

root=Path(__file__).resolve().parents[1]
path=root/'datasets/freight/invoice-pack/01-simple-english/invoice.png'
frame=cv2.imread(str(path))
cv2.setNumThreads(2)
profile=FreightProfile({'stable_frames':2,'required_pages':['invoice']})
def blocked(*args,**kwargs):raise AssertionError('Offline inference attempted networking')
socket.socket.connect=blocked
start=time.monotonic()
durations=[]
result=None
try:
 for i in range(400):
  begin=time.perf_counter()
  result=profile.process(frame,FrameContext(i,time.monotonic(),source_id='image:'+str(path),session_id='async-public'))
  durations.append((time.perf_counter()-begin)*1000)
  if result.status=='accepted':break
  if time.monotonic()-start>45:raise RuntimeError('Public invoice OCR did not finish within45seconds')
  time.sleep(.1)
 duplicate=profile.process(frame,FrameContext(i+1,time.monotonic(),source_id='image:'+str(path),session_id='async-public'))
 report={'source':str(path.relative_to(root)),'network_connect_blocked':True,'processed_guidance_frames':len(durations),'elapsed_s':round(time.monotonic()-start,3),'guidance_p50_ms':round(float(np.percentile(durations,50)),2),'guidance_p95_ms':round(float(np.percentile(durations,95)),2),'guidance_max_ms':round(max(durations),2),'ocr_job_ms':(result.domain_state['ocr'] or {}).get('ocr_job_ms'),'accepted':result.status=='accepted','packet_complete':result.domain_state['complete'],'duplicate_suppressed':duplicate.status=='duplicate' and not duplicate.emit_event,'fields':result.domain_state['ocr']['fields'] if result.domain_state['ocr'] else {},'ocr_error':result.domain_state['ocr_error']}
 out=root/'output/evaluation/freight-async-smoke.json'
 out.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
 if not report['accepted']:raise RuntimeError('Public invoice was not accepted')
finally:
 profile.close(timeout=30)
