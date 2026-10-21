"""Retain real-image CPU adapter receipts for mobile/browser numerical checks.

This reuses TRAIN photographs; results are numerical diagnostics, not accuracy tests.
Run with the parent project's .venv Python from its root.
"""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_task_detector import ConstructionTaskDetector

APP=ROOT/'burhan'
OUT=APP/'tests/fixtures/research'
OUT.mkdir(parents=True,exist_ok=True)
cv2.setNumThreads(1)
sources={
 'research-phele':'output/evaluation/construction-expansion-phele-v1/training-manifest.json',
 'research-fasteners':'datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json',
 'research-stain':'datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json',
}
models=json.loads((APP/'assets/models/manifest.json').read_text(encoding='utf-8'))
receipt=dict(purpose='Reused TRAIN photograph numerical/runtime diagnostic, not independent accuracy or phone qualification',opencv=cv2.__version__,cases=[])
for model in models:
 if model['id'] not in sources:continue
 rows=json.loads((ROOT/sources[model['id']]).read_text())['records']
 train=[r for r in rows if r['split']=='train']
 positive=next(r for r in train if r.get('boxes'))
 negative=next(r for r in train if not r.get('boxes'))
 detector=ConstructionTaskDetector(model['sourceManifest'])
 for variant,row in [('train',positive),('rotated',positive),('target-negative',negative)]:
  path=ROOT/row['image']; image=cv2.imread(str(path)); assert image is not None,path
  if variant=='rotated': image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
  h,w=image.shape[:2]; bound=383 if variant=='rotated' else 417
  if max(h,w)>bound:image=cv2.resize(image,(max(1,round(w*bound/max(h,w))),max(1,round(h*bound/max(h,w)))),interpolation=cv2.INTER_AREA)
  h,w=image.shape[:2]; size=model['inputSize']; scale=min(size/w,size/h); nw,nh=max(1,round(w*scale)),max(1,round(h*scale));px,py=(size-nw)//2,(size-nh)//2
  canvas=np.full((size,size,3),114,np.uint8);canvas[py:py+nh,px:px+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
  tensor=np.ascontiguousarray(canvas[:,:,::-1].astype(np.float32).transpose(2,0,1)[None]/255)
  key=model['id']+'-'+variant
  rgba=cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
  cv2.imwrite(str(OUT/(key+'.png')),image)
  (OUT/(key+'-rgba.gz')).write_bytes(gzip.compress(rgba.tobytes(),mtime=0))
  (OUT/(key+'-input.gz')).write_bytes(gzip.compress(tensor.tobytes(),mtime=0))
  start=time.perf_counter(); outputs=detector.session.run(None,{detector.input_name:tensor});ms=(time.perf_counter()-start)*1000
  entries=[]
  for i,output in enumerate(outputs):
   name=key+f'-output{i}.gz';(OUT/name).write_bytes(gzip.compress(output.tobytes(),mtime=0));entries.append(dict(name=detector.output_names[i],file=name,dims=list(output.shape),dtype=str(output.dtype)))
  detections=detector.infer(image)
  receipt['cases'].append(dict(id=key,modelId=model['id'],source=path.relative_to(ROOT).as_posix(),sourceSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               width=w,height=h,rgba=key+'-rgba.gz',input=key+'-input.gz',image=key+'.png',outputs=entries,observations=detections,cpuInferenceMs=ms))
  print(key,len(detections),f'{ms:.1f}ms',flush=True)
(OUT/'reference.json').write_text(json.dumps(receipt,indent=2)+'\n')
