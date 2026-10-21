"""Calibrate frozen fused neural model only on original calibration items."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory_encoder import FactorySegmentation
from train_factory_surface import samples,matrix
cv2.setNumThreads(2)
directory=ROOT/'output/optimization/factory-accuracy'
artifact=ROOT/'models/factory/kolektor_mobilenet_seg_v7.onnx'
original=json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
protocol=json.loads((directory/'neural-mlp-protocol-before-fit.json').read_text(encoding='utf-8'))
model=FactorySegmentation(artifact);records=[];started=time.perf_counter();times=[]
for path,_,label in samples(ROOT/'datasets/factory/KolektorSDD',original['split_items']['calibration']):
 frame=cv2.imread(str(path));tick=time.perf_counter();scores,_,_=model.predict(frame,.5,1);times.append((time.perf_counter()-tick)*1000)
 areas={}
 for threshold in protocol['threshold_grid']:
  _,_,stats,_=cv2.connectedComponentsWithStats((scores>=threshold).astype(np.uint8),connectivity=8)
  areas[str(threshold)]=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
 records.append({'path':str(path),'label':label,'areas':areas})
def choose(rows):
 normal=sum(not row['label'] for row in rows);choices=[]
 for threshold in protocol['threshold_grid']:
  for area in protocol['minimum_area_grid']:
   result=matrix([r['label'] for r in rows],[r['areas'][str(threshold)]>=area for r in rows]);choices.append({'threshold':threshold,'area':area,**result})
 eligible=[point for point in choices if point['fp']/max(1,normal)<=3/70]
 return max(eligible,key=lambda point:(point['recall'],-point['threshold'],-point['area'])) if eligible else None,choices
point,choices=choose(records);jackknife=[]
for item in original['split_items']['calibration']:
 fit=[r for r in records if Path(r['path']).parent.name!=item];held=[r for r in records if Path(r['path']).parent.name==item]
 chosen,_=choose(fit)
 jackknife.extend({'path':r['path'],'label':r['label'],'prediction':r['areas'][str(chosen['threshold'])]>=chosen['area'] if chosen else None} for r in held)
report={'selected':point,'choices':choices,'calibration_records':records,'calibration_item_jackknife':matrix([r['label'] for r in jackknife],[r['prediction'] for r in jackknife]),
        'jackknife_abstained':sum(r['prediction'] is None for r in jackknife),'neural_predict_ms':{'p50':float(np.median(times)),'p95':float(np.percentile(times,95)),'max':max(times)},
        'execution_devices':model.execution_devices,'elapsed_seconds':time.perf_counter()-started}
if point:
 metadata={**original,'feature_version':'surface_mobilenet_seg_v1','profile_version':'kolektor_mobilenet_seg_v7',
           'model_name':'Frozen ImageNet MobileNet early features + supervised fused neural segmentation head',
           'pixel_threshold':point['threshold'],'minimum_component_pixels':point['area'],
           'input_width':256,'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
           'training_protocol':protocol,'threshold_selection':protocol['selection'],'trees':0,
           'score_meaning':'Learned pointwise sigmoid defect score; not calibrated probability',
           'calibration_selected_point':point,'pretrained_encoder':json.loads((ROOT/'models/factory/encoder_sources/weights-provenance.json').read_text(encoding='utf-8'))}
 artifact.with_suffix('.onnx.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 report.update(model_sha256=metadata['model_sha256'],metadata_sha256=hashlib.sha256(artifact.with_suffix('.onnx.json').read_bytes()).hexdigest())
(directory/'neural-calibration-v7.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'choices','calibration_records'}},indent=2))
