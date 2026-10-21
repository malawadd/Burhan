"""Reproduce calibration component statistics without opening comparison data."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory_defect import SurfaceDefectModel
from train_factory_surface import samples,matrix
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,default=ROOT/'models/factory/kolektor_surface_rf.joblib')
p.add_argument('--dataset',type=Path,default=ROOT/'datasets/factory/KolektorSDD')
p.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-accuracy/calibration-scores-recollected.json')
a=p.parse_args();cv2.setNumThreads(2)
model=SurfaceDefectModel(a.model,backend='native_cascade');thresholds=[.5,.6,.7,.75,.8,.85,.9,.925,.95,.975,.98,.99,.995]
model.pixel_threshold=.5;records=[];started=time.perf_counter()
for path,_,label in samples(a.dataset,model.metadata['split_items']['calibration']):
 scores,_,_=model.predict(cv2.imread(str(path)));areas={}
 for threshold in thresholds:
  _,_,stats,_=cv2.connectedComponentsWithStats((scores>=threshold).astype(np.uint8),connectivity=8)
  areas[str(threshold)]=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
 records.append({'path':str(path),'label':label,'areas':areas,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
choices=[{'threshold':t,'area':4,**matrix([r['label'] for r in records],[r['areas'][str(t)]>=4 for r in records])} for t in thresholds]
report={'protocol':'Calibration-only component statistics for separate preregistered policy selection; native lowerbounds preserve all thresholds>=0.5',
        'model_sha256':model.sha256,'metadata_sha256':model.metadata_sha256,'thresholds':thresholds,
        'records':records,'choices':choices,'elapsed_seconds':time.perf_counter()-started}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'records','choices'}},indent=2))
