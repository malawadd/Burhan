"""Sample frozen actual GPU activations from training items only."""
from pathlib import Path
import json,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory_encoder import FactoryEncoder
from train_factory_surface import samples
cv2.setNumThreads(2);rng=np.random.default_rng(20261025)
metadata=json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
path=ROOT/'models/factory/mobilenetv3_early_512.onnx'
encoder=FactoryEncoder(path,hashlib.sha256(path.read_bytes()).hexdigest())
directory=ROOT/'output/optimization/factory-accuracy';directory.mkdir(exist_ok=True,parents=True)
protocol={'seed':20261025,'training_items':metadata['split_items']['train'],
          'excluded_calibration_items':metadata['split_items']['calibration'],'excluded_comparison_items':metadata['split_items']['test'],
          'normal_per_view':2000,'positive_cap':2000,'positive_repeat':4,
          'positive_train_view_brightness':[1.,.9,1.1],
          'head':'40->16ReLU->1logit, standardizedtrainingfeatures,AdamWlr0.001weightdecay0.0001,15epochs,batch8192,BCEpositiveweight2',
          'selection':'CalibrationFPR<=3/70,maxrecallthenlowervotethresholdandsmallercomponent',
          'threshold_grid':[.5,.7,.8,.9,.95,.98,.99,.995],'minimum_area_grid':[1,2,4,8]}
(directory/'neural-mlp-protocol-before-fit.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
xs,ys=[],[];started=time.perf_counter()
for index,(image,mask,defective) in enumerate(samples(ROOT/'datasets/factory/KolektorSDD',metadata['split_items']['train'])):
 frame=cv2.imread(str(image));target=cv2.resize(mask.astype(np.float32),(256,640),interpolation=cv2.INTER_AREA).ravel()>0
 positive=np.flatnonzero(target);negative=np.flatnonzero(~target)
 for factor in ([1.,.9,1.1] if defective else [1.]):
  view=frame if factor==1 else np.clip(frame.astype(np.float32)*factor,0,255).astype(np.uint8)
  normal=rng.choice(negative,min(2000 if factor==1 else 1000,len(negative)),replace=False)
  chosen_pos=rng.choice(positive,min(2000,len(positive)),replace=False) if len(positive) else positive
  selected=np.concatenate([normal,np.tile(chosen_pos,4)])
  resized=cv2.resize(view,(512,1280),interpolation=cv2.INTER_AREA)
  result=encoder.request.infer({encoder.input:np.ascontiguousarray(resized.transpose(2,0,1)[None])})
  layers=[np.asarray(result[out])[0] for out in encoder.outputs]
  early=next(layer for layer in layers if layer.shape==(16,640,256))
  mid=next(layer for layer in layers if layer.shape==(24,160,64))
  upsampled=cv2.resize(mid.transpose(1,2,0),(256,640),interpolation=cv2.INTER_LINEAR)
  x=np.concatenate([early.reshape(16,-1)[:,selected].T,upsampled.reshape(-1,24)[selected]],axis=1)
  xs.append(x);ys.append(target[selected].astype(np.float32))
 if (index+1)%40==0:print(f'Collectedneuraltrainview{index+1}/239',flush=True)
x=np.concatenate(xs);y=np.concatenate(ys)
np.savez_compressed(directory/'neural-train-samples.npz',x=x,y=y)
report={'protocol':protocol,'rows':len(x),'positive_rows':int(y.sum()),'seconds':time.perf_counter()-started}
(directory/'neural-sample-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='protocol'},indent=2))
