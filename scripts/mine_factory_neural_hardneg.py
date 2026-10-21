"""Mine false-positive TRAIN pixels only; no calibration/comparison fitting."""
from pathlib import Path
import sys,json,hashlib,time,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory_encoder import FactoryEncoder,FactorySegmentation
from train_factory_surface import samples
cv2.setNumThreads(2)
directory=ROOT/'output/optimization/factory-accuracy'
original=json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
protocol=json.loads((directory/'neural-mlp-protocol-before-fit.json').read_text(encoding='utf-8'))
protocol.update(head='40->32ReLU->16ReLU->1logit,standardizedtrainfeatures,AdamW0.001,22epochs,batch8192,BCEpositiveweight2',
                hard_negative_mining='Top1000normalpixels per ORIGINALTRAINview usingfrozenv7scores; repeat4, no calibration/testpixels fitted',
                score_storage='GPU logits; CPU float32 sigmoid avoids FP16 sigmoid output quantization',
                feature_standard_deviation_floor=.05,
                threshold_grid=[.5,.7,.8,.9,.95,.98,.99,.995,.999,.9995,.9999],
                model_version='v8-hard-negative-neural')
protocol['additional_precalibration_logit_grid']=[0,1,2,3,4,5,6,7,8,10,12]
protocol['threshold_grid']=sorted(set(protocol['threshold_grid']+[1/(1+math.exp(-v)) for v in protocol['additional_precalibration_logit_grid']]))
(directory/'neural-v8-protocol-before-fit.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
path=ROOT/'models/factory/mobilenetv3_early_512.onnx'
encoder=FactoryEncoder(path,hashlib.sha256(path.read_bytes()).hexdigest())
miner=FactorySegmentation(ROOT/'models/factory/kolektor_mobilenet_seg_v7.onnx')
xs=[];started=time.perf_counter()
for index,(image,mask,_) in enumerate(samples(ROOT/'datasets/factory/KolektorSDD',original['split_items']['train'])):
 frame=cv2.imread(str(image));target=cv2.resize(mask.astype(np.float32),(256,640),interpolation=cv2.INTER_AREA).ravel()>0
 scores,_,_=miner.predict(frame,.995,1);normal=np.flatnonzero(~target)
 chosen=normal[np.argpartition(scores.ravel()[normal],len(normal)-1000)[-1000:]]
 resized=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA)
 result=encoder.request.infer({encoder.input:np.ascontiguousarray(resized.transpose(2,0,1)[None])})
 layers=[np.asarray(result[out])[0] for out in encoder.outputs]
 early=next(layer for layer in layers if layer.shape==(16,640,256));mid=next(layer for layer in layers if layer.shape==(24,160,64))
 upsampled=cv2.resize(mid.transpose(1,2,0),(256,640),interpolation=cv2.INTER_LINEAR)
 xs.append(np.concatenate([early.reshape(16,-1)[:,chosen].T,upsampled.reshape(-1,24)[chosen]],axis=1))
 if (index+1)%60==0:print(f'Minedhardnormalview{index+1}/239',flush=True)
hard=np.concatenate(xs);data=np.load(directory/'neural-train-samples.npz')
x=np.concatenate([data['x'],np.tile(hard,(4,1))]);y=np.concatenate([data['y'],np.zeros(len(hard)*4,dtype=np.float32)])
np.savez_compressed(directory/'neural-hard-train-samples.npz',x=x,y=y)
(directory/'neural-v8-mining-report.json').write_text(json.dumps({'hard_rows':len(hard),'total_rows':len(x),'positive_rows':int(y.sum()),'seconds':time.perf_counter()-started},indent=2),encoding='utf-8')
