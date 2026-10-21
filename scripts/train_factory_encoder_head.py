"""Frozen ImageNet early features + supervised train-only native forest head."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier
from edge_runtime.profiles.factory_defect import SurfaceDefectModel
from edge_runtime.profiles.factory_encoder import FactoryEncoder,ENCODER_FEATURE_VERSION
from train_factory_surface import samples,matrix

THRESHOLDS=[.95,.975,.98,.99,.995]
AREAS=[1,2,4,8]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/factory/KolektorSDD')
    p.add_argument('--model',type=Path,default=ROOT/'models/factory/kolektor_mobilenet_rf_v6.joblib')
    p.add_argument('--encoder',type=Path,default=ROOT/'models/factory/mobilenetv3_early_512.onnx')
    p.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-accuracy/training-neural-v6.json')
    p.add_argument('--probe',action='store_true')
    a=p.parse_args();cv2.setNumThreads(2);rng=np.random.default_rng(20261024);started=time.perf_counter()
    meta=json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
    encoder=FactoryEncoder(a.encoder,hashlib.sha256(a.encoder.read_bytes()).hexdigest())
    if a.probe:
        path=samples(a.dataset,meta['split_items']['train'])[0][0];frame=cv2.imread(str(path))
        timing=[]
        for i in range(12):
            tick=time.perf_counter();features,shape=encoder.extract(frame);timing.append((time.perf_counter()-tick)*1000)
        report={'scope':'Full frozen GPU encoder+11handcrafted features+fusion, predecoded train image; head/quality/runtime excluded',
                'times_ms':timing,'warm_p50':float(np.median(timing[2:])),'warm_p95':float(np.percentile(timing[2:],95)),
                'columns':features.shape[1],'grid':shape,'devices':encoder.execution_devices,
                'last_stage_timing_ms':encoder.last_timing_ms}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2));return
    protocol={'seed':20261024,'feature_version':ENCODER_FEATURE_VERSION,'frozen_encoder_layers':[0,3],
              'encoder_sha256':encoder.sha256,'input_bgr_shape':[1280,512],'head_grid':[640,256],'feature_columns':51,
              'trees':16,'max_depth':14,'min_samples_leaf':2,'class_weight':None,
              'train_normal_per_view':2000,'train_positive_cap':2000,'positive_repeat':4,
              'positive_view_train_brightness_augmentation':[.9,1.1],'augmentation_normal_pixels':1000,
              'threshold_grid':THRESHOLDS,'component_grid':AREAS,'split_items':meta['split_items'],
              'selection':'Calibration FPR<=3/70; maximize recall then lower vote threshold, smaller area',
              'scope':'Train-only supervised head, ImageNet frozen encoder, no comparison labels or pixels used in fitting/selection'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_name('neural-protocol-before-fit.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    xs,ys=[],[]
    for index,(path,mask,defective) in enumerate(samples(a.dataset,meta['split_items']['train'])):
        frame=cv2.imread(str(path))
        factors=[1.,.9,1.1] if defective else [1.]
        target=cv2.resize(mask.astype(np.float32),(256,640),interpolation=cv2.INTER_AREA).ravel()>0
        pos,neg=np.flatnonzero(target),np.flatnonzero(~target)
        for factor in factors:
            view=frame if factor==1. else np.clip(frame.astype(np.float32)*factor,0,255).astype(np.uint8)
            features,_=encoder.extract(view)
            normal=rng.choice(neg,min(2000 if factor==1. else 1000,len(neg)),replace=False)
            positive=rng.choice(pos,min(2000,len(pos)),replace=False) if len(pos) else pos
            selected=np.concatenate([normal,np.tile(positive,4)])
            xs.append(features[selected]);ys.append(target[selected].astype(np.int32))
        if (index+1)%40==0:print(f'Cached sampledfeatures{index+1}/239trainviews',flush=True)
    x,y=np.concatenate(xs),np.concatenate(ys)
    del xs,ys
    forest=RandomForestClassifier(n_estimators=16,max_depth=14,min_samples_leaf=2,n_jobs=2,random_state=20261024)
    forest.fit(x,y);joblib.dump(forest,a.model,compress=3)
    metadata={**meta,'model_name':'Frozen MobileNetV3 early texture representation + supervised native RF head',
              'feature_version':ENCODER_FEATURE_VERSION,'profile_version':'kolektor_mobilenet_rf_v6',
              'input_width':256,'pixel_threshold':.95,'minimum_component_pixels':1,'trees':16,'maximum_tree_depth':14,
              'encoder_file':a.encoder.name,'encoder_sha256':encoder.sha256,'training_protocol':protocol,
              'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'threshold_selection':protocol['selection']}
    metapath=a.model.with_suffix('.joblib.json');metapath.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    candidate=SurfaceDefectModel(a.model,backend='native_cascade');records=[]
    for path,_,label in samples(a.dataset,meta['split_items']['calibration']):
        scores,_,_=candidate.predict(cv2.imread(str(path)));maximum={}
        for threshold in THRESHOLDS:
            _,_,stats,_=cv2.connectedComponentsWithStats((scores>=threshold).astype(np.uint8),connectivity=8)
            maximum[str(threshold)]=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
        records.append({'path':str(path.relative_to(a.dataset)),'defective':label,'maximum_areas':maximum})
    choices=[]
    for threshold in THRESHOLDS:
        for area in AREAS:
            result=matrix([r['defective'] for r in records],[r['maximum_areas'][str(threshold)]>=area for r in records])
            choices.append({'threshold':threshold,'area':area,**result})
    eligible=[c for c in choices if c['fp']/70<=3/70]
    point=max(eligible,key=lambda c:(c['recall'],-c['threshold'],-c['area'])) if eligible else None
    if point is not None:
        metadata.update(pixel_threshold=point['threshold'],minimum_component_pixels=point['area'])
        metapath.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report={'protocol':protocol,'selected':point,'choices':choices,'calibration_records':records,
            'training_rows':len(x),'positive_rows':int(y.sum()),'model_sha256':metadata['model_sha256'],
            'metadata_sha256':hashlib.sha256(metapath.read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-started}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'protocol','choices','calibration_records'}},indent=2))
    if point is None:raise RuntimeError('No neural calibration point meets the declared false-positive budget')

if __name__=='__main__':main()
