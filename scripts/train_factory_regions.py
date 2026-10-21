"""Fixed high-sensitivity proposals + train-only learned region rejection."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2,joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier
from edge_runtime.profiles.factory_defect import SurfaceDefectModel,extract_features_fast
from edge_runtime.profiles.factory_regions import region_features,REGION_FEATURE_VERSION
from train_factory_surface import samples,matrix

THRESHOLDS=[.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995]

def choose(records):
    normal=sum(not r['defective'] for r in records)
    choices=[]
    for threshold in THRESHOLDS:
        metrics=matrix([r['defective'] for r in records],[r['max_region_score']>=threshold for r in records])
        choices.append({'threshold':threshold,**metrics})
    eligible=[r for r in choices if r['fp']/max(1,normal)<=3/70]
    if not eligible:return None,choices
    return max(eligible,key=lambda r:(r['recall'],-r['threshold'])),choices

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'models/factory/kolektor_surface_rf.joblib')
    p.add_argument('--model',type=Path,default=ROOT/'models/factory/kolektor_surface_rf_v5_regions.joblib')
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/factory/KolektorSDD')
    p.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-accuracy/training-regions-v5.json')
    a=p.parse_args();cv2.setNumThreads(2);started=time.perf_counter()
    base=SurfaceDefectModel(a.baseline,backend='native_cascade');split=base.metadata['split_items']
    protocol={'proposal_trees':'First two original train-only forest trees, fixed order','proposal_vote_threshold':.25,
              'proposal_min_area':1,'region_features':REGION_FEATURE_VERSION,'region_feature_count':19,
              'region_classifier':{'trees':32,'max_depth':10,'min_samples_leaf':2,'class_weight':'balanced_subsample','seed':20261023},
              'region_label':'A proposed connected component overlaps at least one annotated defect grid pixel; source-coverage mask resize',
              'training_items':split['train'],'calibration_items':split['calibration'],'excluded_comparison_items':split['test'],
              'region_thresholds':THRESHOLDS,'selection':'Calibration FPR<=3/70, maximize recall then lower region threshold',
              'purpose':'Human-review observations, not automatic line acceptance; newly learned geometry filter'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_name('region-protocol-before-fit.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    proposal=copy.deepcopy(base.model);proposal.estimators_=proposal.estimators_[:2];proposal.n_estimators=2
    from edge_runtime.profiles.factory_native import NativeCascade
    native=NativeCascade(proposal,base._positive)
    xs,ys=[],[];train_coverage=[]
    def proposals(frame):
        features,shape=extract_features_fast(frame,base.width)
        scores,_=native.score(features,.25);scores=scores.reshape(shape)
        desc,labels,stats=region_features(frame,scores,(scores>=.25).astype(np.uint8))
        return scores,desc,labels
    for index,(path,mask,label) in enumerate(samples(a.dataset,split['train'])):
        frame=cv2.imread(str(path));scores,desc,regions=proposals(frame)
        target=cv2.resize(mask.astype(np.float32),(scores.shape[1],scores.shape[0]),interpolation=cv2.INTER_AREA)>0
        overlap=np.bincount(regions.ravel(),weights=target.ravel().astype(float),minlength=len(desc)+1)[1:]>0
        if len(desc):xs.append(desc);ys.append(overlap.astype(np.int32))
        if label:train_coverage.append(bool(overlap.any()))
        if (index+1)%60==0:print(f'Constructed{index+1} training-region views',flush=True)
    x,y=np.concatenate(xs),np.concatenate(ys)
    classifier=RandomForestClassifier(n_estimators=32,max_depth=10,min_samples_leaf=2,
                                     class_weight='balanced_subsample',n_jobs=2,random_state=20261023)
    classifier.fit(x,y);classifier.n_jobs=1
    calibration=[]
    for path,_,label in samples(a.dataset,split['calibration']):
        _,desc,_=proposals(cv2.imread(str(path)))
        scores=classifier.predict_proba(desc)[:,1] if len(desc) else np.empty(0)
        calibration.append({'path':str(path.relative_to(a.dataset)),'defective':label,
                            'max_region_score':float(scores.max()) if len(scores) else -1.,'proposals':len(desc)})
    point,choices=choose(calibration)
    report={'protocol':protocol,'selected':point,'choices':choices,'training_regions':len(x),'positive_regions':int(y.sum()),
            'train_defect_proposal_coverage':sum(train_coverage)/max(1,len(train_coverage)),
            'calibration_records':calibration,'elapsed_seconds':time.perf_counter()-started}
    if point is not None:
        a.model.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({'kind':REGION_FEATURE_VERSION,'proposal':proposal,'region_classifier':classifier},a.model,compress=3)
        metadata={**base.metadata,'feature_version':REGION_FEATURE_VERSION,'profile_version':'kolektor_rf_v5_regions',
                  'model_name':'Kolektor two-tree proposals + learned real-pixel region classifier',
                  'pixel_threshold':.25,'minimum_component_pixels':1,'region_threshold':point['threshold'],
                  'trees':2,'region_trees':32,'training_protocol':protocol,
                  'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  'score_meaning':'Mean region-forest leaf defect fraction; not calibrated probability',
                  'threshold_selection':protocol['selection']}
        a.model.with_suffix('.joblib.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        report['model_sha256']=metadata['model_sha256']
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'protocol','choices','calibration_records'}},indent=2))
    if point is None:raise RuntimeError('No calibration point met the declared false-positive budget')

if __name__=='__main__':main()
