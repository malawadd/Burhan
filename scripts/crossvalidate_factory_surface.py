"""Fixed-protocol item-disjoint CV; exclude the exposed original comparison.

Each outer item is excluded from BOTH the mining forest and final forest.
This is development cross-validation, not a new external blind test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from edge_runtime.profiles.factory_defect import SurfaceDefectModel, extract_features_fast
from edge_runtime.profiles.factory import FactoryProfile
from edge_runtime.contracts import FrameContext
from train_factory_surface import samples, matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/factory/KolektorSDD')
    p.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-accuracy/item-crossvalidation')
    p.add_argument('--width',type=int,choices=[256,320],default=320)
    p.add_argument('--candidate',choices=['dense','regions','review','review-both'],default='review-both')
    a=p.parse_args()
    cv2.setNumThreads(2)
    original=json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
    names=sorted(original['split_items']['train']+original['split_items']['calibration'])
    rng=np.random.default_rng(20261022)
    shuffled=[names[i] for i in rng.permutation(len(names))]
    groups=[sorted(shuffled[i:i+10]) for i in range(0,40,10)]
    folds=[]
    for i in range(4):
        folds.append({'test':groups[i], 'calibration':groups[(i+1)%4],
                      'train':sorted(sum([groups[j] for j in range(4) if j not in {i,(i+1)%4}],[]))})
    a.output.mkdir(parents=True,exist_ok=True)
    protocol={'seed':20261022,'outer_folds':folds,'excluded_original_comparison_items':original['split_items']['test'],
              'candidate_width':a.width,'baseline':'Original32tree/depth12/leaf4/random2000normal+400positive×2, innercalF1 selection',
              'candidate':a.candidate,
              'candidate_scope':'Fixedcandidateprotocol; proposal/miningforest refit only on innertraining items',
              'scope':'Development item-disjoint cross-validation; lower training count20items/fold vs30final. Hyperparameters designed on thisdataset; not blindexternalvalidation.'}
    (a.output/'protocol-before-cv.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    all_records=[]; started=time.perf_counter()
    for fold,split in enumerate(folds):
        directory=a.output/f'fold-{fold+1}'; directory.mkdir(exist_ok=True)
        basepath=directory/'baseline.joblib'
        # Never use the original forest here: it saw some outer validation items.
        sampling=np.random.default_rng(20261020)
        xs,ys=[],[]
        for path,mask,_ in samples(a.dataset,split['train']):
            features,shape=extract_features_fast(cv2.imread(str(path)),256)
            target=cv2.resize(mask,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST).ravel()>0
            pos,neg=np.flatnonzero(target),np.flatnonzero(~target)
            positive=sampling.choice(pos,min(400,len(pos)),replace=False) if len(pos) else pos
            negative=sampling.choice(neg,min(2000,len(neg)),replace=False)
            chosen=np.concatenate([negative,np.tile(positive,2)])
            xs.append(features[chosen]);ys.append(target[chosen].astype(np.int32))
        forest=RandomForestClassifier(n_estimators=32,max_depth=12,min_samples_leaf=4,n_jobs=2,random_state=20261020)
        forest.fit(np.concatenate(xs),np.concatenate(ys));joblib.dump(forest,basepath,compress=3)
        meta={**original,'model_sha256':sha(basepath),'split_items':split,'pixel_threshold':.5,'minimum_component_pixels':4,
              'profile_version':f'v1-refit-cv-{fold+1}','trained_only_on_inner_training_items':True,
              'trained_utc':datetime.now(timezone.utc).isoformat(),'split_seed':20261022,'training_seed':20261020}
        metapath=basepath.with_suffix('.joblib.json')
        metapath.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        model=SurfaceDefectModel(basepath,backend='native_cascade')
        labels=[];maximum={str(t):[] for t in [.5,.7,.8,.9,.95,.975,.98,.99,.995]}
        for path,_,label in samples(a.dataset,split['calibration']):
            labels.append(label);scores,_,_=model.predict(cv2.imread(str(path)))
            for t in maximum:
                _,_,stats,_=cv2.connectedComponentsWithStats((scores>=float(t)).astype(np.uint8),connectivity=8)
                maximum[t].append(int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0)
        choices=[]
        for t,values in maximum.items():
            if float(t)==.975:continue  # Preserve original baseline threshold grid.
            for area in [4,8,16,32,64,128,256,512]:
                result=matrix(labels,[v>=area for v in values]);choices.append((result['f1'],result['precision'],result['recall'],float(t),area,result))
        _,_,_,threshold,area,calibration=max(choices)
        meta.update(pixel_threshold=threshold,minimum_component_pixels=area,cv_calibration=calibration)
        metapath.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        newpath=directory/f'candidate-{a.candidate}.joblib'
        candidate_paths={}
        if a.candidate in {'review','review-both'}:
            from calibrate_factory_recall import choose as choose_review
            import shutil
            calrows=samples(a.dataset,split['calibration'])
            calibration_records=[{'path':str(row[0]),'label':row[2],
                                  'areas':{t:maximum[t][index] for t in maximum}} for index,row in enumerate(calrows)]
            for budget in ([3,7] if a.candidate=='review-both' else [3]):
                point,review_choices=choose_review(calibration_records,budget)
                newpath=directory/f'candidate-review-{budget}.joblib'
                if point is not None:
                    shutil.copyfile(basepath,newpath)
                    updated={**meta,'pixel_threshold':point['threshold'],'minimum_component_pixels':point['area'],
                             'profile_version':f'v2-review-budget{budget}-refit-cv-{fold+1}',
                             'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False,
                             'review_false_positive_budget':budget/70,'cv_calibration':point,
                             'threshold_selection':f'Inner calibration FPR<={budget}/70; maximize recall then lower threshold and smaller component area'}
                    newpath.with_suffix('.joblib.json').write_text(json.dumps(updated,indent=2),encoding='utf-8')
                (directory/f'candidate-calibration-{budget}.json').write_text(json.dumps({'selected':point,'choices':review_choices},indent=2),encoding='utf-8')
                candidate_paths[f'review_budget{budget}']=newpath if point is not None else None
        else:
            script='train_factory_regions.py' if a.candidate=='regions' else 'improve_factory_surface_v4.py'
            command=[sys.executable,'-X','utf8',str(ROOT/'scripts'/script),
                     '--dataset',str(a.dataset),'--baseline',str(basepath),'--model',str(newpath),
                     '--output',str(directory/'candidate-training.json')]
            if a.candidate=='dense':command.extend(['--width',str(a.width)])
            trained=subprocess.run(command,check=False)
            accepted_candidate=trained.returncode==0
            candidate_paths['after']=newpath if accepted_candidate else None
        profiles={name:FactoryProfile({'defect_model_path':str(path),'defect_backend':'native_cascade','confirmation_frames':1})
                  for name,path in [('before',basepath)]}
        for name,path in candidate_paths.items():
            profiles[name]=(FactoryProfile({'defect_model_path':str(path),'defect_backend':'native_cascade','confirmation_frames':1})
                            if path is not None else None)
        records=[]
        for n,(path,_,label) in enumerate(samples(a.dataset,split['test'])):
            frame=cv2.imread(str(path));record={'path':str(path.relative_to(a.dataset)),'defective':label,'fold':fold+1}
            for name,profile in profiles.items():
                if profile is None:
                    record[name]='inconclusive'
                    record[f'{name}_unavailable_reason']='Calibration/training protocol did not produce an accepted operating point'
                    continue
                result=profile.process(frame,FrameContext(n,time.perf_counter(),unit_id=str(path)))
                record[name]=result.status
            records.append(record)
        all_records.extend(records)
        (directory/'heldout-predictions.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        result={name:matrix([r['defective'] for r in records],[r[name] in {'fault','review'} for r in records]) for name in profiles}
        print(f'Completed outerfold{fold+1}: {json.dumps(result)}',flush=True)
        results={}
        for name in profiles:
            accepted=[r for r in all_records if r[name] in {'fault','review','clear','no_candidate'}]
            results[name]={'accepted_image_metrics':matrix([r['defective'] for r in accepted],[r[name] in {'fault','review'} for r in accepted]),
                           'defect_detection_recall_all_views':sum(r['defective'] and r[name] in {'fault','review'} for r in all_records)/max(1,sum(r['defective'] for r in all_records)),
                           'abstained_defective':sum(r['defective'] and r[name] not in {'fault','review','clear','no_candidate'} for r in all_records),
                           'abstained_normal':sum(not r['defective'] and r[name] not in {'fault','review','clear','no_candidate'} for r in all_records)}
        report={'protocol':protocol,'completed_folds':fold+1,'elapsed_seconds':time.perf_counter()-started,
                'results':results,'records':all_records}
        (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
