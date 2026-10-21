"""Accuracy-only post-CV localization, preserving fold-local model identities."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from edge_runtime.profiles.factory_defect import SurfaceDefectModel
from train_factory_surface import samples
from factory_mask_metrics import component_mask_statistics,at_area,aggregate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cv',type=Path,default=ROOT/'output/optimization/factory-accuracy/item-crossvalidation-review')
a=p.parse_args();cv2.setNumThreads(2)
report=json.loads((a.cv/'report.json').read_text(encoding='utf-8'))
expected={(row['fold'],row['path']):row for row in report['records']}
records=[];manifest={}
for fold,split in enumerate(report['protocol']['outer_folds'][:report['completed_folds']],start=1):
 directory=a.cv/f'fold-{fold}'
 paths={'before':directory/'baseline.joblib','review_budget3':directory/'candidate-review-3.joblib','review_budget7':directory/'candidate-review-7.joblib'}
 models={name:SurfaceDefectModel(path,backend='native_cascade') if path.is_file() else None for name,path in paths.items()}
 manifest[str(fold)]={name:{'weights_sha256':model.sha256,'metadata_sha256':model.metadata_sha256,
                            'threshold':model.pixel_threshold,'area':model.area_threshold} if model else None for name,model in models.items()}
 for path,mask,defective in samples(ROOT/'datasets/factory/KolektorSDD',split['test']):
  relative=str(path.relative_to(ROOT/'datasets/factory/KolektorSDD'));prior=expected[(fold,relative)]
  frame=cv2.imread(str(path));row={'fold':fold,'path':relative,'defective':defective}
  for name,model in models.items():
   if prior[name] not in {'fault','review','clear','no_candidate'}:
    row[name]={'defective':defective,'abstained':True,'flagged':False,'localized':False,
               'foreground_tp':0,'foreground_predicted':0,'foreground_target':int((mask>0).sum()),'box_mask_overlap':False}
    continue
   scores,candidates,_=model.predict(frame)
   values=at_area(component_mask_statistics(scores,model.pixel_threshold,mask),model.area_threshold)
   assert values['flagged']==(prior[name] in {'fault','review'}), 'Frozen CV status disagreement'
   box_overlap=any(mask[max(0,int(c['bbox'][1])):int(c['bbox'][1]+c['bbox'][3]),
                        max(0,int(c['bbox'][0])):int(c['bbox'][0]+c['bbox'][2])].any() for c in candidates)
   row[name]={'defective':defective,'abstained':False,'box_mask_overlap':bool(box_overlap),**values}
  records.append(row)
 print(f'Localized mask evaluation completed fold{fold}',flush=True)
results={}
for name in ['before','review_budget3','review_budget7']:
 rows=[row[name] for row in records]
 results[name]=aggregate(rows)
 positives=[row[name] for row in records if row['defective']]
 results[name]['permissive_display_box_overlap_recall']=sum(row['box_mask_overlap'] for row in positives)/len(positives)
output={'scope':'Accuracy-only second inference over frozen CV models; all supervised fits/calibration remain inner-fold only; no latency claims',
        'pixel_definition':'Retained threshold components (largest20cap), nearest projected to original annotation resolution; no background true negatives used in IoU',
        'completed_folds':report['completed_folds'],'results':results,'models':manifest,'records':records}
(a.cv/'mask-localization-report.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
