"""Accuracy-only frozen comparison with true masks and explicit abstentions."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from edge_runtime.profiles.factory import FactoryProfile
from edge_runtime.contracts import FrameContext
from train_factory_surface import samples,matrix
from factory_mask_metrics import component_mask_statistics,at_area,aggregate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--before',type=Path,default=ROOT/'models/factory/kolektor_surface_rf.joblib')
p.add_argument('--after',type=Path,default=ROOT/'models/factory/kolektor_surface_rf_mask_calibrated_review.joblib')
p.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-accuracy/localized-frozen-comparison.json')
a=p.parse_args();cv2.setNumThreads(2)
profiles={name:FactoryProfile({'defect_model_path':str(path),'defect_backend':'auto','confirmation_frames':1}) for name,path in [('before',a.before),('after',a.after)]}
captures={}
def wrap(model):
 capture={};original=model.predict
 def predict(frame):
  result=original(frame);capture['scores']=result[0];return result
 model.predict=predict
 return capture
for name,profile in profiles.items():captures[name]=wrap(profile._defect_model)
records=[]
for index,(path,mask,defective) in enumerate(samples(ROOT/'datasets/factory/KolektorSDD',profiles['before'].describe()['model_metadata']['split_items']['test'])):
 frame=cv2.imread(str(path));record={'path':str(path.relative_to(ROOT/'datasets/factory/KolektorSDD')),'defective':defective,
                                    'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 for name,profile in profiles.items():
  captures[name].clear();result=profile.process(frame,FrameContext(index,time.perf_counter(),unit_id=str(path)))
  abstained=result.status not in {'fault','review','clear','no_candidate'}
  if abstained:
   values={'abstained':True,'flagged':False,'localized':False,'foreground_tp':0,'foreground_predicted':0,'foreground_target':int((mask>0).sum())}
  else:
   model=profile._defect_model
   growth=model.metadata.get('hysteresis_growth_threshold')
   if growth is None:
    values={'abstained':False,**at_area(component_mask_statistics(captures[name]['scores'],model.pixel_threshold,mask),model.area_threshold)}
   else:
    from factory_localization_audit import retained_mask
    binary=retained_mask(captures[name]['scores'],model.pixel_threshold,model.area_threshold,mask.shape,growth)
    intersection=int((binary&(mask>0)).sum());predicted=int(binary.sum());target=int((mask>0).sum())
    values={'abstained':False,'flagged':bool(predicted),'localized':intersection>0,
            'foreground_tp':intersection,'foreground_predicted':predicted,'foreground_target':target,
            'foreground_pixel_recall':intersection/target if target else None,
            'foreground_pixel_precision':intersection/predicted if predicted else 0.,
            'pixel_iou':intersection/(target+predicted-intersection) if target+predicted else None}
   assert values['flagged']==(result.status in {'fault','review'})
  record[name]={'defective':defective,'status':result.status,**values,
                'findings':[{'bbox':finding.bbox,'score':finding.score} for finding in result.findings]}
 records.append(record)
results={name:aggregate([row[name] for row in records]) for name in profiles}
for name in profiles:
 accepted=[r for r in records if not r[name]['abstained']]
 results[name]['flagged_image_metrics']=matrix([r['defective'] for r in accepted],[r[name]['flagged'] for r in accepted])
output={'scope':'Frozen reused80-image development comparison; accuracy only, no latency claim; no parameters selected using comparison',
        'localization':'Retained predicted component pixels (largest20cap), nearest projected to original mask resolution',
        'models':{name:profile.describe() for name,profile in profiles.items()},'results':results,'records':records}
a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
