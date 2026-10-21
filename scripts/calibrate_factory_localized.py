"""One preregistered mask-aware correction to calibration, no new fitting."""
from pathlib import Path
import hashlib,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from edge_runtime.profiles.factory_defect import SurfaceDefectModel
from train_factory_surface import samples
from factory_mask_metrics import component_mask_statistics,at_area,aggregate
directory=ROOT/'output/optimization/factory-accuracy';directory.mkdir(exist_ok=True,parents=True)
thresholds=[.95,.975,.98,.99,.995];areas=[1,2,4,8]
protocol={'weights':'Original unchanged RF; no new fitting','thresholds':thresholds,'component_areas':areas,
          'maximum_calibration_normal_FPR':7/70,
          'selection':'Max localized defective-view recall; then original-resolution annotated-foreground pixel recall; then foreground pixel precision; then smaller component area; then higher threshold',
          'localization':'Retained threshold-mask pixels overlap original-resolution annotation after nearest projection',
          'candidate_cap':'Largest20components, exactly matching runtime',
          'comparison_used_for_parameters':False,'scope':'One corrected development objective after image-level evaluation was found insufficient; not blind external validation',
          'decision_authority':'MANUAL REVIEW ONLY; no automatic acceptance/rejection'}
(directory/'localized-protocol-before-selection.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
model=SurfaceDefectModel(ROOT/'models/factory/kolektor_surface_rf.joblib',backend='native_cascade');model.pixel_threshold=.95
cv2.setNumThreads(2);records=[];started=time.perf_counter()
for path,mask,label in samples(ROOT/'datasets/factory/KolektorSDD',model.metadata['split_items']['calibration']):
 scores,_,_=model.predict(cv2.imread(str(path)));choices=[]
 for threshold in thresholds:
  stats=component_mask_statistics(scores,threshold,mask)
  for area in areas:
   choices.append({'threshold':threshold,'area':area,'defective':label,**at_area(stats,area)})
 records.append({'path':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'mask_sha256':hashlib.sha256(path.with_name(path.stem+'_label.bmp').read_bytes()).hexdigest(),'choices':choices})
choices=[]
for threshold in thresholds:
 for area in areas:
  values=[next(row for row in record['choices'] if row['threshold']==threshold and row['area']==area) for record in records]
  choices.append({'threshold':threshold,'area':area,**aggregate(values)})
eligible=[row for row in choices if row['normal_view_false_positive_rate']<=7/70]
selected=max(eligible,key=lambda row:(row['localized_view_recall'],row['foreground_pixel_recall'],row['foreground_pixel_precision'],-row['area'],row['threshold'])) if eligible else None
if selected:
 artifact=ROOT/'models/factory/kolektor_surface_rf_mask_calibrated_review.joblib'
 shutil.copyfile(model.path,artifact)
 meta={**model.metadata,'model_name':'Original RF: mask-aware manual-review calibration',
       'profile_version':'kolektor_rf_mask_calibrated_manual_review','pixel_threshold':selected['threshold'],
       'minimum_component_pixels':selected['area'],'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False,
       'threshold_selection':protocol['selection'],'operating_point_protocol':protocol,'calibration_selected_point':selected}
 artifact.with_suffix('.joblib.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
report={'protocol':protocol,'selected':selected,'choices':choices,'calibration_records':records,'elapsed_seconds':time.perf_counter()-started}
(directory/'localized-calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'protocol','choices','calibration_records'}},indent=2))
