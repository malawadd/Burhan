"""Explicit high-recall review operating point, selected only on calibration.

Weights stay unchanged. This spends a declared false-positive budget rather
than presenting threshold selection as a new learned classifier.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_factory_surface import matrix

THRESHOLDS=[.95,.975,.98,.99,.995]
AREAS=[1,2,4,8]

def choose(records,budget=3):
    normal=sum(not r['label'] for r in records)
    choices=[]
    for threshold in THRESHOLDS:
        for area in AREAS:
            metrics=matrix([r['label'] for r in records],[r['areas'][str(threshold)]>=area for r in records])
            choices.append({'threshold':threshold,'area':area,**metrics})
    eligible=[c for c in choices if c['fp']/max(1,normal)<=budget/70]
    if not eligible:return None,choices
    selected=max(eligible,key=lambda c:(c['recall'],-c['threshold'],-c['area']))
    return selected,choices

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=ROOT/'models/factory/kolektor_surface_rf_v2_recall_review.joblib')
    p.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-accuracy/high-recall-calibration.json')
    p.add_argument('--budget',type=int,choices=[3,7],default=3,help='Allowed normal-view false positives /70calibrationnormalviews; manualreviewonly')
    p.add_argument('--cache',type=Path,default=ROOT/'output/optimization/factory-accuracy/calibration-sensitivity.json')
    a=p.parse_args()
    original=ROOT/'models/factory/kolektor_surface_rf.joblib'
    metadata=json.loads(original.with_suffix('.joblib.json').read_text(encoding='utf-8'))
    protocol={'purpose':'High recall, human-review mode; weights unchanged',
              'thresholds':THRESHOLDS,'areas':AREAS,'maximum_calibration_false_positive_rate':a.budget/70,
              'selection':f'Maximize calibration recall; among eligible points prefer lower vote threshold then smaller component area, within <={a.budget}/70 normal-view false positives',
              'comparison_labels_used':False,'production_qualified':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_name(f'high-recall-protocol-{a.budget}-before-selection.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=json.loads(a.cache.read_text(encoding='utf-8'))['records']
    selected,choices=choose(records,a.budget)
    if selected is None:
        a.output.write_text(json.dumps({'protocol':protocol,'accepted':False,'choices':choices},indent=2),encoding='utf-8')
        raise RuntimeError('No review operating point meets the calibration false-positive budget')
    jackknife=[]
    for item in metadata['split_items']['calibration']:
        train=[r for r in records if Path(r['path']).parent.name!=item]
        held=[r for r in records if Path(r['path']).parent.name==item]
        point,_=choose(train,a.budget)
        if point is None:
            jackknife.extend({'path':r['path'],'label':r['label'],'prediction':None,
                              'selected_without_item':None} for r in held)
            continue
        jackknife.extend({'path':r['path'],'label':r['label'],'prediction':r['areas'][str(point['threshold'])]>=point['area'],
                          'selected_without_item':point} for r in held)
    shutil.copyfile(original,a.model)
    metadata.update(pixel_threshold=selected['threshold'],minimum_component_pixels=selected['area'],
                    profile_version=f'kolektor_rf_v2_manual_review_budget{a.budget}',
                    manual_review_only=True,automatic_acceptance=False,automatic_rejection=False,
                    review_false_positive_budget=a.budget/70,
                    operating_point_protocol=protocol,calibration_selected_point=selected,
                    model_name='Original Kolektor forest: explicit high-recall review operating point',
                    model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest())
    a.model.with_suffix('.joblib.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report={'protocol':protocol,'selected':selected,'choices':choices,
            'calibration_item_jackknife':matrix([r['label'] for r in jackknife],[r['prediction'] for r in jackknife]),
            'jackknife_abstained':sum(r['prediction'] is None for r in jackknife),
            'jackknife_records':jackknife,'model_sha256':metadata['model_sha256'],
            'metadata_sha256':hashlib.sha256(a.model.with_suffix('.joblib.json').read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'choices','jackknife_records'}},indent=2))

if __name__=='__main__':main()
