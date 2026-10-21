"""Validate selected DACL class cutoffs through serving infer on exact cached maps."""
import json
import cv2
import numpy as np
from evaluate_dacl_camera_calibration import OUT,CHANNELS,sparse_maps,sha,ROOT
from evaluate_construction_round3 import sampled
from edge_runtime.profiles.construction_dacl import DaclConstructionModel,decode_masks


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf8'))
    maps=np.load(OUT/'two-channel-probabilities.npz')['probabilities']
    model=object.__new__(DaclConstructionModel)
    model.selected=list(CHANNELS);model.threshold=.5;model.minimum_area=4;model.mask_rendering='boxes'
    model.class_thresholds={label:point['threshold'] for label,point in summary['selected'].items()}
    path=ROOT/summary['model']['model_path']
    if sha(path)!=summary['model']['sha256']:raise RuntimeError('Network map asset hash changed')
    signature=lambda rows:sorted((r['label'],tuple(r['bbox']),round(r['score'],7))for r in rows)
    count=0
    for row,two in zip(sampled('calibration'),maps):
        frame=cv2.imread(str(row['image']));probabilities=sparse_maps(two)
        model.predict_maps=lambda frame,p=probabilities:p
        observed=model.infer(frame);expected=[]
        for label in CHANNELS:
            expected.extend(decode_masks(probabilities,frame.shape[:2],[label],model.class_thresholds[label],4,'boxes'))
        if signature(observed)!=signature(expected):raise RuntimeError('Serving postprocessing differs from calibration')
        count+=1
    result={'passed':True,'frames':count,'model_sha256':summary['model']['sha256'],
            'class_confidence':model.class_thresholds,'protocol_sha256':summary['protocol_sha256'],
            'scope':'Actual DaclConstructionModel.infer postprocessing on identical cached network probability maps; no model re-inference or timing claim'}
    (OUT/'serving-cached-parity.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    summary['model']['class_confidence']=model.class_thresholds
    summary['serving_postprocessing_parity']=result
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
