"""Mine false locations from original TRAINING views only, never calibration/test."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory_encoder import FactorySegmentation
from train_factory_surface import samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=ROOT/'models/factory/spatial_unet_v9-step600.onnx')
    parser.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-round2/spatial-train-hardneg.json')
    args=parser.parse_args();cv2.setNumThreads(2)
    original=json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
    protocol={'scope':'TRAIN-only false-location mining; no calibration/test pixels or labels opened',
              'training_items':original['split_items']['train'],'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'maximum_locations_per_view':16,'exclusion_radius_in_score_grid':16,
              'ground_truth_exclusion':'Area-resize mask to score grid, any real coverage positive, then dilate9x9; no near-defect pixels mined',
              'coordinates':'x,y in full512x1280 model input, not original image or physical units'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_name('spatial-hardneg-protocol-before-mining.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    started=time.perf_counter();model=FactorySegmentation(args.model,output_logits=True);records=[]
    for image,mask,defective in samples(ROOT/'datasets/factory/KolektorSDD',original['split_items']['train']):
        scores,_,_=model.predict(cv2.imread(str(image)),.999,32)
        positive=cv2.resize((mask>0).astype(np.float32),(256,640),interpolation=cv2.INTER_AREA)>0
        excluded=cv2.dilate(positive.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
        score=scores.copy();score[excluded]=-1.;locations=[]
        for _ in range(16):
            y,x=np.unravel_index(np.argmax(score),score.shape)
            value=float(score[y,x])
            if value<.1:break
            locations.append({'x':int(x*2+1),'y':int(y*2+1),'score':value})
            score[max(0,y-16):y+17,max(0,x-16):x+17]=-1.
        records.append({'image':str(image.relative_to(ROOT/'datasets/factory/KolektorSDD')),
                        'defective':bool(defective),'locations':locations})
    report={'protocol':protocol,'records':records,'elapsed_seconds':time.perf_counter()-started,
            'views':len(records),'hard_locations':sum(len(r['locations']) for r in records)}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'protocol','records'}}))


if __name__=='__main__':main()
