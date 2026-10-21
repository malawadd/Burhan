"""Render fixed calibration cases from cached semantic maps; no neural inference."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from evaluate_litecrackseg_calibration import OUT
from evaluate_construction_round3 import sampled
from edge_runtime.profiles.construction_lite import camera_exposure,decode_crack_map


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-light-gain',action='store_true')
    args=parser.parse_args()
    if args.low_light_gain:
        OUT=OUT/'low-light-gain'
    stem = 'litecrackseg-tut-384'
    summary = json.loads((OUT/f'{stem}.summary.json').read_text(encoding='utf8'))
    records = json.loads((OUT/f'{stem}.records.json').read_text(encoding='utf8'))
    maps = np.load(OUT/f'{stem}-probabilities.npz')['probabilities']
    threshold = summary['selected']['threshold']
    rows = sampled('calibration')
    # Save exactly retained serving components, including min-area/top100,
    # rather than a raw threshold mask containing discarded tiny speckles.
    retained_masks={}
    for index,row in enumerate(rows):
        shape=cv2.imread(str(row['image'])).shape[:2]
        mask=np.zeros(shape,dtype=bool)
        for component in decode_crack_map(maps[index],shape,threshold,4,masks=True):
            mask|=component['mask']
        retained_masks[f'frame_{index}']=mask.astype(np.uint8)
    np.savez_compressed(OUT/f'{stem}-selected-native-masks.npz',**retained_masks)
    selectors = {
        'first-calibration-crack': lambda row: any(e['label']=='crack' for e in row['expected']),
        'first-calibration-spalling-unsupported': lambda row: any(e['label']=='spalling' for e in row['expected']),
        'first-night-calibration': lambda row: 'night_nov18' in str(row['image']),
        'first-geometrically-thin-calibration': lambda row: any(e['label']=='crack' and min(e['bbox'][2:])<=30 and max(e['bbox'][2:])>=2*min(e['bbox'][2:]) for e in row['expected']),
        'first-verified-negative': lambda row: row['negative'],
    }
    for name, selector in selectors.items():
        index = next(i for i,row in enumerate(rows) if selector(row))
        row, record = rows[index], records[index]
        raw = cv2.imread(str(row['image']))
        height,width = raw.shape[:2]
        gt,pred = raw.copy(),raw.copy()
        for e in row['expected']:
            x,y,w,h = map(round,e['bbox'])
            cv2.rectangle(gt,(x,y),(x+w,y+h),(255,255,0),1)
        native = cv2.resize(maps[index],(width,height),interpolation=cv2.INTER_LINEAR)
        mask=retained_masks[f'frame_{index}'].astype(bool)
        pred[mask]=(pred[mask]*.25+np.array([0,0,255])*.75).astype(np.uint8)
        for p in record['predictions']:
            x,y,w,h=map(round,p['bbox'])
            cv2.rectangle(pred,(x,y),(x+w,y+h),(0,200,0) if p.get('matched_annotation_id') else (0,180,255),1)
        corrected,gain=camera_exposure(raw,'low_light_gain' if args.low_light_gain else 'off')
        panel=np.hstack([raw,corrected,gt,pred]) if args.low_light_gain else np.hstack([raw,gt,pred])
        panel=cv2.copyMakeBorder(panel,90,50,0,0,cv2.BORDER_CONSTANT,value=(23,23,23))
        cv2.putText(panel,name,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(panel,f'Actual frame {row["frame_index"]}; cutoff>{threshold}; global gain{gain:.2f}x; no pixel ground truth',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
        captions=['RAW CAMERA','ACTUAL MODEL INPUT GAIN','PUBLISHER BOXES (cyan)','MODEL MASK (red pixels)'] if args.low_light_gain else ['RAW CAMERA','PUBLISHER BOXES (cyan)','MODEL MASK (red pixels)']
        for j,caption in enumerate(captions):
            cv2.putText(panel,caption,(j*width+5,73),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        counts=record['by_class']['crack']
        text=f'Component BOX IoU.50 proxy: TP{counts["tp"]} / FN{counts["fn"]} / unmatched{counts["fp"]}' if record['evaluated'] else 'UNSCORED: spalling-only annotations do not verify crack presence/absence'
        cv2.putText(panel,text,(8,panel.shape[0]-30),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
        cv2.putText(panel,'Green model box=matched; amber=unmatched. Crack mask is a prediction, never publisher mask GT.',(8,panel.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,230,230),1)
        cv2.imwrite(str(OUT/f'{name}.jpg'),panel)
    print('Five fixed cached calibration panels rendered; no inference')


if __name__=='__main__':main()
