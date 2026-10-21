"""Re-render cached model outputs and separate publisher truth without running inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from evaluate_construction_defect_video import annotate
from prepare_evcivil import encode_video
ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',default='baseline');args=parser.parse_args()
    for name in ['crack_10','crack_89','spalling_35']:
        target=ROOT/f'output/construction-{name}-{args.tag}'
        report=json.loads(target.with_suffix('.json').read_text(encoding='utf8'))
        source=ROOT/'datasets/construction/evcivil'/name
        frames=np.load(source/'frames.npy',allow_pickle=False);times=np.load(source/'timestamps-us.npy',allow_pickle=False)
        annotated=np.asarray([annotate(frame,record['predictions'],name,index,record)
                              for index,(frame,record) in enumerate(zip(frames,report['frames']))])
        report['matching_iou_threshold']=.5
        report['overlay_legend']='Solid green: model IoU match; solid amber: unmatched model box. Dashed red/cyan: separate publisher missed/matched crack. Dashed gray: unsupported publisher damage. Partial crack boxes can fail full-instance IoU.'
        report['derived_video']=encode_video(target.with_suffix('.mp4'),annotated,times,max(1,round(1e6/np.median(np.diff(times)))))
        target.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        cv2.imwrite(str(target.with_name(target.name+'-frame').with_suffix('.jpg')),annotated[len(annotated)//2])
        matched=next((i for i,r in enumerate(report['frames']) if r['localized_matches']),None)
        if matched is not None:cv2.imwrite(str(target.with_name(target.name+'-localized-match').with_suffix('.jpg')),annotated[matched])
        print(name,report['derived_video'],flush=True)
