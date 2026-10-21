"""Offline diagnostic: do saved candidate boxes touch real annotated defects?

This is a permissive display-box check, not pixel IoU or segmentation quality.
No model inference, tuning or synthetic image generation occurs here.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',type=Path,default=ROOT/'output/optimization/factory-accuracy/before-after-review-first.json')
p.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-accuracy/review-box-overlap.json')
a=p.parse_args();report=json.loads(a.report.read_text(encoding='utf-8'));records=[]
for r in report['records']:
 if r['repeat']!=0:continue
 image=ROOT/'datasets/factory/KolektorSDD'/r['path'];mask=cv2.imread(str(image.with_name(image.stem+'_label.bmp')),cv2.IMREAD_GRAYSCALE)>0
 assert bool(mask.any())==r['defective']
 record={'path':r['path'],'defective':r['defective']}
 for name in ['before','after']:
  covered=np.zeros(mask.shape,dtype=np.uint8)
  for box in [f['bbox'] for f in r[name]['findings']]:
   x,y,w,h=map(int,box);covered[max(0,y):min(mask.shape[0],y+h),max(0,x):min(mask.shape[1],x+w)]=1
  overlap=int(np.count_nonzero(mask & (covered>0)))
  record[name]={'flagged':bool(r[name]['findings']),'any_box_overlaps_annotation':overlap>0,
                'annotated_pixels_covered_by_display_boxes':overlap,
                'annotated_pixel_count':int(mask.sum())}
 records.append(record)
summary={}
for name in ['before','after']:
 positive=[r for r in records if r['defective']]
 summary[name]={'defective_views':len(positive),'flagged_defective_views':sum(r[name]['flagged'] for r in positive),
                'defective_views_with_box_mask_overlap':sum(r[name]['any_box_overlaps_annotation'] for r in positive),
                'any_box_overlap_recall':sum(r[name]['any_box_overlaps_annotation'] for r in positive)/len(positive),
                'flagged_defects_without_any_box_mask_overlap':[r['path'] for r in positive if r[name]['flagged'] and not r[name]['any_box_overlaps_annotation']]}
output={'scope':'Permissive offline original-mask overlap of saved display rectangles; not segmentation IoU, blind validation or independent inference',
        'source_report':str(a.report),'summary':summary,'records':records}
a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
