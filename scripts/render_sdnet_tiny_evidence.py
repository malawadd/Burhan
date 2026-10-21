"""Render recorded TRAIN-only predictions without obscuring source image pixels."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/construction-resumption-fastening-tiny-v1'
def panel(row,predictions,title):
    frame=cv2.imread(str(ROOT/row['image']));summary=[]
    if predictions is None:
        for box in row['boxes']:
            color=(0,255,0)if box['class_id']==0 else(255,200,0)
            for polygon in box['polygons']:cv2.polylines(frame,[np.rint(np.array(polygon).reshape(-1,2)).astype(np.int32)],True,color,2)
            x,y,w,h=map(round,box['bbox']);cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        summary.append(f"{len(row['boxes'])} real publisher targets; masks=GT only")
    else:
        chosen=sorted([p for p in predictions if p['score']>=.25],key=lambda p:-p['score'])
        for p in chosen[:20]:
            x,y,w,h=map(round,p['bbox']);cv2.rectangle(frame,(x,y),(x+w,y+h),(0,180,255),2)
            cv2.putText(frame,('L'if p['class_id']==0 else'A')+f" {p['score']:.2f}",(x,max(15,y)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,180,255),1)
        summary.append(f'{len(chosen)} boxes >=.25; top20 shown; L=unseated A=absence')
    banner=np.zeros((55,640,3),np.uint8);cv2.putText(banner,title,(7,21),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);cv2.putText(banner,summary[0],(7,43),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,230),1)
    return np.vstack([banner,frame])
def main():
    cv2.setNumThreads(1);path=OUT/'predictions.json';data=json.loads(path.read_text());cards=[];files=[]
    target=OUT/'full-frame-overlays';target.mkdir(exist_ok=True)
    for row,b,a in zip(data['selected_records'],data['before'],data['after']):
        name='/'.join(row['source_ids']);frame=np.hstack([panel(row,None,'GROUND TRUTH '+name),panel(row,b,'BEFORE SSDv1 - TRAIN PHOTO'),panel(row,a,'AFTER64 - SAME TRAIN PHOTO')])
        dest=target/(row['id']+'.jpg');cv2.imwrite(str(dest),frame);files.append(str(dest.relative_to(ROOT)));cards.append(cv2.resize(frame,(960,348)))
    cv2.imwrite(str(OUT/'full-frame-train-before-after.jpg'),np.vstack(cards))
    receipt={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_images':'All8selectedTRAINphotos; full640x640pixelsretainedbelowexternalbanner','threshold':.25,'display_cap':20,'model_predicted_masks':False,'files':files}
    (OUT/'evidence-render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
