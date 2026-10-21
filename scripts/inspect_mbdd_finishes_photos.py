"""Verify acquired real wall photograph/label geometry and create GT-only sheets."""
from pathlib import Path
import hashlib,json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd2025-wall-photos'


def main():
    cv2.setNumThreads(1)
    manifest_path=DATA/'asset-manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    out=DATA/'qa';out.mkdir(exist_ok=True)
    checked=[]
    for row in manifest['records']:
        image=DATA/row['image']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=row['image_sha256']:raise RuntimeError('Original wall image bytes differ')
        frame=cv2.imread(str(image));h,w=frame.shape[:2]
        if [w,h]!=row['frame_size_wh']:raise RuntimeError('Wall annotation/image dimensions differ')
        for b in row['boxes']:
            x1,y1,x2,y2=b['source_xyxy']
            if not np.isfinite([x1,y1,x2,y2]).all() or not(0<=x1<x2<=w and 0<=y1<y2<=h):raise RuntimeError('Wall bbox exceeds original frame')
        checked.append({'image':row['image'],'labels':sorted({b['label'] for b in row['boxes']}),'frame_size_wh':[w,h],'boxes':len(row['boxes'])})
    for label in manifest['classes']:
        group=[r for r in manifest['records'] if any(b['label']==label for b in r['boxes'])]
        sheet=np.full((len(group)*240,720,3),24,np.uint8)
        for i,row in enumerate(group):
            raw=cv2.imread(str(DATA/row['image']));model=raw.copy();h,w=raw.shape[:2]
            for b in row['boxes']:
                p=b['source_xyxy'];x1,y1,x2,y2=map(round,p)
                cv2.rectangle(model,(x1,y1),(x2,y2),(0,180,255),2)
                cv2.putText(model,b['label'],(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,180,255),1)
            for j,panel in enumerate([raw,model]):
                thumb=cv2.resize(panel,(360,202),interpolation=cv2.INTER_AREA)
                sheet[i*240+25:i*240+227,j*360:(j+1)*360]=thumb
                caption=('ORIGINAL RAW PHOTO' if j==0 else 'PUBLISHER BOXES - NOT MODEL')
                cv2.putText(sheet,caption,(j*360+4,i*240+17),cv2.FONT_HERSHEY_SIMPLEX,.4,(235,235,235),1)
            cv2.putText(sheet,Path(row['image']).name,(4,i*240+238),cv2.FONT_HERSHEY_SIMPLEX,.32,(235,235,235),1)
        cv2.imwrite(str(out/(label+'-original-publisher-gt.jpg')),sheet)
    report={'source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'photos':len(checked),'checks':'Original byte hashes, image/nativeXML dimensions, finitepositiveinbounds boxes allPASS',
            'annotation_coordinates':'Original publisherPascalVOC XYXY; no 1-based conversion or claims of subpixel truth. Rendering diagnostic only.',
            'classes':{label:sum(any(b['label']==label for b in r['boxes']) for r in manifest['records']) for label in manifest['classes']},
            'model_outputs':False,'records':checked}
    (out/'geometry-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
