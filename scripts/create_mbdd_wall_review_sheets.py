"""Fixed source-order raw/publisher-GT sheets, never model predictions."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd-wall-five-class-v1'


def main():
    cv2.setNumThreads(1)
    source=DATA/'asset-manifest.json';manifest=json.loads(source.read_text(encoding='utf-8'))
    out=DATA/'qa';out.mkdir(exist_ok=True)
    selections=[]
    for split in ['train','calibration']:
        for label in manifest['classes']:
            rows=sorted([r for r in manifest['records'] if r['split']==split and r['status']=='eligible_pending_independent_review' and any(b['label']==label for b in r['boxes'])],key=lambda r:r['source_name'])
            chosen=[rows[i] for i in sorted({0,len(rows)//2,len(rows)-1})]
            sheet=np.full((len(chosen)*250,800,3),22,np.uint8)
            for index,row in enumerate(chosen):
                raw=cv2.imread(str(DATA/row['image']));gt=raw.copy()
                for box in row['boxes']:
                    x1,y1,x2,y2=map(round,box['source_xyxy']);cv2.rectangle(gt,(x1,y1),(x2,y2),(0,170,255),2)
                    cv2.putText(gt,box['label'],(x1,max(16,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,170,255),1)
                for panel_index,panel in enumerate([raw,gt]):
                    sheet[index*250+28:index*250+228,panel_index*400:(panel_index+1)*400]=cv2.resize(panel,(400,200),interpolation=cv2.INTER_AREA)
                    caption='ORIGINAL RAW PHOTO' if panel_index==0 else 'PUBLISHER GT - NOT MODEL'
                    cv2.putText(sheet,caption,(panel_index*400+5,index*250+18),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
                cv2.putText(sheet,f"{split} {label} | {row['source_name']} | block{row['capture_block_200']}",(5,index*250+244),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
            target=out/(split+'-'+label+'-first-middle-last.jpg');cv2.imwrite(str(target),sheet)
            selections.append({'split':split,'label':label,'source_order_rule':'Lexical actualsource filename first/middle/last among eligible source-class positives; no model ranking',
                               'photo_names':[r['source_name'] for r in chosen],'sheet':target.relative_to(DATA).as_posix(),
                               'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    diagnostic=ROOT/'datasets/construction/finishes-expansion/mbdd2025-wall-photos'
    old=json.loads((diagnostic/'asset-manifest.json').read_text(encoding='utf-8'))
    paths={r['source_name']:DATA/r['image'] for r in manifest['records']}
    paths.update({r['filename']:diagnostic/r['image'] for r in old['records']})
    for index,flag in enumerate(manifest['crosssplit_flags']):
        sheet=np.full((300,1000,3),22,np.uint8)
        for j,key in enumerate(['a','b']):
            raw=cv2.imread(str(paths[flag[key]]));sheet[25:275,j*500:(j+1)*500]=cv2.resize(raw,(500,250),interpolation=cv2.INTER_AREA)
            cv2.putText(sheet,flag[key]+' '+flag[key+'_split'],(j*500+5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
        cv2.putText(sheet,'Similarity quarantine remains: '+flag['quarantined']+' | NOT MODEL OUTPUT',(5,294),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
        target=out/f'crosssplit-quarantine-{index:02d}.jpg';cv2.imwrite(str(target),sheet)
        selections.append({'flag':flag,'sheet':target.relative_to(DATA).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    report={'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model_outputs':False,'manual_review_status':'Not independently reviewed by this rendering helper',
            'scope':'Bounded fixed30 source observations plus all similarityflags, not all230 visually reviewed or physical-site independence',
            'selections':selections}
    (out/'sheet-selection.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(out/'sheet-selection.json'),'sheets':len(selections)},indent=2),flush=True)


if __name__=='__main__':main()
