"""Display-only fixed shared gain for saved duplicate-review pairs; no NN."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_preprocessing import camera_exposure
OUT=ROOT/'datasets/construction/evcivil-dec31-future'
def main():
    cv2.setNumThreads(1);source=OUT/'qa/duplicate-audit.json';audit=json.loads(source.read_text());pages=[]
    target=OUT/'qa/gain-pair-review';target.mkdir(exist_ok=True)
    if (target/'index.json').exists():raise ValueError('Preserve completed review rendering')
    for start in range(0,len(audit['flagged_pairs']),6):
        rows=[]
        for i,pair in enumerate(audit['flagged_pairs'][start:start+6],start):
            panels=[]
            for side in ('left','right'):
                p=ROOT/pair[side] if side=='right' and pair['kind']=='known_train' else OUT/pair[side]
                raw=cv2.imread(str(p));frame,gain=camera_exposure(raw,'low_light_gain');head=np.zeros((50,520,3),np.uint8)
                cv2.putText(head,f"pair{i}: {side}; shared gain {gain:.2f}; dHash{pair['distance']}",(4,16),0,.44,(255,255,255),1)
                cv2.putText(head,p.name[:65],(4,32),0,.36,(255,255,255),1)
                cv2.putText(head,'DISPLAY INPUT ONLY; NO MODEL OR SYNTHETIC DAMAGE',(4,46),0,.36,(255,255,255),1)
                panels.append(np.vstack([head,cv2.resize(frame,(520,390),interpolation=cv2.INTER_NEAREST)]))
            rows.append(np.hstack(panels))
        p=target/f'fixed-pairs-{start//6:03d}.jpg';cv2.imwrite(str(p),np.vstack(rows));pages.append(str(p.relative_to(OUT)))
    value={'source_audit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'policy':'Immutable camera_exposure low_light_gain, input display only; originals unchanged','pairs':audit['flagged_pairs'],'pages':pages,'model_inference':False}
    (target/'index.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps({'pairs':len(value['pairs']),'pages':len(pages)}))
if __name__=='__main__':main()
