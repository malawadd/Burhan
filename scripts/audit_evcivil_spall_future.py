"""One-thread, non-neural duplicate/visual audit of unused future camera data."""
from pathlib import Path
import hashlib,json
import cv2,h5py,numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/construction'
OUT=BASE/'evcivil-spall-future-r4'

def signature(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
    small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    bits=np.packbits((small[:,1:]>small[:,:-1]).reshape(-1))
    reduced=cv2.resize(gray,(32,24),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    reduced-=reduced.mean(); norm=np.linalg.norm(reduced)
    if norm:reduced/=norm
    digest=hashlib.sha256(str(gray.shape).encode()+gray.tobytes()).hexdigest()
    return digest,bits,reduced

def main():
    cv2.setNumThreads(1)
    source=OUT/'asset-manifest.json'; manifest=json.loads(source.read_text(encoding='utf-8'))
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    references=[]; refsig=[]
    # These are known, already-materialized inputs. No inference/accuracy labels are read.
    for root in [BASE/'evcivil-round3/images',BASE/'evcivil-round3-continuation/images']:
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.suffix.lower() not in {'.png','.jpg','.jpeg'}:continue
                frame=cv2.imread(str(path));
                if frame is None:raise RuntimeError(str(path))
                references.append({'path':path.relative_to(ROOT).as_posix(),'kind':'known_materialized_image'})
                refsig.append(signature(frame))
    for name in ['crack_10','crack_89','spalling_35']:
        path=BASE/'evcivil'/name/(name+'_frames.h5')
        if not path.exists():continue
        with h5py.File(path,'r') as data:
            for index in range(len(data['frames'])):
                references.append({'path':path.relative_to(ROOT).as_posix(),'frame_index':index,'kind':'exposed_original_camera'})
                refsig.append(signature(data['frames'][index]))
    exact={}
    for i,sig in enumerate(refsig):exact.setdefault(sig[0],[]).append(i)
    packed=np.stack([s[1] for s in refsig]); popcount=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    findings=[]; proposal_sigs=[]
    for item in manifest['exposures']:
        frame=cv2.imread(str(OUT/item['image']),cv2.IMREAD_UNCHANGED)
        sig=signature(frame);proposal_sigs.append(sig)
        distance=popcount[np.bitwise_xor(packed,sig[1])].sum(axis=1)
        best=[]
        for idx in np.argsort(distance,kind='stable')[:5]:
            correlation=float(np.sum(sig[2]*refsig[idx][2]))
            best.append({'reference':references[int(idx)],'dhash_hamming':int(distance[idx]),'normalized_gray_correlation':correlation})
        flags=[r for r in best if r['dhash_hamming']<=4 or (r['dhash_hamming']<=8 and r['normalized_gray_correlation']>=.95)]
        findings.append({'image':item['image'],'canonical_id':item['canonical_id'],'source_group':item['source_group'],
                         'exact_known_matches':[references[i] for i in exact.get(sig[0],[])],
                         'closest_known':best,'non_neural_similarity_flags':flags})
    intra=[]
    for a in range(len(proposal_sigs)):
        ia=manifest['exposures'][a]
        for b in range(a):
            ib=manifest['exposures'][b]
            if ia['canonical_id']==ib['canonical_id']:continue
            sa,sb=proposal_sigs[a],proposal_sigs[b]
            distance=int(popcount[np.bitwise_xor(sa[1],sb[1])].sum())
            correlation=float(np.sum(sa[2]*sb[2]))
            if sa[0]==sb[0] or distance<=4 or (distance<=8 and correlation>=.95):
                intra.append({'left':ia['image'],'right':ib['image'],'exact':sa[0]==sb[0],'dhash_hamming':distance,'normalized_gray_correlation':correlation})
    sheets=[]
    for seq in manifest['sequence_summary']:
        records=[i for i in manifest['exposures'] if i['source_group']==seq['group'] and i['canonical_id']==seq['canonical_id']]
        if not records:continue
        selected=[records[int(i)] for i in np.unique(np.linspace(0,len(records)-1,3).round().astype(int))]
        rows=[]
        for item in selected:
            raw=cv2.imread(str(OUT/item['image'])); overlay=raw.copy()
            for box in item['boxes']:
                x,y,w,h=map(round,box['bbox']);cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,255),2)
                cv2.putText(overlay,box['label'],(x,max(14,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            panels=[]
            for title,frame in [('RAW',raw),('PUBLISHER BOXES - NOT MODEL OUTPUT',overlay)]:
                target=cv2.resize(frame,(480,360));header=np.zeros((42,480,3),np.uint8)
                cv2.putText(header,title,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                cv2.putText(header,f"frame {item['original_frame_index']} / PTS {item['source_pts_us']/1e6:.3f}s",(5,34),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                panels.append(np.vstack([header,target]))
            rows.append(np.hstack(panels))
        path=qa/(seq['group']+'-'+seq['canonical_id']+'-fixed-raw-gt.jpg')
        cv2.imwrite(str(path),np.vstack(rows));sheets.append(str(path.relative_to(OUT)))
    report={'manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'method':'No NN. Exact grayscale pixels+geometry; 64-bit dHash; mean-centered normalized32x24 gray correlation. Fixed first/middle/last original exposure sheets.',
            'caveat':'Similarity is a screen, not proof of physical identity or independence. Concrete texture can collide. Known heldout inputs used solely to screen leakage, never to evaluate/select a model.',
            'reference_count':len(references),'proposed_count':len(findings),'exact_known_match_views':sum(bool(f['exact_known_matches']) for f in findings),
            'similarity_flagged_known_views':sum(bool(f['non_neural_similarity_flags']) for f in findings),
            'cross_proposed_sequence_flags':intra,'findings':findings,'fixed_contact_sheets':sheets}
    target=qa/'duplicate-audit.json';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['reference_count','proposed_count','exact_known_match_views','similarity_flagged_known_views']},indent=2))
    print('Cross proposed sequence flags',len(intra));print(target)

if __name__=='__main__':main()
