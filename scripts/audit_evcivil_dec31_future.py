"""Non-neural audit of a separate unused Dec31 proposal; never opens heldout pixels."""
from pathlib import Path
import sys,hashlib,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_preprocessing import camera_exposure
BASE=ROOT/'datasets/construction';OUT=BASE/'evcivil-dec31-future'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def signature(gray):
    small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    bits=np.packbits((small[:,1:]>small[:,:-1]).reshape(-1))
    reduced=cv2.resize(gray,(32,24),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    reduced-=reduced.mean();norm=np.linalg.norm(reduced)
    if norm:reduced/=norm
    return bits,reduced
def main():
    cv2.setNumThreads(1);source=OUT/'asset-manifest.json';manifest=json.loads(source.read_text());qa=OUT/'qa';qa.mkdir(exist_ok=True)
    if (qa/'duplicate-audit.json').exists():raise ValueError('Preserve prior audit output')
    hashes={};trainpaths=set();pinned={}
    def add(value,desc):
        if isinstance(value,str) and len(value)==64:hashes.setdefault(value,[]).append(desc)
    for p in sorted(BASE.rglob('exposures.json')):
        pinned[str(p.relative_to(ROOT))]=sha(p)
        for r in json.loads(p.read_text()):
            desc={'manifest':str(p.relative_to(ROOT)),'split':r.get('split'),'image':r.get('image'),'global_id':r.get('global_defect_id') or r.get('canonical_defect_id')}
            for k in ('sha256_pixels','pixel_sha256','image_sha256','sha256'):add(r.get(k),{**desc,'hash_field':k})
            if r.get('split')=='train' and r.get('image'):trainpaths.add(r['image'])
    for p in sorted(BASE.rglob('native-input-manifest.json')):
        pinned[str(p.relative_to(ROOT))]=sha(p)
        for r in json.loads(p.read_text()):
            desc={'manifest':str(p.relative_to(ROOT)),'split':r.get('split'),'image':r.get('raw_source_image')}
            for k in ('raw_source_file_sha256','raw_native_bgr_sha256','image_sha256','native_model_input_bgr_sha256'):add(r.get(k),{**desc,'hash_field':k})
            if r.get('split')=='train' and r.get('raw_source_image'):trainpaths.add(r['raw_source_image'])
    all_camera_hash_matches=[]
    for sequence in manifest['sequence_summary']:
        for row in sequence['all_camera_statistics']:
            if row['pixel_sha256'] in hashes:
                all_camera_hash_matches.append({'sequence':sequence['sequence'],'original_frame_index':row['original_frame_index'],
                    'matches':hashes[row['pixel_sha256']]})
    refs=[];refsig=[];seen=set()
    for s in sorted(trainpaths):
        p=ROOT/s
        if not p.is_file():raise ValueError(f'Missing known TRAIN source {s}')
        digest=sha(p)
        if digest in seen:continue
        seen.add(digest);gray=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError('Invalid known TRAIN source')
        refs.append({'path':str(p.relative_to(ROOT)),'file_sha256':digest,'scope':'Already-used TRAIN input only'});refsig.append(signature(gray))
    packed=np.stack([s[0] for s in refsig]);pop=np.array([i.bit_count() for i in range(256)],np.uint8)
    findings=[];sigs=[]
    for r in manifest['exposures']:
        p=OUT/r['image'];gray=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if sha(p)!=r['image_sha256'] or hashlib.sha256(gray.tobytes()).hexdigest()!=r['pixel_sha256']:raise ValueError('Proposed original image identity changed')
        bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);s=signature(gray);sigs.append(s)
        exact=[]
        for k,digest in [('original_png',sha(p)),('native_gray_pixels',r['pixel_sha256']),('native_bgr_pixels',hashlib.sha256(bgr.tobytes()).hexdigest())]:
            exact.extend({**match,'candidate_hash_convention':k} for match in hashes.get(digest,[]))
        distances=pop[np.bitwise_xor(packed,s[0])].sum(axis=1);best=[]
        for idx in np.argsort(distances,kind='stable')[:5]:
            best.append({'reference':refs[int(idx)],'dhash_hamming':int(distances[idx]),'normalized_gray_correlation':float(np.sum(s[1]*refsig[idx][1]))})
        flags=[x for x in best if x['dhash_hamming']<=4 or(x['dhash_hamming']<=8 and x['normalized_gray_correlation']>=.95)]
        findings.append({'image':r['image'],'sequence':r['sequence'],'canonical_id':r['canonical_id'],'exact_protected_metadata_matches':exact,'closest_known_train':best,'near_duplicate_flags':flags})
    cross=[]
    for a in range(len(sigs)):
        for b in range(a):
            if manifest['exposures'][a]['sequence']==manifest['exposures'][b]['sequence']:continue
            distance=int(pop[np.bitwise_xor(sigs[a][0],sigs[b][0])].sum());corr=float(np.sum(sigs[a][1]*sigs[b][1]))
            if distance<=4 or(distance<=8 and corr>=.95):cross.append({'left':manifest['exposures'][a]['image'],'right':manifest['exposures'][b]['image'],'dhash_hamming':distance,'normalized_gray_correlation':corr})
    sheets=[]
    for sequence in manifest['sequence_summary']:
        rows=[r for r in manifest['exposures'] if r['sequence']==sequence['sequence']]
        if not rows:continue
        chosen=[rows[i] for i in np.unique(np.linspace(0,len(rows)-1,min(3,len(rows))).round().astype(int))];tiles=[]
        for r in chosen:
            gray=cv2.imread(str(OUT/r['image']),cv2.IMREAD_GRAYSCALE);raw=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);gain,mult=camera_exposure(raw,'low_light_gain');overlay=gain.copy()
            for b in r['boxes']:
                x,y,w,h=map(round,b['bbox']);cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,255),1)
            panels=[]
            for name,frame in [('ORIGINAL RAW',raw),('SHARED GAIN INPUT',gain),('PUBLISHER GT; NO MODEL',overlay)]:
                header=np.zeros((48,415,3),np.uint8);cv2.putText(header,name,(5,15),0,.44,(255,255,255),1)
                cv2.putText(header,f"{sequence['sequence']} #{r['original_frame_index']} PTS {r['source_pts_us']/1e6:.3f}s",(5,30),0,.38,(255,255,255),1)
                cv2.putText(header,f"native p99={r['p99']:.1f}; gain={mult:.2f}; future TRAIN ONLY",(5,44),0,.37,(255,255,255),1)
                panels.append(np.vstack([header,cv2.resize(frame,(415,312),interpolation=cv2.INTER_NEAREST)]))
            tiles.append(np.hstack(panels))
        target=qa/(sequence['sequence']+'-fixed-raw-input-gt.jpg');cv2.imwrite(str(target),np.vstack(tiles));sheets.append(target.relative_to(OUT).as_posix())
    # Pair-review sheets are fixed by metadata ordering, never by model outputs.
    pairs=[{'kind':'known_train','left':r['image'],'right':f['reference']['path'],'distance':f['dhash_hamming']} for r in findings for f in r['near_duplicate_flags']]
    pairs +=[{'kind':'cross_candidate','left':r['left'],'right':r['right'],'distance':r['dhash_hamming']}for r in cross]
    pages=[]
    for start in range(0,len(pairs),8):
        tiles=[]
        for i,pair in enumerate(pairs[start:start+8],start):
            panels=[]
            for side in ('left','right'):
                p=(ROOT/pair[side]) if side=='right' and pair['kind']=='known_train' else OUT/pair[side]
                frame=cv2.imread(str(p));header=np.zeros((42,415,3),np.uint8)
                cv2.putText(header,f"pair {i} {pair['kind']} dHash={pair['distance']} {side}",(4,15),0,.38,(255,255,255),1)
                cv2.putText(header,p.name[:56],(4,32),0,.34,(255,255,255),1)
                panels.append(np.vstack([header,cv2.resize(frame,(415,312))]))
            tiles.append(np.hstack(panels))
        p=qa/f'flagged-pairs-{start//8:03d}.jpg';cv2.imwrite(str(p),np.vstack(tiles));pages.append(p.relative_to(OUT).as_posix())
    report={'manifest_sha256':sha(source),'protected_reference_manifest_sha256':pinned,'known_train_reference_images':len(refs),'reference_hash_values':len(hashes),'proposed_views':len(findings),
        'exact_protected_match_views':sum(bool(r['exact_protected_metadata_matches']) for r in findings),'all_camera_gray_hash_matches':all_camera_hash_matches,
        'near_flagged_known_train_views':sum(bool(r['near_duplicate_flags']) for r in findings),
        'cross_candidate_flags':cross,'findings':findings,'fixed_sequence_contact_sheets':sheets,'flagged_pairs':pairs,'pair_review_pages':pages,
        'heldout_pixel_opening':False,'calibration_pixel_opening':False,'neural_inference':False,
        'limitations':'Exact overlap checked against stored prepared-source file/gray/BGR hashes of all splits. Near screening uses known TRAIN plus cross-candidate only; no stored universal heldout perceptual signatures found. Physical site independence unproven, not an automatic rejection; actual unresolved duplicate/alias flags remain acceptance blockers.'}
    write(qa/'duplicate-audit.json',report)
    print(json.dumps({k:report[k]for k in ('known_train_reference_images','reference_hash_values','proposed_views','exact_protected_match_views','near_flagged_known_train_views')},indent=2));print('Cross candidate flags',len(cross),'pair pages',len(pages))
if __name__=='__main__':main()
