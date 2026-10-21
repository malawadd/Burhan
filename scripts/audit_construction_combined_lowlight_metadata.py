"""Independent combined-data metadata/label-text audit; no image bytes/NN."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,re
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'datasets/construction'
NEW=BASE/'evcivil-round4-real-lowlight-gain';OLD=BASE/'evcivil-round4-broad-adapted-gain'
ACCEPTED={16,17,18,20,24,27,35,37,41,42,46,48,51};QUARANTINE={30,34,40,45,49,50,52}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    errors=[];checks={};paths={}
    def require(condition,message):
        if not condition:errors.append(message)
    def pin(p,expected):
        actual=sha(p);paths[str(p.relative_to(ROOT))]=actual;require(actual==expected,'Pinned JSON identity differs: '+str(p));return load(p)
    protocol=load(NEW/'protocol.json');paths[str((NEW/'protocol.json').relative_to(ROOT))]=sha(NEW/'protocol.json')
    data=pin(NEW/'data-protocol.json',protocol['prepared_data_protocol_sha256'])
    native=pin(NEW/'native-input-manifest.json',protocol['native_input_manifest_sha256'])
    exposures=pin(NEW/'exposures.json',protocol['exposures_sha256'])
    for value in protocol['source_pins'].values():pin(ROOT/value['path'],value['sha256'])
    decpath=BASE/'evcivil-dec31-future/acceptance-v1.json';dec=pin(decpath,data['dec31_acceptance_sha256'])
    decmanifest=pin(BASE/'evcivil-dec31-future/asset-manifest.json',dec['source_manifest_sha256'])
    pair=pin(BASE/'evcivil-dec31-future/qa/pair-review-v2.json',dec['independent_pair_review_v2_sha256'])
    lcpath=BASE/'lcsd-low-light-future-r4/acceptance-v1.json';lc=pin(lcpath,data['lcsd_acceptance_sha256'])
    lcmanifest=pin(BASE/'lcsd-low-light-future-r4/asset-manifest.json',lc['source_manifest_sha256'])
    lcproto=pin(BASE/'lcsd-low-light-future-r4/protocol-before-pixels.json',lc['acquisition_protocol_sha256'])
    olddata=pin(OLD/'data-protocol.json',data['base_prepared_data_protocol_sha256'])
    old=pin(OLD/'native-input-manifest.json',olddata['native_input_manifest_sha256'])
    oldexp=pin(OLD/'exposures.json',olddata['exposures_sha256'])
    require(sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py')==data['preprocessing_source_sha256'],'Shared exposure source changed')
    require(data['classes']=={'0':'spalling','1':'crack'},'Combined class map changed')
    expected={str((ROOT/r['image']).resolve()):r['boxes']for r in exposures if r['split']in('train','calibration')}
    counts={s:Counter()for s in('train','calibration')};maxerror=0.;minimum=[math.inf,math.inf];lclabels=0;declabels=0
    newbykey={};decrows=[];lcrows=[];photoids=set()
    for row in native:
        split=row['split'];counts[split]['images']+=1;counts[split]['gain_applied']+=row['gain']>1
        key=(split,str((ROOT/row['raw_source_image']).resolve()));require(key not in newbykey,'Duplicate raw-source identity in split');newbykey[key]=row
        label=ROOT/row['label'];content=label.read_bytes();require(hashlib.sha256(content).hexdigest()==row['label_sha256'],'Actual label bytes differ: '+str(label))
        require((ROOT/row['image']).is_file(),'Prepared image absent: '+row['image'])
        actual=[];w,h=row['shape_wh']
        for line in content.decode().splitlines():
            if not line.strip():continue
            values=list(map(float,line.split()));require(len(values)==5 and all(math.isfinite(v)for v in values),'Malformed normalized label: '+str(label))
            if len(values)!=5:continue
            cls,x,y,bw,bh=values
            require(cls in(0,1)and min(bw,bh)>0 and min(x-bw/2,y-bh/2)>=-1e-6 and max(x+bw/2,y+bh/2)<=1+1e-6,'Invalid normalized class/geometry: '+str(label))
            actual.append((int(cls),(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h));counts[split]['boxes_'+str(int(cls))]+=1
            minimum=[min(minimum[0],bw*w),min(minimum[1],bh*h)]
        source=expected.get(str((ROOT/row['image']).resolve()),[])
        wanted=[(b.get('class_id',{'spalling':0,'crack':1}[b['label']]),*b['bbox'])for b in source]
        require(len(actual)==len(wanted),'Exposure metadata/label count differs: '+str(label));remaining=set(range(len(wanted)))
        for a in actual:
            options=[(max(abs(x-y)for x,y in zip(a,wanted[i])),i)for i in remaining if a[0]==wanted[i][0]]
            delta,index=min(options,default=(math.inf,None));maxerror=max(maxerror,delta)
            require(delta<=1e-3,'Native box/class mismatch: '+str(label))
            if index is not None:remaining.remove(index)
        counts[split]['positive_images'if actual else'negative_images']+=1
        raw=row['raw_source_image']
        if 'evcivil-dec31-future'in raw:decrows.append(row);declabels+=len(actual)
        if 'lcsd-low-light-future-r4'in raw:lcrows.append(row);lclabels+=len(actual)
        if split=='train':
            if not actual:counts[split]['negative_photos']+=1
            elif 'dacl10k'in raw or 'lcsd-low-light-future-r4'in raw:
                counts[split]['positive_photo_views']+=1
                match=re.search(r'dacl10k_v2_train_\d+|LCSD_\d+',raw)
                if match:photoids.add(match[0])
            else:counts[split]['camera_exposures']+=1
    require(len(native)==1293 and counts['train']['images']==1171 and counts['calibration']['images']==122,'Combined input split counts differ')
    require(counts['train']['camera_exposures']==990 and counts['train']['positive_photo_views']==161 and len(photoids)==121 and counts['train']['negative_photos']==20,'Media/source counts differ')
    require(counts['train']['gain_applied']==296 and counts['calibration']['gain_applied']==32,'Actual recorded gain counts differ')
    require((counts['train']['boxes_0'],counts['train']['boxes_1'])==(585,1016),'Actual TRAIN class label counts differ')
    require((counts['calibration']['boxes_0'],counts['calibration']['boxes_1'])==(38,125),'Actual CAL class label counts differ')
    preserved=Counter()
    for row in old:
        key=(row['split'],str((ROOT/row['raw_source_image']).resolve()));other=newbykey.get(key);require(other is not None,'Historical880 input missing')
        if other:
            for field in ['raw_source_file_sha256','raw_native_bgr_sha256','image_sha256','label_sha256','gain','native_model_input_bgr_sha256','shape_wh']:
                require(row[field]==other[field],'Historical native/label contract changed: '+field)
            preserved[row['split']]+=1
    require(preserved=={'train':880,'calibration':122},'Historical880/CAL copy cohort differs')
    # Compare original exposures/GT after path-only remapping; no heldout labels/pixels read.
    def stable(row,root):return json.dumps({k:(v.replace(str(root.relative_to(ROOT)), '<DATA>')if isinstance(v,str)else v)for k,v in row.items()},sort_keys=True)
    newset={stable(r,NEW)for r in exposures}
    require(all(stable(r,OLD)in newset for r in oldexp),'Historical source exposure/GT metadata changed')
    require(sha(NEW/'night-holdout-protocol.json')==sha(OLD/'night-holdout-protocol.json'),'Heldout protocol byte identity changed')
    # All Dec31 rows must map exactly once to the accepted original raw exposures.
    decaccepted={str((BASE/'evcivil-dec31-future'/r['image']).resolve()):r for r in dec['accepted_views']}
    require(len(decrows)==250 and declabels==405,'Dec31 imported view/box count differs')
    ids=set()
    for row in decrows:
        original=decaccepted.get(str((ROOT/row['raw_source_image']).resolve()));require(original is not None,'Unaccepted Dec31 source imported')
        ident=int(row['canonical_defect_id'].split('_')[-1]);ids.add(ident);require(ident in ACCEPTED and ident not in QUARANTINE,'Forbidden Dec31 ID imported')
        require(row['split']=='train'and row['gain']>1,'Dec31 source outside dark TRAIN scope')
        if original:
            for a,b in [('raw_source_file_sha256','image_sha256'),('gain','gain'),('source_camera_sha256','camera_source_sha256'),('source_annotation_sha256','annotation_source_sha256'),('original_frame_index','original_frame_index'),('source_pts_us','source_pts_us')]:require(row[a]==original[b],'Dec31 source/timing/gain receipt differs')
    require(ids==ACCEPTED and set(data['dec31_accepted_ids'])==ACCEPTED,'Exact accepted13 Dec31 IDs differ')
    require(set(data['dec31_quarantined_ids'])==QUARANTINE,'Seven Dec31 quarantines differ')
    # LCSD source-code and component evidence are audited as retained metadata,
    # not re-decoded masks in this explicitly metadata-only independent audit.
    accepted={r['source_id']:r for r in lc['accepted_records']};originals={r['source_id']:r for r in lcmanifest['records']}
    require(len(lcrows)==41 and lclabels==92 and len(accepted)==41,'LCSD exact41 views/92 components differ')
    require(not lc['bounded_visual_review']['all41_manually_reviewed'] and len(lc['bounded_visual_review']['fixed_source_ids'])==3,'LCSD bounded-review scope inflated')
    require(lc['derived_box_count']==92 and lc['original_author_partition']=='VAL/TEST','LCSD repurposing/component contract differs')
    seen_lc=set()
    for row in lcrows:
        ident=row['source_photo_id'];seen_lc.add(ident);source=accepted[ident];oldsource=originals[ident]
        require(row['split']=='train'and row['media_kind']=='actual_still_photograph'and row['original_frame_index']is None,'LCSD media/partition incorrectly represented')
        require(row['original_author_partition']=='VAL/TEST'and row['gain']==source['shared_gain'],'LCSD author partition/gain differs')
        require(source['mask_code_counts'].keys()<={'0','2'} and source['mask_code_counts'].get('2',0)>0 and source['ignore_pixels']==0,'LCSD retained mask-code metadata unresolved')
        require(sum(source['mask_code_counts'].values())==160000,'LCSD retained mask pixel total differs')
        require(source['original_mask_sha256']==oldsource['original_mask_sha256']==row['source_mask_sha256'],'LCSD original mask identity differs')
        require((ROOT/row['source_mask']).is_file(),'Original LCSD mask not retained')
        require(row['raw_source_file_sha256']==source['original_image_sha256']and row['raw_native_bgr_sha256']==source['original_native_bgr_sha256']and row['native_model_input_bgr_sha256']==source['actual_gained_native_bgr_sha256'],'LCSD source/native gain identities differ')
        boxes=expected[str((ROOT/row['image']).resolve())];require(boxes==source['candidate_component_boxes'],'LCSD all-component boxes changed')
    require(seen_lc==set(accepted)==set(data['lcsd_source_ids']),'Exact LCSD source set differs')
    calhashes={r['raw_source_file_sha256']for r in native if r['split']=='calibration'};trainhashes={r['raw_source_file_sha256']for r in native if r['split']=='train'}
    require(not calhashes&trainhashes,'Exact raw-source file hash overlap TRAIN/CAL')
    result={'passed':not errors,'errors':errors,'scope':'Metadata JSON/file existence/actual label TEXT only; no image or mask byte scans/decoding, NN, heldout labels or active changes',
        'pins':paths,'counts':{s:dict(c)for s,c in counts.items()},'unique_positive_photo_ids':len(photoids),'historical_native_copy_contract_preserved':dict(preserved),
        'dec31':{'views':len(decrows),'publisher_crack_boxes':declabels,'accepted_ids':sorted(ids),'quarantined_ids_excluded':sorted(QUARANTINE)},
        'lcsd':{'views':len(lcrows),'derived_component_boxes':lclabels,'original_mask_retained_records':len(lcrows),'mask_code_semantics':'Stored source metadata0background/2crack; static builder explicitly compares mask==2, no inversion, no morphology; masks not independently decoded here','bounded_manual_review':'Three fixed sheets plus one flagged pair, not all41'},
        'maximum_native_label_coordinate_error_pixels':maxerror,'minimum_label_wh_native_pixels':minimum,
        'raw_source_file_hash_overlap_train_cal':len(calhashes&trainhashes),'limitations':['Native pixel-copy/gain identities are compared via pinned manifests; this audit does not independently decode or scan PNG/JPEG/mask bytes. Trainer separately verified actual sample hashes before optimizer.','Publisher boxes and component-derived boxes are distinct scopes, not pixel/object metric interchange.','No physical site independence, invisible damage assurance or LCSD heldout benchmark credit.']}
    target=ROOT/'output/evaluation/construction-combined-lowlight-independent-metadata-audit.json'
    if target.exists():raise ValueError('Preserve independent audit result')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items()if k not in('pins','limitations')},indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
