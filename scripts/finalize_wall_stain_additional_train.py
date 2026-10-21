"""Conservative eligibility refinement: include quarantined TEST exposure; no fit."""
from pathlib import Path
import hashlib,json,sys
from collections import Counter
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_mbdd_wall_training as source
BASE=ROOT/'datasets/construction/expansion-public-v1'
V1=BASE/'wall-stain-additional-train-v1'
OUT=BASE/'wall-stain-additional-train-v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if (OUT/'prepared/manifest.json').exists():raise ValueError('Preserve final eligibility manifest')
    cv2.setNumThreads(1)
    prior=json.loads((V1/'prepared/manifest.json').read_text());old=json.loads((BASE/'wall-stain/prepared/manifest.json').read_text());selection_path=BASE/'wall-stain/protocol-before-new-photo-acquisition.json';exposed=json.loads(selection_path.read_text())
    oldids={r['id']for r in old['records']};newrows=[r for r in prior['records']if r['id']not in oldids]
    all_test_blocks={r['capture_block_200']for r in exposed['selection']}
    pre={'rule':'Exclude newTRAIN from ALL120 previouslyselectedTEST blocks, including22 subsequentlyquarantined views; thenremove originalSHA/pHash<=6 duplicatesofoldTRAIN. No replacement/backfill or modelinference.',
         'old_manifest_sha256':sha(BASE/'wall-stain/prepared/manifest.json'),'acquisition_manifest_sha256':sha(V1/'prepared/manifest.json'),'all120test_selection_sha256':sha(selection_path),'source_script_sha256':sha(__file__),'all_test_blocks':sorted(all_test_blocks),'new_candidate_ids':[r['id']for r in newrows],
         'why_followup':'Initial acquisition excluded surviving98TEST blocks; this stricter eligibility review also protects22 quarantinedTEST source groups. Initial pre-pixel source selection is retained.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'protocol-before-eligibility-refinement.json').write_text(json.dumps(pre,indent=2)+'\n')
    trainrefs=[]
    for r in old['records']:
        if r['split']!='train':continue
        p=ROOT/r['original'];trainrefs.append({'id':r['id'],'sha256':sha(p),'phash63':source.visual_hash(cv2.imread(str(p)))})
    quarantine=[];accepted=[]
    for r in newrows:
        if r['capture_block_200']in all_test_blocks:
            quarantine.append({'id':r['id'],'reason':'blockexposedbyold120TESTselection','block':r['capture_block_200']});continue
        hit=None
        for other in trainrefs:
            dist=sum(a!=b for a,b in zip(r['phash63'],other['phash63']))
            if r['original_sha256']==other['sha256']or dist<=6:
                hit={'id':r['id'],'reason':'near_duplicate_of_existingTRAIN','reference':other['id'],'phash_distance':dist};break
        if hit:quarantine.append(hit)
        else:accepted.append(r)
    records=old['records']+accepted
    result={**{k:v for k,v in old.items()if k not in ['records','split_counts','positive_images']},'records':records,'split_counts':dict(Counter(r['split']for r in records)),'positive_images':dict(Counter(r['split']for r in records if r['boxes'])),
        'additional_training':{'accepted_images':len(accepted),'positive':sum(bool(r['boxes'])for r in accepted),'other_defect_negative':sum(not r['boxes']for r in accepted),'blocks':sorted({r['capture_block_200']for r in accepted}),'new_ids':[r['id']for r in accepted],'cal_test_records_unchanged':True,'no_independence_gain_claimed':True,'eligibility_quarantine':quarantine,'acquisition_quarantine':prior['additional_training']['quarantine']},
        'addition_protocol_sha256':sha(V1/'protocol-before-photo-acquisition.json'),'eligibility_protocol_sha256':sha(OUT/'protocol-before-eligibility-refinement.json'),'validation_status':'research_only'}
    (OUT/'prepared').mkdir(exist_ok=True);(OUT/'prepared/manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['split_counts','positive_images','additional_training']}),flush=True)
if __name__=='__main__':main()
