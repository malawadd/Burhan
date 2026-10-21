"""Acquire all41 author-linked LCSD dark still pairs for FUTURE review only.

Freeze metadata protocol before pixels. No model/fit/inference or active data
change. Original author VAL/TEST may only be proposed as our TRAIN transfer;
no LCSD evaluation claim, fake video, guessed inversion or negative labels.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_preprocessing import camera_exposure
from construction_image_signatures import signature

OUT=ROOT/'datasets/construction/lcsd-low-light-future-r4'
REV='254e9a987d1ee9d925472a42750c6f5f58112d9f'
GHREV='49310ef4a6e6648d1a3c50284034883aaddce5e2'
HF=f'https://huggingface.co/datasets/Chrisathy/LCSD/resolve/{REV}/'
GH=f'https://raw.githubusercontent.com/zyaocoder/CrackNex/{GHREV}/'


def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):return digest(path.read_bytes())
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'LocalResearchDatasetAudit/1.0'})
    with urllib.request.urlopen(request,timeout=30) as response:return response.read()


def source_references():
    paths=set()
    native=json.loads((ROOT/'datasets/construction/evcivil-round4-broad-adapted-gain/native-input-manifest.json').read_text(encoding='utf8'))
    paths.update(r['raw_source_image'] for r in native)
    for row in json.loads((ROOT/'datasets/construction/evcivil-round3/exposures.json').read_text(encoding='utf8')):
        if row['split']=='heldout':paths.add(row['image'])
    backgrounds=json.loads((ROOT/'datasets/construction/evcivil-round3/background-import.json').read_text(encoding='utf8'))
    paths.update(r['derived_image'] for r in backgrounds['records'] if r['split']=='heldout')
    references=[]
    for value in sorted(paths):
        path=(ROOT/value).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise RuntimeError('Reference path unavailable/outside workspace')
        references.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    return references


def main():
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'asset-manifest.json').exists():raise RuntimeError('Refusing to repeat completed acquisition')
    info=json.loads(fetch(f'https://huggingface.co/api/datasets/Chrisathy/LCSD/revision/{REV}'))
    if info['sha']!=REV or info.get('cardData',{}).get('license')!='apache-2.0':raise RuntimeError('Pinned distribution/license declaration differs')
    readme=fetch(HF+'README.md');split=fetch(HF+'val.txt');loader=fetch(GH+'dataset/fewshot.py');author_readme=fetch(GH+'README.md')
    if b'huggingface.co/datasets/Chrisathy/LCSD' not in author_readme:raise RuntimeError('Actual official author link not verified')
    if b'#class 1: normal light crack; class 2: lowlight crack;' not in loader or b'mask_q[mask_q == cls] = 1' not in loader:
        raise RuntimeError('Reviewed author processed-mask convention changed')
    ids=[line.strip() for line in split.decode().splitlines() if line.strip()]
    if len(ids)!=41 or len(set(ids))!=41 or any('/' in name or '\\' in name for name in ids):raise RuntimeError('Exact source split differs')
    files={r['rfilename'] for r in info['siblings']}
    pairs=[{'source_id':name,'image':f'val/image/{name}.jpg','mask':f'val/mask/{name}.png'} for name in ids]
    if any(p[field] not in files for p in pairs for field in ['image','mask']):raise RuntimeError('Source pairing incomplete')
    for name,value in [('dataset-card.md',readme),('original-val.txt',split),('author-README.md',author_readme),('author-fewshot.py',loader)]:
        path=OUT/name
        if path.exists() and path.read_bytes()!=value:raise RuntimeError('Preserved source metadata changed')
        path.write_bytes(value)
    references=source_references();write(OUT/'known-reference-files.json',references)
    protocol={'schema_version':1,'frozen_before_pixel_acquisition':True,'status':'FUTURE ONLY; raw acquisition/proposal, no active import/fit',
        'hf_revision':REV,'author_code_revision':GHREV,'pairs':pairs,'count':41,
        'rights':'Author-linked dataset card explicitly declares Apache-2.0; this is distribution declaration, not independent clearance of unlisted source material',
        'source_card_sha256':digest(readme),'source_split_sha256':digest(split),'author_loader_sha256':digest(loader),'author_readme_sha256':digest(author_readme),
        'capture':'Author-described genuine iPad Pro campus low-light stills, released400x400; not video or proof every image is outdoor night',
        'repurposing':'Original author VAL/TEST proposed solely for our future TRAIN transfer. Zero LCSD test/validation accuracy claims. Entire ev-CIVIL CAL/heldout and original832/880 datasets remain untouched.',
        'selection':'All41 original val.txt pairs in original order, no model-dependent subset',
        'mask_contract':{'read':'Unconverted PIL array preserves indexed/grayscale source codes; original mask bytes retained',
            'direct_verified_processed_codes':{'0':'background','2':'low-light crack','255':'ignore'},
            'acceptance':'Derive proposed boxes ONLY if2D source codes subset0,2,255 and class2 present. Otherwise encoding_unresolved; no inversion/nonzero guess, no empty negative labels.',
            'component_box_rule':'Every8-connected class2 component, minimum1 nativepixel, no morphology/merging/topK. Tightinclusive-pixel bbox [xmin,ymin,xmax-xmin+1,ymax-ymin+1]. All original pixel masks retained.',
            'ignore_rule':'Any255ignore pixel makes sample ineligible for ordinary YOLO box training unless a separately declared ignore-aware trainer is used',
            'annotation_scope':'Crack-only masks; other damage/material status unverified; no invented sound/spall-negative labels'},
        'duplicate_screen':{'references_sha256':sha(OUT/'known-reference-files.json'),'references':len(references),
            'method':'Exact sourcefile and native gray+shape SHA,64bitdHash, normalized32x24gray correlation; flag dHash<=4 or<=8 withcorr>=.95; all within-source and vs knownmaterialized inputs',
            'scope':'Protected images read only for leakage screening, never NN/model selection; bounded sample screen cannot prove physical-scene independence'},
        'camera_policy':'Original RGB/raw bytes preserved; shared full-native grayp99/gain audited separately, no GTROI or spatial generation',
        'shared_preprocessing_source_sha256':sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py'),
        'visual_review':'Full native FIRST/MIDDLE/LAST raw/mask-code/overlay/gained sheets plus all-source overview before root accepts any later data version',
        'prohibitions':'No neural forward, immediate fitcopy, inferenceFPS, generatedstillloop, guessed mask inversion or labelnegative from empty/unresolved masks'}
    protocol_path=OUT/'protocol-before-pixels.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Frozen acquisition protocol changed')
    write(protocol_path,protocol);print('FROZEN BEFORE PIXELS',sha(protocol_path),flush=True)
    records=[];signatures=[]
    for pair in pairs:
        paths={}
        for field in ['image','mask']:
            path=OUT/'original'/pair[field];path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(fetch(HF+pair[field]))
            paths[field]=path
        frame=cv2.imread(str(paths['image']))
        with Image.open(paths['mask']) as image:mask_mode=image.mode;mask=np.array(image)
        if frame is None or frame.shape[:2]!=(400,400) or mask.shape[:2]!=(400,400):raise RuntimeError('Released image/mask geometry differs')
        values,counts=np.unique(mask,return_counts=True);codes={int(v):int(c) for v,c in zip(values,counts)}
        valid=mask.ndim==2 and set(codes)<={0,2,255} and 2 in codes
        boxes=[];ignore=int(np.count_nonzero(mask==255)) if mask.ndim==2 else None
        if valid:
            count,labels,stats,_=cv2.connectedComponentsWithStats((mask==2).astype(np.uint8),8)
            boxes=[{'component_id':index,'label':'crack','bbox':[int(x) for x in stats[index,:4]],'mask_pixels':int(stats[index,4])} for index in range(1,count)]
        corrected,gain=camera_exposure(frame,'low_light_gain');gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        records.append({**pair,'original_image_sha256':sha(paths['image']),'original_mask_sha256':sha(paths['mask']),
            'mask_mode':mask_mode,'mask_shape':list(mask.shape),'mask_code_counts':codes,'mask_encoding_status':'processed_class2_verified' if valid else 'encoding_unresolved_no_boxes',
            'candidate_component_boxes':boxes,'ordinary_box_training_eligible':bool(valid and ignore==0),
            'original_author_partition':'VAL/TEST','future_partition':'proposedTRAINonly_pendingrootreview',
            'source_shape_wh':[400,400],'gray_p99':float(np.percentile(gray,99)),'shared_gain':gain,
            'original_native_bgr_sha256':digest(frame.tobytes()),'actual_gained_native_bgr_sha256':digest(corrected.tobytes()),
            'ignore_pixels':ignore,'source':'https://huggingface.co/datasets/Chrisathy/LCSD','license':'Apache-2.0 declared by author-linked dataset card'})
        signatures.append(signature(frame))
    # No GT/NN is used for similarity. Known protected pixels are leakage checks only.
    reference_signatures=[signature(cv2.imread(str(ROOT/r['path']))) for r in references]
    packed=np.stack([s[1] for s in reference_signatures]);pop=np.array([n.bit_count() for n in range(256)],np.uint8)
    flags=[];exact=[]
    for record,sig in zip(records,signatures):
        distance=pop[np.bitwise_xor(packed,sig[1])].sum(axis=1)
        for index in np.flatnonzero(distance<=8):
            correlation=float(np.sum(sig[2]*reference_signatures[index][2]))
            is_exact=sig[0]==reference_signatures[index][0] or record['original_image_sha256']==references[index]['sha256']
            if is_exact:exact.append({'source_id':record['source_id'],'reference':references[index]})
            if is_exact or distance[index]<=4 or correlation>=.95:
                flags.append({'source_id':record['source_id'],'reference':references[index],'exact':is_exact,'dhash_hamming':int(distance[index]),'normalized_gray_correlation':correlation})
    intra=[]
    for i,left in enumerate(signatures):
        for j,right in enumerate(signatures[:i]):
            distance=int(pop[np.bitwise_xor(left[1],right[1])].sum());correlation=float(np.sum(left[2]*right[2]))
            if left[0]==right[0] or distance<=4 or(distance<=8 and correlation>=.95):
                intra.append({'left':records[i]['source_id'],'right':records[j]['source_id'],'exact':left[0]==right[0],'dhash_hamming':distance,'normalized_gray_correlation':correlation})
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    sheets=[]
    for index in sorted({0,len(records)//2,len(records)-1}):
        row=records[index];frame=cv2.imread(str(OUT/'original'/row['image']))
        with Image.open(OUT/'original'/row['mask']) as image:mask=np.array(image)
        color=np.zeros((400,400,3),np.uint8);overlay=frame.copy()
        if mask.ndim==2:
            color[mask==2]=(0,0,255);color[mask==255]=(255,0,255)
            overlay[mask!=0]=(overlay[mask!=0].astype(np.float32)*.5+color[mask!=0].astype(np.float32)*.5).astype(np.uint8)
        gained,gain=camera_exposure(frame,'low_light_gain');canvas=np.full((465,1600,3),24,np.uint8)
        labels=['RAW RELEASED STILL','SOURCE CODE2 RED /255 MAGENTA','RAW + SOURCE CODES (NOT MODEL)',f'ACTUAL SHARED INPUT gain{gain:.3f}']
        for k,(array,title) in enumerate(zip([frame,color,overlay,gained],labels)):
            canvas[50:450,k*400:(k+1)*400]=array
            cv2.putText(canvas,title,(k*400+5,38),cv2.FONT_HERSHEY_SIMPLEX,.40,(240,240,240),1)
        cv2.putText(canvas,f"FUTURE ONLY / {row['source_id']} / {row['mask_encoding_status']} / raw source masks, not predictions",(5,15),cv2.FONT_HERSHEY_SIMPLEX,.44,(240,240,240),1)
        path=qa/f'fixed-{index:02d}-{row["source_id"]}-raw-mask-input.jpg';cv2.imwrite(str(path),canvas);sheets.append(str(path.relative_to(OUT)))
    manifest={'protocol_sha256':sha(protocol_path),'status':'FUTURE ONLY; no training acceptance, activeimport, NN or LCSD evaluation',
        'records':records,'count':len(records),'mask_encodings_resolved':sum(r['mask_encoding_status']=='processed_class2_verified' for r in records),
        'ordinary_box_training_eligible':sum(r['ordinary_box_training_eligible'] for r in records),
        'gain_applied':sum(r['shared_gain']>1 for r in records),'exact_known_matches':exact,'known_similarity_flags':flags,'within_source_similarity_flags':intra,
        'known_reference_count':len(references),'fixed_native_raw_mask_sheets':sheets,
        'repurposing':protocol['repurposing'],'limits':'Still imagery only; no scene IDs, physical independence not proven, maskencoding and actual damage alignment require root visual acceptance before later proposal.'}
    write(OUT/'asset-manifest.json',manifest)
    print(json.dumps({'manifest_sha256':sha(OUT/'asset-manifest.json'),'pairs':len(records),'resolved_mask_encodings':manifest['mask_encodings_resolved'],
        'training_eligible_before_visualreview':manifest['ordinary_box_training_eligible'],'gain_applied':manifest['gain_applied'],'known_exact_matches':len(exact),'similarity_flags':len(flags)},indent=2),flush=True)


if __name__=='__main__':main()
