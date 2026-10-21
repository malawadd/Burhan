"""Prepare accepted real spall training expansion only; never start a fit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import yaml

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'datasets/construction/evcivil-round3-continuation'
SOURCE=ROOT/'datasets/construction/evcivil-spall-future-r4'
NEW=ROOT/'datasets/construction/evcivil-round4'
MANIFEST_SHA='e98ecea5e66da8cc4ef3c09db0ff5f46320d110b09893d870ac9eeca250f8e26'
ACCEPTANCE_SHA='4aaea999944cfc20495244b02f709ff210c26b51896b6be787dbebb9481c217a'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    manifest_path=SOURCE/'asset-manifest.json';review_path=SOURCE/'acceptance-review-v1.json'
    if sha(manifest_path)!=MANIFEST_SHA or sha(review_path)!=ACCEPTANCE_SHA:
        raise RuntimeError('Frozen acquisition/review identity mismatch')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    review=json.loads(review_path.read_text(encoding='utf8'))
    accepted={r['canonical_id']:r for r in review['source_acceptance']if r['acceptance']=='eligible_for_future_train_only'}
    if len(accepted)!=9 or any(r['material_scope']!='concrete_surface'for r in accepted.values()):
        raise RuntimeError('Need exactly9 reviewed concrete capture groups; ambiguous/masonry/coating data excluded')
    prior=json.loads((OLD/'exposures.json').read_text(encoding='utf8'))
    reserved=set(review['reserved_canonical_ids'])
    if set(accepted)&reserved:raise RuntimeError('Reserved canonical source identity overlap')
    additions=[r for r in manifest['exposures']if r['canonical_id']in accepted]
    if len(additions)!=216:raise RuntimeError('Need all216 frozen selected exposures')
    protocol={'schema_version':1,'status':'PREPARED DATA ONLY; model/training protocol and compute handoff still required',
        'frozen_before_future_fit':True,'source_manifest_sha256':MANIFEST_SHA,'acceptance_review_sha256':ACCEPTANCE_SHA,
        'previous_exposures_sha256':sha(OLD/'exposures.json'),'previous_protocol_sha256':sha(OLD/'protocol.json'),
        'selected_canonical_ids':sorted(accepted),'selected_views':216,'classes':{'0':'spalling','1':'crack'},
        'excluded_materials':['spalling_38: ambiguous shallow/coating appearance','spalling_24 and27: render/masonry/paving, separate scope'],
        'scope_change':'dset1/2 now allowed ONLY in distinct future fitting experiment. All old exposed clips become development diagnostics; no independent generalization credit from those source folders.',
        'preserved_evaluation':'Originaldset6 calibration, dset5 heldout, Nov18 calibrationnight, Jan2 heldoutnight and same20/10/10 true-negative split unchanged. Whole capture/globalID exclusions retained.',
        'independence':'No exact duplicate or same visible layout confirmed in121 reviewed flagged pairs; physical project independence and unseen-frame near-duplicates remain unknown.',
        'licenses':'ev-CIVILCCBY4.0, priorDACLdataCCBYNC4.0, initialUltralyticsAGPL/sourceweight lineage caveats remain',
        'no_heldout_predictions_opened':True}
    NEW.mkdir(exist_ok=True,parents=True);target=NEW/'data-protocol.json'
    if target.exists()and json.loads(target.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Frozen futuredata selection changed')
    target.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    images=NEW/'images/train';labels=NEW/'labels/train';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
    cross_split={r['sha256_pixels']for r in prior if r['split']!='train'and 'sha256_pixels'in r}
    result=list(prior)
    for row in additions:
        src=(SOURCE/row['image']).resolve();lab=(SOURCE/row['label']).resolve()
        if not src.is_relative_to(SOURCE.resolve())or not lab.is_relative_to(SOURCE.resolve()):raise RuntimeError('Source outside accepted root')
        if sha(src)!=row['image_sha256']or sha(lab)!=row['label_sha256']:raise RuntimeError('Accepted image/label bytes changed')
        frame=cv2.imread(str(src),cv2.IMREAD_GRAYSCALE)
        pixel_sha=hashlib.sha256(frame.tobytes()).hexdigest()
        if pixel_sha in cross_split:raise RuntimeError('Exact cross-evaluation pixel duplicate')
        if row['source_group']not in {'dset_1','dset_2'}:raise RuntimeError('Unexpected capturegroup')
        image_target=images/src.name;label_target=labels/lab.name
        for source,dest in [(src,image_target),(lab,label_target)]:
            if dest.exists()and sha(dest)!=sha(source):raise RuntimeError('Refusing to replace different prepared sample')
            shutil.copyfile(source,dest)
        result.append({'source_group':row['source_group'],'sequence':row['canonical_id'],'global_defect_id':row['canonical_id'],
                       'split':'train','original_frame_index':row['original_frame_index'],'timestamp_us':row['timestamp_us'],
                       'image':str(image_target.relative_to(ROOT)),'sha256_pixels':pixel_sha,'boxes':row['boxes'],
                       'source_camera_sha256':row['camera_source_sha256'],'source_annotation_sha256':row['label_source_sha256'],
                       'material_scope':'Reviewed concrete surface loss; publisher classspalling; severity/physicalsite unverified'})
    if len(list(images.glob('*.png')))!=216 or len(list(labels.glob('*.txt')))!=216:raise RuntimeError('Stale/unexpected prepared sample count')
    (NEW/'exposures.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for name in ['background-import.json','night-holdout-protocol.json']:
        shutil.copyfile(OLD/name,NEW/name)
    data=yaml.safe_load((OLD/'dataset.yaml').read_text(encoding='utf8'))
    old_train=data['train']if isinstance(data['train'],list)else[data['train']]
    data['train']=old_train+[images.as_posix()]
    (NEW/'dataset.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf8')
    summary={'status':'Prepared only; no fit started','new_training_views':216,'total_training_images':832,
             'actual_video_training_exposures':692,'positive_photographic_views':120,'unique_positive_photos':80,'true_negative_train_photos':20,
             'data_protocol_sha256':sha(target),'exposures_sha256':sha(NEW/'exposures.json'),
             'unchanged_val':data['val'],'unchanged_test':data['test']}
    (NEW/'preparation-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
