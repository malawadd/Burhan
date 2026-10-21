"""Read-only source/label hash review for a broader-material stage2 proposal.

No pixels are decoded or copied; active inputs, labels and fitting stay intact.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'datasets/construction/evcivil-spall-future-r4'
OUT=ROOT/'output/evaluation/construction-round4-broad-stage2-proposal.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path=SOURCE/'asset-manifest.json';review_path=SOURCE/'acceptance-review-v1.json'
    if sha(manifest_path)!='e98ecea5e66da8cc4ef3c09db0ff5f46320d110b09893d870ac9eeca250f8e26' or sha(review_path)!='4aaea999944cfc20495244b02f709ff210c26b51896b6be787dbebb9481c217a':
        raise RuntimeError('Acquisition/review identity changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'));review=json.loads(review_path.read_text(encoding='utf8'))
    allowed={'spalling_24':'render_over_masonry','spalling_27':'paving_and_lower_wall_mixed'}
    scopes={r['canonical_id']:r for r in review['source_acceptance'] if r['canonical_id'] in allowed}
    if set(scopes)!=set(allowed) or set(allowed)&set(review['reserved_canonical_ids']):raise RuntimeError('Material/source exclusion mismatch')
    for name,row in scopes.items():
        if row['acceptance']!='separate_material_scope_not_concrete_spall' or row['material_scope']!=allowed[name]:
            raise RuntimeError('Verified material review differs')
    additions=[r for r in manifest['exposures'] if r['canonical_id'] in allowed]
    if len(additions)!=48 or any(sum(r['canonical_id']==name for r in additions)!=24 for name in allowed):
        raise RuntimeError('Need exact48 originally selected real views')
    for row in additions:
        if row['source_group']!='dset_2':raise RuntimeError('Unexpected source capture folder')
        for field in ['image','label']:
            path=(SOURCE/row[field]).resolve()
            if not path.is_relative_to(SOURCE.resolve()) or sha(path)!=row[field+'_sha256']:
                raise RuntimeError('Exact reviewed source image/label differs')
    proposal={'schema_version':1,'status':'PROPOSED ONLY; requires parent review before distinct training-data/protocol freeze; no fit started',
        'source_manifest_sha256':sha(manifest_path),'visual_acceptance_sha256':sha(review_path),
        'source':'https://doi.org/10.11583/DTU.25924228','license':'ev-CIVIL CC BY4.0, priorDACL CC BY-NC4.0 and source-weight lineage restrictions retained',
        'active_head_protocol_sha256':sha(ROOT/'datasets/construction/evcivil-round4-gain/protocol.json'),
        'active_head_unchanged':True,'preserved_same_data_stage2_script':'scripts/freeze_construction_round4_stage2.py',
        'proposed_data_root':'datasets/construction/evcivil-round4-broad-adapted-gain',
        'proposed_candidate':'evcivil-yolov8n-r4-broad-adapted-640-gain',
        'trigger':'Only after complete head operating-point calibration requires further adaptation; no head/heldout feedback used to choose these48views',
        'semantic_scope':'Publisher-labelled visible surface material loss across reviewed concrete, render/masonry and paving. Not a concrete-cover-only qualification or structural-cause diagnosis; class0 remainsspalling andclass1 remainscrack.',
        'material_groups':[{k:v for k,v in scopes[name].items() if k!='images'} for name in sorted(allowed)],
        'additional_views':48,'additional_publisher_spalling_boxes':sum(len(r['boxes']) for r in additions),
        'expected_fit_counts':{'images':880,'actual_camera_views':740,'positive_photo_views':120,'unique_positive_photos':80,'negative_photos':20,
            'spalling_boxes':585,'crack_boxes':519},
        'excluded':['spalling_38 remains quarantined: ambiguous coating/bulge; no confirmed missing material'],
        'preserved_evaluation':'Original122CAL and heldout sources/GT untouched, whole source/global-ID boundaries and image-only input policy unchanged; all new48views stay fit-only',
        'preparation_contract':'Separate PNGs through same pinned camera_exposure exactlyonce; retain originalraw SHA/nativepixel hashes, gainednative SHA/gainfactor, copiedlabel SHA, originalframeindex/PTS, sourcecamera/annotation SHA. Never overwrite active832head inputs/protocol or same-data stage2history.',
        'split_limits':'Reviewed exact/near-duplicate checks found no confirmed shared visible damage layout; physical project independence remains unknown. dset1/2 training makes old exposed clips development only.',
        'fit_proposal':{'epochs':4,'freeze':4,'image_size':640,'exposure_policy':'low_light_gain','batch':4,'nominal_batch':4,'learning_rate':0.0005,'warmup':0,'wallcap':None,
            'start':'Retain actual completed head best.pt, no class reset. Source hash pinned only after head completion.',
            'memory_format_and_threads':'Select using previously declared4-trial TRAIN-only old-data probe; efficiency only, no accuracy choice'},
        'compute_order':'No probe/new fit until root releases after factoryD final pairedcapacity/runtime; freeze exact selected format/threads/start and newinputmanifest before fit',
        'reviewed_original_records':additions}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists() and json.loads(OUT.read_text(encoding='utf8'))!=proposal:raise RuntimeError('Refusing to replace different proposal')
    OUT.write_text(json.dumps(proposal,indent=2),encoding='utf8')
    print(json.dumps({'proposal':str(OUT.relative_to(ROOT)),'sha256':sha(OUT),'views':48,'spalling_boxes':proposal['additional_publisher_spalling_boxes'],'active_fit_changed':False},indent=2))


if __name__=='__main__':main()
