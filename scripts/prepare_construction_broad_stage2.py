"""Prepare approved generic surface-loss material coverage; no training or inference.

The active832-image head dataset and every evaluation source/label stay intact.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_preprocessing import camera_exposure
OLD=ROOT/'datasets/construction/evcivil-round4-gain'
NEW=ROOT/'datasets/construction/evcivil-round4-broad-adapted-gain'
SOURCE=ROOT/'datasets/construction/evcivil-spall-future-r4'
PROPOSAL=ROOT/'output/evaluation/construction-round4-broad-stage2-proposal.json'
PROPOSAL_SHA='7771c7d92452d70003149219cbf4cca848f95fa39df4144be6fced9ea0d035de'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')


def remap(value):
    if isinstance(value,dict):return {key:remap(item) for key,item in value.items()}
    if isinstance(value,list):return [remap(item) for item in value]
    if isinstance(value,str):
        for prefix in [str(OLD.relative_to(ROOT)),OLD.relative_to(ROOT).as_posix()]:
            if value.startswith(prefix+'\\') or value.startswith(prefix+'/'):
                return str(NEW.relative_to(ROOT)/Path(value).relative_to(Path(prefix)))
    return value


def main():
    cv2.setNumThreads(1)
    if sha(PROPOSAL)!=PROPOSAL_SHA:raise RuntimeError('Approved material proposal changed')
    proposal=json.loads(PROPOSAL.read_text(encoding='utf8'))
    if sha(OLD/'protocol.json')!=proposal['active_head_protocol_sha256']:raise RuntimeError('Active head identity changed')
    if sha(SOURCE/'asset-manifest.json')!=proposal['source_manifest_sha256'] or sha(SOURCE/'acceptance-review-v1.json')!=proposal['visual_acceptance_sha256']:
        raise RuntimeError('Reviewed source provenance changed')
    prepared=json.loads((OLD/'data-protocol.json').read_text(encoding='utf8'))
    shared=ROOT/'edge_runtime/profiles/construction_preprocessing.py'
    if sha(shared)!=prepared['preprocessing_source_sha256'] or sha(OLD/'native-input-manifest.json')!=prepared['native_input_manifest_sha256']:
        raise RuntimeError('Pinned shared gain or original native manifest changed')
    native=json.loads((OLD/'native-input-manifest.json').read_text(encoding='utf8'))
    additions=proposal['reviewed_original_records']
    if len(native)!=954 or len(additions)!=48 or {r['canonical_id'] for r in additions}!={'spalling_24','spalling_27'}:
        raise RuntimeError('Exact approved input count/source set changed')
    for row in native:
        for field in ['image','label']:
            source=(ROOT/row[field]).resolve()
            if not source.is_relative_to(OLD.resolve()) or sha(source)!=row[field+'_sha256']:raise RuntimeError('Existing head input/label changed')
    for row in additions:
        for field in ['image','label']:
            source=(SOURCE/row[field]).resolve()
            if not source.is_relative_to(SOURCE.resolve()) or sha(source)!=row[field+'_sha256']:raise RuntimeError('Reviewed additional source/label changed')
    if NEW.exists():raise RuntimeError('Refusing to replace any previously prepared broad-material version')
    NEW.mkdir()
    for row in native:
        for field in ['image','label']:
            destination=ROOT/remap(row[field]);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/row[field],destination)
            if sha(destination)!=row[field+'_sha256']:raise RuntimeError('Byte-identical original input copy failed')
    records=remap(native);exposures=remap(json.loads((OLD/'exposures.json').read_text(encoding='utf8')))
    cal_raw_bgr={r['raw_native_bgr_sha256'] for r in native if r['split']=='calibration'}
    held_gray={r['sha256_pixels'] for r in exposures if r['split']=='heldout' and 'sha256_pixels' in r}
    material={'spalling_24':'render_over_masonry','spalling_27':'paving_and_lower_wall_mixed'}
    for row in additions:
        source=SOURCE/row['image'];label=SOURCE/row['label'];raw=cv2.imread(str(source))
        if raw is None or raw.shape[:2]!=tuple(row['source_shape']):raise RuntimeError('Original additional shape differs')
        raw_sha=hashlib.sha256(raw.tobytes()).hexdigest();gray_sha=hashlib.sha256(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).tobytes()).hexdigest()
        if raw_sha in cal_raw_bgr or gray_sha in held_gray:raise RuntimeError('Exact additional/evaluation native pixel overlap')
        if gray_sha!=row['pixel_sha256']:raise RuntimeError('Original publisher image pixels changed')
        corrected,gain=camera_exposure(raw,'low_light_gain')
        image_target=NEW/'images/train'/source.name;label_target=NEW/'labels/train'/label.name
        if image_target.exists() or label_target.exists():raise RuntimeError('Additional source collides with existing sample identity')
        cv2.imwrite(str(image_target),corrected);shutil.copy2(label,label_target)
        if not np.array_equal(cv2.imread(str(image_target)),corrected) or sha(label_target)!=row['label_sha256']:
            raise RuntimeError('Prepared additional pixel/label parity failed')
        records.append({'split':'train','raw_source_image':str(source.relative_to(ROOT)),'raw_source_file_sha256':sha(source),
            'raw_native_bgr_sha256':raw_sha,'image':str(image_target.relative_to(ROOT)),'image_sha256':sha(image_target),
            'label':str(label_target.relative_to(ROOT)),'label_sha256':sha(label_target),'gain':gain,
            'native_model_input_bgr_sha256':hashlib.sha256(corrected.tobytes()).hexdigest(),'shape_wh':[raw.shape[1],raw.shape[0]],
            'source_canonical_id':row['canonical_id'],'material_scope':material[row['canonical_id']],
            'original_frame_index':row['original_frame_index'],'original_timestamp_us':row['timestamp_us'],
            'source_camera_sha256':row['camera_source_sha256'],'source_annotation_sha256':row['label_source_sha256']})
        exposures.append({'split':'train','source_group':row['source_group'],'global_defect_id':row['canonical_id'],
            'canonical_defect_id':row['canonical_id'],'original_frame_index':row['original_frame_index'],
            'original_timestamp_us':row['timestamp_us'],'source_frame_period_us':row['period_us'],
            'image':str(image_target.relative_to(ROOT)),'sha256_pixels':gray_sha,'boxes':row['boxes'],
            'material_scope':material[row['canonical_id']],'source_camera_sha256':row['camera_source_sha256'],
            'source_annotation_sha256':row['label_source_sha256'],'input_exposure_gain':gain})
    if sum(r['split']=='train' for r in records)!=880 or sum(r['split']=='calibration' for r in records)!=122:
        raise RuntimeError('Prepared split counts differ')
    for split,count in [('train',880),('calibration',122)]:
        if len(list((NEW/f'images/{split}').glob('*.png')))!=count or len(list((NEW/f'labels/{split}').glob('*.txt')))!=count:
            raise RuntimeError('Unexpected stale/missing prepared file count')
    write(NEW/'native-input-manifest.json',records);write(NEW/'exposures.json',exposures)
    write(NEW/'background-import.json',remap(json.loads((OLD/'background-import.json').read_text(encoding='utf8'))))
    shutil.copy2(OLD/'night-holdout-protocol.json',NEW/'night-holdout-protocol.json')
    dataset=yaml.safe_load((OLD/'dataset.yaml').read_text(encoding='utf8'))
    dataset.update(path=NEW.as_posix(),train=(NEW/'images/train').as_posix(),val=(NEW/'images/calibration').as_posix())
    (NEW/'dataset.yaml').write_text(yaml.safe_dump(dataset),encoding='utf8')
    protocol={'schema_version':1,'status':'Prepared only; conditional model/training protocol still not frozen, no new fit started',
        'approved_proposal_sha256':PROPOSAL_SHA,'parent_head_protocol_sha256':sha(OLD/'protocol.json'),
        'parent_prepared_data_protocol_sha256':sha(OLD/'data-protocol.json'),'parent_native_manifest_sha256':sha(OLD/'native-input-manifest.json'),
        'policy':'low_light_gain','preprocessing_source_sha256':sha(shared),
        'native_input_manifest_sha256':sha(NEW/'native-input-manifest.json'),'exposures_sha256':sha(NEW/'exposures.json'),
        'semantic_scope':proposal['semantic_scope'],'new_material_groups':material,'additional_real_views':48,'excluded':'Ambiguousspalling38 remains quarantined',
        'counts':{split:{'images':sum(r['split']==split for r in records),'gain_applied':sum(r['split']==split and r['gain']>1 for r in records)} for split in ['train','calibration']},
        'preserved_evaluation':proposal['preserved_evaluation'],'split_limits':proposal['split_limits'],
        'material_source_manifest_sha256':proposal['source_manifest_sha256'],'material_acceptance_review_sha256':proposal['visual_acceptance_sha256'],
        'classes':{'0':'spalling','1':'crack'},'license':proposal['license'],'no_new_fit_or_heldout_predictions':True}
    write(NEW/'data-protocol.json',protocol)
    summary={'prepared_data_protocol_sha256':sha(NEW/'data-protocol.json'),'native_input_manifest_sha256':sha(NEW/'native-input-manifest.json'),
        'exposures_sha256':sha(NEW/'exposures.json'),'training_images':880,'calibration_images':122,
        'new_real_source_views':48,'active_head_changed':False,'new_fit_started':False}
    write(NEW/'preparation-summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
