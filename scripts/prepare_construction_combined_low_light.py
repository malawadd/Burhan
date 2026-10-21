"""Prepare explicitly approved broad+actual-dark source version; no NN or fit."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from construction_native_dataset import NativeDataset,ROOT,sha,write_json

BASE=ROOT/'datasets/construction/evcivil-round4-broad-adapted-gain'
TARGET=ROOT/'datasets/construction/evcivil-round4-real-lowlight-gain'
DEC=ROOT/'datasets/construction/evcivil-dec31-future'
LCSD=ROOT/'datasets/construction/lcsd-low-light-future-r4'


def main():
    cv2.setNumThreads(1)
    dec_path=DEC/'acceptance-v1.json';lcsd_path=LCSD/'acceptance-v1.json'
    if sha(dec_path)!='e3afb0d2d765238a7a0c240198a89775122746cfa1aec597aa0c26fec6a2bf7d' or sha(lcsd_path)!='28b13decfb62bd695dcc5a68af3cc7cbf5874191108e9184342d943d218e355f':
        raise RuntimeError('Explicit reviewed source acceptance changed')
    dec=json.loads(dec_path.read_text(encoding='utf8'));lcsd=json.loads(lcsd_path.read_text(encoding='utf8'))
    if len(dec['accepted_views'])!=250 or len(lcsd['accepted_records'])!=41 or lcsd['derived_box_count']!=92:
        raise RuntimeError('Exact accepted source counts differ')
    if sha(DEC/'asset-manifest.json')!=dec['source_manifest_sha256'] or sha(LCSD/'asset-manifest.json')!=lcsd['source_manifest_sha256']:
        raise RuntimeError('Accepted original source manifests differ')
    plan={'schema_version':1,'source_acceptances_pinned_before_combined_copy':True,'durable_plan_written_before_added_sources':True,'status':'Prepared data only, no candidate-fit protocol or training',
        'base_data_protocol_sha256':'1da0533a85aa7f322a90e8ea87ae1f6b27a28b148cc04ca1523793d2cde1e935',
        'dec31_acceptance_sha256':sha(dec_path),'dec31_source_manifest_sha256':dec['source_manifest_sha256'],
        'dec31_accepted_ids':dec['accepted_source_ids'],'dec31_quarantined_ids':dec['quarantined_source_ids'],
        'lcsd_acceptance_sha256':sha(lcsd_path),'lcsd_source_manifest_sha256':lcsd['source_manifest_sha256'],
        'lcsd_ids':[r['source_id'] for r in lcsd['accepted_records']],
        'intended_counts':{'train':1171,'calibration':122,'actual_camera_train':990,'photo_views_train':161,'unique_photos':121,'negative_photos':20,
            'spalling_boxes':585,'crack_boxes':1016},
        'annotation_scope':'Original publisher camera boxes plus disclosed DACL polygon/LCSD component-derived regions. No native object/pixelmetric interchange.',
        'lcsd_box_rule':lcsd['mask_contract'],'lcsd_repurposing':lcsd['repurposing'],
        'preserved':'All original832/880 roots/history and original122CAL/heldout identities/GT remain unchanged. New sources stayTRAIN only.',
        'limits':'Source-specific visual samples/near-pair review; physical-project independence unproven. Original exposed folders are development only; LCSD is still imagery, never fabricated video.',
        'license':'ev-CIVIL CC BY4.0; LCSD declaredApache2.0; existingDACL CC BY-NC4.0/sourceweightsAGPL lineage restrictions retained',
        'fit_condition':'Only after failed R4head minimumguard, boundedTRAINlayoutprobe, explicit rootresource release and a distinct fixed4epochfreeze4/640/gain trainingprotocol'}
    # Validate positive source/mask identities before any output is created.
    for row in dec['accepted_views']:
        path=(DEC/row['image']).resolve()
        if not path.is_relative_to(DEC.resolve()) or sha(path)!=row['image_sha256']:raise RuntimeError('Accepted camera PNG differs')
    for row in lcsd['accepted_records']:
        for key in ['image','mask']:
            path=(LCSD/row[key]).resolve()
            if not path.is_relative_to(LCSD.resolve()) or sha(path)!=row[f'original_{key}_sha256']:raise RuntimeError('Accepted LCSD pair differs')
    builder=NativeDataset(BASE,TARGET,plan['base_data_protocol_sha256'],preparation_plan=plan)
    for row in dec['accepted_views']:
        metadata={'source_group':row['source_group'],'sequence':row['sequence'],'global_defect_id':row['canonical_id'],
            'canonical_defect_id':row['canonical_id'],'original_frame_index':row['original_frame_index'],
            'original_timestamp_us':row['timestamp_us'],'source_pts_us':row['source_pts_us'],
            'source_camera_sha256':row['camera_source_sha256'],'source_annotation_sha256':row['annotation_source_sha256'],
            'media_kind':'actual_camera_exposure'}
        builder.add(DEC/row['image'],row['image_sha256'],row['boxes'],'night-dec31-'+Path(row['image']).stem,
            row['annotation_scope'],metadata,expected_gain=row['gain'],expected_gray_sha=row['pixel_sha256'])
    for row in lcsd['accepted_records']:
        with Image.open(LCSD/row['mask']) as image:mask=np.array(image)
        if mask.shape!=(400,400) or not set(np.unique(mask))<={0,2} or not np.any(mask==2):raise RuntimeError('Direct original LCSD0/2 encoding differs')
        n,labels,stats,_=cv2.connectedComponentsWithStats((mask==2).astype(np.uint8),8)
        actual=[{'component_id':i,'label':'crack','bbox':[int(v) for v in stats[i,:4]],'mask_pixels':int(stats[i,4])} for i in range(1,n)]
        if actual!=row['candidate_component_boxes']:raise RuntimeError('Exact proposed all-component mask boxes differ')
        metadata={'source_group':'LCSD_authorVAL_repurposedTRAIN','global_defect_id':'lcsd:'+row['source_id'],
            'original_frame_index':None,'source_photo_id':row['source_id'],'source_mask':str((LCSD/row['mask']).relative_to(ROOT)),
            'source_mask_sha256':row['original_mask_sha256'],'original_author_partition':'VAL/TEST',
            'media_kind':'actual_still_photograph','license':'Apache2.0 declared by author-linked dataset card'}
        builder.add(LCSD/row['image'],row['original_image_sha256'],actual,'lcsd-lowlight-'+row['source_id'],
            'Original class2 crack pixelmask -> every8connected min1pixel tightcomponentbox; no other-damage/negative claim; no LCSDbenchmark',metadata,expected_gain=row['shared_gain'])
    result=builder.finish({'schema_version':1,'status':'Prepared only; no candidate training protocol/fit started',
        'preparation_plan_sha256':sha(TARGET/'preparation-plan.json'),'base_prepared_data_protocol_sha256':plan['base_data_protocol_sha256'],
        'dec31_acceptance_sha256':sha(dec_path),'lcsd_acceptance_sha256':sha(lcsd_path),
        'dec31_accepted_ids':plan['dec31_accepted_ids'],'dec31_quarantined_ids':plan['dec31_quarantined_ids'],
        'lcsd_source_ids':plan['lcsd_ids'],'counts_by_source':plan['intended_counts'],
        'annotation_scope':plan['annotation_scope'],'lcsd_box_rule':plan['lcsd_box_rule'],'lcsd_repurposing':plan['lcsd_repurposing'],
        'preserved_evaluation':plan['preserved'],'independence_limits':plan['limits'],'license':plan['license'],
        'classes':{'0':'spalling','1':'crack'},'no_new_fit_or_heldout_predictions':True},1171)
    write_json(TARGET/'preparation-summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
