"""Record explicit bounded root visual acceptance; no copies, pixels or model work."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction/lcsd-low-light-future-r4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=DATA/'asset-manifest.json'
    if sha(source)!='6a138d4a26a84ec5ed2da7eef93b3b9519d3d745df1d7fc7428e54aef7478ce1':raise RuntimeError('Reviewed LCSD manifest changed')
    manifest=json.loads(source.read_text(encoding='utf8'))
    protocol=DATA/'protocol-before-pixels.json'
    if sha(protocol)!=manifest['protocol_sha256'] or len(manifest['records'])!=41:
        raise RuntimeError('Exact acquisition protocol/count changed')
    accepted=[]
    for record in manifest['records']:
        if record['mask_encoding_status']!='processed_class2_verified' or not record['ordinary_box_training_eligible'] or record['ignore_pixels']!=0:
            raise RuntimeError('Root accepted only directly verified0/2 foreground masks with no ignore')
        row=dict(record)
        for key in ['image','mask']:
            path=(DATA/'original'/record[key]).resolve()
            if not path.is_relative_to(DATA.resolve()) or sha(path)!=record[f'original_{key}_sha256']:
                raise RuntimeError('Accepted source image/mask identity changed')
            row[key]=str(path.relative_to(DATA))
        accepted.append(row)
    sheets=[{'path':name,'sha256':sha(DATA/name)} for name in manifest['fixed_native_raw_mask_sheets']]
    pair_image=DATA/'original/val/image/LCSD_0231.jpg'
    reference=ROOT/'datasets/construction/evcivil-round4/images/train/dset_1-spalling_13-frame00320.png'
    acceptance={'schema_version':1,'status':'Accepted future TRAIN transfer only; no active dataset import, new fit or independent evaluation',
        'reviewer':'Root agent explicit visual acceptance in parent chat,20October2026',
        'source_manifest_sha256':sha(source),'acquisition_protocol_sha256':sha(protocol),
        'source_card_sha256':sha(DATA/'dataset-card.md'),'rights':'Apache-2.0 declared by author-linked LCSD distribution; source lineage caveat retained',
        'bounded_visual_review':{'fixed_source_ids':['LCSD_0227','LCSD_0254','LCSD_0285'],'fixed_sheets':sheets,
            'finding':'Genuine visible cracks align with source masks; gain increases noise but does not alter geometry',
            'near_pair':{'source_id':'LCSD_0231','source_image_sha256':sha(pair_image),'reference':str(reference.relative_to(ROOT)),
                'reference_sha256':sha(reference),'finding':'Distinct diagonal branching crack layout versus mottled spall; no confirmed duplicate'},
            'all41_manually_reviewed':False,'physical_scene_independence_established':False,
            'scope':'Root accepted bounded3-source+flagged-pair review, not a claim of full manual41-image damage adjudication'},
        'mask_contract':'Original2D codes0background/2crack, no inversion; every8connected foregroundcomponent retained min1nativepixel, no morphology/merge; originalmasks preserved',
        'derived_box_count':sum(len(r['candidate_component_boxes']) for r in accepted),'accepted_count':41,
        'training_class_map':{'0':'spalling','1':'crack'},'original_author_partition':'VAL/TEST',
        'repurposing':'All41 used only as our futureTRAIN transfer. No LCSDtest/validation benchmark claim; ev-CIVIL CAL/heldout fixed.',
        'annotation_scope':'Crack-only masks, no invented negativeimage or claim other damage absent',
        'media_scope':'Actual released still photographs only; never converted to claimedrealvideo or generatedstillloop',
        'accepted_records':accepted}
    target=DATA/'acceptance-v1.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8'))!=acceptance:raise RuntimeError('Refusing to replace different acceptance')
    target.write_text(json.dumps(acceptance,indent=2),encoding='utf8')
    print(json.dumps({'acceptance_sha256':sha(target),'accepted':41,'proposed_component_boxes':acceptance['derived_box_count'],'fit_started':False},indent=2))


if __name__=='__main__':main()
