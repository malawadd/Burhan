"""Freeze approved broader-material adaptation after actual head calibration/probe.

No inference or training. Parent must release compute after factory timing before
the resulting four-epoch fit starts. Active head and same-data proposal stay intact.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction/evcivil-round4-broad-adapted-gain'
HEAD_DATA=ROOT/'datasets/construction/evcivil-round4-gain'
HEAD=ROOT/'models/construction/evcivil-yolov8n-r4-head-640-gain.onnx'
NAME='evcivil-yolov8n-r4-broad-adapted-640-gain'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_candidate(data=DATA, name=NAME, expected_data_sha='1da0533a85aa7f322a90e8ea87ae1f6b27a28b148cc04ca1523793d2cde1e935', combined=False):
    data=Path(data).resolve()
    meta=json.loads(HEAD.with_suffix('.provenance.json').read_text(encoding='utf8'))
    op_path=HEAD.with_suffix('.operating-point.json');op=json.loads(op_path.read_text(encoding='utf8'))
    guard_path=HEAD_DATA/'stage2-continuation-guard.json';decision=op.get('minimum_continuation_guard',{})
    if sha(HEAD)!='33e7ab167a93c7ee9c9e80e544951ec2e61fcdaf977377ea56e321dd4fea41ad' or sha(HEAD)!=meta['onnx_sha256'] or op['model_sha256']!=sha(HEAD) or meta['training']['completed_epochs']!=4:
        raise RuntimeError('Complete exact four-epoch head/calibration required')
    if not op.get('serving_decoder_parity_verified') or not decision.get('requires_distinct_stage2') or decision.get('protocol_sha256')!=sha(guard_path):
        raise RuntimeError('Exact complete-calibration continuation guard must require stage2')
    if sha(HEAD_DATA/'protocol.json')!=meta['split_protocol_sha256']:
        raise RuntimeError('Head training protocol changed')
    checkpoint=(Path(meta['save_dir'])/'weights/best.pt').resolve()
    if not checkpoint.is_relative_to((ROOT/'output/training/construction-round3').resolve()) or not checkpoint.is_file():
        raise RuntimeError('Only actual locally learned head can seed adaptation')
    if sha(checkpoint)!='5e48033ee0122d39a3a6301aa926dad6bd18b141e89468185a66d1871a5eef6c':
        raise RuntimeError('The exact retained four-epoch head checkpoint is required')
    capacity_path=ROOT/'output/training/construction-round4-layout-capacity/summary.json'
    capacity=json.loads(capacity_path.read_text(encoding='utf8'))
    if capacity['source_checkpoint_sha256']!=sha(checkpoint) or capacity['selected_memory_format'] not in ['contiguous','channels_last'] or capacity['selected_cpu_threads'] not in [3,4]:
        raise RuntimeError('Actual learned head/layout probe identity mismatch')
    if sha(data/'data-protocol.json')!=expected_data_sha:
        raise RuntimeError('Approved prepared broad-material dataset changed')
    prepared=json.loads((data/'data-protocol.json').read_text(encoding='utf8'))
    if not combined and prepared['approved_proposal_sha256']!='7771c7d92452d70003149219cbf4cca848f95fa39df4144be6fced9ea0d035de':
        raise RuntimeError('Broader-material scope/proposal differs')
    if sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py')!=prepared['preprocessing_source_sha256'] or sha(data/'native-input-manifest.json')!=prepared['native_input_manifest_sha256']:
        raise RuntimeError('Prepared camera-input contract changed')
    if sha(data/'exposures.json')!=prepared['exposures_sha256']:
        raise RuntimeError('Prepared source exposure manifest changed')
    rows=json.loads((data/'native-input-manifest.json').read_text(encoding='utf8'))
    train_count=1171 if combined else 880
    if len(rows)!=train_count+122 or sum(r['split']=='train' for r in rows)!=train_count or sum(r['split']=='calibration' for r in rows)!=122:
        raise RuntimeError('Exact prepared split counts differ')
    # Preparation verified all bytes. This freezer verifies manifest identity and
    # safe paths only; the actual trainer rehashes every image/label before fit.
    for row in rows:
        for field in ['image','label']:
            path=(ROOT/row[field]).resolve()
            if not path.is_relative_to(data) or not row.get(field+'_sha256'):
                raise RuntimeError('Prepared image/label path or hash is invalid')
    config={'epochs':4,'image_size':640,'batch':4,'nominal_batch':4,'threads':capacity['selected_cpu_threads'],
        'memory_format':capacity['selected_memory_format'],'freeze':4,'learning_rate':.0005,
        'max_minutes':0.0,'fraction':1.0,'name':name,'data_root':str(data.relative_to(ROOT)),
        'continue_from':str(checkpoint.relative_to(ROOT)),'exposure_policy':'low_light_gain'}
    parent=json.loads((HEAD_DATA/'protocol.json').read_text(encoding='utf8'))
    protocol={'schema_version':1,'candidate':name,'stage':'distinct4_epoch_real_lowlight_material_adaptation' if combined else 'distinct4_epoch_broader_material_backbone_adaptation',
        'frozen_before_candidate_training':True,'strict_configuration':True,'configuration':config,
        'training_script_sha256':sha(ROOT/'scripts/train_construction_damage_round3.py'),
        'freeze_helper_sha256':sha(Path(__file__)),
        'starting_checkpoint_sha256':sha(checkpoint),'head_model_sha256':sha(HEAD),'head_operating_point_sha256':sha(op_path),
        'continuation_guard_sha256':sha(guard_path),'parent_head_protocol_sha256':sha(HEAD_DATA/'protocol.json'),
        'layout_capacity_summary_sha256':sha(capacity_path),'layout_capacity_protocol_sha256':capacity['protocol_sha256'],
        'layout_probe_scope':'Four oldTRAIN examples only, no new material/CAL accuracy selection; capacity choice only',
        'prepared_data_protocol_sha256':sha(data/'data-protocol.json'),'native_input_manifest_sha256':sha(data/'native-input-manifest.json'),
        'exposures_sha256':sha(data/'exposures.json'),'preprocessing_source_sha256':prepared['preprocessing_source_sha256'],
        'prepared_sample_verification':'Preparation verified sample bytes; trainer must reverify every native image/label hash before optimizer work.',
        'approved_material_proposal_sha256':prepared.get('approved_proposal_sha256'),
        'semantic_scope':prepared.get('semantic_scope','Publisher-labelled surface material loss, including reviewed render/masonry/paving, plus visible cracks. Verify substrate; no structural diagnosis.'),'material_groups':prepared.get('new_material_groups',['concrete surface loss','render over masonry','paving/lower wall']),
        'source_basis':'Retain all learned class heads; adapt blocks4 onward. Add only48 separately reviewed genuine render/masonry/paving TRAIN views, no ambiguous38, no evaluation label changes.',
        'selection':'Four fixed epochs, no wall cap; best checkpoint by original122CAL identities/GT through exact sharedgain. Exact raw-source operating-point search and serving parity before heldout.',
        'validation_scope':parent['validation_scope']+' Task now explicitly includes verified render/masonry/paving surface loss; material-specific scores and physical-project independence remain limited.',
        'licenses':parent['licenses'],'resource_requirement':'Start only after explicit root release following final factory timing. Selected3/4CPU layout recorded; actual contention disclosed.'}
    if combined:
        source_pins={
            'broad_data':(DATA/'data-protocol.json',prepared['base_prepared_data_protocol_sha256']),
            'dec31_acceptance':(ROOT/'datasets/construction/evcivil-dec31-future/acceptance-v1.json',prepared['dec31_acceptance_sha256']),
            'lcsd_acceptance':(ROOT/'datasets/construction/lcsd-low-light-future-r4/acceptance-v1.json',prepared['lcsd_acceptance_sha256']),
            'combined_preparation_plan':(data/'preparation-plan.json',prepared['preparation_plan_sha256']),
            'combined_preparation_order':(data/'preparation-order-observation.json',prepared['preparation_order_observation_sha256']),
        }
        for label,(path,digest) in source_pins.items():
            if sha(path)!=digest:raise RuntimeError(f'Accepted source identity differs: {label}')
        protocol.update({
            'source_pins':{key:{'path':str(path.relative_to(ROOT)),'sha256':digest} for key,(path,digest) in source_pins.items()},
            'counts_by_source':prepared['counts_by_source'],'dec31_accepted_ids':prepared['dec31_accepted_ids'],
            'dec31_quarantined_ids':prepared['dec31_quarantined_ids'],'lcsd_source_ids':prepared['lcsd_source_ids'],
            'source_basis':'Retain all learned heads; adapt blocks4 onward. Approved broad880 plus250 genuine dark camera exposures from13 accepted Dec31 captures and41 real low-light still photographs. No generated video; ambiguous/quarantined sources excluded.',
            'licenses':prepared['license'],'lcsd_repurposing':prepared['lcsd_repurposing'],'lcsd_box_rule':prepared['lcsd_box_rule'],
            'data_preparation_plan_sha256':prepared['preparation_plan_sha256'],
            'preparation_order_observation_sha256':prepared['preparation_order_observation_sha256'],
            'preparation_sources':{str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'scripts/prepare_construction_combined_low_light.py',ROOT/'scripts/construction_native_dataset.py',ROOT/'scripts/freeze_construction_real_lowlight_stage2.py']},
            'resource_requirement':'Start only after explicit root release following factory F preflight/parity. Four CPU threads/channels_last selected by bounded TRAIN-only probe; actual fit contention disclosed.',
            'timing_estimate_only':'Tiny-probe rate suggests about30min bare optimizer for4x1171 images, excluding augmentation/validation/checkpoints/export and any shared-host contention. Not a completion promise or inference speed.',
        })
    target=data/'protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8'))!=protocol:
        raise RuntimeError('Refusing to replace any different frozen broader-material fit')
    target.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    print(json.dumps({'protocol_sha256':sha(target),'configuration':config},indent=2))


def main():
    freeze_candidate()


if __name__=='__main__':main()
