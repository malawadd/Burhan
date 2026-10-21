"""Prepare a distinct four-epoch backbone adaptation only after R4 head calibration.

This copies already verified native inputs without changing pixels or splits.
It does not start fitting, run inference, or inspect heldout images.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction/evcivil-round4-gain'
TARGET=ROOT/'datasets/construction/evcivil-round4-adapted-gain'
NAME='evcivil-yolov8n-r4-adapted-640-gain'
HEAD=ROOT/'models/construction/evcivil-yolov8n-r4-head-640-gain.onnx'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf8')


def remap(value):
    if isinstance(value,dict):return {key:remap(item) for key,item in value.items()}
    if isinstance(value,list):return [remap(item) for item in value]
    if isinstance(value,str):
        for prefix in [str(DATA.relative_to(ROOT)),DATA.relative_to(ROOT).as_posix()]:
            if value.startswith(prefix+'\\') or value.startswith(prefix+'/'):
                return str(TARGET.relative_to(ROOT)/Path(value).relative_to(Path(prefix)))
    return value


def main():
    meta=json.loads(HEAD.with_suffix('.provenance.json').read_text(encoding='utf8'))
    op_path=HEAD.with_suffix('.operating-point.json')
    op=json.loads(op_path.read_text(encoding='utf8'))
    guard_path=DATA/'stage2-continuation-guard.json'
    if sha(HEAD)!=meta['onnx_sha256'] or op['model_sha256']!=meta['onnx_sha256']:
        raise RuntimeError('Actual head model/calibration identity changed')
    decision=op.get('minimum_continuation_guard',{})
    if decision.get('protocol_sha256')!=sha(guard_path) or not decision.get('requires_distinct_stage2'):
        raise RuntimeError('Frozen complete-calibration continuation guard does not require stage2')
    if not op.get('serving_decoder_parity_verified'):
        raise RuntimeError('Head serving parity must pass before continuation')
    if meta['training']['completed_epochs']!=4 or meta['exposure_policy']!='low_light_gain':
        raise RuntimeError('Unexpected head fit or camera policy')
    if sha(DATA/'protocol.json')!=meta['split_protocol_sha256']:
        raise RuntimeError('Head training protocol changed')
    checkpoint=(Path(meta['save_dir'])/'weights/best.pt').resolve()
    if not checkpoint.is_relative_to((ROOT/'output/training/construction-round3').resolve()) or not checkpoint.is_file():
        raise RuntimeError('Only the locally produced learned head can seed this continuation')
    capacity_path=ROOT/'output/training/construction-round4-layout-capacity/summary.json'
    capacity=json.loads(capacity_path.read_text(encoding='utf8'))
    if capacity['source_checkpoint_sha256']!=sha(checkpoint) or capacity['selected_memory_format'] not in ['contiguous','channels_last'] or capacity['selected_cpu_threads'] not in [3,4]:
        raise RuntimeError('Actual retained head must match bounded layout/thread probe')
    prepared=json.loads((DATA/'data-protocol.json').read_text(encoding='utf8'))
    if sha(DATA/'data-protocol.json')!=meta['prepared_data_protocol_sha256']:
        raise RuntimeError('Prepared head inputs changed')
    shared=ROOT/'edge_runtime/profiles/construction_preprocessing.py'
    if sha(shared)!=prepared['preprocessing_source_sha256'] or sha(DATA/'native-input-manifest.json')!=prepared['native_input_manifest_sha256']:
        raise RuntimeError('Shared policy/input manifest changed')
    native=json.loads((DATA/'native-input-manifest.json').read_text(encoding='utf8'))
    if TARGET.exists():raise RuntimeError('Refusing to replace any existing stage2 preparation')
    # Verify every source before creating the distinct destination.
    for row in native:
        for field in ['image','label']:
            source=(ROOT/row[field]).resolve()
            if not source.is_relative_to(DATA.resolve()) or sha(source)!=row[field+'_sha256']:
                raise RuntimeError('Native image/label identity changed')
    TARGET.mkdir()
    for row in native:
        for field in ['image','label']:
            source=ROOT/row[field];destination=ROOT/remap(row[field])
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            if sha(destination)!=row[field+'_sha256']:raise RuntimeError('Copied input differs')
    for filename in ['exposures.json','background-import.json','night-holdout-protocol.json']:
        write(TARGET/filename,remap(json.loads((DATA/filename).read_text(encoding='utf8'))))
    write(TARGET/'native-input-manifest.json',remap(native))
    prepared.update(status='Distinct adaptation inputs, same pixels/classes/splits as head fit',
        parent_prepared_data_protocol_sha256=sha(DATA/'data-protocol.json'),
        native_input_manifest_sha256=sha(TARGET/'native-input-manifest.json'),
        exposures_sha256=sha(TARGET/'exposures.json'))
    write(TARGET/'data-protocol.json',prepared)
    dataset=yaml.safe_load((DATA/'dataset.yaml').read_text(encoding='utf8'))
    dataset.update(path=TARGET.as_posix(),train=(TARGET/'images/train').as_posix(),val=(TARGET/'images/calibration').as_posix())
    (TARGET/'dataset.yaml').write_text(yaml.safe_dump(dataset),encoding='utf8')
    config={'epochs':4,'image_size':640,'batch':4,'nominal_batch':4,'threads':capacity['selected_cpu_threads'],'freeze':4,
        'memory_format':capacity['selected_memory_format'],
        'learning_rate':.0005,'max_minutes':0.0,'fraction':1.0,'name':NAME,
        'data_root':str(TARGET.relative_to(ROOT)),'continue_from':str(checkpoint.relative_to(ROOT)),
        'exposure_policy':'low_light_gain'}
    parent=json.loads((DATA/'protocol.json').read_text(encoding='utf8'))
    protocol={'schema_version':1,'candidate':NAME,'stage':'distinct4_epoch_backbone_adaptation',
        'frozen_before_candidate_training':True,'strict_configuration':True,'configuration':config,
        'training_script_sha256':sha(ROOT/'scripts/train_construction_damage_round3.py'),
        'starting_checkpoint_sha256':sha(checkpoint),'head_model_sha256':sha(HEAD),
        'head_operating_point_sha256':sha(op_path),'continuation_guard_sha256':sha(guard_path),
        'layout_capacity_summary_sha256':sha(capacity_path),'layout_capacity_protocol_sha256':capacity['protocol_sha256'],
        'parent_head_protocol_sha256':sha(DATA/'protocol.json'),
        'prepared_data_protocol_sha256':sha(TARGET/'data-protocol.json'),
        'native_input_manifest_sha256':sha(TARGET/'native-input-manifest.json'),
        'source_basis':'Same verified pixels, labels and source identities; retain all learned head parameters, adapt blocks4 onward. No additional data or heldout selection.',
        'selection':'Four fixed epochs, no wall cap; best checkpoint by same122 calibration-source validation mAP. Exact raw-source/shared-gain operating-point calibration follows before any heldout.',
        'validation_scope':parent['validation_scope'],'licenses':parent['licenses'],
        'resource_requirement':'Start only after coordinated factoryD isolated timing release; selected3/4CPU threads, no surprise concurrent jobs, actual training contention disclosed.'}
    write(TARGET/'protocol.json',protocol)
    print(json.dumps({'protocol_sha256':sha(TARGET/'protocol.json'),'configuration':config},indent=2))


if __name__=='__main__':main()
