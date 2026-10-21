"""Add source/license/result provenance without rewriting model or original metadata.

Small metadata files only. Run after all four epochs and final122-source parity.
The calibration loader already verifies the ONNX bytes against its model hash.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/construction/evcivil-yolov8n-r4-real-lowlight-640-gain.onnx'
DATA=ROOT/'datasets/construction/evcivil-round4-real-lowlight-gain'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata=MODEL.with_suffix('.provenance.json');point=MODEL.with_suffix('.operating-point.json')
    meta=json.loads(metadata.read_text(encoding='utf8'));op=json.loads(point.read_text(encoding='utf8'))
    protocol=json.loads((DATA/'protocol.json').read_text(encoding='utf8'))
    if meta['onnx_sha256']!=op['model_sha256'] or meta['training']['completed_epochs']!=4 or not op['serving_decoder_parity_verified']:
        raise RuntimeError('Actual complete model/calibration parity required')
    if sha(DATA/'protocol.json')!=meta['split_protocol_sha256'] or meta['training_script_sha256']!=protocol['training_script_sha256']:
        raise RuntimeError('Original protected fit identity differs')
    archive=Path(meta['save_dir'])/'executed-trainer-source.py'
    if sha(archive)!=meta['training_script_sha256']:raise RuntimeError('Executed source archive differs')
    receipt={'schema_version':1,'candidate':meta['experiment'],'model_sha256':meta['onnx_sha256'],
        'original_provenance_sha256':sha(metadata),'frozen_operating_point_sha256':sha(point),
        'fit_protocol_sha256':meta['split_protocol_sha256'],'evaluation_guard_sha256':op['minimum_accuracy_review_guard']['receipt_sha256'],
        'source_pins':protocol['source_pins'],'prepared_data_protocol_sha256':protocol['prepared_data_protocol_sha256'],
        'native_input_manifest_sha256':protocol['native_input_manifest_sha256'],'preprocessing_source_sha256':protocol['preprocessing_source_sha256'],
        'executed_trainer_archive':str(archive.relative_to(ROOT)),'executed_trainer_archive_sha256':sha(archive),
        'license_addendum':'LCSD author-linked distribution declares Apache-2.0; original author val/test photographs are repurposed solely as our TRAIN transfer with no LCSD evaluation claim. This is a source declaration, not universal rights clearance. Inherited Ultralytics/initial-weight AGPL lineage, ev-CIVIL CC BY4.0 and DACL CC BY-NC4.0 restrictions remain; initial source-weight data rights are not independently cleared.',
        'counts_by_source':protocol['counts_by_source'],'class_confidence':op['class_confidence'],
        'final_full_calibration':op['selected_runtime_summary'],'minimum_guard':op['minimum_accuracy_review_guard'],
        'training_seconds':meta['training']['train_seconds'],'fit_and_export_seconds':meta['training']['fit_and_export_seconds'],
        'timing_scope':'Observed full walltime under authorized shared CPU/GPU-host work; no subtraction or inference-capacity claim.',
        'heldout_forward':False,'status':'FAILED minimum accuracy review; no promotion, hidden extension or claim that thin cracks/spalling are fixed',
        'original_metadata_weights_and_policy_unchanged':True}
    target=MODEL.with_suffix('.source-license-addendum.json')
    if target.exists() and json.loads(target.read_text(encoding='utf8'))!=receipt:raise RuntimeError('Refusing changed final source receipt')
    target.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps({'path':str(target.relative_to(ROOT)),'sha256':sha(target)},indent=2))


if __name__=='__main__':main()
