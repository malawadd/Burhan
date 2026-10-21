"""Freeze the single four-epoch640+gain head fit after TRAIN-only capacity choice."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'datasets/construction/evcivil-round4-gain'
CAPACITY = ROOT/'output/training/construction-round4-capacity/summary.json'
SOURCE = ROOT/'output/training/construction-round3/evcivil-yolov8n-r3/weights/best.pt'
NAME = 'evcivil-yolov8n-r4-head-640-gain'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    capacity=json.loads(CAPACITY.read_text(encoding='utf8'))
    batch=capacity['selected_batch_size']
    if batch not in [4,8,16]:
        raise RuntimeError('Training batch was not one of the bounded measured choices')
    if sha(SOURCE)!='ac3072242f4d7890312ba227f3d394fc034d491f8dfe95f665ad6795802a7182':
        raise RuntimeError('First retained learned source identity changed')
    config={'epochs':4,'image_size':640,'batch':batch,'nominal_batch':batch,'threads':3,'freeze':10,
            'learning_rate':.0005,'max_minutes':0.0,'fraction':1.0,'name':NAME,
            'data_root':str(DATA.relative_to(ROOT)),'continue_from':str(SOURCE.relative_to(ROOT)),
            'exposure_policy':'low_light_gain'}
    protocol={'schema_version':1,'candidate':NAME,'stage':'head_only_first4_epochs',
              'frozen_before_candidate_training':True,'strict_configuration':True,
              'starting_checkpoint_sha256':sha(SOURCE),'configuration':config,
              'prepared_data_protocol_sha256':sha(DATA/'data-protocol.json'),
              'native_input_manifest_sha256':sha(DATA/'native-input-manifest.json'),
              'capacity_summary_sha256':sha(CAPACITY),'capacity_protocol_sha256':capacity['protocol_sha256'],
              'training_contention':'Training uses three CPU threads alongside the factory fixed-step one-thread fit. The batch probe used four contended threads; it is not a prediction of full-fit throughput.',
              'source_basis':'Retain the first failed head with actual learned spalling behavior; no reset or channel relabelling. Nine additional concrete capture groups and 216 views, with the shared camera policy.',
              'selection':'Four fixed epochs, no wall cap. Best checkpoint by validation mAP on the same 122 source identities and labels with the prepared shared camera policy. No heldout or old exposed inference.',
              'threshold_selection':'Exact serving calibration on the original 122 raw sources plus shared gain; per-class grid and guards, strict NMS, global top100, and selected-point runtime parity.',
              'stage2_trigger':'Any class fails its precision/negative guard or retains low recall. A later freeze4 continuation requires a distinct configuration and version before training, rather than a silent extension.',
              'validation_scope':'Capture-group/global-defect-ID exclusions preserve dset6 calibration, dset5 heldout, Nov18 night calibration and Jan2 night heldout. Physical project independence and initial pretrained overlap remain unknown. New dset1/2 fitting makes old exposed recordings development only.',
              'licenses':'ev-CIVIL CC BY4.0; DACL CC BY-NC4.0; Ultralytics AGPL and source-weight lineage caveats remain.'}
    target=DATA/'protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8'))!=protocol:
        raise RuntimeError('Refusing to rewrite frozen head-fit plan')
    target.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    print(json.dumps({'protocol_sha256':sha(target),'configuration':config},indent=2))


if __name__=='__main__':
    main()
