"""One separately preregistered final-epoch export/CAL-only candidate, no fit/test.

Shared frozen evaluator remains untouched. The exported fit's experiment name
stays the original run identity; this distinct artifact explicitly selects last4.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/construction/evcivil-yolov8n-r4-real-lowlight-640-gain.provenance.json'
RUN=ROOT/'output/training/construction-round3/evcivil-yolov8n-r4-real-lowlight-640-gain'
MODEL=ROOT/'models/construction/evcivil-yolov8n-r4-last4-640-gain.onnx'
OUT=ROOT/'output/evaluation/construction-last4-candidate'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['freeze','export','calibration'],required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=OUT/'protocol-before-export-calibration.json'
    if args.stage=='freeze':
        meta=json.loads(SOURCE.read_text(encoding='utf-8'))
        protocol={'candidate':'evcivil-r4-last4-640-gain','selection':'ONE exact final fourth-epoch checkpoint from already completed fixed4epoch1171fit, selected before new export/CAL by explicit user-task continuation; no training restart, no selecting among additional epochs/models',
                  'source_metadata_sha256':sha(SOURCE),'source_pt':(RUN/'weights/last.pt').relative_to(ROOT).as_posix(),'source_pt_sha256':sha(RUN/'weights/last.pt'),
                  'fit_protocol_sha256':meta['split_protocol_sha256'],'original_calibration_protocol_sha256':sha(ROOT/'datasets/construction/evcivil-round3/protocol.json'),
                  'evaluator_sha256':sha(ROOT/'scripts/evaluate_construction_round3.py'),
                  'preprocessing_sha256':sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py'),
                  'script_sha256':sha(Path(__file__)),'input_size':640,'classes':['spalling','crack'],'exposure_policy':'low_light_gain',
                  'calibration_scope':'Exactly unchanged122 rawCAL sources; all wrong-class/unmatched candidates retained, actual-infer parity; shared grid .10,.15,.20,.25,.35,.45,.55,.65,.75,.85 with P80/negative10% rule',
                  'minimum_review':'Recall>=.50 for EACHclass and geometric-thin subset, precision>=.80 each, combined10true-negative FPR<=.10, all122 serving parity. This is not a solved/release criterion.',
                  'prohibited':'No heldout or genericcompare()/video_cases(), no additional fitting, no changed example/case selection, no overwriting bestepoch1 artifacts or eight videos',
                  'license':'Inherited AGPL/ev-CIVILCCBY4/DACLNC/LCSDsourceApache declaration and unresolved initial source rights; originaladdendum retained separately'}
        if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Preserve frozen last4 candidate')
        write(protocol_path,protocol)
        print('FROZEN',sha(protocol_path),flush=True)
        return
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    for path,expected in [(SOURCE,protocol['source_metadata_sha256']),(ROOT/protocol['source_pt'],protocol['source_pt_sha256']),
                          (ROOT/'scripts/evaluate_construction_round3.py',protocol['evaluator_sha256']),
                          (ROOT/'edge_runtime/profiles/construction_preprocessing.py',protocol['preprocessing_sha256']),
                          (Path(__file__),protocol['script_sha256'])]:
        if sha(path)!=expected:raise RuntimeError('Protected last4 candidate input/source differs: '+str(path))
    if args.stage=='export':
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if MODEL.exists():raise RuntimeError('Preserve exported last4 candidate')
        wrapper=YOLO(str(ROOT/protocol['source_pt']),task='detect')
        if wrapper.model.names!={0:'spalling',1:'crack'}:raise RuntimeError('Locally trained final head semantics differ')
        wrapper.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,batch=1)
        exported=(ROOT/protocol['source_pt']).with_suffix('.onnx')
        MODEL.write_bytes(exported.read_bytes())
        meta=json.loads(SOURCE.read_text(encoding='utf-8'))
        meta.update(model_variant=protocol['candidate'],model_path=str(MODEL),onnx_sha256=sha(MODEL),
                    source_final_pt_sha256=protocol['source_pt_sha256'],candidate_protocol_sha256=sha(protocol_path),
                    selection='Exact last.pt epoch4, preregistered single alternative to original generic-cal-mAP bestepoch1; no extra fit, no test selection',
                    parent_best_artifact_metadata_sha256=sha(SOURCE))
        meta.pop('results',None)
        for key in ['monitored_backbone_weight_l2_change','spall_classifier_parameter_l2_change','learned_spall_classifier_sha256']:
            meta['training'].pop(key,None)
        meta['training']['checkpoint_selection']='Exact completed epoch4; originalfitduration includes all4epochs; bestepoch1-specific parameter deltas omitted'
        write(MODEL.with_suffix('.provenance.json'),meta)
        print('EXPORTED',sha(MODEL),flush=True)
        return
    if MODEL.with_suffix('.operating-point.json').exists():raise RuntimeError('Preserve completed last4CAL')
    sys.path.insert(0,str(ROOT/'scripts'))
    from evaluate_construction_round3 import calibrate
    calibrate(MODEL)  # Only explicit CAL; never calls compare()/video_cases().


if __name__=='__main__':main()
