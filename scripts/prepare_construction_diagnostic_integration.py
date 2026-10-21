"""Write reviewable model/preset/recording blocks; never mutate app registries."""
import json
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/construction-round3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    models={};recordings=[];presets=[]
    for historical in [True,False]:
        stem='evcivil-yolov8n-r3'if historical else'evcivil-yolov8n-r3-adapted'
        variant='evcivil_r3_historical_failed'if historical else'evcivil_r3_adapted_failed'
        meta=json.loads((ROOT/f'models/construction/{stem}.provenance.json').read_text(encoding='utf8'))
        op=json.loads((ROOT/f'models/construction/{stem}.operating-point.json').read_text(encoding='utf8'))
        models[variant]={'display_name':'Historical failed calibration · seen-source diagnostic'if historical else'Adapted failed calibration · research diagnostic',
                         'file':f'{stem}.onnx','sha256':meta['onnx_sha256'],'source':meta['source'],
                         'classes':meta['classes'],'capabilities':meta['capabilities'],'decoder':'yolo_detect','input_size':416,
                         'confidence':.05,'class_confidence':op['class_confidence'],'license':meta['license'],
                         'validation_scope':'FAILED unchanged camera calibration; only seen-training/calibration diagnostics, not independent field validation; '+meta.get('validation_scope','Known current-fit source-folder/globalID exclusions; physical project independence and initial pretrained overlap unverified; no Saudi-site qualification'),
                         'experimental':True,'diagnostic_only':True}
        for sequence,role in [('spalling_54','SEEN TRAINING'),('spalling_75','CAL FAILURE'),('crack_129','CAL THIN FAILURE')]:
            prefix=f'construction-round3-diagnostic-{stem}-{sequence}'if historical else f'construction-round3-diagnostic-{sequence}'
            report_path=ROOT/f'output/{prefix}.json';report=json.loads(report_path.read_text(encoding='utf8'))
            if report['model']['sha256']!=meta['onnx_sha256']:raise RuntimeError('Diagnostic report model identity mismatch')
            label=f'Construction diagnostic · {"historical"if historical else"adapted"} / {role} / {sequence}'
            summary=report['summary'];byclass=summary['by_class']
            note=f'Failed-calibration model. {role}: crack{byclass["crack"]["tp"]}TP/{byclass["crack"]["fn"]}FN/{byclass["crack"]["fp"]}unmatched; spall{byclass["spalling"]["tp"]}TP/{byclass["spalling"]["fn"]}FN/{byclass["spalling"]["fp"]}unmatched. Complete real source, publisher boxes; no independent accuracy credit.'
            recording={'id':f'{variant}-{sequence}','label':label,'path':f'output/{prefix}.mp4',
                       'evidence_path':str(report_path.relative_to(ROOT)).replace('\\','/'),'purpose':'failure_diagnostic',
                       'issue_summary':note,'note':note,'source_url':meta['source'],
                       'license_url':'https://creativecommons.org/licenses/by/4.0/',
                       'attribution':'Gamage et al., ev-CIVIL, DTU, CC BY4.0; labels Xuanni Huo; actual camera exposures/local model overlays; see evidence for all authors.',
                       'source_preset_label':label,'video_sha256':sha(ROOT/f'output/{prefix}.mp4'),'report_sha256':sha(report_path)}
            recordings.append(recording)
            source=report['source']['lossless_replay_input']
            presets.append({'label':label,'profile':'construction','path':source['path'].replace('\\','/'),
                            'input_fps':15,'deadline_ms':500,'preview_fps':8,'video_mode':'every_frame',
                            'purpose':'failure_diagnostic','note':note,
                            'profile_options':{'mode':'damage','damage_variant':variant,'engine':'openvino','openvino_device':'GPU',
                                               'mask_rendering':'boxes','class_confidence':op['class_confidence'],'confirm_frames':1},
                            'independent_accuracy_credit':False})
    proposal={'status':'Reviewable integration proposal only; root owns registries. Sixdevelopmentrows maximum, no new primary category reference.',
              'damage_variants':models,'recording_rows':recordings,'public_preset_rows':presets,
              'source_role':'Three fixed source recordings, each evaluated with two calibration-failed candidates. No heldout/exposed evaluation.',
              'important':'Do not add a uniform confidence to presets: it overrides calibration class cutoffs. Every-frame review can run slower than original time; no real-time qualification.'}
    path=OUT/'diagnostic-integration-proposal.json';path.write_text(json.dumps(proposal,indent=2),encoding='utf8')
    print(path)


if __name__=='__main__':main()
