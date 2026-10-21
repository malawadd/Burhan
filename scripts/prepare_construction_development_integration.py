"""Build actual eight-report integration proposal; never edits app registries.

Run after the explicit four-source/two-model development export completes.
Missing reports/videos fail closed; no placeholder is registered.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/construction-round3'
CASES=['crack_89','crack_129','spalling_75','ros_crack_21']
MODELS=[('BEFORE','evcivil-yolov8n-r4-head-640-gain','evcivil_r4_head_failed'),
        ('AFTER','evcivil-yolov8n-r4-real-lowlight-640-gain','evcivil_r4_real_lowlight_failed')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol=OUT/'four-development-video-comparison-protocol.json'
    if sha(protocol)!='0006e730fc4951044d395a7921796b1c226884d574ae10ed4d8a09604d19ee7b':raise RuntimeError('Four-source development protocol differs')
    variants={};recordings=[];presets=[]
    for tag,stem,variant in MODELS:
        base=ROOT/'models/construction'/stem
        meta=json.loads(base.with_suffix('.provenance.json').read_text(encoding='utf8'))
        point=json.loads(base.with_suffix('.operating-point.json').read_text(encoding='utf8'))
        if point['model_sha256']!=meta['onnx_sha256'] or not point['serving_decoder_parity_verified']:raise RuntimeError('Complete model/operating point mismatch')
        license_text=meta['license']
        if tag=='AFTER':license_text+='; '+json.loads(base.with_suffix('.source-license-addendum.json').read_text(encoding='utf8'))['license_addendum']
        variants[variant]={'display_name':f'{tag.title()} failed camera calibration · development review',
            'file':stem+'.onnx','sha256':meta['onnx_sha256'],'source':meta['source'],
            'classes':meta['classes'],'capabilities':meta['capabilities'],'decoder':'yolo_detect','input_size':640,
            'confidence':.05,'class_confidence':point['class_confidence'],'exposure_policy':'low_light_gain',
            'license':license_text,'validation_scope':'FAILED fixed122-source camera calibration. Four full calibration/exposed recordings are development only; no independent field/physical-site/Saudi qualification. '+meta['validation_scope'],
            'experimental':True,'diagnostic_only':True}
        for sequence in CASES:
            prefix=f'construction-development-{stem}-{sequence}'
            evidence=ROOT/'output'/f'{prefix}.json';video=evidence.with_suffix('.mp4')
            report=json.loads(evidence.read_text(encoding='utf8'))
            if report['sequence']!=sequence or report['model']['sha256']!=meta['onnx_sha256'] or report['selection_protocol_sha256']!=sha(protocol) or report['candidate_status']!='FAILED CALIBRATION':raise RuntimeError('Exact failed development report identity mismatch')
            if report['derived_video']['sha256']!=sha(video) or report['summary']['unsupported_annotations']!=0:raise RuntimeError('Video identity or supported-class scope differs')
            source=report['source']['lossless_replay_input'];by_class=report['summary']['by_class']
            note=f'FAILED camera calibration; {report["source"]["evaluation_role"]}, not independent. Crack {by_class["crack"]["tp"]}TP/{by_class["crack"]["fn"]}FN/{by_class["crack"]["fp"]} unmatched; surface loss {by_class["spalling"]["tp"]}TP/{by_class["spalling"]["fn"]}FN/{by_class["spalling"]["fp"]} unmatched. Four separate RAW/NNinput/sourceGT/MODEL panes. Verify material, no automatic acceptance.'
            label=f'Construction development · {tag.title()} / {sequence} / FAILED'
            recordings.append({'id':f'{variant}-{sequence}','label':label,'path':str(video.relative_to(ROOT)).replace('\\','/'),
                'evidence_path':str(evidence.relative_to(ROOT)).replace('\\','/'),'purpose':'failure_diagnostic',
                'issue_summary':note,'note':note,'source_url':meta['source'],'license_url':'https://creativecommons.org/licenses/by/4.0/',
                'attribution':report['source']['authors']+'; ev-CIVIL CC BY4.0; external publisher GT separate from model.',
                'source_preset_label':label,'video_sha256':sha(video),'report_sha256':sha(evidence)})
            presets.append({'label':label,'profile':'construction','path':source['path'].replace('\\','/'),
                'input_fps':15,'deadline_ms':500,'preview_fps':8,'video_mode':'every_frame','purpose':'failure_diagnostic',
                'note':note+' Every-frame replay can be slower than source time; input15FPS is not measured capacity.',
                'profile_options':{'mode':'damage','stage':'structure','damage_variant':variant,'engine':'openvino','openvino_device':'GPU',
                    'mask_rendering':'boxes','exposure_policy':'low_light_gain','class_confidence':point['class_confidence'],'confirm_frames':1},
                'independent_accuracy_credit':False})
    proposal={'status':'Actual eight-report review proposal only; root owns app registrations. Both models FAILED, diagnostic_only, never auto-default.',
        'damage_variants':variants,'recording_rows':recordings,'public_preset_rows':presets,
        'case_selection_protocol_sha256':sha(protocol),'independent_accuracy_credit':False,
        'important':'No uniform confidence in preset; it would override class cuts. Preserve old artifacts; no heldout, training-success selection, production or15FPS claim.'}
    target=OUT/'four-development-video-integration-proposal.json'
    target.write_text(json.dumps(proposal,indent=2),encoding='utf8');print(target)


if __name__=='__main__':main()
