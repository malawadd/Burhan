"""Explicit four-recording development audit; never loads heldout/default cases.

Freeze the whitelist before postfit scoring. Export only after coordinated GPU
release and both complete operating-point calibrations. No production claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from evaluate_construction_round3 import ROOT, BASE, OUT, load_new, score_frame, summarize
from construction_annotations import publisher_labels_at_or_after
from edge_runtime.profiles.construction_preprocessing import camera_exposure

HEAD=ROOT/'models/construction/evcivil-yolov8n-r4-head-640-gain.onnx'
AFTER=ROOT/'models/construction/evcivil-yolov8n-r4-real-lowlight-640-gain.onnx'
DATA=ROOT/'datasets/construction/evcivil-round4-real-lowlight-gain'
TARGET=OUT/'four-development-video-comparison-protocol.json'
SELECTION=[('exposed','crack_89'),('dset_6','crack_129'),('dset_6','spalling_75'),('night_nov18','ros_crack_21')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_sources():
    """Validate identities/roles BEFORE opening any camera arrays or HDF5 file."""
    original=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
    calibration={(r['group'],r['sequence']) for r in original['sequences']['calibration']}
    heldout={(r['group'],r['sequence']) for r in original['sequences']['heldout']}
    night=BASE/'night-holdout-protocol.json'
    if night.exists():heldout.update((r['group'],r['sequence']) for r in json.loads(night.read_text(encoding='utf8'))['sequences'])
    rows=[]
    for group,name in SELECTION:
        if (group,name) in heldout:raise RuntimeError('Protected heldout recording forbidden')
        if group=='exposed':
            if name!='crack_89':raise RuntimeError('Only the declared exposed challenge is allowed')
            folder=ROOT/'datasets/construction/evcivil'/name
            files=[folder/'frames.npy',folder/'timestamps-us.npy',folder/'annotations.json']
        else:
            if (group,name) not in calibration:raise RuntimeError('Explicit calibration whitelist differs')
            folder=BASE/'raw'/group/name
            label=folder/f'{name}_frame_label.npy'
            if not label.exists():label=folder/f'{name}_label.npy'
            files=[folder/f'{name}_frames.h5',label]
        rows.append({'group':group,'sequence':name,'role':'reused_exposed_development' if group=='exposed' else 'calibration_development',
                     'files':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    return rows


def freeze():
    sources=selected_sources()
    if sha(HEAD)!='33e7ab167a93c7ee9c9e80e544951ec2e61fcdaf977377ea56e321dd4fea41ad':raise RuntimeError('Before model differs')
    if sha(DATA/'protocol.json')!='f77168288653f986ddf23a8c601141767d84a172ee8e0d31f9d5059c95da625d':raise RuntimeError('After fit protocol differs')
    point=json.loads(HEAD.with_suffix('.operating-point.json').read_text(encoding='utf8'))
    if point['model_sha256']!=sha(HEAD) or not point['serving_decoder_parity_verified']:raise RuntimeError('Before frozen operating point differs')
    protocol={'schema_version':1,'selected_before_combined_postfit_operating_point_search':True,
        'observed_stage':'Combined four-epoch fit already running; ordinary epoch validation may select checkpoints. No combined postfit calibration/video inference yet.',
        'source_protocol_sha256':sha(BASE/'protocol.json'),'sources':sources,
        'before_model_sha256':sha(HEAD),'before_operating_point_sha256':sha(HEAD.with_suffix('.operating-point.json')),
        'after_fit_protocol_sha256':sha(DATA/'protocol.json'),'after_evaluation_guard_sha256':sha(DATA/'evaluation-minimum-guard.json'),
        'after_model':str(AFTER.relative_to(ROOT)),
        'selection_basis':'Fixed old exposed thin89, first day calibration crack129/materialloss75 and first night calibration ros21. No training-success or favorable-model selection/replacement.',
        'scoring':'Both models support crack and spalling/materialloss; class-aware publisher boxIoU>=0.50. Retain all exposures/PTS, misses/wrongclass/unmatched; unlabelled exposures unscored. No pixel metric.',
        'inputs':'Untouched raw source; exact native sharedgain once before letterbox. RAW/GAIN/externalGT/MODEL separate; original camera indices/timestamps preserved.',
        'scope':'Exposed/calibration development only, no independent accuracy credit or heldout access; contended accuracy/export, never15FPS qualification.',
        'execution':'After all four epochs, full122 calibration, serving parity and frozen cuts, AND factory GPU release. Failed minimum still yields explicitly FAILED diagnostic exports, no promotion.'}
    if TARGET.exists() and json.loads(TARGET.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Refusing changed development selection/protocol')
    TARGET.parent.mkdir(parents=True,exist_ok=True);TARGET.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    return protocol


def load_cases(protocol):
    """Only paths already checked/hash-pinned by the four-case whitelist."""
    import h5py
    publisher=json.loads((BASE/'publisher-record.json').read_text(encoding='utf8'))
    authors='; '.join(a['full_name'] for a in publisher['authors'])+'; annotation contributor Xuanni Huo'
    cases=[]
    for row in protocol['sources']:
        for relative,digest in row['files'].items():
            if sha(ROOT/relative)!=digest:raise RuntimeError('Selected source changed')
        group,name=row['group'],row['sequence']
        if (group,name) not in SELECTION:raise RuntimeError('Unselected source refused before load')
        if group=='exposed':
            folder=ROOT/'datasets/construction/evcivil'/name
            frames=np.load(folder/'frames.npy',allow_pickle=False);times=np.load(folder/'timestamps-us.npy',allow_pickle=False)
            truth=json.loads((folder/'annotations.json').read_text(encoding='utf8'))['frames']
        else:
            folder=BASE/'raw'/group/name
            with h5py.File(folder/f'{name}_frames.h5','r') as src:
                frames=src['frames'][:];times=src['frame_timestamp'][:].astype(np.int64)
            label=folder/f'{name}_frame_label.npy'
            if not label.exists():label=folder/f'{name}_label.npy'
            labels=np.load(label,allow_pickle=False);period=float(np.median(np.diff(times)));truth=[]
            for index,timestamp in enumerate(times):
                expected=[]
                for j,b in enumerate(publisher_labels_at_or_after(labels,timestamp,period)):
                    x,y,w,h=map(float,b[2:6]);right=min(frames.shape[2],x+w);bottom=min(frames.shape[1],y+h);x=max(0,x);y=max(0,y)
                    if right>x and bottom>y:expected.append({'id':f'{name}:{index}:{j}','label':'crack' if int(b[1])==0 else 'spalling','bbox':[x,y,right-x,bottom-y],'annotation_timestamp_us':int(b[0])})
                truth.append({'source_frame_index':index,'video_time_s':(int(timestamp)-int(times[0]))/1e6,'source_timestamp_us':int(timestamp),'expected':expected})
        if len(frames)!=len(times) or len(truth)!=len(frames) or np.any(np.diff(times)<=0):raise RuntimeError('Complete original timebase/count contract failed')
        cases.append({'name':name,'frames':frames,'times':times,'truth':truth,'source':{
            'source':'https://doi.org/10.11583/DTU.25924228','authors':authors,'license':'CC BY4.0',
            'source_group':group,'evaluation_role':row['role'],'independent_accuracy_credit':False,
            'original_camera_frames':len(frames),'timebase':'Original microsecond camera exposure PTS shifted tozero; one record/frame per original exposure',
            'source_files':row['files']}})
    return cases


def panel_video(raw,native,record,name,tag,status):
    """Native pixels, separate source GT and model, no GT-derived model input."""
    # Two rows keep real low-resolution chips/hairlines visible on a laptop.
    # Nearest2x is display only; model input and saved identities remain native.
    canvas=np.full((1300,1420,3),(24,27,30),np.uint8)
    gt=raw.copy();pred=raw.copy();matched={p.get('matched_annotation_id') for p in record['predictions'] if p.get('matched_annotation_id')}
    def box(image,item,color,label):
        x,y,w,h=map(round,item['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),color,1)
        cv2.putText(image,label,(max(0,x),max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.32,color,1)
    for item in record['expected']:box(gt,item,(245,210,30) if item['id'] in matched else (70,70,245),'SOURCE '+item['label'])
    for item in record['predictions']:box(pred,item,(60,205,90) if item.get('matched_annotation_id') else (30,170,245),f"{item['label']} {item['score']:.2f}")
    for i,(array,label) in enumerate([(raw,'RAW CAMERA - unchanged'),(native,f'NN INPUT - gain {record["input_exposure_gain"]:.2f}x'),(gt,'EXTERNAL PUBLISHER GT'),(pred,'MODEL ONLY - actual outputs')]):
        x=8+(i%2)*708;y=100+(i//2)*560
        canvas[y:y+520,x:x+692]=cv2.resize(array,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas,label+' (native pixels shown2x)',(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,.56,(235,235,235),1)
    header=[f'REAL DAMAGE / {tag} / {name} / {status}',f'Source frame {record["source_frame_index"]} at {record["video_time_s"]:.6f}s | DEVELOPMENT ONLY - not independent validation']
    counts=f'Matched {record["localized_matches"]} / missed {record["missed_defects"]} / unmatched {record["false_alerts"]}' if record['evaluated'] else 'UNSCORED: no publisher GT; matches/misses/unmatched not scored. Amber candidates unverified.'
    footer=[counts,'External GT: cyan matched / red missed. Model: green localized / amber unmatched. BoxIoU >=0.50; no pixel accuracy claim.',
            'Crack: verify joint/shadow and measure with scale. Surface loss: verify actual substrate/coating; inspect loose material. No automatic code acceptance.',
            'Gamage et al., ev-CIVIL (DTU), CC BY4.0. No source frame repeated/generated; actual input intensity adjustment shown separately.']
    for i,line in enumerate(header):cv2.putText(canvas,line,(8,25+i*25),cv2.FONT_HERSHEY_SIMPLEX,.56,(235,235,235),1)
    for i,line in enumerate(footer):cv2.putText(canvas,line,(8,1204+i*26),cv2.FONT_HERSHEY_SIMPLEX,.46,(225,225,225),1)
    return canvas


def verify_replay(case,verification,folder):
    """Read-only validation against this exact whitelist's camera arrays/PTS."""
    import av
    from fractions import Fraction
    target=(ROOT/verification['path']).resolve()
    if not target.is_relative_to(folder.resolve()) or verification['source_sequence']!=case['name']:
        raise RuntimeError('Replay source sequence/path differs')
    expected_times=case['times']-int(case['times'][0]);count=0
    with av.open(str(target)) as source:
        for index,frame in enumerate(source.decode(video=0)):
            if index>=len(expected_times) or Fraction(frame.pts)*frame.time_base*1_000_000!=int(expected_times[index]) or not np.array_equal(frame.to_ndarray(format='gray'),case['frames'][index]):
                raise RuntimeError('Replay pixels/PTS differ from explicit raw camera source')
            count+=1
    if count!=len(case['frames']):raise RuntimeError('Replay exposure count differs')
    capture=cv2.VideoCapture(str(target));decoded=0
    try:
        while True:
            okay,bgr=capture.read()
            if not okay:break
            if decoded>=count or not np.array_equal(bgr,cv2.cvtColor(case['frames'][decoded],cv2.COLOR_GRAY2BGR)):
                raise RuntimeError('Actual OpenCV replay differs from raw camera pixels')
            decoded+=1
    finally:capture.release()
    if decoded!=count:raise RuntimeError('Actual OpenCV replay count differs')
    return {'all_camera_exposures':count,'all_pyav_gray_and_opencv_bgr_pixels_equal_raw':True,
            'all_original_microsecond_pts_exact':True,'scope':'Read-only validation against the actual whitelisted raw arrays before development video inference; old replay artifacts unmodified.'}


def export():
    from prepare_evcivil import encode_video
    from prepare_evcivil_lossless import encode_and_verify
    protocol=freeze();cases=load_cases(protocol);aggregate={}
    after_op=json.loads(AFTER.with_suffix('.operating-point.json').read_text(encoding='utf8'))
    if not after_op.get('serving_decoder_parity_verified') or not after_op.get('minimum_accuracy_review_guard'):raise RuntimeError('Complete selected-point calibration/review guard required before video inference')
    for case in cases:
        exposed=case['source']['source_group']=='exposed'
        folder=ROOT/'datasets/construction/evcivil'/case['name'] if exposed else BASE/'replay'/case['name']
        proof=folder/f'{case["name"]}-lossless-frames.provenance.json'
        if not proof.exists():
            if exposed or folder.exists():raise RuntimeError('Refusing to overwrite a missing/incomplete existing replay proof')
            folder.mkdir(parents=True)
            np.save(folder/'frames.npy',case['frames'],allow_pickle=False);np.save(folder/'timestamps-us.npy',case['times'],allow_pickle=False)
            annotations={**case['source'],'frames':case['truth']}
            (folder/'annotations.json').write_text(json.dumps(annotations,indent=2),encoding='utf8')
            verification=encode_and_verify(folder)
        else:verification=json.loads(proof.read_text(encoding='utf8'))
        if verification['original_camera_exposures']!=len(case['frames']) or not verification['all_opencv_bgr_pixels_equal_raw_gray_to_bgr'] or verification['max_presentation_timestamp_error_us']!=0 or sha(ROOT/verification['path'])!=verification['sha256']:
            raise RuntimeError('Lossless replay identity proof differs')
        case['source']['lossless_replay_input']=verification
        case['source']['replay_identity_reverified_against_raw']=verify_replay(case,verification,folder)
    for tag,path in [('BEFORE',HEAD),('AFTER',AFTER)]:
        point=json.loads(path.with_suffix('.operating-point.json').read_text(encoding='utf8'))
        if point['model_sha256']!=sha(path) or not point['serving_decoder_parity_verified']:raise RuntimeError('Frozen operating point differs')
        status='FAILED CALIBRATION' if tag=='BEFORE' or not point['minimum_accuracy_review_guard']['all_minimum_guards_pass'] else 'MINIMUM REVIEW PASSED - NOT QUALIFIED'
        model=load_new(path,point['class_confidence']);reports=[]
        license_receipt=None
        if tag=='AFTER':
            receipt_path=path.with_suffix('.source-license-addendum.json')
            license_receipt=json.loads(receipt_path.read_text(encoding='utf8'))
            if license_receipt['model_sha256']!=sha(path) or license_receipt['original_provenance_sha256']!=sha(path.with_suffix('.provenance.json')):
                raise RuntimeError('Additive source/license receipt differs')
        for case in cases:
            target=ROOT/f'output/construction-development-{path.stem}-{case["name"]}'
            # Metadata-only recovery may reuse completed immutable recordings;
            # never overwrite/reinfer a frozen successful export.
            existing=target.with_suffix('.json')
            if existing.exists():
                prior=json.loads(existing.read_text(encoding='utf8'))
                if prior['model']['sha256']!=sha(path) or prior['class_confidence']!=point['class_confidence'] or prior['selection_protocol_sha256']!=sha(TARGET) or prior['source']['source_files']!=case['source']['source_files'] or prior['derived_video']['sha256']!=sha(target.with_suffix('.mp4')):
                    raise RuntimeError('Existing completed export differs; refusing overwrite')
                reports.append({'sequence':case['name'],'model_sha256':sha(path),'status':prior['candidate_status'],'summary':prior['summary'],
                    'video':str(target.with_suffix('.mp4').relative_to(ROOT)),'video_sha256':sha(target.with_suffix('.mp4')),
                    'report':str(existing.relative_to(ROOT)),'report_sha256':sha(existing)})
                print(json.dumps({'reused_immutable_recording':str(existing.relative_to(ROOT))}),flush=True)
                continue
            records=[];canvases=[]
            for gray,truth in zip(case['frames'],case['truth']):
                raw=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);predictions=model.infer(raw)
                native,gain=camera_exposure(raw,model.exposure_policy)
                if gain!=model.last_input_exposure_gain:raise RuntimeError('Displayed native-input gain differs from actual inference')
                record={**truth,**score_frame(predictions,truth['expected'],{'crack','spalling'}),'frame_size_wh':[346,260],
                    'input_exposure_policy':model.exposure_policy,'input_exposure_gain':gain,
                    'raw_native_bgr_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'model_input_native_bgr_sha256':hashlib.sha256(native.tobytes()).hexdigest()}
                if record['unsupported_defects']:raise RuntimeError('Both declared classes must be supported')
                records.append(record);canvases.append(panel_video(raw,native,record,case['name'],tag,status))
            report={'schema_version':1,'purpose':'failure_diagnostic' if status=='FAILED CALIBRATION' else 'development_diagnostic',
                'matching_iou_threshold':.5,'sequence':case['name'],'candidate_status':status,'independent_accuracy_credit':False,
                'issue_summary':f'Actual publisher-labelled {"surface material loss" if "spalling" in case["name"] else "crack"}; {status}. All misses and unmatched predictions retained.',
                'why_it_matters':'Visible surface damage needs close inspection; grayscale appearance alone cannot establish structural severity or material.',
                'next_check':'Verify surface, substrate and scale with a qualified inspector; this development diagnostic cannot accept construction work.',
                'annotation_scope':case['source']['evaluation_role']+'; original publisher boxes, no pixel ground truth; unlabelled exposures unscored',
                'frame_size_wh':[346,260],'source':case['source'],'model':model.describe(),'class_confidence':point['class_confidence'],
                'annotated_canvas_layout':'Two-by-two RAW / actual nativeGAIN / externalGT / MODEL; native346x260 shown2x nearest for visibility. Model input/identities unchanged.',
                'selection_protocol_sha256':sha(TARGET),'summary':summarize(records),'frames':records}
            if license_receipt:
                report['model']['license']=model.spec.get('license','')+'; '+license_receipt['license_addendum']
                report['model_source_license_receipt']={'path':str(receipt_path.relative_to(ROOT)),'sha256':sha(receipt_path)}
            report['derived_video']=encode_video(target.with_suffix('.mp4'),np.asarray(canvases),case['times'],max(1,round(1e6/np.median(np.diff(case['times'])))))
            first=next((i for i,r in enumerate(records) if r['expected']),0)
            preview=target.with_name(target.name+'-fixed-first-annotated').with_suffix('.jpg')
            if not cv2.imwrite(str(preview),canvases[first]):raise RuntimeError('Fixed source preview could not be saved')
            report['fixed_preview']={'selection':'First publisher-annotated exposure, never selected for favorable model output',
                'source_frame_index':records[first]['source_frame_index'],'path':str(preview.relative_to(ROOT)),'sha256':sha(preview)}
            target.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
            result={'sequence':case['name'],'model_sha256':sha(path),'status':status,'summary':report['summary'],
                    'video':str(target.with_suffix('.mp4').relative_to(ROOT)),'video_sha256':sha(target.with_suffix('.mp4')),
                    'report':str(target.with_suffix('.json').relative_to(ROOT)),'report_sha256':sha(target.with_suffix('.json'))}
            reports.append(result);print(json.dumps(result,indent=2),flush=True)
        aggregate[tag]=reports;del model
    (OUT/'four-development-video-paired-summary.json').write_text(json.dumps(aggregate,indent=2),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--export',action='store_true');args=parser.parse_args();cv2.setNumThreads(1)
    if args.freeze_only:freeze();print(json.dumps({'protocol':str(TARGET),'sha256':sha(TARGET)},indent=2))
    elif args.export:export()
    else:parser.error('Choose --freeze-only or explicitly coordinated --export')
