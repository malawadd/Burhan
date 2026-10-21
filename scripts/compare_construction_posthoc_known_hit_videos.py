"""Exactly two post-hoc known-hit real recordings, never independent validation.

Freeze before any new camera loading/model forward. Sources/cuts/failed gates
remain unchanged. The existing best-epoch1 crack129 timeline is reused without
NN; all other timelines retain every original exposure, miss and unmatched box.
"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'output/evaluation/construction-posthoc-known-hit-videos'
BASE=ROOT/'datasets/construction/evcivil-round3'
SEQUENCES=['crack_129','spalling_76']
MODELS={'BEFORE':'evcivil-yolov8n-r4-real-lowlight-640-gain','AFTER':'evcivil-yolov8n-r4-last4-640-gain'}
CACHE=ROOT/'output/construction-development-evcivil-yolov8n-r4-real-lowlight-640-gain-crack_129.json'
TARGET=OUT/'two-posthoc-known-hit-protocol-before-forward.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def freeze():
    protocol_path=BASE/'protocol.json';original=json.loads(protocol_path.read_text())
    known={(row['group'],row['sequence']) for row in original['sequences']['calibration']}
    held={(row['group'],row['sequence']) for row in original['sequences']['heldout']}
    sources=[]
    for name in SEQUENCES:
        if ('dset_6',name) not in known or ('dset_6',name) in held:raise RuntimeError('Only exact declared CAL sources permitted before camera loading')
        folder=BASE/'raw/dset_6'/name
        label=folder/(name+'_frame_label.npy')
        if not label.exists():label=folder/(name+'_label.npy')
        paths=[folder/(name+'_frames.h5'),label]
        sources.append({'sequence':name,'group':'dset_6','role':'POST-HOC KNOWN-HIT CAL illustration',
                        'files':{path.relative_to(ROOT).as_posix():sha(path) for path in paths}})
    models={}
    for tag,stem in MODELS.items():
        path=ROOT/'models/construction'/(stem+'.onnx');point=path.with_suffix('.operating-point.json')
        p=json.loads(point.read_text())
        if p['model_sha256']!=sha(path) or not p['serving_decoder_parity_verified'] or p['class_confidence']!={'crack':.10,'spalling':.25}:
            raise RuntimeError('Exact fixed before/after model point differs')
        models[tag]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'point_sha256':sha(point),'metadata_sha256':sha(path.with_suffix('.provenance.json')),'class_confidence':p['class_confidence']}
    cached=json.loads(CACHE.read_text())
    if cached['model']['sha256']!=models['BEFORE']['sha256'] or cached['class_confidence']!=models['BEFORE']['class_confidence'] or cached['sequence']!='crack_129':
        raise RuntimeError('Original complete beforecrack129 cache identity differs')
    paths=[Path(__file__),ROOT/'scripts/compare_construction_development_videos.py',ROOT/'scripts/evaluate_construction_round3.py',ROOT/'scripts/construction_annotations.py',ROOT/'scripts/prepare_evcivil.py',ROOT/'scripts/prepare_evcivil_lossless.py',ROOT/'edge_runtime/profiles/construction_model.py',ROOT/'edge_runtime/profiles/construction_preprocessing.py']
    protocol={'schema_version':1,'selection_stage':'After old1171last4 CAL results are known, BEFORE any new selected whole-video forward',
              'selection_reason':'EXPLICIT POST-HOC KNOWN-HIT ILLUSTRATION: crack129 contains the1/62geometric-thin CAL match; spall76 contains4/5spall CAL matches. Outcome-based source selection is disclosed, not an independent benchmark or fix.',
              'independent_accuracy_credit':False,'changes_to_prior_cases_metrics_gates':False,'source_protocol_sha256':sha(protocol_path),
              'known_hit_metadata_sha256':sha(ROOT/'output/evaluation/construction-last4-candidate/known-calibration-hits-metadata.json'),
              'sources':sources,'models':models,'before_crack129_cache':{'path':CACHE.relative_to(ROOT).as_posix(),'sha256':sha(CACHE),'reuse':'Exact complete frozen prediction timeline, zero new NN for this pair member; recheck raw/NN-input hashes and original GT/PTS'},
              'executed_sources':{path.relative_to(ROOT).as_posix():sha(path) for path in paths},
              'scoring':'Class-aware one-to-one native publisherboxIoU>=.50; everyexposure/miss/wrongclass/unmatched retained; unlabelledframes unscored. No pixelGTmetric.',
              'visuals':'Unchanged RAW / actualsharednativeGAIN / externalpublisherGT / actualMODEL separatepanes; explicitPOST-HOC+FAILEDlabels. No sourcecrop, repeated/generatedframes or replacedcases.',
              'qualification':'Both models failed originalminimumguards. This illustrativepair does not change cuts, gates, all122results, old4caseaudit, heldoutstatus or modelpromotion.',
              'execution':'Only after explicitroot GPU/mediarelease; accuracy/encodingnotcapacity; failclosed partialattempts preserved.'}
    OUT.mkdir(parents=True,exist_ok=True)
    if TARGET.exists() and json.loads(TARGET.read_text())!=protocol:raise RuntimeError('Preserve fixed post-hoc protocol')
    write(TARGET,protocol);return protocol


def load_cases(protocol):
    import h5py,numpy as np
    from construction_annotations import publisher_labels_at_or_after
    publisher=json.loads((BASE/'publisher-record.json').read_text(encoding='utf-8'))
    authors='; '.join(a['full_name'] for a in publisher['authors'])+'; annotation contributor Xuanni Huo'
    cases=[]
    if [row['sequence'] for row in protocol['sources']]!=SEQUENCES:raise RuntimeError('Explicit two-source whitelist changed')
    for row in protocol['sources']:
        for path,digest in row['files'].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Known-hit camera/label source differs')
        name=row['sequence'];paths=[ROOT/p for p in row['files']]
        with h5py.File(next(path for path in paths if path.suffix=='.h5'),'r') as src:
            frames=src['frames'][:];times=src['frame_timestamp'][:].astype(np.int64)
        labels=np.load(next(path for path in paths if path.suffix=='.npy'),allow_pickle=False)
        if len(frames)!=len(times) or np.any(np.diff(times)<=0):raise RuntimeError('Original camera exposure/PTS contract differs')
        truth=[];period=float(np.median(np.diff(times)))
        for index,timestamp in enumerate(times):
            expected=[]
            for j,b in enumerate(publisher_labels_at_or_after(labels,timestamp,period)):
                x,y,w,h=map(float,b[2:6]);right=min(frames.shape[2],x+w);bottom=min(frames.shape[1],y+h);x=max(0,x);y=max(0,y)
                if right>x and bottom>y:expected.append({'id':f'{name}:{index}:{j}','label':'crack' if int(b[1])==0 else 'spalling','bbox':[x,y,right-x,bottom-y],'annotation_timestamp_us':int(b[0])})
            truth.append({'source_frame_index':index,'source_timestamp_us':int(timestamp),'video_time_s':(int(timestamp)-int(times[0]))/1e6,'expected':expected})
        cases.append({'name':name,'frames':frames,'times':times,'truth':truth,'source':{'source':'https://doi.org/10.11583/DTU.25924228','authors':authors,'license':'CC BY4.0','source_group':'dset_6','evaluation_role':row['role'],'original_camera_frames':len(frames),'source_files':row['files'],'independent_accuracy_credit':False,'timebase':'Original microsecond exposure PTS shifted tozero; complete genuine moving-camera footage'}})
    return cases


def export():
    import cv2,numpy as np
    from compare_construction_development_videos import panel_video,verify_replay
    from evaluate_construction_round3 import load_new,score_frame,summarize
    from edge_runtime.profiles.construction_preprocessing import camera_exposure
    from prepare_evcivil import encode_video
    from prepare_evcivil_lossless import encode_and_verify
    protocol=freeze();attempt=OUT/'export-started.json'
    if attempt.exists():raise RuntimeError('Preserve prior partial/completed post-hoc export; no silent retry')
    write(attempt,{'state':'started','protocol_sha256':sha(TARGET)})
    cases=load_cases(protocol);cached=json.loads(CACHE.read_text());aggregate={}
    if len(cached['frames'])!=len(next(case for case in cases if case['name']=='crack_129')['frames']):raise RuntimeError('Cached complete before timeline count differs')
    for case in cases:
        folder=BASE/'replay'/case['name'];proof=folder/(case['name']+'-lossless-frames.provenance.json')
        if not proof.exists():
            if folder.exists():raise RuntimeError('Preserve incomplete existing replay')
            folder.mkdir(parents=True);np.save(folder/'frames.npy',case['frames'],allow_pickle=False);np.save(folder/'timestamps-us.npy',case['times'],allow_pickle=False)
            write(folder/'annotations.json',{**case['source'],'frames':case['truth']});verification=encode_and_verify(folder)
        else:verification=json.loads(proof.read_text())
        case['source']['lossless_replay_input']=verification
        case['source']['replay_identity_reverified_against_raw']=verify_replay(case,verification,folder)
    for tag,identity in protocol['models'].items():
        model=load_new(ROOT/identity['path'],identity['class_confidence']);reports=[]
        for case in cases:
            target=ROOT/f'output/construction-posthoc-{tag.lower()}-{case["name"]}'
            if target.with_suffix('.json').exists() or target.with_suffix('.mp4').exists():raise RuntimeError('Preserve existing partial or complete post-hoc artifact')
            records=[];panels=[]
            for index,(gray,truth) in enumerate(zip(case['frames'],case['truth'])):
                raw=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);native,gain=camera_exposure(raw,model.exposure_policy)
                if tag=='BEFORE' and case['name']=='crack_129':
                    old=cached['frames'][index]
                    if old['source_frame_index']!=index or old['source_timestamp_us']!=truth['source_timestamp_us'] or old['expected']!=truth['expected'] or old['raw_native_bgr_sha256']!=hashlib.sha256(raw.tobytes()).hexdigest() or old['model_input_native_bgr_sha256']!=hashlib.sha256(native.tobytes()).hexdigest():raise RuntimeError('Cached before timeline/source/input identity differs')
                    predictions=old['predictions']
                else:predictions=model.infer(raw)
                record={**truth,**score_frame(predictions,truth['expected'],{'crack','spalling'}),'frame_size_wh':[346,260],'input_exposure_policy':model.exposure_policy,'input_exposure_gain':gain,'raw_native_bgr_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'model_input_native_bgr_sha256':hashlib.sha256(native.tobytes()).hexdigest()}
                records.append(record);panels.append(panel_video(raw,native,record,case['name'],'POST-HOC '+tag,'FAILED CALIBRATION / KNOWN-HIT ILLUSTRATION'))
            report={'schema_version':1,'purpose':'failure_diagnostic','matching_iou_threshold':.5,'sequence':case['name'],'candidate_status':'FAILED CALIBRATION / POST-HOC KNOWN-HIT','independent_accuracy_credit':False,'issue_summary':'Actual publisher-labelled crack or surface material loss; source selected post-hoc for known CAL hits, all failures retained.','why_it_matters':'Visible damage requires inspection; this illustration does not establish generalization or code acceptance.','next_check':'Verify actual material/substrate and dimensions; inspect source cause with appropriate instruments.','annotation_scope':'Post-hoc calibration development only; complete publisherboxes, unlabelledframes unscored; no pixelGT.','frame_size_wh':[346,260],'source':case['source'],'model':model.describe(),'class_confidence':identity['class_confidence'],'selection_protocol_sha256':sha(TARGET),'summary':summarize(records),'frames':records,'posthoc_known_hit_selection':True,'cached_predictions_reused_without_nn':tag=='BEFORE' and case['name']=='crack_129'}
            report['derived_video']=encode_video(target.with_suffix('.mp4'),np.asarray(panels),case['times'],20)
            first=next((i for i,r in enumerate(records) if r['expected']),0);preview=target.with_name(target.name+'-fixed-first-annotated').with_suffix('.jpg')
            if not cv2.imwrite(str(preview),panels[first]):raise RuntimeError('Could not save fixed first publisher-annotated preview')
            report['fixed_preview']={'selection':'First publisher-annotated exposure, never chosen for favorable inference','source_frame_index':first,'path':preview.relative_to(ROOT).as_posix(),'sha256':sha(preview)}
            write(target.with_suffix('.json'),report);reports.append({'sequence':case['name'],'report':target.with_suffix('.json').relative_to(ROOT).as_posix(),'report_sha256':sha(target.with_suffix('.json')),'video':target.with_suffix('.mp4').relative_to(ROOT).as_posix(),'video_sha256':sha(target.with_suffix('.mp4')),'summary':report['summary']})
        aggregate[tag]=reports;del model
    write(OUT/'complete-posthoc-paired-summary.json',aggregate);write(attempt,{'state':'completed','protocol_sha256':sha(TARGET),'summary_sha256':sha(OUT/'complete-posthoc-paired-summary.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['freeze','export'],required=True);args=parser.parse_args()
    if args.stage=='freeze':freeze();print(sha(TARGET),flush=True)
    else:export()
