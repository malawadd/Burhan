"""Independent JSON-only construction recording count/geometry/timeline audit.

No video/image pixels, model assets, inference framework or producer scorer is
loaded. Existing registries and evidence reports are never modified.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def identifier(value):return isinstance(value,(str,int)) and not isinstance(value,bool) and value!=''
def bbox(value,size=None):
    if not isinstance(value,(list,tuple)) or len(value)!=4 or not all(number(v) for v in value):raise ValueError('bbox must contain four finite numbers')
    x,y,w,h=value
    # Publisher projection can retain roundoff such as -1.5e-14 at an edge.
    if x < -1e-6 or y < -1e-6 or w<=0 or h<=0:raise ValueError('bbox has invalid nonnegative xy/positive wh geometry')
    if size and (x+w>size[0]+1e-5 or y+h>size[1]+1e-5):raise ValueError('bbox exceeds declared source dimensions')
    return value

def iou(left,right):
    x,y,w,h=left;xx,yy,ww,hh=right
    intersection=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return intersection/(w*h+ww*hh-intersection)

def fresh_class():return {'tp':0,'fn':0,'fp':0,'unsupported_annotations':0}

def protect_inputs(inputs,outputs):
    originals={Path(path).resolve() for path in inputs}
    destinations=[Path(path).resolve() for path in outputs]
    if len(set(destinations))!=len(destinations) or any(path in originals for path in destinations):
        raise ValueError('Audit output paths must be distinct and cannot overwrite the registry or evidence inputs')

def compare_counts(declared,actual,location,errors):
    if not isinstance(declared,dict):errors.append(location+' must be an object');return
    aliases={'unsupported_defects':'unsupported_annotations'}
    for key,value in declared.items():
        mapped=aliases.get(key,key)
        if mapped not in actual:continue
        expected=actual[mapped]
        if value is None and key=='false_alerts' and not actual.get('evaluated',True):continue
        if isinstance(expected,bool):valid=isinstance(value,bool) and value==expected
        elif isinstance(expected,float) or expected is None:
            valid=(value is None if expected is None else number(value) and abs(value-expected)<=1e-8)
        else:valid=isinstance(value,int) and not isinstance(value,bool) and value==expected
        if not valid:errors.append(f'{location}.{key}: declared {value!r}, recomputed {expected!r}')

def audit_document(document):
    errors=[];warnings=[];computed=[];class_totals=defaultdict(fresh_class)
    if not isinstance(document,dict):return {'passed':False,'errors':['evidence must be an object'],'warnings':[],'recomputed':{}}
    frames=document.get('frames')
    if not isinstance(frames,list):return {'passed':False,'errors':['frames must be an array'],'warnings':[],'recomputed':{}}
    threshold=document.get('matching_iou_threshold',.5)
    if not number(threshold) or abs(threshold-.5)>1e-8:errors.append('matching_iou_threshold must be the declared0.50 whole-instance box metric')
    previous_index=previous_time=previous_timestamp=None
    first_timestamp=first_video_time=None
    all_indices=[]
    for position,frame in enumerate(frames):
        location=f'frames[{position}]'
        try:
            if not isinstance(frame,dict):raise ValueError('frame is not an object')
            evaluated=frame.get('evaluated') is True
            if 'evaluated' not in frame or not isinstance(frame['evaluated'],bool):raise ValueError('evaluated must be an explicit boolean')
            size=frame.get('frame_size_wh',document.get('frame_size_wh'))
            if size is not None and (not isinstance(size,list) or len(size)!=2 or not all(number(v) and v>0 for v in size)):raise ValueError('frame_size_wh is invalid')
            expected=frame.get('expected');predictions=frame.get('predictions')
            if not isinstance(expected,list) or not isinstance(predictions,list):raise ValueError('expected/predictions must be arrays')
            annotations={};known=[];counts=defaultdict(fresh_class)
            for truth in expected:
                key=truth.get('id');label=truth.get('label')
                if not identifier(key) or key in annotations:raise ValueError('annotation IDs must be unique valid strings/integers per frame')
                if not isinstance(label,str) or not label:raise ValueError('annotation label missing')
                if truth.get('evaluation_status') not in {None,'unsupported'}:raise ValueError('unknown annotation evaluation_status')
                bbox(truth.get('bbox'),size);annotations[key]=truth
                if truth.get('evaluation_status')=='unsupported':counts[label]['unsupported_annotations']+=1
                else:known.append(truth)
            if known and not evaluated:warnings.append(f'{location}: {len(known)} supported annotations explicitly unevaluated; excluded from accuracy')
            remaining=set(range(len(known)));claimed=set();prediction_ids=set()
            tp=fp=0;unscored=0
            for p_index,prediction in enumerate(predictions):
                label=prediction.get('label');match=prediction.get('matched_annotation_id');status=prediction.get('evaluation_status')
                if not isinstance(label,str) or not label:raise ValueError('prediction label missing')
                counts[label]
                if 'matched_annotation_id' not in prediction or (match is not None and not identifier(match)):raise ValueError('matched_annotation_id missing or invalid')
                bbox(prediction.get('bbox'),size)
                if not number(prediction.get('score')) or not 0<=prediction['score']<=1:raise ValueError('prediction score invalid')
                if not number(prediction.get('iou')) or not 0<=prediction['iou']<=1:raise ValueError('prediction IoU invalid')
                if 'id' in prediction:
                    key=prediction['id']
                    if not identifier(key) or key in prediction_ids:raise ValueError('prediction IDs must be unique per frame')
                    prediction_ids.add(key)
                if status not in {None,'matched','false_alert','unscored'}:raise ValueError('unknown prediction evaluation_status')
                scored=evaluated and status!='unscored'
                if not scored:
                    unscored+=1
                    if match is not None or status=='matched':raise ValueError('unscored/unevaluated prediction cannot claim a match')
                    continue
                # Independent geometric assignment in recorded candidate order,
                # matching the declared greedy protocol, with deterministic GT ties.
                choices=[(iou(prediction['bbox'],truth['bbox']),index) for index,truth in enumerate(known)
                         if index in remaining and truth['label']==label]
                overlap,index=max(choices,default=(0,None))
                recomputed_match=known[index]['id'] if index is not None and overlap>=.5 else None
                if match!=recomputed_match:raise ValueError(f'prediction{p_index} claimed match {match!r}, actual class-aware greedy match {recomputed_match!r}')
                if match is not None:
                    if not identifier(match) or match not in annotations or match in claimed:raise ValueError('duplicate/unknown matched annotation ID')
                    truth=annotations[match];actual=iou(prediction['bbox'],truth['bbox'])
                    if truth.get('evaluation_status')=='unsupported' or truth['label']!=label or actual+1e-8<.5:raise ValueError('invalid class/unsupported/low-IoU match')
                    if abs(prediction['iou']-actual)>1e-5:raise ValueError(f'matched IoU does not equal actual geometry ({actual})')
                    if status=='false_alert':raise ValueError('matched prediction marked false_alert')
                    claimed.add(match);remaining.remove(index);counts[label]['tp']+=1;tp+=1
                else:
                    if status=='matched':raise ValueError('unmatched prediction marked matched')
                    best=max((iou(prediction['bbox'],truth['bbox']) for truth in known if truth['label']==label),default=0.)
                    if abs(prediction['iou']-best)>1e-5:raise ValueError(f'unmatched IoU does not equal actual best same-class geometry ({best})')
                    counts[label]['fp']+=1;fp+=1
            if evaluated:
                for index in remaining:counts[known[index]['label']]['fn']+=1
            fn=sum(c['fn'] for c in counts.values());unsupported=sum(c['unsupported_annotations'] for c in counts.values())
            actual={'evaluated':evaluated,'localized_matches':tp,'missed_defects':fn,'false_alerts':fp,'unsupported_annotations':unsupported}
            compare_counts(frame,actual,location,errors)
            for label,declared in frame.get('by_class',{}).items():compare_counts(declared,counts[label],location+'.by_class.'+label,errors)
            for label,c in counts.items():
                for key,value in c.items():class_totals[label][key]+=value
            computed.append({**actual,'annotated':bool(expected),'annotations':len(expected),'unscored_predictions':unscored})
            index=frame.get('source_frame_index');time=frame.get('video_time_s');timestamp=frame.get('source_timestamp_us')
            if not isinstance(index,int) or isinstance(index,bool) or index<0:raise ValueError('source_frame_index invalid')
            if not number(time) or time<0:raise ValueError('video_time_s invalid')
            if previous_index is not None and index<=previous_index:raise ValueError('source_frame_index must strictly increase')
            if previous_time is not None and time<=previous_time:raise ValueError('video_time_s must strictly increase')
            all_indices.append(index);previous_index=index;previous_time=time
            if timestamp is not None:
                if not number(timestamp):raise ValueError('source_timestamp_us invalid')
                if previous_timestamp is not None and timestamp<=previous_timestamp:raise ValueError('source timestamps must strictly increase')
                if first_timestamp is None:first_timestamp,first_video_time=timestamp,time
                if abs((time-first_video_time)-(timestamp-first_timestamp)/1e6)>1e-6:raise ValueError('video_time_s intervals differ from original source timestamp intervals')
                previous_timestamp=timestamp
        except (ValueError,TypeError,KeyError,AttributeError) as exc:errors.append(f'{location}: {exc}')
    summary={key:sum(row[key] for row in computed) for key in ['localized_matches','missed_defects','false_alerts','unsupported_annotations']}
    summary.update(frames=len(frames),audited_frames=len(computed),annotated_frames=sum(row['annotated'] for row in computed),
                   evaluated_frames=sum(row['evaluated'] for row in computed),annotations=sum(row['annotations'] for row in computed),
                   unscored_predictions=sum(row['unscored_predictions'] for row in computed))
    tp,fn,fp=summary['localized_matches'],summary['missed_defects'],summary['false_alerts']
    summary.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                   all_annotated_class_localization_coverage=tp/(tp+fn+summary['unsupported_annotations']) if tp+fn+summary['unsupported_annotations'] else None)
    declared=document.get('summary',{})
    if not isinstance(declared,dict):errors.append('summary must be an object');declared={}
    for key in ['annotated_frames','localized_matches','missed_defects','false_alerts']:
        if key not in declared:errors.append('summary missing '+key)
    if not any(key in declared for key in ['unsupported_annotations','unsupported_defects']):errors.append('summary missing unsupported annotation count or historical alias')
    compare_counts(declared,summary,'summary',errors)
    declared_classes=declared.get('by_class',{})
    if not isinstance(declared_classes,dict):errors.append('summary.by_class must be an object');declared_classes={}
    for label in declared_classes:class_totals[label]
    for label,values in class_totals.items():
        a,b,c=values['tp'],values['fn'],values['fp'];values.update(precision=a/(a+c) if a+c else None,recall=a/(a+b) if a+b else None)
    for label,values in declared_classes.items():compare_counts(values,class_totals[label],'summary.by_class.'+label,errors)
    source=document.get('source',{});original=source.get('original_camera_frames')
    timebase=''.join(source.get('timebase','').lower().split())
    complete=document.get('complete_sequence') is True or any(text in timebase for text in ['oneframeperexposure','oneframerecordperoriginalexposure'])
    if complete and original is not None:
        if not isinstance(original,int) or isinstance(original,bool) or original<0:errors.append('Complete original sequence count must be a nonnegative integer')
        elif len(frames)!=original or all_indices!=list(range(original)):errors.append('Complete original sequence declaration disagrees with record count or contiguous source indices')
        if frames and number(frames[0].get('video_time_s')) and abs(frames[0]['video_time_s'])>1e-6 and 'shiftedtozero' in timebase:errors.append('Complete zero-shifted sequence must start video_time_s atzero')
    checks=[('derived_video.encoded_frames',document.get('derived_video',{}).get('encoded_frames')),
            ('source.encoding_verification.encoded_frames',source.get('encoding_verification',{}).get('encoded_frames'))]
    if complete:
        replay=source.get('lossless_replay_input',{})
        checks.extend((f'source.lossless_replay_input.{key}',replay.get(key)) for key in ['original_camera_exposures','opencv_decoded_frames'])
    for key,value in checks:
        if value is not None and value!=len(frames):errors.append(f'{key}: {value} differs from{len(frames)} saved frame records')
    summary['by_class']=dict(class_totals)
    return {'passed':not errors,'errors':errors,'warnings':warnings,'recomputed':summary,
            'timeline_scope':'JSON source indices, PTS and declared encoding/replay counts only; actual container/pixels are not reopened or independently verified here.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,default=ROOT/'config/construction-videos.json')
    parser.add_argument('--evidence-paths',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path,default=ROOT/'output/evaluation/construction-recording-count-audit.json')
    parser.add_argument('--markdown',type=Path,default=ROOT/'docs/construction-recording-count-audit.md')
    args=parser.parse_args();registry=json.loads(args.registry.read_text(encoding='utf-8'));paths={};context=[]
    for row in registry:
        if row.get('evidence_path'):paths[(ROOT/row['evidence_path']).resolve()]=row['id']
        else:context.append(row['id'])
    for path in args.evidence_paths:paths[path.resolve()]='optional diagnostic'
    protect_inputs([args.registry,*paths],[args.output,args.markdown])
    rows=[]
    for path,label in paths.items():
        try:
            raw=path.read_bytes();result=audit_document(json.loads(raw.decode('utf-8')))
            rows.append({'path':str(path),'registry_id':label,'sha256':hashlib.sha256(raw).hexdigest(),**result})
        except (OSError,ValueError,TypeError) as exc:rows.append({'path':str(path),'registry_id':label,'passed':False,'errors':[str(exc)],'warnings':[],'recomputed':{}})
    report={'scope':'Independent read-only JSON count/actual-box-IoU audit. No NN, media pixels, model assets or active report/registry edits.',
            'matching':'Class-aware greedy in recorded candidate order, strict one-to-one at IoU0.50; evaluated=false frames and unscored predictions excluded from accuracy; unsupported labels counted separately. FP means unmatched model boxes under publisher annotation scope, not independently verified material false alarms.',
            'registry_sha256':hashlib.sha256(args.registry.read_bytes()).hexdigest(),'context_only_without_evidence':context,
            'passed':all(r['passed'] for r in rows),'recordings':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Independent construction recording count audit','',report['scope'],'',report['matching'],'',
           '**'+('PASS' if report['passed'] else 'FAILURES FOUND')+'**. This verifies saved counts and JSON timeline consistency, not detector quality, real-time speed, pixel segmentation, physical-site independence or Saudi code compliance.','',
           '| Evidence | Result | Frames | TP | FN | Unmatched boxes (FP) | Unsupported |','|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        c=row['recomputed'];lines.append(f"| [{Path(row['path']).name}]({row['path'].replace(chr(92),'/')}) | {'PASS' if row['passed'] else 'FAIL'} | {c.get('frames','?')} | {c.get('localized_matches','?')} | {c.get('missed_defects','?')} | {c.get('false_alerts','?')} | {c.get('unsupported_annotations','?')} |")
    for row in rows:
        if row['errors'] or row['warnings']:
            lines.extend(['','## '+Path(row['path']).name,''])
            lines.extend('- ERROR: '+value for value in row['errors']);lines.extend('- NOTE: '+value for value in row['warnings'])
    lines.extend(['','Recordings without external evidence JSON were skipped, without granting any accuracy credit: '+', '.join(context)+'.',
                  '',f"[Full machine-readable audit]({args.output.resolve().as_posix()})",'',
                  'Unmatched boxes include partial coverage that fails whole-instance IoU and predictions outside the available publisher labels. Counts are unchanged; they are not verified material false alarms. Class-negative false-positive rates require a separate completely annotated negative dataset, such as the reserved DACL polygon-negative scope.', '',
                  'Actual encoded container frame counts/timestamps and source pixels are not independently re-decoded by this audit; only their saved declarations are checked against frame records.'])
    args.markdown.parent.mkdir(parents=True,exist_ok=True);args.markdown.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'audited':len(rows),'failures':sum(not r['passed'] for r in rows),'json':str(args.output),'markdown':str(args.markdown)},indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
