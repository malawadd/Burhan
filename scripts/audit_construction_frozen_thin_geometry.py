"""Frozen R4head actual-box diagnostic; no NN, pixels or threshold selection."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/evaluation/construction-round3/evcivil-yolov8n-r4-head-640-gain/calibration-runtime-records.json'
OP=ROOT/'models/construction/evcivil-yolov8n-r4-head-640-gain.operating-point.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def relation(gt,pred):
    x,y,w,h=gt;a,b,c,d=pred
    overlap=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
    union=w*h+c*d-overlap;dx=a+c/2-(x+w/2);dy=b+d/2-(y+h/2)
    return {'iou':overlap/union if union else 0,'gt_box_coverage':overlap/(w*h),'prediction_box_coverage':overlap/(c*d),
        'pred_to_gt_area_ratio':c*d/(w*h),'center_dx_native_pixels':dx,'center_dy_native_pixels':dy,
        'center_distance_native_pixels':math.hypot(dx,dy),'center_distance_over_gt_diagonal':math.hypot(dx,dy)/math.hypot(w,h),
        'width_ratio':c/w,'height_ratio':d/h}
def matches(rows,expected,cut):
    remaining=set(range(len(expected)));result={}
    for i,p in enumerate(rows):
        options=[(relation(e['bbox'],p['bbox'])['iou'],j)for j,e in enumerate(expected)if j in remaining and e['label']==p['label']]
        value,j=max(options,default=(0,None))
        if j is not None and value>=cut:remaining.remove(j);result[i]=expected[j]['id']
    return result
def percentiles(values):
    if not values:return None
    values=sorted(values);result={}
    for name,q in [('min',0),('p25',.25),('median',.5),('p75',.75),('p95',.95),('max',1)]:
        i=(len(values)-1)*q;low=int(i);f=i-low;result[name]=values[low]*(1-f)+values[min(low+1,len(values)-1)]*f
    return result
def main():
    records=json.loads(SOURCE.read_text());op=json.loads(OP.read_text());all_ids=set();thin_ids=set();thin=[];errors=[];groups=Counter();by_source=defaultdict(Counter)
    primary=Counter();supplementary={str(c):Counter()for c in(.1,.25,.5)}
    for row_index,row in enumerate(records):
        if not row['evaluated']:raise ValueError('Frozen calibration row unexpectedly unscored')
        predictions=row['predictions'];expected=row['expected'];native=matches(predictions,expected,.5)
        for i,p in enumerate(predictions):
            if p['score']+1e-7<op['class_confidence'][p['label']]:errors.append('Frozen confidence contract differs')
            if p.get('matched_annotation_id')!=native.get(i):errors.append('Original strict matching identity differs')
        for e in expected:
            if e['id']in all_ids:raise ValueError('Duplicate annotation identity')
            all_ids.add(e['id'])
        truths=[e for e in expected if e['label']=='crack'and min(e['bbox'][2:])<=30 and max(e['bbox'][2:])>=2*min(e['bbox'][2:])]
        frame_thin_ids={e['id']for e in truths};thin_ids|=frame_thin_ids
        for cut in(.1,.25,.5):
            found=matches(predictions,expected,cut);supplementary[str(cut)]['thin_localized']+=sum(i in frame_thin_ids for i in found.values())
            supplementary[str(cut)]['all_crack_localized']+=sum(e['id']in found.values()for e in expected if e['label']=='crack')
        for e in truths:
            same=[(i,p,relation(e['bbox'],p['bbox']))for i,p in enumerate(predictions)if p['label']=='crack']
            wrong=[(i,p,relation(e['bbox'],p['bbox']))for i,p in enumerate(predictions)if p['label']!='crack']
            # At zero IoU, choose nearest center only as a declared diagnostic
            # tie-break; no box, confidence, match or operating point is changed.
            best=max(same,key=lambda v:(v[2]['iou'],-v[2]['center_distance_over_gt_diagonal'],-v[0]))if same else None
            bad=max(wrong,key=lambda v:(v[2]['iou'],-v[0]))if wrong else None
            anyoverlap=bool(best and best[2]['iou']>0)
            category=('no_predictions'if not predictions else'wrong_class_predictions_only'if not same else
                'same_class_present_without_overlap'if not anyoverlap else'same_class_overlap_below_0_1'if best[2]['iou']<.1 else
                'same_class_iou_0_1_to_0_25'if best[2]['iou']<.25 else'same_class_iou_0_25_to_0_5'if best[2]['iou']<.5 else'same_class_best_iou_ge_0_5')
            groups[category]+=1;by_source[row['source']][category]+=1
            primary['primary_iou_0_5_localized']+=e['id']in native.values()
            thin.append({'row_index':row_index,'source':row['source'],'original_frame_index':row['original_frame_index'],'annotation_id':e['id'],'gt_bbox':e['bbox'],
                'diagnostic_category':category,'same_class_prediction_count':len(same),'all_prediction_count':len(predictions),
                'same_class_any_positive_intersection':anyoverlap,'best_same_class_prediction':None if best is None else{'record_index':best[0],'id':best[1]['id'],'bbox':best[1]['bbox'],'score':best[1]['score'],**best[2]},
                'best_wrong_class_prediction':None if bad is None else{'id':bad[1]['id'],'label':bad[1]['label'],'bbox':bad[1]['bbox'],'score':bad[1]['score'],**bad[2]}})
    if len(records)!=122 or len(thin)!=62 or primary['primary_iou_0_5_localized']!=0 or op['geometric_thin_crack_audit']['localized_matches']!=0:errors.append('Expected122/62/0 frozen scope differs')
    withsame=[r for r in thin if r['best_same_class_prediction']];overlap=[r for r in withsame if r['same_class_any_positive_intersection']]
    aggregate={}
    for scope,rows in [('any_same_class_box_in_frame',withsame),('same_class_positive_intersection',overlap)]:
        aggregate[scope]={'annotations':len(rows),'geometry':{k:percentiles([r['best_same_class_prediction'][k]for r in rows])for k in('iou','gt_box_coverage','prediction_box_coverage','pred_to_gt_area_ratio','center_distance_native_pixels','center_distance_over_gt_diagonal','width_ratio','height_ratio')}}
    wrong_overlap=[r for r in thin if r['best_wrong_class_prediction']and r['best_wrong_class_prediction']['iou']>0]
    for scope,rows in [('wrong_class_box_anywhere',[r for r in thin if r['best_wrong_class_prediction']]),('wrong_class_positive_intersection',wrong_overlap)]:
        aggregate[scope]={'annotations':len(rows),'geometry':{k:percentiles([r['best_wrong_class_prediction'][k]for r in rows])for k in('iou','gt_box_coverage','prediction_box_coverage','pred_to_gt_area_ratio','center_distance_native_pixels','center_distance_over_gt_diagonal','width_ratio','height_ratio')}}
    result={'passed_consistency':not errors,'errors':errors,'sources':{'frozen_runtime_records':str(SOURCE.relative_to(ROOT)),'runtime_records_sha256':sha(SOURCE),'operating_point':str(OP.relative_to(ROOT)),'operating_point_sha256':sha(OP),
        'actual_model_sha256_from_operating_point':op['model_sha256'],'frozen_class_confidence':op['class_confidence'],'nms_iou':op['nms_iou'],'primary_matching_iou':.5},
        'scope':'Frozen actual serving boxes on original122 CAL only; no raw NN outputs/pixel decode/threshold changes/heldout. GT/prediction box overlap, not pixel-mask accuracy or physical crack width.',
        'thin_definition':'Native publisher GTbox short side<=30px and long side>=2*short; same62 as frozen audit',
        'counts':{'annotations':len(thin),'frames_with_thin':len({r['row_index']for r in thin}),'primary_iou_0_5_localized':primary['primary_iou_0_5_localized'],'same_class_box_anywhere_in_frame':len(withsame),'same_class_box_intersects_GT':len(overlap),'wrong_class_box_intersects_GT':len(wrong_overlap),'only_wrong_class_overlap_no_same_class_overlap':sum(not r['same_class_any_positive_intersection']for r in wrong_overlap)},
        'diagnostic_categories':dict(groups),'by_source':{k:dict(v)for k,v in by_source.items()},'descriptive_only_fixed_prediction_matching':{cut:dict(v)for cut,v in supplementary.items()},
        'distribution_scope':'For each GT, best actual same-class IoU; ties nearest center then saved prediction order. Zero-overlap geometry is conditional on same-class prediction existing, not meaningful localization.',
        'geometry_distributions':aggregate,'records':thin,
        'interpretation_limits':['Secondary IoU.1/.25 counts are descriptive geometry diagnostics, never new acceptance criteria or threshold selection.','Frozen prediction boxes/classes/scores are unchanged. Wrong-class overlap can indicate confusion under partial publisher scope; it does not prove a verified material false alarm.','YOLO boxes are continuous decoded coordinates before serving rounding; geometric-thin publisher boxes do not establish crack stroke width or a stride-imposed minimum.','Frames/annotations from one source capture are correlated; no new independent validation credit.']}
    target=ROOT/'output/evaluation/construction-r4head-frozen-thin-geometry-v2.json'
    result['initial_diagnostic_sha256']=sha(ROOT/'output/evaluation/construction-r4head-frozen-thin-geometry.json')
    if target.exists():raise ValueError('Preserve prior diagnostic')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:result[k]for k in('passed_consistency','counts','diagnostic_categories','by_source','descriptive_only_fixed_prediction_matching','geometry_distributions')},indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
