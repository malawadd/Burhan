"""L actual-GPU checkpoint maps and unchanged exact mask metrics; CAL only."""
import argparse,json,time,sys
import cv2,numpy as np
from factory_stfpm_bn_common import ROOT,OUT,NAME,SAVE_AT,CAL_SHA,sha,identity,write_new,verify_protocol,protocol_path,model_path,metadata,rank,validate_search_report,validate_bn_rows
sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.factory_encoder import FactorySegmentation,seeded_growth_mask
from factory_mask_metrics import component_mask_statistics,at_area,aggregate,original_pixel_weights
from train_factory_surface import samples
from probe_factory_stfpm_bn import parity


def summarize(rows):
    result=aggregate(rows);positive=[r for r in rows if r['defective']]
    result.update(meaningful_iou_0_1_recall=sum((r['pixel_iou'] or 0)>=.1 for r in positive)/len(positive),
                  coverage_50pct_recall=sum((r['foreground_pixel_recall'] or 0)>=.5 for r in positive)/len(positive),
                  worst_positive_iou=min((r['pixel_iou'] or 0) for r in positive),
                  worst_positive_coverage=min((r['foreground_pixel_recall'] or 0) for r in positive))
    return result


def validate_checkpoint(step,protocol):
    from prepare_factory_stfpm_bn import verify_bn_fit
    fit_receipt,_,fit=verify_bn_fit(protocol)
    path=model_path(step);receipt=OUT/f'checkpoint{step}-training.json';value=json.loads(receipt.read_text(encoding='utf-8'))
    export=OUT/f'{NAME}-step{step}-export.json';e=json.loads(export.read_text(encoding='utf-8'))
    execution=OUT/'training-execution-before-optimizer.json';x=json.loads(execution.read_text(encoding='utf-8'))
    if value['step']!=step or value['execution_receipt_sha256']!=sha(execution) or x['protocol_sha256']!=sha(protocol_path()):
        raise ValueError('Learned checkpoint/execution lineage changed')
    if not value['teacher_unchanged'] or value['teacher_state_sha256']!=x['initial_teacher_state_sha256']:
        raise ValueError('Teacher state must remain exactly frozen')
    if sha(path)!=value['onnx_sha256'] or sha(path)!=e['model_sha256'] or sha(path.with_suffix('.pt'))!=value['state_sha256']:
        raise ValueError('Learned model/state changed')
    if e['protocol_sha256']!=sha(protocol_path()) or sha(path.with_suffix('.onnx.json'))!=e['initial_metadata_sha256']:
        raise ValueError('Initial learned metadata/protocol changed before selection')
    for row in [value,e,x]:
        if row['BN_fit_receipt_sha256']!=sha(fit_receipt) or row['BN_fitted_state_sha256']!=fit['state_sha256']:raise ValueError('Learned BN fitting lineage changed')
    validate_bn_rows(value['BN_rows'],protocol['actual_BN_factory']['student_BN'],64+step)
    validate_bn_rows(x['initial_BN'],protocol['actual_BN_factory']['student_BN'],64)
    return path,receipt


def cached_rows(step,protocol,path):
    cache_path=OUT/f'checkpoint{step}-actualGPU-CAL-scoremaps.npz';identities=[];rows=[]
    # Existing maps may recover only the fixed CPU search; never repeat NN.
    saved=np.load(cache_path,allow_pickle=False) if cache_path.exists() else None
    if saved is not None:
        if str(saved['model_sha256'])!=sha(path) or str(saved['protocol_sha256'])!=sha(protocol_path()):raise ValueError('Score cache binding changed')
        saved_sources=json.loads(str(saved['source_identities_json']));saved_scores=saved['scores']
        if saved_scores.shape!=(80,640,256) or not np.isfinite(saved_scores).all() or saved_scores.min()<0 or saved_scores.max()>1:
            raise ValueError('Invalid cached native anomaly maps')
    base=ROOT/'datasets/factory/KolektorSDD';times=[];pending=[]
    for index,(image,truth,defective) in enumerate(samples(base,protocol['split_items']['calibration'])):
        frame=cv2.imread(str(image));mask=image.with_name(image.stem+'_label.bmp')
        item={'image':str(image.relative_to(base)),'defective':bool(defective),'image_sha256':sha(image),'mask_sha256':sha(mask),
              'native_image_shape':list(frame.shape),'native_mask_shape':list(truth.shape)}
        identities.append(item)
        if saved is not None and item!=saved_sources[index]:raise ValueError('CAL source/GT/native geometry drift')
        pending.append((frame,truth,bool(defective),item))
    if len(identities)!=80 or sum(r['defective'] for r in identities)!=10 or identity(identities)!=CAL_SHA:
        raise ValueError('Only protected ordered80 original CAL permitted')
    # Validate the complete identity BEFORE compiling/running any CAL NN.
    if saved is None:model=FactorySegmentation(path,output_logits=False,precision_hint='f16',disable_winograd_convolution=True)
    for index,(frame,truth,defective,item) in enumerate(pending):
        if saved is not None:score=saved_scores[index]
        else:
            tick=time.perf_counter();score,_,_=model.predict(frame,.05,4);times.append(1000*(time.perf_counter()-tick))
        rows.append({'image':item['image'],'defective':defective,'truth':truth,'scores':score,
                     'projection_weights':original_pixel_weights(truth,score.shape)})
    for item in identities:
        source=base/item['image'];mask=source.with_name(source.stem+'_label.bmp')
        if sha(source)!=item['image_sha256'] or sha(mask)!=item['mask_sha256']:raise ValueError('CAL source changed during inference')
    if saved is None:
        with cache_path.open('xb') as stream:
            np.savez_compressed(stream,scores=np.stack([r['scores'] for r in rows]),model_sha256=np.asarray(sha(path)),
                                protocol_sha256=np.asarray(sha(protocol_path())),source_identities_json=np.asarray(json.dumps(identities)))
    else:saved.close()
    return rows,identities,cache_path,times


def select(cache,protocol):
    raw=[]
    for high in protocol['raw_high_grid']:
        per_area={area:[] for area in protocol['area_grid']}
        for row in cache:
            statistics=component_mask_statistics(row['scores'],high,row['truth'],row['projection_weights'])
            for area in per_area:per_area[area].append({'defective':row['defective'],**at_area(statistics,area)})
        for area,rows in per_area.items():raw.append({'high':high,'area':area,**summarize(rows)})
    seeds=[]
    for high in protocol['raw_high_grid']:
        eligible=[r for r in raw if r['high']==high and r['normal_fp']<=1]
        if eligible:seeds.append(max(eligible,key=rank))
    seeds=sorted(seeds,key=rank,reverse=True)[:6]
    grown=[];records={}
    for seed in seeds:
        for low in protocol['growth_grid']:
            if low is not None and low>seed['high']:continue
            rows=[]
            for row in cache:
                binary=seeded_growth_mask(row['scores'],seed['high'],seed['area'],low)
                values=at_area(component_mask_statistics(binary.astype(np.float32),.5,row['truth'],row['projection_weights']),1)
                rows.append({'image':row['image'],'defective':row['defective'],**values})
            point={'high':seed['high'],'area':seed['area'],'growth':low,**summarize(rows)}
            if point['normal_fp']!=seed['normal_fp']:raise ValueError('Growth must not create a flagged normal view')
            grown.append(point);records[(seed['high'],seed['area'],low)]=rows
    eligible=[p for p in grown if p['normal_fp']<=1 and p['foreground_pixel_precision']>=.65]
    winner=max(eligible,key=rank) if eligible else None
    return raw,grown,winner,records.get((winner['high'],winner['area'],winner['growth']),[]) if winner else []


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--step',type=int,choices=SAVE_AT,required=True)
    parser.add_argument('--root-cal-released',action='store_true');parser.add_argument('--contention-note',required=True);args=parser.parse_args()
    if not args.root_cal_released:raise ValueError('Actual coordinated CAL GPU release is required')
    cv2.setNumThreads(1);_,protocol=verify_protocol();step=args.step
    target=OUT/f'checkpoint{step}-calibration.json'
    if target.exists():raise ValueError('Preserve completed CAL report')
    path,training=validate_checkpoint(step,protocol)
    parity_path=OUT/f'{NAME}-step{step}-gpu-parity.json'
    if not parity_path.exists():parity(step,protocol)
    guard=json.loads(parity_path.read_text(encoding='utf-8'))
    if not guard['all_checks_passed'] or guard['model_sha256']!=sha(path) or guard['protocol_sha256']!=sha(protocol_path()):
        raise ValueError('Fitted numeric guard failed/stale; no CAL access')
    started=time.perf_counter();cache,sources,cache_path,times=cached_rows(step,protocol,path)
    raw,grown,point,records=select(cache,protocol)
    report={'step':step,'checkpoint':str(path),'model_sha256':sha(path),'state_sha256':sha(path.with_suffix('.pt')),
            'protocol_sha256':sha(protocol_path()),'training_report_sha256':sha(training),'parity_report_sha256':sha(parity_path),
            'scorecache_sha256':sha(cache_path),'calibration_source_identity':{'ordered_sources':sources,'sha256':identity(sources)},
            'selected':point,'selected_records':records,'raw_grid':raw,'grown_grid':grown,'elapsed_seconds':time.perf_counter()-started,
            'scope':args.contention_note+'; accuracy-only, no capacity claim','actual_GPU_maps':True,
            'raw_predict_ms':{'p50':float(np.median(times)),'p95':float(np.percentile(times,95))} if times else None}
    validate_search_report(report,protocol)
    if point:
        initial=path.with_suffix('.onnx.json');history=OUT/f'checkpoint{step}-initial-metadata.json'
        write_new(history,json.loads(initial.read_text(encoding='utf-8')))
        meta=metadata(path,protocol,point)
        meta.update(calibration_selected_point=point,calibration_source_identity_sha256=CAL_SHA,
                    calibration_protocol_sha256=sha(protocol_path()),learned_state_sha256=report['state_sha256'],
                    training_execution_receipt_sha256=sha(OUT/'training-execution-before-optimizer.json'))
        temporary=initial.with_name(initial.name+'.selected-writing')
        if temporary.exists():raise ValueError('Preserve interrupted metadata write')
        temporary.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');temporary.replace(initial)
        report['metadata_sha256']=sha(initial)
    write_new(target,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['raw_grid','grown_grid','selected_records','calibration_source_identity']},indent=2))


if __name__=='__main__':main()
