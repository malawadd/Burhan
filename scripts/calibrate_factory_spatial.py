"""Freeze spatial U-Net checkpoint/threshold using calibration masks only.

No comparison data is read. Genuine mask overlap is required; image flagging
without touching a defect cannot improve localized recall.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from edge_runtime.profiles.factory_encoder import FactorySegmentation
from factory_mask_metrics import component_mask_statistics,at_area,aggregate
from train_factory_surface import samples


def evaluate(path,original,protocol):
    started=time.perf_counter();model=FactorySegmentation(path,output_logits=True)
    startup=time.perf_counter()-started;cache=[];times=[]
    for image,mask,defective in samples(ROOT/'datasets/factory/KolektorSDD',original['split_items']['calibration']):
        frame=cv2.imread(str(image));tick=time.perf_counter()
        scores,_,_=model.predict(frame,.5,1);times.append((time.perf_counter()-tick)*1000)
        row={'path':str(image.relative_to(ROOT/'datasets/factory/KolektorSDD')),'defective':bool(defective),'points':{}}
        for threshold in protocol['threshold_grid']:
            statistics=component_mask_statistics(scores,threshold,mask)
            for area in protocol['minimum_area_grid']:
                row['points'][f'{threshold}:{area}']=at_area(statistics,area)
        cache.append(row)
    choices=[]
    for threshold in protocol['threshold_grid']:
        for area in protocol['minimum_area_grid']:
            rows=[{'defective':row['defective'],**row['points'][f'{threshold}:{area}']} for row in cache]
            result=aggregate(rows)
            positives=[row for row in rows if row['defective']]
            result['meaningful_localization_recall_iou_0_1']=sum((row['pixel_iou'] or 0)>=.1 for row in positives)/max(1,len(positives))
            result['gt_coverage_10pct_recall']=sum((row['foreground_pixel_recall'] or 0)>=.1 for row in positives)/max(1,len(positives))
            result['gt_coverage_50pct_recall']=sum((row['foreground_pixel_recall'] or 0)>=.5 for row in positives)/max(1,len(positives))
            choices.append({'threshold':threshold,'area':area,**result})
    eligible=[row for row in choices if row['normal_view_false_positive_rate']<=3/70]
    selected=max(eligible,key=lambda row:(row['meaningful_localization_recall_iou_0_1'],row['localized_view_recall'],row['micro_pixel_iou'],
                                          row['foreground_pixel_recall'],-row['normal_fp'],row['threshold'])) if eligible else None
    return {'checkpoint':str(path),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'selected':selected,'choices':choices,
            'records':cache,'startup_seconds':startup,
            'inference_ms':{'p50':float(np.median(times)),'p95':float(np.percentile(times,95)),'mean':float(np.mean(times))},
            'execution_devices':model.execution_devices,'elapsed_seconds':time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='spatial_unet_v9')
    parser.add_argument('--steps',type=int,nargs='+',default=[300,600])
    args=parser.parse_args();cv2.setNumThreads(2)
    directory=ROOT/'output/optimization/factory-round2'
    original=json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
    protocol=json.loads((directory/(args.name+'-protocol-before-fit.json')).read_text(encoding='utf-8'))
    selection={'written_before_calibration_inference':True,'checkpoints':args.steps,
               'normal_false_positive_budget':3/70,'selection_order':['View pixelIoU>=.1 recall','Any-overlap localized view recall','micro pixelIoU','foreground pixel recall','fewer normal false positives'],
               'localization_rule':'Retained predicted mask pixels nearest projected to original-resolution, no validation-mask dilation',
               'comparison_images_opened':0}
    (directory/(args.name+'-calibration-protocol-before-scoring.json')).write_text(json.dumps(selection,indent=2),encoding='utf-8')
    reports=[]
    for step in args.steps:
        path=ROOT/'models/factory'/f'{args.name}-step{step}.onnx'
        report=evaluate(path,original,protocol);reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k not in {'records','choices'}}),flush=True)
        (directory/(args.name+'-calibration.json')).write_text(json.dumps({'scope':'Calibration-only checkpoint/threshold selection; no comparison data opened','reports':reports},indent=2),encoding='utf-8')
    eligible=[r for r in reports if r['selected'] is not None]
    if not eligible:
        raise RuntimeError('No checkpoint meets calibration false-positive budget')
    selected=max(eligible,key=lambda r:(r['selected']['meaningful_localization_recall_iou_0_1'],r['selected']['localized_view_recall'],r['selected']['micro_pixel_iou'],
                                        r['selected']['foreground_pixel_recall'],-r['selected']['normal_fp']))
    path=Path(selected['checkpoint']);point=selected['selected']
    provenance={key:original[key] for key in ['split_items','split_seed','supported_height_width_ratio','dataset','dataset_url','dataset_archive_url','dataset_archive_sha256','license']}
    metadata={**provenance,'feature_version':'surface_mobilenet_seg_v1','profile_version':args.name,
              'model_name':'End-to-end fine-tuned spatial MobileNet U-Net defect segmentation',
              'trained_utc':datetime.now(timezone.utc).isoformat(),'field_validated':False,
              'trust':'Locally trained/exported ONNX; application verifies artifact SHA-256 and explicit Intel GPU; training checkpoints are local-only weights, never untrusted pickle inputs',
              'output_logits':True,'pixel_threshold':point['threshold'],'minimum_component_pixels':point['area'],
              'input_width':256,'model_sha256':selected['model_sha256'],'training_protocol':protocol,
              'threshold_selection':selection,'trees':0,
              'score_meaning':'Spatial neural sigmoid defect score; not calibrated probability',
              'calibration_selected_point':point,'manual_review_only':True,
              'automatic_acceptance':False,'automatic_rejection':False,
              'localization_validation':'Original-resolution ground-truth mask pixels; any overlap and pixelIoU reported separately',
              'comparison_scope':'Original comparison items repeatedly exposed; reused development comparison, not blind'}
    if protocol.get('warmstart_checkpoint'):
        pretrain_name=Path(protocol['warmstart_checkpoint']).stem.rsplit('-step',1)[0]
        lineage=directory/(pretrain_name+'-training.json')
        if lineage.is_file():
            metadata['pretraining_lineage']={'training_report':str(lineage),'training_report_sha256':hashlib.sha256(lineage.read_bytes()).hexdigest(),
                                            'protocol':json.loads(lineage.read_text(encoding='utf-8'))['protocol']}
    path.with_suffix('.onnx.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (directory/(args.name+'-frozen-selection.json')).write_text(json.dumps({'checkpoint':str(path),'point':point,'model_sha256':selected['model_sha256'],
                                                                       'metadata_sha256':hashlib.sha256(path.with_suffix('.onnx.json').read_bytes()).hexdigest(),
                                                                       'comparison_images_opened':0},indent=2),encoding='utf-8')
    print('FROZEN '+str(path),flush=True)


if __name__=='__main__':main()
