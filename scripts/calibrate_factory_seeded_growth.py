"""Calibration-only hysteresis refinement; raw neural scores remain unmodified."""
from pathlib import Path
import hashlib,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory_encoder import FactorySegmentation,seeded_growth_mask
from factory_mask_metrics import component_mask_statistics,at_area,aggregate
from train_factory_surface import samples


def main():
    cv2.setNumThreads(2)
    directory=ROOT/'output/optimization/factory-round2'
    original=ROOT/'models/factory/spatial_unet_v10_sdd2-step300.onnx'
    metadata=json.loads(original.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    high,area=metadata['pixel_threshold'],metadata['minimum_component_pixels']
    grid=[None,.1,.25,.5,.75,.9,.95,.98,.99,.995]
    protocol={'scope':'Calibration only; no original80 comparison/officialSDD2test opened',
              'frozen_seed_model_sha256':metadata['model_sha256'],'seed_threshold':high,'seed_minimum_area':area,
              'growth_threshold_grid':grid,'retention':'Largest20 high-seed components; keep low-score connected components only if they touch a retained seed; never modify raw scores',
              'selection':'Maximize views pixelIoU>=.1, then micro pixelIoU, then foreground recall; include no-growth baseline. Normal flags must exactly match no-growth for each view.',
              'comparison_images_opened':0}
    (directory/'seeded-growth-protocol-before-scoring.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    started=time.perf_counter();model=FactorySegmentation(original,output_logits=True);records=[]
    for path,truth,defective in samples(ROOT/'datasets/factory/KolektorSDD',metadata['split_items']['calibration']):
        scores,_,_=model.predict(cv2.imread(str(path)),high,area)
        result={}
        for low in grid:
            binary=seeded_growth_mask(scores,high,area,low)
            values=at_area(component_mask_statistics(binary.astype(np.float32),.5,truth),1)
            result[str(low)]=values
            if low is not None:assert values['flagged']==result['None']['flagged'],'Growth cannot create a new flagged view'
        records.append({'image':str(path.relative_to(ROOT/'datasets/factory/KolektorSDD')),'defective':bool(defective),'points':result})
    choices=[]
    for low in grid:
        rows=[{'defective':row['defective'],**row['points'][str(low)]} for row in records]
        value=aggregate(rows);positive=[row for row in rows if row['defective']]
        value['meaningful_localization_recall_iou_0_1']=sum((row['pixel_iou'] or 0)>=.1 for row in positive)/max(1,len(positive))
        value['gt_coverage_50pct_recall']=sum((row['foreground_pixel_recall'] or 0)>=.5 for row in positive)/max(1,len(positive))
        choices.append({'growth_threshold':low,**value})
    selected=max(choices,key=lambda row:(row['meaningful_localization_recall_iou_0_1'],row['micro_pixel_iou'],row['foreground_pixel_recall']))
    report={'protocol':protocol,'selected':selected,'choices':choices,'records':records,'elapsed_seconds':time.perf_counter()-started,
            'raw_score_maps_modified':False,'new_flagged_views':0}
    if selected['growth_threshold'] is not None:
        destination=ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx'
        shutil.copyfile(original,destination)
        updated={**metadata,'profile_version':'spatial_unet_v10h_sdd2',
                 'model_name':'End-to-end spatial MobileNet U-Net with calibrated seeded foreground growth',
                 'hysteresis_growth_threshold':selected['growth_threshold'],
                 'hysteresis_calibration_protocol':protocol,'hysteresis_calibration_point':selected,
                 'score_meaning':'Unmodified spatial neural sigmoid scores; grown foreground requires connected low-score support plus a retained high-score seed; not calibrated probabilities'}
        destination.with_suffix('.onnx.json').write_text(json.dumps(updated,indent=2),encoding='utf-8')
        report.update(model=str(destination),model_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      metadata_sha256=hashlib.sha256(destination.with_suffix('.onnx.json').read_bytes()).hexdigest())
    else:report['model']=str(original)
    (directory/'seeded-growth-calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in {'protocol','records','choices'}},indent=2))


if __name__=='__main__':main()
