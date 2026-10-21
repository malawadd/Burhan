"""Additive actual wrapper-vs-original raw-mask audit; no refit/threshold change."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path,PurePosixPath
import hashlib,json,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_tile_adapter import TileConstructionModel
from factory_localization_audit import mask_metrics
DATA=ROOT/'datasets/construction/finishes-expansion/mvtec-tile'
OUT=ROOT/'output/evaluation/construction-finishes-tile/serving-audit'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'report.json').exists():raise RuntimeError('Preserve completed actualtile serving audit')
    metadata=ROOT/'models/construction/finishes-tile/model.json'
    manifest_path=DATA/'asset-manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    original=ROOT/'output/evaluation/construction-finishes-tile/report.json'
    protocol={'script_sha256':sha(Path(__file__)),'original_report_sha256':sha(original),'metadata_sha256':sha(metadata),
              'source_manifest_sha256':sha(manifest_path),'wrapper_sha256':sha(ROOT/'edge_runtime/profiles/construction_tile_adapter.py'),
              'backend_sha256':sha(ROOT/'edge_runtime/profiles/construction_tile.py'),
              'scope':'Samefrozen175photo source/48normalfit/10normalcal/all117test. ActualwrapperinferCPU1; captureitsrealrawmap thencompare rawthresholdmask with unionofactualfilteredmasks. No secondmodel/threshold/modelsearch.',
              'reason':'Originalpixel evaluationusedrawthresholdmask; servingbackend dropsnativecomponents<4pixels. Verifydifferencesratherthanassumeparity.',
              'performance_scope':'Accuracyonly underotherprojectwork; NOT a speed benchmark'}
    write(OUT/'protocol-before-inference.json',protocol)
    spec={'file':'finishes-tile/model.json','sha256':sha(metadata),'classes':['tile_surface_anomaly'],
          'source':'https://www.mvtec.com/research-teaching/datasets/mvtec-ad','diagnostic_only':True}
    model=TileConstructionModel({'damage_variant':'tile_surface_anomaly_research','cpu_threads':1,'mask_rendering':'masks'},spec)
    real_predict=model.backend.predict_maps;captured={}
    def capture(frame):
        result=real_predict(frame);captured['map']=result;return result
    model.backend.predict_maps=capture
    rows=[]
    for row in manifest['records']:
        if row['partition']!='test':continue
        path=DATA/'original'/str(PurePosixPath(row['source_path']).relative_to('MVTec-AD/tile'))
        if sha(path)!=row['sha256']:raise RuntimeError('Originaltilephoto changed')
        frame=cv2.imread(str(path));predictions=model.infer(frame)
        raw=captured['map']>model.backend.threshold
        served=np.zeros(raw.shape,dtype=bool)
        for p in predictions:
            if not(0<=p['score']<=1) or p['label']!='tile_surface_anomaly':raise RuntimeError('Wrappercandidate contract differs')
            served|=p['mask']
        if row['mask_source_path']:
            target=DATA/'original'/str(PurePosixPath(row['mask_source_path']).relative_to('MVTec-AD/tile'))
            if sha(target)!=row['mask_sha256']:raise RuntimeError('Originaltilemask changed')
            expected=cv2.imread(str(target),cv2.IMREAD_GRAYSCALE)>0
        else:expected=np.zeros(raw.shape,dtype=bool)
        rows.append({'source_path':row['source_path'],'defect_type':row['defect_type'],'changed_pixels':int(np.count_nonzero(raw!=served)),
                     'metrics':mask_metrics(served,expected),'actual_detections':[{k:v for k,v in p.items() if k!='mask'} for p in predictions]})
    summary={}
    for name in sorted({r['defect_type'] for r in rows}):
        selected=[r for r in rows if r['defect_type']==name];tp=sum(r['metrics']['pixel_tp'] for r in selected)
        pred=sum(r['metrics']['pixel_predicted'] for r in selected);gt=sum(r['metrics']['pixel_target'] for r in selected)
        summary[name]={'photos':len(selected),'changed_pixels':sum(r['changed_pixels'] for r in selected),'pixel_tp':tp,'pixel_fp':pred-tp,'pixel_fn':gt-tp,
                       'pixel_precision':tp/pred if pred else None,'pixel_recall':tp/gt if gt else None,'pixel_iou':tp/(pred+gt-tp) if pred+gt-tp else None,
                       'derived_region_matches_iou_0_5':sum(r['metrics']['region_matches_at_iou']['0.5'] for r in selected),
                       'class_gt_regions':sum(r['metrics']['gt_regions'] for r in selected)}
    result={'protocol_sha256':sha(OUT/'protocol-before-inference.json'),'model':model.describe(),'all117_rawthreshold_actualinfer_masks_equal':all(not r['changed_pixels'] for r in rows),
            'changed_pixels':sum(r['changed_pixels'] for r in rows),'by_defect_type':summary,'records':rows,'scope':protocol['scope']}
    write(OUT/'report.json',result)
    print(json.dumps({k:result[k] for k in ['all117_rawthreshold_actualinfer_masks_equal','changed_pixels','by_defect_type']},indent=2),flush=True)


if __name__=='__main__':main()
