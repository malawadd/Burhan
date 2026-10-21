"""Record fixed TRAIN/CAL guard saturation after parity; never refit/tune."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from check_factory_c_density_parity import compile_graph, run
from factory_c_density_identity import validate_density_artifacts
from factory_c_density_math import normalized_log_distance
from train_factory_surface import samples

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/optimization/factory-round3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    name='spatial_round3k_c200_density_f16'
    protocol,fit,_=validate_density_artifacts(name=name)
    parity=json.loads((OUT/(name+'-fitted-gpu-parity.json')).read_text(encoding='utf-8'))
    if not parity['all_checks_passed'] or parity['normal_bank_sha256']!=fit['bank_sha256']:
        raise ValueError('No CAL access before fitted numerical guard passes')
    cv2.setNumThreads(1);started=time.perf_counter()
    root=ROOT/'datasets/factory/KolektorSDD'
    path=ROOT/'models/factory'/(name+'-offline-diagnostic.onnx')
    graph=compile_graph(path)
    with np.load(OUT/(name+'-normal-bank.npz')) as bank:
        mean,white=bank['mean'],bank['whitening']
    rows=[('TRAIN-normal',root/row['image'],False) for row in fit['normal_sources']]
    calibration_identities=[]
    for image,truth,defective in samples(root,protocol['split_items']['calibration']):
        frame=cv2.imread(str(image));mask=image.with_name(image.stem+'_label.bmp')
        calibration_identities.append({'image':str(image.relative_to(root)),'defective':bool(defective),
            'image_sha256':sha(image),'mask_sha256':sha(mask),'native_image_shape':list(frame.shape),'native_mask_shape':list(truth.shape)})
        rows.append(('CAL-positive' if defective else 'CAL-normal',image,bool(defective)))
    groups={};records=[]
    for index,(group,image,defective) in enumerate(rows):
        frame=cv2.imread(str(image));value=np.ascontiguousarray(cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA).transpose(2,0,1)[None])
        raw,evidence,fine=run(graph,value)
        if not all(np.isfinite(a).all() for a in (raw,evidence,fine)):
            raise ValueError('Nonfinite diagnostic output')
        _,counts=normalized_log_distance(fine[0],mean,white,diagnostics=True)
        counts.update(evidence_at_guard=int((np.abs(evidence)>=8).sum()),evidence_pixels=int(evidence.size),views=1)
        summary=groups.setdefault(group,{key:0 for key in counts})
        for key in counts:summary[key]+=counts[key]
        records.append({'image':str(image.relative_to(root)),'group':group,**counts})
        if (index+1)%50==0:print(json.dumps({'diagnostic_views':index+1,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    for group,counts in groups.items():
        counts['centered_fraction_clipped']=counts['centered_values_clipped']/counts['channel_values']
        counts['whitened_fraction_clipped']=counts['whitened_values_clipped']/counts['channel_values']
        counts['actual_evidence_guard_saturation_fraction']=counts['evidence_at_guard']/counts['evidence_pixels']
    validate_density_artifacts(name=name)
    report={'scope':'Fixed guards, no tuning or refit. Actual diagnostic GPU evidence boundary saturation; independent full-context feature-based centered/white clipping counts. Not serving latency or defect accuracy.',
        'bank_sha256':fit['bank_sha256'],'diagnostic_graph_sha256':sha(path),
        'calibration_source_identity_sha256':hashlib.sha256(json.dumps(calibration_identities,sort_keys=True).encode()).hexdigest(),
        'groups':groups,'records':records,'elapsed_seconds':time.perf_counter()-started,
        'observed_workload':'Concurrent construction CPU4 fitting and ordinary host services; accuracy-only GPU window',
        'normal_bank_unchanged':True,'comparison_images_opened':0,'official_sdd2_test_still_sealed':True}
    target=OUT/(name+'-guard-clipping-audit.json')
    if target.exists():raise ValueError('Preserve the first guard diagnostic report')
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
