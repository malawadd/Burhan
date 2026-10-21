"""Compare frozen v1/v3 candidates on reused real public item images.

Do not use this script to select model parameters: the comparison split has
already been exposed during development and is not independent validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.factory import FactoryProfile
from train_factory_surface import samples, matrix


def times(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": max(values),
            "mean": float(np.mean(values)), "serial_capacity_fps": 1000/float(np.mean(values))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, default=ROOT/'models/factory/kolektor_surface_rf.joblib')
    p.add_argument('--after', type=Path, default=ROOT/'models/factory/kolektor_surface_rf_v3_hardneg.joblib')
    p.add_argument('--dataset', type=Path, default=ROOT/'datasets/factory/KolektorSDD')
    p.add_argument('--output', type=Path, default=ROOT/'output/optimization/factory-accuracy/before-after.json')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--split', choices=['test','calibration'], default='test')
    a = p.parse_args()
    cv2.setNumThreads(2)
    profiles = {name: FactoryProfile({'defect_model_path': str(path), 'defect_backend': 'auto',
                                     'confirmation_frames': 1}) for name,path in [('before',a.before),('after',a.after)]}
    split = profiles['before'].describe()['model_metadata']['split_items']
    assert split == profiles['after'].describe()['model_metadata']['split_items']
    decoded = [(path, cv2.imread(str(path)), label) for path,_,label in samples(a.dataset,split[a.split])]
    for profile in profiles.values():
        for i in range(2):
            profile.process(decoded[0][1], FrameContext(i,time.perf_counter(),unit_id=f'warm-{i}'))
        profile.reset()
    records = []
    a.output.parent.mkdir(parents=True,exist_ok=True)
    saved = set()
    for repeat in range(a.repeat):
        for i,(path,frame,label) in enumerate(decoded):
            record = {'path': str(path.relative_to(a.dataset)), 'defective': label, 'repeat': repeat,
                      'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for name in (['before','after'] if (i+repeat)%2==0 else ['after','before']):
                started = time.perf_counter()
                result = profiles[name].process(frame,FrameContext(i,started,session_id='paired-public',unit_id=f'{repeat}/{i}'))
                ms = (time.perf_counter()-started)*1000
                record[name] = {'status': result.status, 'process_ms': ms,
                                'findings': [{'bbox': f.bbox,'score': f.score} for f in result.findings],
                                'metrics': result.metrics}
                if name=='after' and repeat==0:
                    case = 'true-positive' if label and result.status in {'fault','review'} else 'miss' if label else 'false-positive' if result.status in {'fault','review'} else 'normal'
                    if case not in saved:
                        cv2.imwrite(str(a.output.parent/f'{a.after.stem}-{case}.jpg'),result.annotated_frame)
                        saved.add(case)
            if repeat==0 and record['before']['status']=='clear' and label and record['after']['status'] in {'fault','review'}:
                # Save genuine newly detected public image pixels, no generated data.
                result=profiles['after'].process(frame,FrameContext(i,time.perf_counter(),unit_id=f'review/{i}'))
                cv2.imwrite(str(a.output.parent/f'{a.after.stem}-new-detection-{path.parent.name}-{path.stem}.jpg'),result.annotated_frame)
            records.append(record)
        print(f'Completed paired repeat {repeat+1}/{a.repeat}',flush=True)
    report = {'scope': 'Paired full profile timings on predecoded real public images; excludes disk decode/capture/queue/UI/evidence/actuation. No moving-video or independent-test accuracy claim.',
              'split': a.split,
              'accuracy_units': '80 unique views, 10 physical items; repetitions add timing observations only',
              'selection': 'After model and operating point frozen using train/calibration only; no comparison retuning',
              'repeats': a.repeat, 'images': len(decoded), 'models': {k:v.describe() for k,v in profiles.items()},
              'results': {}, 'records': records}
    for name in profiles:
        unique = [r for r in records if r['repeat']==0]
        accepted = [r for r in unique if r[name]['status'] in {'fault','review','clear','no_candidate'}]
        report['results'][name] = {'image_metrics': matrix([r['defective'] for r in accepted],[r[name]['status'] in {'fault','review'} for r in accepted]),
                                  'abstained_defective': sum(r['defective'] for r in unique if r not in accepted),
                                  'abstained_normal': sum(not r['defective'] for r in unique if r not in accepted),
                                  'process_ms': times([r[name]['process_ms'] for r in records]),
                                  'misses': [r['path'] for r in unique if r['defective'] and r[name]['status'] not in {'fault','review'}],
                                  'false_positives': [r['path'] for r in unique if not r['defective'] and r[name]['status'] in {'fault','review'}]}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['results'],indent=2))


if __name__=='__main__':
    main()
