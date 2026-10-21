"""Nonconstant full-shape preflight on TRAIN pixels only; no accuracy claim."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory import FactoryProfile
from edge_runtime.contracts import FrameContext


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packed',action='store_true');parser.add_argument('--phase-max',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(1)
    directory=ROOT/'output/optimization/factory-round3';name='spatial_round3e_phase_max_refine' if args.phase_max else 'spatial_round3d_packed_refine' if args.packed else 'spatial_round3c_fullres_refine'
    protocol=json.loads((directory/(name+'-protocol-before-fit.json')).read_text(encoding='utf-8'))
    path=ROOT/'models/factory'/(name+'-capacity-probe.onnx')
    metadata=json.loads((ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json').read_text(encoding='utf-8'))
    metadata.update(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),profile_version=name+'_capacity_probe',
        model_name='NONCONSTANT capacity probe, not an accuracy/deployment candidate',training_protocol=protocol,
        architecture=protocol['architecture'],input_width=512,input_height=1280,input_shape=[1,3,1280,512],
        output_shape=protocol['output_shape'],model_grid_width=protocol['output_shape'][-1],model_grid_height=protocol['output_shape'][-2],
        pixel_threshold=.995,minimum_component_pixels=32 if args.phase_max else 128,hysteresis_growth_threshold=.9)
    path.with_suffix('.onnx.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    selected=[]
    for item in protocol['split_items']['train']:
        selected.extend(sorted((ROOT/'datasets/factory/KolektorSDD'/item).glob('*.jpg'))[:2])
        if len(selected)>=16:break
    decoded=[(p,cv2.imread(str(p))) for p in selected]
    tick=time.perf_counter();profile=FactoryProfile({'defect_model_path':str(path),'defect_backend':'auto','confirmation_frames':1})
    profile.prepare_worker();startup=time.perf_counter()-tick
    for index,(_,frame) in enumerate(decoded[:4]):profile.process(frame,FrameContext(index,time.perf_counter(),unit_id=f'preflight-warm-{index}'))
    profile.reset();records=[]
    for index in range(64):
        image,frame=decoded[index%len(decoded)];started=time.perf_counter()
        result=profile.process(frame,FrameContext(index,started,unit_id=f'preflight-{index}'))
        records.append({'image':str(image.relative_to(ROOT/'datasets/factory/KolektorSDD')),'profile_ms':(time.perf_counter()-started)*1000,
            'status':result.status,'stage_ms':result.metrics.get('factory_neural_stage_ms')})
    values=[r['profile_ms'] for r in records]
    report={'scope':'Pre-fit nonconstant full-shape graph on16TRAIN views/64calls; noGTread, no accuracy or final speed claim',
        'contended':True,'construction_cpu_threads':3,'factory_cpu_threads':1,'gpu_exclusive':True,
        'model_sha256':metadata['model_sha256'],'protocol_sha256':hashlib.sha256((directory/(name+'-protocol-before-fit.json')).read_bytes()).hexdigest(),
        'startup_seconds':startup,'initialization':profile.describe().get('model_initialization'),
        'profile_ms':{'mean':float(np.mean(values)),'p50':float(np.median(values)),'p95':float(np.percentile(values,95)),
                      'p99':float(np.percentile(values,99)),'max':max(values),'serial_calls_per_second':1000/float(np.mean(values))},
        'records':records,'comparison_images_opened':0,'official_sdd2_test_still_sealed':True}
    target=directory/(name+'-fullprofile-preflight.json');target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
