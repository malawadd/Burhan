"""Actual L GPU numeric parity and separately authorized fullprofile preflight."""
import argparse,json,time,sys
import numpy as np
import cv2
from factory_stfpm_bn_common import ROOT,OUT,NAME,sha,write_new,verify_protocol,protocol_path,model_path
sys.path.insert(0,str(ROOT))


def parity(step,protocol):
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    from prepare_factory_stfpm_bn import verify_bn_fit
    from probe_factory_stfpm_winograd import property_audit
    fit_receipt,_,fit=verify_bn_fit(protocol)
    path=model_path(step);target=OUT/f'{NAME}-step{step}-gpu-parity.json'
    if target.exists():raise ValueError('Preserve first numeric parity result')
    export=OUT/f'{NAME}-step{step}-export.json';receipt=json.loads(export.read_text(encoding='utf-8'))
    ref=OUT/f'{NAME}-step{step}-torch-reference.npz'
    if (receipt['model_sha256']!=sha(path) or receipt['reference_sha256']!=sha(ref) or receipt['protocol_sha256']!=sha(protocol_path())
            or receipt['initial_metadata_sha256']!=sha(path.with_suffix('.onnx.json'))):
        raise ValueError('Export/source/reference binding changed')
    if receipt['BN_fit_receipt_sha256']!=sha(fit_receipt) or receipt['BN_fitted_state_sha256']!=fit['state_sha256']:raise ValueError('BN fitted source changed')
    model=FactorySegmentation(path,output_logits=False,precision_hint='f16',disable_winograd_convolution=True)
    differences=[];allfinite=True;allbounded=True;actual_ranges=[]
    with np.load(ref,allow_pickle=False) as data:
        if data['inputs_bgr'].shape!=(3,1280,512,3) or data['scores'].shape!=(3,640,256):raise ValueError('Exactly three fixed numeric cases required')
        for frame,expected in zip(data['inputs_bgr'],data['scores']):
            actual,_,_=model.predict(frame,.05,4)
            if actual.shape!=(640,256):raise ValueError('Actual GPU geometry differs')
            allfinite=allfinite and bool(np.isfinite(actual).all())
            allbounded=allbounded and bool(actual.min()>=0 and actual.max()<=1)
            differences.append(float(np.max(np.abs(actual-expected))))
            actual_ranges.append([float(actual.min()),float(actual.max())])
    nonconstant=(actual_ranges[0][1]-actual_ranges[0][0])>=1e-7
    value={'step':step,'model_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),'export_receipt_sha256':sha(export),'protocol_sha256':sha(protocol_path()),
           'max_abs_errors':differences,'actual_score_ranges':actual_ranges,'all_finite':allfinite,'all_bounded_0_1':allbounded,
           'preflight_nonconstant':nonconstant,'score_meaning':protocol['score'],'actual_execution_devices':model.execution_devices,
           'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'queried_configuration_and_execution':property_audit(model),
           'all_checks_passed':bool(allfinite and allbounded and max(differences)<=.01 and (step!=0 or nonconstant)),
           'scope':'Fixed first realTRAIN normal + zero/all255 numerical initialization cases; no accuracy claim'}
    write_new(target,value)
    if not value['all_checks_passed']:raise ValueError('Fresh actual GPU numeric guard failed; stop before fit/CAL')
    return value


def preflight(protocol):
    from edge_runtime.profiles.factory import FactoryProfile
    from edge_runtime.contracts import FrameContext
    target=OUT/'fullprofile-preflight.json'
    if target.exists():raise ValueError('Preserve first capacity result including tails')
    from prepare_factory_stfpm_bn import verify_bn_fit
    fit_receipt,_,fit=verify_bn_fit(protocol)
    parity_path=OUT/f'{NAME}-step0-gpu-parity.json';numeric=json.loads(parity_path.read_text(encoding='utf-8'))
    if (not numeric['all_checks_passed'] or numeric['protocol_sha256']!=sha(protocol_path()) or numeric['model_sha256']!=sha(model_path(0))
            or numeric['BN_fit_receipt_sha256']!=sha(fit_receipt) or numeric['metadata_sha256']!=sha(model_path(0).with_suffix('.onnx.json'))):raise ValueError('New L4 numeric pass required BEFORE capacity')
    path=model_path(0);rows=protocol['preflight']['source_rows'];base=ROOT/'datasets/factory/KolektorSDD'
    frames=[]
    for row in rows:
        source=base/row['image']
        if sha(source)!=row['image_sha256']:raise ValueError('Preflight source changed')
        frame=cv2.imread(str(source))
        if frame is None:raise ValueError('Cannot decode original TRAIN source')
        frames.append(frame)
    start=time.perf_counter();profile=FactoryProfile({'defect_model_path':str(path),'defect_backend':'auto','confirmation_frames':1})
    profile.prepare_worker();startup=time.perf_counter()-start
    # Sidecar must carry the exact approved GPU recipe, rather than serving defaults.
    model_meta=json.loads(path.with_suffix('.onnx.json').read_text(encoding='utf-8'))
    if model_meta['openvino_inference_precision']!='f16' or model_meta['openvino_disable_winograd_convolution'] is not True:raise ValueError('L4 capacity GPU recipe mismatch')
    for index,frame in enumerate(frames[:4]):profile.process(frame,FrameContext(index,time.perf_counter(),unit_id=f'Lwarm-{index}'))
    profile.reset();records=[]
    for index in range(64):
        tick=time.perf_counter();result=profile.process(frames[index%16],FrameContext(index,tick,unit_id=f'Lprobe-{index}'))
        records.append({'source':rows[index%16]['image'],'profile_ms':1000*(time.perf_counter()-tick),'status':result.status,
                        'neural_stage_ms':result.metrics.get('factory_neural_stage_ms')})
    values=np.asarray([r['profile_ms'] for r in records]);all_inferred=all(r['neural_stage_ms'] is not None for r in records)
    value={'protocol_sha256':sha(protocol_path()),'model_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),
           'initialization':profile.describe()['model_initialization'],'startup_seconds':startup,'records':records,
           'numeric_parity_sha256':sha(parity_path),'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'scope':'Root-authorized project-quiescent first nonconstant twoencoder fullFactoryProfile;16realTRAIN/64calls, ordinary host notexclusive; no GT/accuracy claim',
           'profile_ms':{'mean':float(values.mean()),'p50':float(np.median(values)),'p95':float(np.percentile(values,95)),
                         'p99':float(np.percentile(values,99)),'max':float(values.max()),'serial_calls_per_second':1000/float(values.mean())},
           'all_calls_inferred':all_inferred,'capacity_gate_passed':bool(all_inferred and 1000/values.mean()>=15),
           'comparison_images_opened':0,'official_sdd2_test_still_sealed':True}
    write_new(target,value)
    if not value['capacity_gate_passed']:raise ValueError('L capacity gate failed; stop before fit')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--parity-step',type=int,choices=[0,200,400,600]);parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if not args.root_released or args.preflight==(args.parity_step is not None):raise ValueError('One explicit authorized stage required')
    cv2.setNumThreads(1);_,protocol=verify_protocol()
    result=preflight(protocol) if args.preflight else parity(args.parity_step,protocol)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
