"""Fixed all3 learned mixed GPU/CPU numerical gates then passing-only CAL."""
import argparse,json,time
import numpy as np
import cv2
from prepare_factory_partitioned_learned import (ROOT,OUT,NAME,STEPS,L7,PROTOCOL,source_path,model_path,sha,write_new,verify_protocol,numeric_acceptance)
from factory_partitioned_segmentation import adapter_type
from probe_factory_stfpm_partition import numeric_case,L4_REF,configuration_audit
from factory_stfpm_common import identity,CAL_SHA,validate_search_report
from calibrate_factory_stfpm_learn import select
from factory_mask_metrics import original_pixel_weights
from train_factory_surface import samples

def transformed(step,protocol,selected=False):
    complete=OUT/'all3-transforms-complete.json';c=json.loads(complete.read_text())
    if c['protocol_sha256']!=sha(PROTOCOL) or [r['step'] for r in c['checkpoints']]!=STEPS:raise ValueError('Complete all3 transformed artifacts required before NN')
    path=model_path(step);receipt=OUT/f'checkpoint{step}-transform.json';r=json.loads(receipt.read_text())
    if c['checkpoints'][STEPS.index(step)]['receipt_sha256']!=sha(receipt):raise ValueError('All3 transform lineage changed')
    ref=L7/f'spatial_round3l7_learned_stfpm-step{step}-torch-reference.npz'
    if (r['protocol_sha256']!=sha(PROTOCOL) or r['model_sha256']!=sha(path) or r['source_model_sha256']!=sha(source_path(step))
        or r['source_PT_sha256']!=sha(source_path(step).with_suffix('.pt')) or r['reference_sha256']!=sha(ref)
        or not r['extraction']['all_retained_node_initializer_input_opset_bytes_identical']):raise ValueError('Transformed source/model/reference changed')
    if not selected and r['initial_metadata_sha256']!=sha(path.with_suffix('.onnx.json')):raise ValueError('Initial partition metadata changed')
    return path,ref,receipt,r

def terminal_numeric_pass(report):
    passed=numeric_acceptance(report['cases'],report['compiled_provenance'])
    if report['fixed_cases_attempted']!=3 or report['all_checks_passed']!=passed:raise ValueError('Numeric outcome arithmetic/casecount changed')
    return passed

def numeric(step,protocol,scope):
    path,ref,receipt,_=transformed(step,protocol);target=OUT/f'checkpoint{step}-numeric.json';started=OUT/f'checkpoint{step}-numeric-attempt.json'
    if target.exists() or started.exists():raise ValueError('Preserve first learned numeric attempt')
    write_new(started,{'step':step,'protocol_sha256':sha(PROTOCOL),'scope':scope,'fixed_cases':3})
    with np.load(ref,allow_pickle=False) as data,np.load(L4_REF,allow_pickle=False) as initial:
        if (data['inputs_bgr'].dtype!=np.uint8 or data['inputs_bgr'].shape!=(3,1280,512,3)
            or not np.array_equal(data['inputs_bgr'],initial['inputs_bgr']) or data['scores'].shape!=(3,640,256)
            or not np.isfinite(data['scores']).all() or data['scores'].min()<0 or data['scores'].max()>1):raise ValueError('Own learned scores/exact three L4 inputs required')
        model=adapter_type(sha(path))(path);audit=configuration_audit(model.compiled,'GPU')
        if not audit['actual_configuration_passed']:raise ValueError('Actual mixed GPU/CPU recipe differs before cases')
        cases=[numeric_case(model,frame,expected) for frame,expected in zip(data['inputs_bgr'],data['scores'])]
    passed=numeric_acceptance(cases,audit)
    report={'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),
        'transform_receipt_sha256':sha(receipt),'reference_sha256':sha(ref),'fixed_cases_attempted':3,'cases':cases,'compiled_provenance':audit,
        'all_checks_passed':passed,'execution':'GPU raw features + CPU NumPyfloat32 score head','scope':scope+'; no accuracy/FPS claim'}
    write_new(target,report)
    if not passed:write_new(OUT/f'checkpoint{step}-terminal.json',{'step':step,'status':'numeric_failed_no_CAL','numeric_sha256':sha(target),'CAL_inputs_opened':0,'protocol_sha256':sha(PROTOCOL)})
    print(json.dumps({'step':step,'numeric_pass':passed,'cases':cases},indent=2))

def calibration(step,protocol,scope):
    # All3 numeric outcomes precede any threshold selection, including failures.
    for required in STEPS:
        report=json.loads((OUT/f'checkpoint{required}-numeric.json').read_text());terminal_numeric_pass(report)
        if report['step']!=required or report['protocol_sha256']!=sha(PROTOCOL) or report['model_sha256']!=sha(model_path(required)):raise ValueError('All3 numeric source identities required before CAL')
    numeric_path=OUT/f'checkpoint{step}-numeric.json';n=json.loads(numeric_path.read_text())
    if not terminal_numeric_pass(n):raise ValueError('Numeric-failed checkpoint forbidden from CAL')
    path,_,receipt,_=transformed(step,protocol);target=OUT/f'checkpoint{step}-calibration.json';started=OUT/f'checkpoint{step}-CAL-attempt.json'
    if any(p.exists() for p in [target,started,OUT/f'checkpoint{step}-terminal.json']):raise ValueError('Preserve first partial/failed CAL')
    write_new(started,{'step':step,'protocol_sha256':sha(PROTOCOL),'numeric_sha256':sha(numeric_path),'scope':scope})
    base=ROOT/'datasets/factory/KolektorSDD';sources=[];rows=[]
    for image,truth,defective in samples(base,protocol['split_items']['calibration']):
        frame=cv2.imread(str(image));mask=image.with_name(image.stem+'_label.bmp')
        if frame is None:raise ValueError('Protected CAL source decode failed')
        sources.append({'image':str(image.relative_to(base)),'defective':bool(defective),'image_sha256':sha(image),'mask_sha256':sha(mask),
            'native_image_shape':list(frame.shape),'native_mask_shape':list(truth.shape)})
        rows.append({'image':sources[-1]['image'],'defective':bool(defective),'truth':truth,'frame':frame})
    if len(sources)!=80 or sum(r['defective'] for r in sources)!=10 or identity(sources)!=CAL_SHA or protocol['calibration_source_identity_sha256']!=CAL_SHA:raise ValueError('Only exact CAL80 geometry/image/GT identities permitted before NN')
    model=adapter_type(sha(path))(path);audit=configuration_audit(model.compiled,'GPU')
    if not audit['actual_configuration_passed']:raise ValueError('Actual mixed backend configuration changed')
    for row in rows:
        row['scores']=model.predict(row.pop('frame'),.05,4)[0]
        row['projection_weights']=original_pixel_weights(row['truth'],row['scores'].shape)
    for row in sources:
        image=base/row['image']
        if sha(image)!=row['image_sha256'] or sha(image.with_name(image.stem+'_label.bmp'))!=row['mask_sha256']:raise ValueError('CAL image/GT changed during inference')
    cache=OUT/f'checkpoint{step}-actual-mixed-scoremaps.npz'
    with cache.open('xb') as stream:np.savez_compressed(stream,scores=np.stack([r['scores'] for r in rows]),source_identity_json=np.asarray(json.dumps(sources)),model_sha256=np.asarray(sha(path)))
    raw,grown,point,selected_records=select(rows,protocol)
    report={'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'numeric_sha256':sha(numeric_path),
        'transform_receipt_sha256':sha(receipt),'scorecache_sha256':sha(cache),'calibration_source_identity':{'ordered_sources':sources,'sha256':identity(sources)},
        'raw_grid':raw,'grown_grid':grown,'selected':point,'selected_records':selected_records,
        'actual_partitioned_maps':True,'execution':'GPU raw features + CPU NumPyfloat32 score head','compiled_provenance':audit,'scope':scope+'; no inference-capacity claim'}
    validate_search_report(report,protocol)
    if point:
        meta_path=path.with_suffix('.onnx.json');meta=json.loads(meta_path.read_text());write_new(OUT/f'checkpoint{step}-initial-metadata.json',meta)
        meta.update(pixel_threshold=point['high'],minimum_component_pixels=point['area'],hysteresis_growth_threshold=point['growth'],
            calibration_selected_point=point,calibration_source_identity_sha256=CAL_SHA,partitioned_protocol_sha256=sha(PROTOCOL))
        temporary=meta_path.with_name(meta_path.name+'.selected-writing')
        write_new(temporary,meta);temporary.replace(meta_path);report['metadata_sha256']=sha(meta_path)
    write_new(target,report)
    write_new(OUT/f'checkpoint{step}-terminal.json',{'step':step,'status':'calibration_complete','protocol_sha256':sha(PROTOCOL),
        'numeric_sha256':sha(numeric_path),'calibration_sha256':sha(target),'CAL_inputs_opened':80})
    print(json.dumps({'step':step,'selected':point},indent=2))

def build_profile(path):
    """Root-owned audit/bench helper; never installs a shared/default adapter."""
    from unittest.mock import patch
    import edge_runtime.profiles.factory_encoder as module
    from edge_runtime.profiles.factory import FactoryProfile
    original=module.FactorySegmentation;cls=adapter_type(sha(path))
    with patch.object(module,'FactorySegmentation',cls):profile=FactoryProfile({'defect_model_path':str(path),'defect_backend':'auto','confirmation_frames':1})
    if module.FactorySegmentation is not original or type(profile._defect_model.neural) is not cls:raise ValueError('Local profile adapter binding/restoration failed')
    return profile

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['numeric','calibration'],required=True)
    parser.add_argument('--step',choices=STEPS,type=int,required=True);parser.add_argument('--root-released',action='store_true');parser.add_argument('--contention-note',required=True);a=parser.parse_args()
    if not a.root_released:raise ValueError('Separate root learned GPU release required')
    cv2.setNumThreads(1);protocol=verify_protocol()
    try:(numeric if a.stage=='numeric' else calibration)(a.step,protocol,a.contention_note)
    except Exception as error:
        failed=OUT/f'checkpoint{a.step}-{a.stage}-failure.json'
        if not failed.exists():write_new(failed,{'error':str(error),'protocol_sha256':sha(PROTOCOL),'no_automatic_retry':True})
        raise

if __name__=='__main__':main()
