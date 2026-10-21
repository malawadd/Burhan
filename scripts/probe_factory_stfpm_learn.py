"""Actual L GPU numeric parity and separately authorized fullprofile preflight."""
import argparse,json,time,sys
import numpy as np
import cv2
from factory_stfpm_learn_common import ROOT,OUT,NAME,sha,write_new,verify_protocol,protocol_path,model_path,gpu_configuration_passes,validate_numeric_receipt,L4_REF
sys.path.insert(0,str(ROOT))

def numeric_case(model,frame,expected):
    """Only the adapter's exact nonfinite-output error is a numeric outcome."""
    try:actual,_,_=model.predict(frame,.05,4)
    except ValueError as error:
        if str(error)!='Factory neural model produced invalid scores':raise
        return {'max_abs_error':None,'range':[None,None],'finite':False,'bounded':False,'numeric_failure':str(error)}
    if actual.shape!=(640,256):raise ValueError('Actual GPU geometry differs')
    if not np.isfinite(actual).all():
        return {'max_abs_error':None,'range':[None,None],'finite':False,'bounded':False,'numeric_failure':'Actual returned map contains nonfinite scores'}
    low,high=float(actual.min()),float(actual.max())
    return {'max_abs_error':float(np.max(np.abs(actual-expected))),'range':[low,high],'finite':True,
        'bounded':low>=0 and high<=1,'numeric_failure':None}


def parity(step,protocol):
    from factory_stfpm_learn_common import require_complete_training
    require_complete_training(protocol)
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    from prepare_factory_stfpm_learn import verify_bn_fit
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
    differences=[];allfinite=True;allbounded=True;actual_ranges=[];numeric_failures=[]
    with np.load(ref,allow_pickle=False) as data:
        if (data['inputs_bgr'].shape!=(3,1280,512,3) or data['inputs_bgr'].dtype!=np.uint8 or data['scores'].shape!=(3,640,256)
                or not np.isfinite(data['scores']).all() or data['scores'].min()<0 or data['scores'].max()>1):raise ValueError('Exactly three fixed finite bounded reference cases required')
        with np.load(L4_REF,allow_pickle=False) as original:
            if not np.array_equal(data['inputs_bgr'],original['inputs_bgr']):raise ValueError('Reference inputs must remain byte-identical to L4')
        if data['inputs_bgr'][1].any() or not (data['inputs_bgr'][2]==255).all():raise ValueError('Initialization reference cases changed')
        for index,(frame,expected) in enumerate(zip(data['inputs_bgr'],data['scores'])):
            case=numeric_case(model,frame,expected)
            allfinite=allfinite and case['finite'];allbounded=allbounded and case['bounded']
            differences.append(case['max_abs_error']);actual_ranges.append(case['range'])
            if case['numeric_failure']:numeric_failures.append({'case_index':index,'error':case['numeric_failure']})
    nonconstant=actual_ranges[0][0] is not None and actual_ranges[0][1]-actual_ranges[0][0]>=1e-7
    audit=property_audit(model)
    for key in ['NUM_STREAMS','PERFORMANCE_HINT']:
        try:audit[key]={'supported':True,'value':str(model.compiled.get_property(key))}
        except Exception as error:audit[key]={'supported':False,'error':str(error)}
    config_passed=gpu_configuration_passes(audit)
    value={'step':step,'model_sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.onnx.json')),'export_receipt_sha256':sha(export),'protocol_sha256':sha(protocol_path()),
           'max_abs_errors':differences,'actual_score_ranges':actual_ranges,'all_finite':allfinite,'all_bounded_0_1':allbounded,
           'preflight_nonconstant':nonconstant,'score_meaning':protocol['score'],'actual_execution_devices':model.execution_devices,
           'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'queried_configuration_and_execution':audit,'actual_GPU_configuration_passed':config_passed,
           'reference_inputs_byte_identical_to_L4':True,'reference_scores_finite_bounded':True,
           'actual_numeric_output_failures':numeric_failures,'fixed_cases_attempted':3,
           'all_checks_passed':bool(allfinite and allbounded and max(differences)<=.01 and nonconstant and config_passed),
           'scope':'Fixed first realTRAIN normal + zero/all255 numerical initialization cases; no accuracy claim'}
    validate_numeric_receipt(value)
    write_new(target,value)
    if not value['all_checks_passed']:raise ValueError('Learned GPU numeric guard failed; exclude this checkpoint without CAL')
    return value



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=[200,400,600],required=True)
    parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if not args.root_released:raise ValueError('Explicit coordinated GPU release required')
    cv2.setNumThreads(1);_,protocol=verify_protocol();parity(args.step,protocol)

if __name__=='__main__':main()
