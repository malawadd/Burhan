"""L7 source-pinned proposal and checkpoint export; no initial NN/BN repeat."""
import argparse,json
from factory_stfpm_learn_common import ROOT,OUT,NAME,sha,write_new,protocol_path,protocol_value,verify_protocol,model_path,metadata,validate_initialization
from prepare_factory_stfpm_cpu import verify_bn_fit

def export_model(model,step,protocol):
    """Future gradient checkpoints only; source graph/BN initialization unchanged."""
    import torch,numpy as np
    from prepare_factory_stfpm import reference_inputs,module_digest
    if step not in [200,400,600]:raise ValueError('Only declared learned checkpoints may be exported')
    fit_receipt,_,fit=verify_bn_fit(protocol)
    if module_digest(model.teacher)!=fit['teacher_sha256']:raise ValueError('Export teacher must remain unchanged')
    path=model_path(step);ref=OUT/f'{NAME}-step{step}-torch-reference.npz';receipt=OUT/f'{NAME}-step{step}-export.json'
    if any(p.exists() for p in [path,path.with_suffix('.onnx.json'),ref,receipt]):raise ValueError('Preserve future L7 learned export')
    path.parent.mkdir(parents=True,exist_ok=True);model.eval();arrays=reference_inputs(protocol);maps=[]
    from factory_stfpm_learn_common import L4_REF
    with np.load(L4_REF,allow_pickle=False) as original:
        if not np.array_equal(arrays,original['inputs_bgr']):raise ValueError('Three reference inputs must remain byte-identical to L4')
    with torch.inference_mode():
        for frame in arrays:maps.append(model(torch.from_numpy(frame.transpose(2,0,1)[None].copy())).numpy()[0,0])
        values=np.stack(maps)
        if values.shape!=(3,640,256) or not np.isfinite(values).all() or values.min()<0 or values.max()>1:raise ValueError('Learned Torch references invalid')
        torch.onnx.export(model,torch.from_numpy(arrays[0].transpose(2,0,1)[None].copy()),str(path),opset_version=17,input_names=['bgr'],output_names=['anomaly_score'],dynamo=False,do_constant_folding=True)
    with ref.open('xb') as stream:np.savez_compressed(stream,inputs_bgr=arrays,scores=values)
    write_new(path.with_suffix('.onnx.json'),metadata(path,protocol))
    value={'step':step,'protocol_sha256':sha(protocol_path()),'model_sha256':sha(path),'reference_sha256':sha(ref),
           'teacher_state_sha256':module_digest(model.teacher),'student_state_sha256':module_digest(model.student),
           'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'real_TRAIN_map_range':[float(values[0].min()),float(values[0].max())],'output_logits':False}
    write_new(receipt,value);return value



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['proposal','record-protocol'],required=True)
    args=parser.parse_args()
    value=protocol_value()
    if args.stage=='proposal':write_new(OUT/'proposed-protocol-before-fit.json',value)
    else:
        approval=OUT/'parent-approved-fit-decision.json'
        decision=json.loads(approval.read_text(encoding='utf-8'))
        if not decision.get('approved_exact_600') or decision['proposal_sha256']!=sha(OUT/'proposed-protocol-before-fit.json'):
            raise ValueError('Explicit exact600 parent decision required')
        if value!=json.loads((OUT/'proposed-protocol-before-fit.json').read_text(encoding='utf-8')):raise ValueError('Approved proposal content must equal actual protocol')
        write_new(protocol_path(),value)
    print(json.dumps({'stage':args.stage,'NN_executed':False,'optimizer_updates':0}))

if __name__=='__main__':main()
