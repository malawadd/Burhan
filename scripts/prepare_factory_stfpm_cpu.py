"""L5 exact retained L4 graph/reference copy; no BN fitting stage exists here."""
import argparse,json
from factory_stfpm_cpu_common import ROOT,OUT,NAME,L4,L4_PROTOCOL_SHA,L4_BN_SHA,L4_STATE_SHA,L4_NN,L4_NN_SHA,L4_REF,L4_REF_SHA,sha,write_new,protocol_path,protocol_value,verify_protocol,model_path,metadata,allowed_bn_keys,check_only_bn_changed,validate_bn_rows


def verify_bn_fit(protocol):
    receipt=L4/'BN-fitting-complete.json';state=L4/'real-TRAIN-BN64-fitted-state.pt'
    if sha(receipt)!=L4_BN_SHA or sha(state)!=L4_STATE_SHA:raise ValueError('Exact original BN fit state/receipt changed')
    value=json.loads(receipt.read_text(encoding='utf-8'))
    if (value['protocol_sha256']!=L4_PROTOCOL_SHA or value['state_sha256']!=L4_STATE_SHA or value['forward_passes_completed']!=64
            or value['source_identity_sha256']!=protocol['BN_fit_source_identity_sha256'] or not value['all_invariants_passed']
            or value['optimizer_created'] or value['gradient_updates']!=0):raise ValueError('Established real64BNfit lineage changed')
    validate_bn_rows(value['BN_after'],protocol['actual_BN_factory']['student_BN'],64)
    check_only_bn_changed(value['before_state_sha256_by_key'],value['after_state_sha256_by_key'],allowed_bn_keys(protocol['actual_BN_factory']['student_BN']))
    return receipt,state,value


def copy_preflight(protocol):
    fit_receipt,_,fit=verify_bn_fit(protocol);path=model_path(0);ref=OUT/f'{NAME}-step0-torch-reference.npz';receipt=OUT/f'{NAME}-step0-export.json'
    if any(p.exists() for p in [path,path.with_suffix('.onnx.json'),ref,receipt]):raise ValueError('Preserve initial CPU4 graph/ref copy')
    if sha(L4_NN)!=L4_NN_SHA or sha(L4_REF)!=L4_REF_SHA:raise ValueError('Exact retained NN/reference changed')
    path.parent.mkdir(parents=True,exist_ok=True);ref.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(L4_NN.read_bytes())
    with ref.open('xb') as stream:stream.write(L4_REF.read_bytes())
    write_new(path.with_suffix('.onnx.json'),metadata(path,protocol))
    original=json.loads((L4/'spatial_round3l4_stfpm_bn64-step0-export.json').read_text(encoding='utf-8'))
    if sha(L4/'spatial_round3l4_stfpm_bn64-step0-export.json')!='bfc53b4cc52a58531bfabf780160987528ca0b82c7cfc14d1a5b109a3338a536':raise ValueError('Preserved L4 export lineage changed')
    value={**original,'model_sha256':sha(path),'reference_sha256':sha(ref),'protocol_sha256':sha(protocol_path()),
           'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
           'byte_exact_initial_graph_and_three_references':True,'new_BN_forwards':0,'source_export_receipt_sha256':sha(L4/'spatial_round3l4_stfpm_bn64-step0-export.json')}
    if value['model_sha256']!=L4_NN_SHA or value['reference_sha256']!=L4_REF_SHA:raise ValueError('Byte-exact copy invariant failed')
    write_new(receipt,value);return value


def export_model(model,step,protocol):
    """Future gradient checkpoints only; source graph/BN initialization unchanged."""
    import torch,numpy as np
    from prepare_factory_stfpm import reference_inputs,module_digest
    if step not in [200,400,600]:raise ValueError('Initial L5 graph must be copied byte-exact, never re-exported')
    fit_receipt,_,fit=verify_bn_fit(protocol)
    if module_digest(model.teacher)!=fit['teacher_sha256']:raise ValueError('Export teacher must remain unchanged')
    path=model_path(step);ref=OUT/f'{NAME}-step{step}-torch-reference.npz';receipt=OUT/f'{NAME}-step{step}-export.json'
    if any(p.exists() for p in [path,path.with_suffix('.onnx.json'),ref,receipt]):raise ValueError('Preserve future CPU4 learned export')
    path.parent.mkdir(parents=True,exist_ok=True);model.eval();arrays=reference_inputs(protocol);maps=[]
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['record-protocol','copy-preflight'],required=True);args=parser.parse_args()
    if args.stage=='record-protocol':write_new(protocol_path(),protocol_value());print(json.dumps({'protocol_sha256':sha(protocol_path()),'NN_executed':False}));return
    _,p=verify_protocol();print(json.dumps(copy_preflight(p),indent=2))


if __name__=='__main__':main()
