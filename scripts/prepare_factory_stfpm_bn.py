"""L4: metadata/protocol preparation; real64-view BN fitting requires root release."""
import argparse,json,hashlib
from pathlib import Path
from factory_stfpm_bn_common import ROOT,OUT,NAME,sha,identity,write_new,source_metadata,select64,protocol_value,protocol_path,verify_protocol,model_path,metadata,allowed_bn_keys,check_only_bn_changed,validate_bn_rows


def record_module_metadata():
    import importlib.metadata,torch
    from torch import nn
    from torchvision.models import mobilenet_v3_small
    source=Path(importlib.metadata.distribution('torchvision').locate_file('torchvision/models/mobilenetv3.py'))
    modules=nn.ModuleList(list(mobilenet_v3_small(weights=None).features.children())[:7])
    rows=[{'name':name,'eps':m.eps,'momentum':m.momentum,'num_features':m.num_features,
           'initial_counter':int(m.num_batches_tracked)} for name,m in modules.named_modules() if isinstance(m,nn.BatchNorm2d)]
    value={'torch_version':torch.__version__,'torchvision_version':importlib.metadata.version('torchvision'),
           'torchvision_source_path':str(source),'torchvision_source_sha256':sha(source),'student_BN':rows,
           'forward_calls':0,'weights_loaded':False,'scope':'ActualinstalledMobileNetfactorymoduleconstruction+BNattributesonly,noimage/NNforward'}
    write_new(OUT/'actual-BN-factory-metadata.json',value);print(json.dumps(value,indent=2))


def record_selection():
    old=source_metadata();rows=select64(old['normal_sources'],old['split_items']['train'])
    from collections import Counter
    base=ROOT/'datasets/factory/KolektorSDD'
    for row in rows:
        p=base/row['image'];mask=p.with_name(p.stem+'_label.bmp')
        if sha(p)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('ExactselectedsourceSHA changed')
    value={'sources':rows,'source_identity_sha256':identity(rows),'all208_normal_identity_sha256':old['normal_source_identity_sha256'],
           'counts_by_item':dict(Counter(Path(r['image']).parts[0] for r in rows)),
           'selection':'2normalviewsper30TRAINitems,then3rdnormalfirst4items;metadataonlyfixedorder;nooutput-basedselection',
           'image_decodes':0,'NN_forward_calls':0}
    write_new(OUT/'selected64-before-any-forward.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='sources'},indent=2))


def state_hashes(model):
    return {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name,value in model.state_dict().items()}


def check_bn_spec(model,protocol):
    from torch import nn
    rows=[{'name':n,'eps':m.eps,'momentum':m.momentum,'num_features':m.num_features,'initial_counter':int(m.num_batches_tracked)}
          for n,m in model.student.named_modules() if isinstance(m,nn.BatchNorm2d)]
    if rows!=protocol['actual_BN_factory']['student_BN']:raise ValueError('ActualsourceBNfactory differsfromprotectedmetadata')
    return [(n,m) for n,m in model.student.named_modules() if isinstance(m,nn.BatchNorm2d)]


def verify_bn_fit(protocol):
    receipt=OUT/'BN-fitting-complete.json';value=json.loads(receipt.read_text(encoding='utf-8'))
    state=OUT/'real-TRAIN-BN64-fitted-state.pt'
    if value['protocol_sha256']!=sha(protocol_path()) or value['state_sha256']!=sha(state):raise ValueError('BNfittedstate/protocolchanged')
    if value['forward_passes_completed']!=64 or value['source_identity_sha256']!=protocol['BN_fit_source_identity_sha256']:
        raise ValueError('Exact64fittedsourcesrequired')
    if not value['all_invariants_passed'] or value['optimizer_created'] or value['gradient_updates']!=0:raise ValueError('BN fitting guards failed')
    validate_bn_rows(value['BN_after'],protocol['actual_BN_factory']['student_BN'],64)
    check_only_bn_changed(value['before_state_sha256_by_key'],value['after_state_sha256_by_key'],allowed_bn_keys(protocol['actual_BN_factory']['student_BN']))
    return receipt,state,value


def fit_bn(protocol):
    import cv2,numpy as np,torch
    from factory_stfpm_graph import PyramidTeacherStudent
    from prepare_factory_stfpm import module_digest
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    target=OUT/'BN-fitting-complete.json';state=OUT/'real-TRAIN-BN64-fitted-state.pt'
    if target.exists() or state.exists() or (OUT/'BN-fitting-started-before-forward.json').exists():raise ValueError('Preservepartial/fittedBNstate;neverrepeatpass')
    base=ROOT/'datasets/factory/KolektorSDD';frames=[]
    for row in protocol['BN_fit_sources']:
        p=base/row['image'];mask=p.with_name(p.stem+'_label.bmp')
        if sha(p)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('SelectedBNfitsourcechanged')
        frame=cv2.imread(str(p));truth=cv2.imread(str(mask),0)
        if frame is None or truth is None or truth.any() or frame.shape[:2]!=truth.shape:raise ValueError('AllselectedBNfitinputs mustberealverifiednormalTRAIN')
        frames.append(cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA))
    model=PyramidTeacherStudent(protocol['teacher']);spec=check_bn_spec(model,protocol)
    export=json.loads((ROOT/'output/optimization/factory-round3l/spatial_round3l_stfpm_mobile-step0-export.json').read_text(encoding='utf-8'))
    if sha(ROOT/'output/optimization/factory-round3l/spatial_round3l_stfpm_mobile-step0-export.json')!='df98576f1ef7f21c4bf5a8206afefb7224305e5bb907ed88901ca05883d51a0b':raise ValueError('Immutable original seeded-state receipt changed')
    if module_digest(model.teacher)!=export['teacher_state_sha256'] or module_digest(model.student)!=export['student_state_sha256']:
        raise ValueError('Pre-BNteacher/randomstudentmustmatchoriginalseededstates')
    before=state_hashes(model);teacher_before=module_digest(model.teacher)
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval();model.student.train()
    for _,bn in spec:bn.reset_running_stats();bn.momentum=None
    write_new(OUT/'BN-fitting-started-before-forward.json',{'protocol_sha256':sha(protocol_path()),'source_identity_sha256':protocol['BN_fit_source_identity_sha256'],
             'before_state_sha256_by_key':before,'actual_BN_before':protocol['actual_BN_factory']['student_BN'],
             'teacher_before_sha256':teacher_before,'optimizer_created':False,'gradients_enabled':False,'declared_forward_passes':64})
    observations=[]
    with torch.inference_mode():
        for row,frame in zip(protocol['BN_fit_sources'],frames):
            raw=torch.from_numpy(frame.transpose(2,0,1)[None].copy()).float()
            value=(raw[:,[2,1,0]]/255.-model.mean)/model.std
            for block in model.student:value=block(value)
            if not torch.isfinite(value).all():raise ValueError('NonfiniteBNfittingfeatures')
            observations.append({'image':row['image'],'last_feature_min':float(value.min()),'last_feature_max':float(value.max()),
                                 'last_feature_rms':float(torch.mean(value*value).sqrt())})
    bn_after=[]
    for name,bn in spec:
        bn.momentum=.01
        finite=bool(torch.isfinite(bn.running_mean).all() and torch.isfinite(bn.running_var).all())
        bn_after.append({'name':name,'counter':int(bn.num_batches_tracked),'finite':finite,
                         'variance_min':float(bn.running_var.min()),'variance_max':float(bn.running_var.max()),
                         'preserved_eps':bn.eps,'restored_momentum':bn.momentum})
    model.eval();after=state_hashes(model)
    changed=check_only_bn_changed(before,after,allowed_bn_keys(protocol['actual_BN_factory']['student_BN']))
    validate_bn_rows(bn_after,protocol['actual_BN_factory']['student_BN'],64)
    if module_digest(model.teacher)!=teacher_before:raise ValueError('Teacher changed during BN fit')
    for row in protocol['BN_fit_sources']:
        p=base/row['image'];mask=p.with_name(p.stem+'_label.bmp')
        if sha(p)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('TRAINsourcechangedduringBNfit')
    torch.save(model.state_dict(),state)
    value={'protocol_sha256':sha(protocol_path()),'state_sha256':sha(state),'forward_passes_completed':len(observations),
           'source_identity_sha256':protocol['BN_fit_source_identity_sha256'],'BN_after':bn_after,
           'before_state_sha256_by_key':before,'after_state_sha256_by_key':after,'changed_state_keys':changed,
           'teacher_sha256':teacher_before,'teacher_unchanged':True,'all_parameters_unchanged':True,'normalization_buffers_unchanged':True,
           'observations':observations,'all_invariants_passed':True,'optimizer_created':False,'gradient_updates':0,
           'stage':'ActualTRAIN-basedBNfitting,NOTunfitted/noNN;fittedstateusedexactlyfornewexport/future600'}
    write_new(target,value);print(json.dumps({k:v for k,v in value.items() if k not in ['before_state_sha256_by_key','after_state_sha256_by_key','observations']},indent=2))


def export_model(model,step,protocol):
    import torch,numpy as np
    from prepare_factory_stfpm import reference_inputs,module_digest
    fit_receipt,_,fit=verify_bn_fit(protocol)
    if step==0 and state_hashes(model)!=fit['after_state_sha256_by_key']:raise ValueError('Export must start from EXACT saved BN-fitted state')
    if module_digest(model.teacher)!=fit['teacher_sha256']:raise ValueError('Export teacher must remain unchanged')
    path=model_path(step);reference=OUT/f'{NAME}-step{step}-torch-reference.npz';target=OUT/f'{NAME}-step{step}-export.json'
    if any(p.exists() for p in [path,path.with_suffix('.onnx.json'),reference,target]):raise ValueError('Preservepartial/newfittedexport')
    path.parent.mkdir(parents=True,exist_ok=True);model.eval();arrays=reference_inputs(protocol);maps=[]
    with torch.inference_mode():
        for frame in arrays:maps.append(model(torch.from_numpy(frame.transpose(2,0,1)[None].copy())).numpy()[0,0])
        values=np.stack(maps)
        if values.shape!=(3,640,256) or not np.isfinite(values).all() or values.min()<0 or values.max()>1:raise ValueError('FittedTorchreferenceinvalid')
        if step==0 and float(np.ptp(values[0]))<1e-7:raise ValueError('Fittedcapacitygraphmustbenonconstant')
        torch.onnx.export(model,torch.from_numpy(arrays[0].transpose(2,0,1)[None].copy()),str(path),opset_version=17,
                          input_names=['bgr'],output_names=['anomaly_score'],dynamo=False,do_constant_folding=True)
    with reference.open('xb') as stream:np.savez_compressed(stream,inputs_bgr=arrays,scores=values)
    write_new(path.with_suffix('.onnx.json'),metadata(path,protocol))
    value={'step':step,'protocol_sha256':sha(protocol_path()),'model_sha256':sha(path),'reference_sha256':sha(reference),
           'teacher_state_sha256':module_digest(model.teacher),'student_state_sha256':module_digest(model.student),
           'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),'BN_fit_receipt_sha256':sha(fit_receipt),
           'BN_fitted_state_sha256':fit['state_sha256'],'real_TRAIN_map_range':[float(values[0].min()),float(values[0].max())],
           'reference_scope':'SameTRAIN/zero/all255INPUTS,newcorrespondingBN-fittedTorchscores;initializationcasesneverinBNfit',
           'output_logits':False}
    write_new(target,value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=['module-metadata','select64','record-protocol','fit-bn','export-preflight'])
    parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if args.stage=='module-metadata':record_module_metadata();return
    if args.stage=='select64':record_selection();return
    if args.stage=='record-protocol':write_new(protocol_path(),protocol_value());print(json.dumps({'protocol':str(protocol_path()),'sha256':sha(protocol_path()),'forward_calls':0}));return
    if not args.root_released:raise ValueError('ActualrootBN-fitting/exportrelease required')
    _,protocol=verify_protocol()
    try:
        if args.stage=='fit-bn':fit_bn(protocol)
        else:
            import torch
            from factory_stfpm_graph import PyramidTeacherStudent
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            _,state,_=verify_bn_fit(protocol);model=PyramidTeacherStudent(protocol['teacher'])
            model.load_state_dict(torch.load(state,map_location='cpu',weights_only=True),strict=True)
            print(json.dumps(export_model(model,0,protocol),indent=2))
    except Exception as error:
        target=OUT/f'{args.stage}-failure.json'
        if not target.exists():write_new(target,{'error':str(error),'type':type(error).__name__,'protocol_sha256':sha(protocol_path()),'automatic_retry':False})
        raise


if __name__=='__main__':main()
