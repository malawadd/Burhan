"""L7 fixed600 CPU1 normal-only fit; post-fit GPU gates are never waived."""
import argparse,hashlib,json,time
from collections import OrderedDict,Counter
import cv2,numpy as np,torch
from factory_stfpm_learn_common import ROOT,OUT,NAME,SEED,SAVE_AT,sha,write_new,verify_protocol,protocol_path,model_path,validate_bn_rows
from factory_stfpm_graph import PyramidTeacherStudent
from prepare_factory_stfpm_learn import export_model,verify_bn_fit
from prepare_factory_stfpm import module_digest


def teacher_digest(model):
    digest=hashlib.sha256()
    for name,value in model.teacher.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def require_prefit(protocol):
    from factory_stfpm_learn_common import validate_initialization
    return validate_initialization(protocol)

def bn_rows(model):
    return [{'name':name,'counter':int(bn.num_batches_tracked),
             'finite':bool(torch.isfinite(bn.running_mean).all() and torch.isfinite(bn.running_var).all()),
             'variance_min':float(bn.running_var.min()),'variance_max':float(bn.running_var.max()),
             'preserved_eps':bn.eps,'restored_momentum':bn.momentum}
            for name,bn in model.student.named_modules() if isinstance(bn,torch.nn.BatchNorm2d)]


def verify_normal_files(protocol,decode=False):
    base=ROOT/'datasets/factory/KolektorSDD';stats=[]
    for row in protocol['normal_sources']:
        image=base/row['image'];mask=image.with_name(image.stem+'_label.bmp')
        if sha(image)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('NormalTRAIN source changed')
        if decode:
            frame=cv2.imread(str(image));truth=cv2.imread(str(mask),0)
            if frame is None or truth is None or truth.any() or frame.shape[:2]!=truth.shape:raise ValueError('Fitting source is not verified normal')
            stats.append({'image':row['image'],'native_shape':list(frame.shape),'native_std':float(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).std())})
    return stats


def sample_plan(count,seed=SEED,steps=600):
    rng=np.random.default_rng(seed);plan=[]
    while len(plan)<steps:
        for index in rng.permutation(count):
            draw=float(rng.random())
            y=0 if draw<.1 else 640 if draw<.2 else int(rng.integers(0,641))
            plan.append((int(index),y))
            if len(plan)==steps:break
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-fit-released',action='store_true')
    parser.add_argument('--contention-note',required=True);args=parser.parse_args()
    if not args.root_fit_released:raise ValueError('Actual coordinated root FITLRELEASE is required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
    _,protocol=verify_protocol();prereq=require_prefit(protocol)
    receipt=OUT/'training-execution-before-optimizer.json'
    if receipt.exists() or any(model_path(step).exists() or model_path(step).with_suffix('.pt').exists() for step in SAVE_AT):
        raise ValueError('Preserve previous complete/partial fixed fit; no silent restart')
    source_stats=verify_normal_files(protocol,decode=True)
    fit_receipt,fitted_state,fit=verify_bn_fit(protocol)
    model=PyramidTeacherStudent(protocol['teacher'])
    # Momentum is a Python module attribute, absent from state_dict: restore explicitly.
    model.load_state_dict(torch.load(fitted_state,map_location='cpu',weights_only=True),strict=True)
    for bn in model.student.modules():
        if isinstance(bn,torch.nn.BatchNorm2d):bn.momentum=.01
    initial_bn=bn_rows(model);validate_bn_rows(initial_bn,protocol['actual_BN_factory']['student_BN'],64)
    initial_teacher=teacher_digest(model)
    probe_export=json.loads((ROOT/'output/optimization/factory-round3l4-bn64/spatial_round3l4_stfpm_bn64-step0-export.json').read_text(encoding='utf-8'))
    if initial_teacher!=probe_export['teacher_state_sha256'] or module_digest(model.student)!=probe_export['student_state_sha256']:
        raise ValueError('Exact saved BN-fitted teacher/student differ from pinned L4 graph')
    if any(p.requires_grad for p in model.teacher.parameters()):raise ValueError('Teacher must be frozen')
    plan=sample_plan(208);counts=Counter(index for index,_ in plan)
    if len(counts)!=208 or min(counts.values())!=2 or max(counts.values())!=3:raise ValueError('Fixed600 normal coverage broken')
    write_new(receipt,{'protocol_sha256':sha(protocol_path()),'prerequisites':prereq,'source_stats':source_stats,
                       'normal_source_identity_sha256':protocol['normal_source_identity_sha256'],'sample_plan':plan,
                       'initial_teacher_state_sha256':initial_teacher,'actual_source_pins':protocol['source_pins'],
                       'initial_student_state_sha256':module_digest(model.student),'initial_BN':initial_bn,
                       'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
                       'execution_scope':args.contention_note,'capacity_evidence':False,'torch_version':torch.__version__,
                       'trainable_parameters':sum(p.numel() for p in model.student.parameters()),'cpu_threads':torch.get_num_threads()})
    from factory_stfpm_learn_common import archive_executed_sources
    archive_executed_sources(protocol,receipt)
    optimizer=torch.optim.AdamW(model.student.parameters(),lr=.001,weight_decay=.00001)
    history=[];cache=OrderedDict();base=ROOT/'datasets/factory/KolektorSDD';started=time.perf_counter()
    for step,(index,y) in enumerate(plan,1):
        if index not in cache:
            frame=cv2.imread(str(base/protocol['normal_sources'][index]['image']))
            if frame is None:raise ValueError('Training source decode failed')
            cache[index]=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA)
            if len(cache)>8:cache.popitem(last=False)
        frame=cache[index];cache.move_to_end(index)
        crop=np.ascontiguousarray(frame[y:y+640].transpose(2,0,1)[None])
        if crop.shape!=(1,3,640,512):raise ValueError('Matched normal crop is not actual observed geometry')
        model.train_normals();optimizer.zero_grad(set_to_none=True)
        if model.teacher.training or not model.student.training:raise ValueError('Teacher/student BN policy changed')
        loss=model.normal_loss(torch.from_numpy(crop))
        if not torch.isfinite(loss):raise ValueError('Nonfinite normal loss')
        loss.backward()
        if any(p.grad is not None for p in model.teacher.parameters()):raise ValueError('Teacher gradients forbidden')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.student.parameters()):raise ValueError('Nonfinite student gradient')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.student.state_dict().values()):raise ValueError('Nonfinite student parameter/BN buffer state')
        history.append({'step':step,'image':protocol['normal_sources'][index]['image'],'crop_y':y,'loss':float(loss.detach()),
                        'crop_gray_std':float(crop.mean(axis=1).std())})
        if step%50==0:print(json.dumps({'step':step,'loss':history[-1]['loss'],'elapsed_contended_s':time.perf_counter()-started}),flush=True)
        if step in SAVE_AT:
            if teacher_digest(model)!=initial_teacher:raise ValueError('Frozen teacher or BN state changed')
            counters=bn_rows(model);validate_bn_rows(counters,protocol['actual_BN_factory']['student_BN'],64+step)
            model.eval();target=model_path(step).with_suffix('.pt')
            if target.exists():raise ValueError('Never overwrite learned checkpoint')
            torch.save(model.state_dict(),target)
            torch.save({'step':step,'optimizer':optimizer.state_dict(),'sample_plan':plan,'torch_rng':torch.get_rng_state()},target.with_suffix('.optimizer.pt'))
            exported=export_model(model,step,protocol)
            write_new(OUT/f'checkpoint{step}-training.json',{'step':step,'state_sha256':sha(target),
                      'onnx_sha256':exported['model_sha256'],'teacher_state_sha256':initial_teacher,'teacher_unchanged':True,
                      'BN_rows':counters,'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
                      'execution_receipt_sha256':sha(receipt),'history_prefix':history,'student_BN_saved_in_state':True})
    verify_normal_files(protocol,decode=False)
    if teacher_digest(model)!=initial_teacher:raise ValueError('Teacher final-state drift')
    write_new(OUT/'training-complete.json',{'fixed_steps_completed':600,'save_at':SAVE_AT,'history':history,
               'normal_view_visits':{protocol['normal_sources'][i]['image']:counts[i] for i in sorted(counts)},
               'normal_sources':208,'normal_source_identity_sha256':protocol['normal_source_identity_sha256'],
               'teacher_state_sha256':initial_teacher,'teacher_unchanged':True,'execution_receipt_sha256':sha(receipt),
               'protocol_sha256':sha(protocol_path()),'elapsed_contended_seconds':time.perf_counter()-started,'capacity_evidence':False,
               'BN_fit_receipt_sha256':sha(fit_receipt),'BN_fitted_state_sha256':fit['state_sha256'],
               'checkpoint_reports':{str(s):sha(OUT/f'checkpoint{s}-training.json') for s in SAVE_AT}})
    print('Fixed600 complete; all3 GPU numeric outcomes, passing-only CAL and globalfreeze remain required.',flush=True)


if __name__=='__main__':main()
