"""L protocol recording is metadata-only. Export requires explicit root release."""
import argparse,json,hashlib
from pathlib import Path
from factory_stfpm_common import ROOT,OUT,NAME,sha,write_new,protocol_value,protocol_path,verify_protocol,metadata,model_path


def reference_inputs(protocol):
    import cv2,numpy as np
    row=protocol['normal_sources'][0];base=ROOT/'datasets/factory/KolektorSDD'
    image=base/row['image'];mask=image.with_name(image.stem+'_label.bmp')
    if sha(image)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('Fixed TRAIN reference changed')
    frame=cv2.imread(str(image));truth=cv2.imread(str(mask),0)
    if frame is None or truth is None or truth.any() or frame.shape[:2]!=truth.shape:raise ValueError('Reference must be real verified normal TRAIN')
    actual=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA)
    return np.stack([actual,np.zeros_like(actual),np.full_like(actual,255)])


def module_digest(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_model(model,step,protocol):
    """Only called in separately authorized NN stage; every output is new."""
    import torch,numpy as np
    path=model_path(step)
    ref=OUT/f'{NAME}-step{step}-torch-reference.npz'
    receipt=OUT/f'{NAME}-step{step}-export.json'
    if any(p.exists() for p in [path,path.with_suffix('.onnx.json'),ref,receipt]):raise ValueError('Preserve existing/partial export')
    path.parent.mkdir(parents=True,exist_ok=True);model.eval()
    arrays=reference_inputs(protocol);outputs=[]
    with torch.inference_mode():
        for frame in arrays:
            outputs.append(model(torch.from_numpy(frame.transpose(2,0,1)[None].copy())).numpy()[0,0])
        values=np.stack(outputs)
        if values.shape!=(3,640,256) or not np.isfinite(values).all() or values.min()<0 or values.max()>1:
            raise ValueError('Torch numerical/geometry/score bounds failed')
        if step==0 and float(np.ptp(values[0]))<1e-7:raise ValueError('Capacity probe must be actually nonconstant')
        torch.onnx.export(model,torch.from_numpy(arrays[0].transpose(2,0,1)[None].copy()),str(path),
                          opset_version=17,input_names=['bgr'],output_names=['anomaly_score'],dynamo=False,do_constant_folding=True)
    with ref.open('xb') as stream:np.savez_compressed(stream,inputs_bgr=arrays,scores=values)
    write_new(path.with_suffix('.onnx.json'),metadata(path,protocol))
    value={'step':step,'model_sha256':sha(path),'reference_sha256':sha(ref),'protocol_sha256':sha(protocol_path()),
           'teacher_state_sha256':module_digest(model.teacher),'student_state_sha256':module_digest(model.student),
           'source_teacher_sha256':protocol['teacher']['sha256'],'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),
           'reference_source':protocol['normal_sources'][0],'initialization_cases':['actual_first_normal_TRAIN','zero_initialization_only','all255_initialization_only'],
           'torch_finite_bounded':True,'real_TRAIN_score_range':[float(values[0].min()),float(values[0].max())],
           'output_logits':False,'score_semantics':protocol['score']}
    write_new(receipt,value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--record-protocol',action='store_true');choice.add_argument('--export-preflight',action='store_true')
    parser.add_argument('--root-released',action='store_true');args=parser.parse_args()
    if args.record_protocol:
        value=protocol_value();write_new(protocol_path(),value)
        print(json.dumps({'protocol':str(protocol_path()),'sha256':sha(protocol_path()),'NN_executed':False},indent=2));return
    if not args.root_released:raise ValueError('Export requires actual separate coordinator PREFLIGHTLRELEASE')
    _,protocol=verify_protocol()
    import torch
    from factory_stfpm_graph import PyramidTeacherStudent
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=PyramidTeacherStudent(protocol['teacher'])
    value=export_model(model,0,protocol)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
