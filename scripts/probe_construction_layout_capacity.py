"""Four bounded TRAIN-only layout/thread trials; execute after parent compute release."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from probe_construction_train_capacity import build_batch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/training/construction-round4-layout-capacity'
HEAD=ROOT/'models/construction/evcivil-yolov8n-r4-head-640-gain.onnx'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1);torch.set_num_interop_threads(1)
    plan_path=OUT/'plan.json';plan=json.loads(plan_path.read_text(encoding='utf8'))
    expected_trials=[{'memory_format':layout,'cpu_threads':threads} for layout in ['contiguous','channels_last'] for threads in [3,4]]
    if plan['trials']!=expected_trials or plan['maximum_steps_per_trial']!=4 or plan['batch_size']!=4 or plan['freeze_first_blocks']!=4:
        raise RuntimeError('Bounded four-trial design changed')
    source=ROOT/plan['source_checkpoint'];data=ROOT/plan['data_root']
    meta=json.loads(HEAD.with_suffix('.provenance.json').read_text(encoding='utf8'))
    op_path=HEAD.with_suffix('.operating-point.json');op=json.loads(op_path.read_text(encoding='utf8'))
    if meta['training']['completed_epochs']!=4 or sha(HEAD)!=meta['onnx_sha256'] or op['model_sha256']!=sha(HEAD):
        raise RuntimeError('Complete exact head fit and calibration are prerequisites')
    if not op.get('serving_decoder_parity_verified') or not op.get('minimum_continuation_guard',{}).get('requires_distinct_stage2'):
        raise RuntimeError('Frozen calibration does not require a valid stage2 continuation')
    if source.resolve()!=(Path(meta['save_dir'])/'weights/best.pt').resolve():
        raise RuntimeError('Probe must use the actual retained head checkpoint')
    prepared=json.loads((data/'data-protocol.json').read_text(encoding='utf8'))
    native_path=data/'native-input-manifest.json'
    if sha(data/'data-protocol.json')!=meta['prepared_data_protocol_sha256'] or sha(native_path)!=prepared['native_input_manifest_sha256']:
        raise RuntimeError('Prepared inputs changed')
    selected=sorted([r for r in json.loads(native_path.read_text(encoding='utf8')) if r['split']=='train'],key=lambda r:r['raw_source_file_sha256'])[:4]
    protocol={**plan,'source_checkpoint_sha256':sha(source),'plan_sha256':sha(plan_path),'probe_script_sha256':sha(Path(__file__)),
        'head_operating_point_sha256':sha(op_path),'native_input_manifest_sha256':sha(native_path),
        'ordered_training_images':[r['image'] for r in selected],'frozen_before_probe':True}
    protocol_path=OUT/'preregistered-protocol.json'
    if protocol_path.exists():raise RuntimeError('Refusing to repeat an already bound layout probe')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    process=psutil.Process();results=[]
    for trial in plan['trials']:
        torch.set_num_threads(trial['cpu_threads']);torch.manual_seed(plan['seed'])
        network=YOLO(str(source),task='detect').model.float().train()
        network.args=get_cfg(overrides=dict(network.args));network.args.cls_pw=.5
        if network.names!={0:'spalling',1:'crack'}:raise RuntimeError('Retained head semantics changed')
        for name,parameter in network.named_parameters():
            frozen='.dfl.' in name or any(name.startswith(f'model.{i}.') for i in range(plan['freeze_first_blocks']))
            parameter.requires_grad_(not frozen)
        memory_format=torch.channels_last if trial['memory_format']=='channels_last' else torch.contiguous_format
        network.to(memory_format=memory_format)
        optimizer=torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=.0005)
        batch=build_batch(selected);batch['img']=batch['img'].contiguous(memory_format=memory_format)
        times=[];losses=[];error=None;memory=process.memory_info();peak=max(memory.rss,getattr(memory,'peak_wset',memory.rss))
        try:
            for step in range(plan['maximum_steps_per_trial']):
                started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                loss,_=network.loss(batch)
                if not torch.isfinite(loss).all():raise RuntimeError('Non-finite actual training loss')
                loss.sum().backward();optimizer.step();times.append(time.perf_counter()-started)
                losses.append(float(loss.detach().sum()));memory=process.memory_info()
                peak=max(peak,memory.rss,getattr(memory,'peak_wset',memory.rss))
                if peak>plan['maximum_rss_bytes']:raise RuntimeError('Exceeded6GiB process memory guard')
        except RuntimeError as failure:error=str(failure)
        result={**trial,'batch_size':4,'steps_completed':len(times),'step_seconds':times,'losses':losses,
            'peak_process_rss_bytes':peak,'training_images_per_second':4/float(np.mean(times[1:])) if len(times)>1 else None,
            'status':'passed' if error is None and len(times)==4 else 'rejected','error':error,
            'memory_scope':'Windows lifetime process peak working set includes preceding trials; conservative6GiB guard'}
        results.append(result);print(json.dumps(result,indent=2),flush=True)
        del network,optimizer,batch
        if 'loss' in locals():del loss
        gc.collect()
    passed=[r for r in results if r['status']=='passed']
    if not passed:raise RuntimeError('No safe finite-loss layout trial')
    selected=max(passed,key=lambda r:r['training_images_per_second'])
    summary={'protocol_sha256':sha(protocol_path),'results':results,
        'selected_memory_format':selected['memory_format'],'selected_cpu_threads':selected['cpu_threads'],
        'source_checkpoint_sha256':sha(source),'scope':'TRAIN capacity only; reset identical checkpoint/seed. No saved weights, calibration or heldout performance.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
