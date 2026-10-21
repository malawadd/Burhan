"""Predeclared attempt2: real SDD2 context warmup, rehearsal and FN-aware loss.

Only frozen original SDD1 train and SDD2 inner-fit images are opened. No GPU
calibration/speed measurement occurs here. Run after a coordinated CPU handoff.
"""
from pathlib import Path
from collections import OrderedDict
import argparse,hashlib,json,random,time
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from train_factory_round3 import ContextSpatialUNet,load_training,make_batch
from train_factory_spatial_unet import train_mode,export
from factory_training_keys import hardnegative_keys,require_matched_hardnegatives

ROOT=Path(__file__).resolve().parents[1]
NAME='spatial_round3b_context_sdd2_fn'
OUT=ROOT/'output/optimization/factory-round3'
SDD2=ROOT/'datasets/factory/KolektorSDD2'
SPLIT_SHA='ecee83db4becdac3133a2d4c0950ef383c511d143b5509aa8c1fdf61d445a47c'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class Sdd2Fit:
    """Bounded real-image cache; never load reserved calibration/test records."""
    def __init__(self,records):
        self.records=records;self.cache=OrderedDict()
        self.positive=[i for i,r in enumerate(records) if r['defective']]
        self.normal=[i for i,r in enumerate(records) if not r['defective']]
    def get(self,index):
        if index in self.cache:
            value=self.cache.pop(index);self.cache[index]=value;return value
        row=self.records[index];image=cv2.imread(str(SDD2/row['image']));truth=cv2.imread(str(SDD2/row['mask']),0)
        if image is None or truth is None or bool(np.any(truth))!=bool(row['defective']):raise ValueError('Invalid frozen SDD2 fit pair')
        image=cv2.resize(image,(512,1280),interpolation=cv2.INTER_AREA)
        grid=cv2.resize((truth>0).astype(np.float32),(256,640),interpolation=cv2.INTER_AREA)>0
        target=np.repeat(np.repeat(grid,2,axis=0),2,axis=1).astype(np.uint8);ys,xs=np.nonzero(target)
        value=(image,target,ys,xs,row['image']);self.cache[index]=value
        if len(self.cache)>64:self.cache.popitem(last=False)
        return value


def sdd2_batch(dataset,rng):
    images=[];targets=[];height,width=640,256
    for positive in [True,False]:
        pool=dataset.positive if positive else dataset.normal
        image,truth,ys,xs,_=dataset.get(pool[int(rng.integers(len(pool)))])
        if positive:
            p=int(rng.integers(len(ys)))
            top=int(np.clip(ys[p]-rng.integers(64,height-64),0,1280-height))
            left=int(np.clip(xs[p]-rng.integers(32,width-32),0,512-width))
        else:top,left=int(rng.integers(1280-height+1)),int(rng.integers(512-width+1))
        top-=top%2;left-=left%2
        frame=image[top:top+height,left:left+width].copy();target=truth[top:top+height,left:left+width].copy()
        if rng.random()<.5:frame,target=frame[:,::-1],target[:,::-1]
        if rng.random()<.5:frame,target=frame[::-1],target[::-1]
        frame=np.clip((frame.astype(np.float32)-127.5)*rng.uniform(.4,1.4)+127.5+rng.uniform(-30,30),0,255)
        if rng.random()<.15:frame=cv2.GaussianBlur(frame,(3,3),float(rng.uniform(.3,.8)))
        images.append(np.clip(frame,0,255).astype(np.uint8).transpose(2,0,1));targets.append(target[None])
    return torch.from_numpy(np.ascontiguousarray(images)),torch.from_numpy(np.ascontiguousarray(targets)).float()


def sensitivity_loss(logits,target):
    positive=target>.5;neg=F.softplus(logits)[~positive];pos=F.softplus(-logits)[positive]
    k=max(1,min(len(neg),max(1024,len(neg)//10)))
    loss=2*(pos.mean() if len(pos) else logits.sum()*0)+2*neg.topk(k,sorted=False).values.mean()
    has=positive.flatten(1).any(1)
    if has.any():
        probability=logits[has].sigmoid();truth=target[has]
        tp=(probability*truth).flatten(1).sum(1)
        fp=(probability*(1-truth)).flatten(1).sum(1);fn=((1-probability)*truth).flatten(1).sum(1)
        loss=loss+(1-(tp+1)/(tp+.3*fp+.7*fn+1)).mean()
    return loss


def optimizer_for(model,fine=False):
    encoder=list(model.encoder.parameters());context=list(model.context.parameters())+list(model.context_projection.parameters())
    known={id(p) for p in encoder+context};decoder=[p for p in model.parameters() if id(p) not in known]
    return torch.optim.AdamW([{'params':encoder,'lr':5e-5 if fine else 1e-4},
                             {'params':context,'lr':1e-4 if fine else 2e-4},
                             {'params':decoder,'lr':2e-4 if fine else 3e-4}],weight_decay=1e-4)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-threads',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--contended-training',action='store_true');args=parser.parse_args()
    torch.set_num_threads(args.cpu_threads);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    seed=20261022;torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    OUT.mkdir(parents=True,exist_ok=True)
    split_path=SDD2/'train-inner-split-before-fit.json'
    if sha(split_path)!=SPLIT_SHA:raise ValueError('Frozen SDD2 inner-fit split changed')
    split2=json.loads(split_path.read_text(encoding='utf-8'))
    original=json.loads((ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json').read_text(encoding='utf-8'))
    warm=ROOT/'models/factory/spatial_unet_v10_sdd2-step300.pt';weights=ROOT/'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
    hard_path=OUT/'v10-train-hardneg.json'
    protocol={'name':NAME,'seed':seed,'architecture':'Compact context graph as attempt1; no output-resolution change',
              'warmstart':str(warm),'warmstart_sha256':sha(warm),'pretrained_sha256':sha(weights),
              'split_items':original['split_items'],'sdd2_inner_split_sha256':SPLIT_SHA,
              'sdd2_fit_images':len(split2['fit']),'sdd2_fit_defective':sum(r['defective'] for r in split2['fit']),
              'sdd2_reserved_calibration_and_official_test_opened':False,'official_sdd2_test_still_sealed':True,
              'pretrain_steps':300,'fine_steps':600,'save_at':[200,400,600],'batch_size':2,'patch':[640,256],
              'pretrain_sampling':'SDD2 inner-fit only: one positive-centered plus one random normal patch/batch',
              'fine_sampling':'Every5th batch SDD2 rehearsal; other80% original SDD1 train with50/25/25 positive/hard/unbiased sampling',
              'mask_projection':'Binary any-overlap output-cell occupancy; even crops; unchanged original masks in evaluation',
              'loss':'2*positive softplus +2*top10%negative softplus(min1024) +positive-patch Tversky(FPalpha.3,FNbeta.7,smooth1)',
              'augmentation':'Train-only matching flips, contrast.4:1.4,brightness+-30,15%mild3x3Gaussianblur; no synthesized damage or altered validation',
              'pretrain_lrs':{'encoder':1e-4,'context':2e-4,'decoder':3e-4},'fine_lrs':{'encoder':5e-5,'context':1e-4,'decoder':2e-4},
              'optimizer':'AdamW,weight_decay1e-4,frozenBNrunningstatistics,clipnorm5; optimizer reset before fine stage',
              'hardnegative_sha256':sha(hard_path),'cpu_threads':args.cpu_threads,'contended_training':args.contended_training,
              'selection_gate':'NormalcalFP<=1/70 and aggregateforegroundprecision>=.65',
              'selection_rank':['meaningfulIoU>=.1viewrecall','worst_positive_IoU','coverage50pct_viewrecall','worst_positive_GTcoverage','microIoU','fewerNormalFP','foregroundrecall'],
              'raw_high_grid':[.05,.1,.25,.5,.7,.9,.95,.98,.99,.995,.999,.9999,.99999,.999999],
              'area_grid':[1,2,4,8,16,32],'growth_grid':[None,.05,.1,.25,.5,.7,.8,.9,.95,.98,.99],
              'seed_search':'Best eligiblearea per high by same weakest-positive rank; top6 distinct highs plus no-growth; final fixed rank; growth>=0 and<=high',
              'comparison_scope':'Attempt1 failed frozen development comparison retained; no comparison-ID mining/selection or officialSDD2test access'}
    (OUT/(NAME+'-protocol-before-fit.json')).write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=ContextSpatialUNet(weights)
    missing,unexpected=model.load_state_dict(torch.load(warm,map_location='cpu',weights_only=True),strict=False)
    if unexpected or any(not(k.startswith('context.') or k.startswith('context_projection.')) for k in missing):raise ValueError('Warmstart keys changed')
    rows=load_training(protocol['split_items']['train']);sdd2=Sdd2Fit(split2['fit'])
    hard=hardnegative_keys(json.loads(hard_path.read_text(encoding='utf-8'))['records']);require_matched_hardnegatives([r[4] for r in rows],hard)
    started=time.perf_counter();history=[];train_mode(model)
    for phase,steps in [('pretrain',300),('fine',600)]:
        optimizer=optimizer_for(model,phase=='fine');phase_started=time.perf_counter()
        for step in range(1,steps+1):
            source='sdd2' if phase=='pretrain' or step%5==0 else 'sdd1'
            image,target=sdd2_batch(sdd2,rng) if source=='sdd2' else make_batch(rows,rng,hard,count=2,step=step,contrast=(.4,1.4))
            target=F.max_pool2d(target,2);optimizer.zero_grad(set_to_none=True);loss=sensitivity_loss(model(image),target)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss{phase}/{step}')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if step%25==0:
                row={'phase':phase,'step':step,'loss':float(loss.detach()),'phase_elapsed_seconds':time.perf_counter()-phase_started,'elapsed_seconds':time.perf_counter()-started}
                history.append(row);(OUT/(NAME+'-training-progress.json')).write_text(json.dumps({'protocol':protocol,'history':history},indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
            if phase=='fine' and step in protocol['save_at']:
                path=ROOT/'models/factory'/f'{NAME}-step{step}.onnx';torch.save(model.state_dict(),path.with_suffix('.pt'));export(model,path);train_mode(model)
        if phase=='pretrain':
            pretrain=ROOT/'models/factory'/(NAME+'-pretrain300.pt');torch.save(model.state_dict(),pretrain)
            (OUT/(NAME+'-pretrain.json')).write_text(json.dumps({'protocol':protocol,'weights_sha256':sha(pretrain),'elapsed_seconds':time.perf_counter()-started},indent=2),encoding='utf-8')
    report={'protocol':protocol,'history':history,'elapsed_seconds':time.perf_counter()-started,'calibration_images_opened':0,'comparison_images_opened':0,'official_sdd2_test_still_sealed':True}
    (OUT/(NAME+'-training.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['protocol','history']},indent=2))


if __name__=='__main__':main()
