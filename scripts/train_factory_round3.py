"""Round-three learned factory models: thin-mask coverage, context and OHEM.

Only original training physical items are opened. Separate scripts calibrate and
audit frozen models. Run in the isolated Torch environment.
"""
from pathlib import Path
import argparse,hashlib,json,random,time
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small
from train_factory_spatial_unet import SpatialUNet,export,train_mode
from factory_training_keys import hardnegative_keys,require_matched_hardnegatives

ROOT=Path(__file__).resolve().parents[1]


class ContextSpatialUNet(SpatialUNet):
    def __init__(self,pretrained):
        super().__init__(pretrained)
        backbone=mobilenet_v3_small(weights=None)
        backbone.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True))
        self.context=nn.ModuleList(list(backbone.features.children())[7:12])
        self.context_projection=nn.Conv2d(96,40,1)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)

    def forward(self,bgr):
        x=(bgr.to(torch.float32)[:,[2,1,0]]/255.-self.mean)/self.std
        fine=middle=None
        for index,block in enumerate(self.encoder):
            x=block(x)
            if index==0:fine=x
            elif index==3:middle=x
        coarse=x
        for block in self.context:x=block(x)
        coarse=coarse+F.interpolate(self.context_projection(x),size=coarse.shape[-2:],mode='bilinear',align_corners=False)
        x=F.interpolate(self.coarse(coarse),size=middle.shape[-2:],mode='bilinear',align_corners=False)
        x=self.middle_spatial(F.relu(x+self.middle(middle)))
        x=F.interpolate(x,size=fine.shape[-2:],mode='bilinear',align_corners=False)
        return self.classifier(self.fine_spatial(F.relu(x+self.fine(fine))))


def load_training(items):
    rows=[]
    for item in items:
        for path in sorted((ROOT/'datasets/factory/KolektorSDD'/item).glob('*.jpg')):
            image=cv2.imread(str(path));truth=cv2.imread(str(path.with_name(path.stem+'_label.bmp')),0)
            if image is None or truth is None:raise ValueError(f'Invalid real training pair {path}')
            image=cv2.resize(image,(512,1280),interpolation=cv2.INTER_AREA)
            # Direct source -> output-cell coverage avoids aliasing introduced
            # by nearest upscaling followed by pooling. Repeat to input grid so
            # even-aligned crops/flips preserve exact stride-two target cells.
            grid=cv2.resize((truth>0).astype(np.float32),(256,640),interpolation=cv2.INTER_AREA)>0
            target=np.repeat(np.repeat(grid,2,axis=0),2,axis=1).astype(np.uint8)
            ys,xs=np.nonzero(target)
            rows.append((image,target,ys,xs,f'{item}/{path.name}'))
    return rows


def make_batch(rows,rng,hardneg,count=2,step=0,contrast=(.6,1.4)):
    positives=[row for row in rows if len(row[2])]
    hard=[row for row in rows if hardneg.get(row[4])]
    images,targets=[],[];h,w=640,256
    for index in range(count):
        # Batch2 keeps one positive + alternating hard/unbiased patches, for
        # the same50/25/25 long-run ratio as batch4's2positive+hard+unbiased.
        role=([0,1 if step%2==0 else 2] if count==2 else [0,0,1,2])[index]
        pool=positives if role==0 else hard if role==1 else rows
        image,truth,ys,xs,name=pool[int(rng.integers(len(pool)))]
        if role==0:
            point=int(rng.integers(len(ys)))
            top=int(np.clip(ys[point]-rng.integers(64,h-64),0,1280-h))
            left=int(np.clip(xs[point]-rng.integers(32,w-32),0,512-w))
        elif role==1:
            points=hardneg[name];point=points[int(rng.integers(len(points)))]
            top=int(np.clip(point['y']-h//2+rng.integers(-64,65),0,1280-h))
            left=int(np.clip(point['x']-w//2+rng.integers(-32,33),0,512-w))
        else:
            top,left=int(rng.integers(1280-h+1)),int(rng.integers(512-w+1))
        top-=top%2;left-=left%2
        frame=image[top:top+h,left:left+w].copy();target=truth[top:top+h,left:left+w].copy()
        if rng.random()<.5:frame,target=frame[:,::-1],target[:,::-1]
        if rng.random()<.5:frame,target=frame[::-1],target[::-1]
        frame=np.clip((frame.astype(np.float32)-127.5)*rng.uniform(*contrast)+127.5+rng.uniform(-30,30),0,255)
        if rng.random()<.15:
            frame=cv2.GaussianBlur(frame,(3,3),float(rng.uniform(.3,.8)))
        frame=np.clip(frame,0,255).astype(np.uint8)
        images.append(frame.transpose(2,0,1));targets.append(target[None])
    return torch.from_numpy(np.ascontiguousarray(images)),torch.from_numpy(np.ascontiguousarray(targets)).float()


def loss_function(logits,target):
    positive=target>.5;negative=~positive
    pos=F.softplus(-logits)[positive]
    neg=F.softplus(logits)[negative]
    # Focus gradients on real difficult background instead of overwhelming
    # them with the many already-easy normal pixels.
    k=max(1,min(len(neg),max(1024,len(neg)//10)))
    negative_loss=neg.topk(k,sorted=False).values.mean()
    positive_loss=pos.mean() if len(pos) else logits.sum()*0
    probability=logits.sigmoid();has=positive.flatten(1).any(1)
    dice=logits.sum()*0
    if has.any():
        p,t=probability[has],target[has]
        dice=1-((2*(p*t).flatten(1).sum(1)+1)/(p.flatten(1).sum(1)+t.flatten(1).sum(1)+1)).mean()
    return positive_loss+2*negative_loss+dice


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['shallow','context'],default='context')
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--save-at',type=int,nargs='+',default=[200,400,600])
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--batch-size',type=int,choices=[2,4],default=2)
    parser.add_argument('--cpu-threads',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--contended-training',action='store_true',help='Record an explicitly coordinated concurrent fit; never an isolated speed benchmark')
    parser.add_argument('--hardneg',type=Path,default=ROOT/'output/optimization/factory-round3/v10-train-hardneg.json')
    args=parser.parse_args();torch.set_num_threads(args.cpu_threads);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(20261021);random.seed(20261021);rng=np.random.default_rng(20261021)
    directory=ROOT/'output/optimization/factory-round3';directory.mkdir(parents=True,exist_ok=True)
    weights=ROOT/'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
    warmstart=ROOT/'models/factory/spatial_unet_v10_sdd2-step300.pt'
    original=json.loads((ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json').read_text(encoding='utf-8'))
    model=ContextSpatialUNet(weights) if args.variant=='context' else SpatialUNet(weights)
    missing,unexpected=model.load_state_dict(torch.load(warmstart,map_location='cpu',weights_only=True),strict=False)
    if unexpected or any(not(key.startswith('context.') or key.startswith('context_projection.')) for key in missing):
        raise ValueError(f'Unexpected warmstart mismatch: {missing}, {unexpected}')
    name=f'spatial_round3_{args.variant}'
    protocol={'seed':20261021,'name':name,'variant':args.variant,'steps':args.steps,'save_at':args.save_at,
              'cpu_threads':args.cpu_threads,'contended_training':args.contended_training,
              'training_time_scope':'Coordinated concurrent CPU fitting, not an isolated capacity/speed measurement' if args.contended_training else 'CPU fit; actual background preparation overlap documented separately',
              'architecture':'v10 spatial U-Net plus trainable blocks7:12 ImageNet context with zero-initialized40channel residual fusion' if args.variant=='context' else 'v10 spatial U-Net controlled shallow ablation',
              'warmstart':str(warmstart),'warmstart_sha256':hashlib.sha256(warmstart.read_bytes()).hexdigest(),
              'pretrained_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
              'split_items':original['split_items'],'training_items':original['split_items']['train'],
              'patch':[640,256],'batch_size':args.batch_size,'sampling':'50% positive-centered,25% v10 train-mined hard-location,25% unbiased training-view; batch2alternateshard/unbiasedsecondpatch',
              'mask_projection':'Binary any-overlap occupancy from original annotation INTER_AREA resize into256x640 output cells (>0, not fractional targets); duplicate2x2; even-aligned crops; original evaluation labels unchanged',
              'loss':'Mean positive softplus + 2*mean top10% negative softplus (min1024pixels) + positive-patch softDice',
              'optimizer':'AdamW encoder1e-4 decoder3e-4 newcontext2e-4; weight_decay1e-4; frozen BN running statistics',
              'augmentation':'Train-only flips; contrast.6:1.4, brightness+-30,15% mild3x3Gaussianblur; original validation images untouched',
              'selection':'Calibration only: <=1/70normalFP; maximize views pixelIoU>=.1, microIoU, pixelrecall; comparison never read',
              'raw_high_grid':[.5,.7,.9,.95,.98,.99,.995,.999,.9999,.99999,.999999],
              'area_grid':[1,2,4,8,16,32],'growth_grid':[None,.5,.8,.9,.95,.98,.99],
              'growth_search':'For each high threshold choose eligible area by any-mask localized recall,foregroundrecall,microIoU; evaluate growth only for top4 different high thresholds; final rank meaningfulIoUrecall,fewerFP,microIoU,pixelrecall',
              'comparison_scope':'Original80 exposed/reused development comparison, never training/calibration selection. OfficialSDD2test remains excluded.'}
    if not args.probe:
        protocol['hardnegative_source_sha256']=hashlib.sha256(args.hardneg.read_bytes()).hexdigest()
    (directory/(name+'-protocol-before-fit.json')).write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.probe:
        if args.variant=='context':
            # Keep the added branch nonconstant in capacity probes. The actual
            # fit starts at exactly zero residual; trained weights won't stay
            # zero, so allowing constant folding would understate its cost.
            with torch.no_grad():model.context_projection.weight.normal_(0,.01)
        artifact=ROOT/'models/factory'/(name+'-capacity-probe.onnx');export(model,artifact)
        train_mode(model);values=[]
        for _ in range(5):
            start=time.perf_counter();out=model(torch.zeros((args.batch_size,3,640,256),dtype=torch.uint8));out.mean().backward();model.zero_grad(set_to_none=True)
            values.append(time.perf_counter()-start)
        report={'parameters':sum(p.numel() for p in model.parameters()),'probe_onnx':str(artifact),'patch_batch_size':args.batch_size,
                'patch_forward_backward_seconds':values,'warm_median':float(np.median(values[2:])),'scope':'Capacity only; zero inputs not accuracy examples'}
        (directory/(name+f'-training-capacity-batch{args.batch_size}.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2));return
    rows=load_training(protocol['training_items'])
    hardneg=hardnegative_keys(json.loads(args.hardneg.read_text(encoding='utf-8'))['records'])
    require_matched_hardnegatives([row[4] for row in rows],hardneg)
    encoder_ids={id(p) for p in model.encoder.parameters()};context=list(model.context.parameters())+list(model.context_projection.parameters()) if args.variant=='context' else []
    context_ids={id(p) for p in context}
    optimizer=torch.optim.AdamW([{'params':list(model.encoder.parameters()),'lr':1e-4},
                                {'params':[p for p in model.parameters() if id(p) not in encoder_ids|context_ids],'lr':3e-4},
                                {'params':context,'lr':2e-4}],weight_decay=1e-4)
    train_mode(model);start=time.perf_counter();history=[]
    for step in range(1,args.steps+1):
        image,target=make_batch(rows,rng,hardneg,count=args.batch_size,step=step);target=F.max_pool2d(target,2)
        optimizer.zero_grad(set_to_none=True);logits=model(image);loss=loss_function(logits,target)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss at step{step}')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%25==0:
            row={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-start};history.append(row);print(json.dumps(row),flush=True)
            (directory/(name+'-training-progress.json')).write_text(json.dumps({'protocol':protocol,'history':history},indent=2),encoding='utf-8')
        if step in args.save_at:
            path=ROOT/'models/factory'/f'{name}-step{step}.onnx';torch.save(model.state_dict(),path.with_suffix('.pt'));export(model,path);train_mode(model)
    report={'protocol':protocol,'history':history,'elapsed_seconds':time.perf_counter()-start,'training_views':len(rows),'positive_training_views':sum(bool(len(r[2])) for r in rows),'calibration_images_opened':0,'comparison_images_opened':0}
    (directory/(name+'-training.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in {'protocol','history'}},indent=2))


if __name__=='__main__':main()
