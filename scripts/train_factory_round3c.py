"""Attempt3: learned full-resolution surface refinement; train partitions only."""
from pathlib import Path
from collections import OrderedDict
import argparse,hashlib,json,random,time
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from train_factory_round3 import ContextSpatialUNet,make_batch
from train_factory_round3b import Sdd2Fit,sdd2_batch,sensitivity_loss,SPLIT_SHA
from train_factory_spatial_unet import export,train_mode
from factory_training_keys import hardnegative_keys,require_matched_hardnegatives

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/optimization/factory-round3'
NAME='spatial_round3c_fullres_refine'
ARCHITECTURE='Compact context U-Net plus8channel fullres14-to8 pointwise,depthwise3x3,pointwise8,pointwise1 residual refinement;12upsampleddecoderfeatures+rawchannelmean+5x5localcontrast; noGT input'
SERVING_POOL=1
WARMSTART_FREEZE=OUT/'winner-attempt2-freeze-before-comparison.json'
WARMSTART_SOURCE=ROOT/'models/factory/spatial_round3b_context_sdd2_fn-step400.pt'
WARMSTART_ORIGIN='Immutable attempt2 calibration-selected step400, not a comparison-selected losing checkpoint'
SEED=20261023
STEPS=600
SAVE_AT=[200,400,600]
PROTOCOL_OVERRIDES={}
SDD2=ROOT/'datasets/factory/KolektorSDD2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class FullResolutionRefiner(ContextSpatialUNet):
    def __init__(self,pretrained):
        super().__init__(pretrained)
        self.refine=nn.Sequential(nn.Conv2d(14,8,1),nn.ReLU(),
            nn.Conv2d(8,8,3,padding=1,groups=8),nn.ReLU(),
            nn.Conv2d(8,8,1),nn.ReLU(),nn.Conv2d(8,1,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)

    def forward(self,bgr):
        raw=bgr.to(torch.float32)/255.
        x=(raw[:,[2,1,0]]-self.mean)/self.std
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
        features=self.fine_spatial(F.relu(x+self.fine(fine)))
        logits=F.interpolate(self.classifier(features),size=bgr.shape[-2:],mode='bilinear',align_corners=False)
        features=F.interpolate(features,size=bgr.shape[-2:],mode='bilinear',align_corners=False)
        gray=raw.mean(1,keepdim=True)
        detail=gray-F.avg_pool2d(gray,5,stride=1,padding=2,count_include_pad=False)
        return logits+self.refine(torch.cat((features,gray,detail),dim=1))


def training_rows(items):
    rows=[]
    for item in items:
        for path in sorted((ROOT/'datasets/factory/KolektorSDD'/item).glob('*.jpg')):
            frame=cv2.imread(str(path));truth=cv2.imread(str(path.with_name(path.stem+'_label.bmp')),0)
            if frame is None or truth is None:raise ValueError(f'Invalid training pair {path}')
            frame=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA)
            target=(cv2.resize((truth>0).astype(np.float32),(512,1280),interpolation=cv2.INTER_AREA)>0).astype(np.uint8)
            ys,xs=np.nonzero(target);rows.append((frame,target,ys,xs,f'{item}/{path.name}'))
    return rows


class FullResolutionSdd2(Sdd2Fit):
    def get(self,index):
        if index in self.cache:
            value=self.cache.pop(index);self.cache[index]=value;return value
        row=self.records[index];frame=cv2.imread(str(SDD2/row['image']));truth=cv2.imread(str(SDD2/row['mask']),0)
        if frame is None or truth is None or bool(np.any(truth))!=bool(row['defective']):raise ValueError('Invalid frozen SDD2 fit pair')
        frame=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA)
        target=(cv2.resize((truth>0).astype(np.float32),(512,1280),interpolation=cv2.INTER_AREA)>0).astype(np.uint8)
        ys,xs=np.nonzero(target);value=(frame,target,ys,xs,row['image']);self.cache[index]=value
        if len(self.cache)>64:self.cache.popitem(last=False)
        return value


def initialize(expected_source_sha256=None):
    weights=ROOT/'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
    freeze=WARMSTART_FREEZE
    winner=json.loads(freeze.read_text(encoding='utf-8'))['winner']
    source=Path(winner['model_path']).with_suffix('.pt')
    if sha(Path(winner['model_path']))!=winner['model_sha256']:raise ValueError('Frozen warmstart ONNX changed')
    if expected_source_sha256 is not None and sha(source)!=expected_source_sha256:raise ValueError('Loaded warmstart differs from protected training protocol')
    model=FullResolutionRefiner(weights)
    missing,unexpected=model.load_state_dict(torch.load(source,map_location='cpu',weights_only=True),strict=False)
    if unexpected or any(not key.startswith('refine.') for key in missing):raise ValueError('Fullres warmstart keys changed')
    return model,source,freeze


def export_serving(model,path):
    serving=nn.Sequential(model,nn.MaxPool2d(SERVING_POOL)) if SERVING_POOL>1 else model
    export(serving,path)


def training_batch(rows,sdd2,rng,hard,step):
    return sdd2_batch(sdd2,rng) if step%5==0 else make_batch(rows,rng,hard,count=2,step=step,contrast=(.4,1.4))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',action='store_true');parser.add_argument('--record-protocol',action='store_true')
    parser.add_argument('--contended-training',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    seed=SEED;torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    splitpath=SDD2/'train-inner-split-before-fit.json'
    if sha(splitpath)!=SPLIT_SHA:raise ValueError('Frozen SDD2 split changed')
    original=json.loads((ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json').read_text(encoding='utf-8'))
    source=WARMSTART_SOURCE
    freeze=WARMSTART_FREEZE
    protocol={'name':NAME,'seed':seed,'architecture':ARCHITECTURE,
      'warmstart':str(source),'warmstart_sha256':sha(source),'warmstart_origin':WARMSTART_ORIGIN,
      'warmstart_freeze_sha256':sha(freeze),'split_items':original['split_items'],'sdd2_inner_split_sha256':SPLIT_SHA,
      'input_shape':[1,3,1280,512],'output_shape':[1,1,1280//SERVING_POOL,512//SERVING_POOL],
      'fine_steps':STEPS,'save_at':SAVE_AT,'batch_size':2,'patch':[640,256],
      'sampling':'Every5th batch SDD2 inner-fit rehearsal;other80% SDD1 originaltrain50/25/25positive/hard/unbiased; no comparison mining',
      'mask_projection':'Direct originalGT to512x1280 INTER_AREA>0 binary any-overlap occupancy; no stride2 pooling or validation dilation',
      'loss':'2*positive softplus+2*top10%negative softplus(min1024)+positivepatchTversky(FP.3,FN.7,smooth1)',
      'augmentation':'Train matchingflips,contrast.4:1.4,brightness+-30,15%mildGaussianblur; real unchanged evaluation inputs',
      'optimizer':'AdamW encoder/context5e-5,decoder1e-4,newrefiner3e-4,weightdecay1e-4; frozenBN; clipnorm5',
      'cpu_threads':1,'contended_training':args.contended_training,
      'selection_gate':'NormalcalFP<=1/70 and aggregateforegroundprecision>=.65',
      'selection_rank':['meaningfulIoU>=.1viewrecall','worst_positive_IoU','coverage50pct_viewrecall','worst_positive_GTcoverage','microIoU','fewerNormalFP','foregroundrecall'],
      'raw_high_grid':[.05,.1,.25,.5,.7,.9,.95,.98,.99,.995,.999,.9999,.99999,.999999],
      'area_grid':[1,2,4,8,16,32] if SERVING_POOL>1 else [1,2,4,8,16,32,64,128],'growth_grid':[None,.05,.1,.25,.5,.7,.8,.9,.95,.98,.99],
      'seed_search':'Best eligiblearea perhigh bysameweakest-positive rank;top6distincthighs plusnogrowth;finalfixedrank',
      'capacity_gate':'Nonconstant fullshape preflight beforefit; provisional contendedcalcapacity may permitaccuracyonly; finalisolatedpairedcapacity+runtime mandatory',
      'official_sdd2_test_still_sealed':True,'comparison_images_used_for_fit_selection':0}
    if SERVING_POOL>1:
        protocol.update(training_output_shape=[1,1,1280,512],serving_score_reduction='Maximum learned logit over each2x2full-resolution cell; sigmoid remainsmonotonic; nativehalfgridmaskcalibration, notfullreslocalization')
    protocol.update(PROTOCOL_OVERRIDES)
    protocolpath=OUT/(NAME+'-protocol-before-fit.json')
    if protocolpath.exists() and json.loads(protocolpath.read_text(encoding='utf-8'))!=protocol:raise ValueError('Protected attempt3 protocol differs')
    if not protocolpath.exists():protocolpath.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.record_protocol:print(json.dumps(protocol,indent=2));return
    model,source,freeze=initialize(protocol['warmstart_sha256'])
    if args.probe:
        # Nonconstant residual prevents zero-head constant folding understating serving cost.
        with torch.no_grad():model.refine[-1].weight.normal_(0,.01)
        path=ROOT/'models/factory'/(NAME+'-capacity-probe.onnx');export_serving(model,path)
        train_mode(model);times=[]
        for _ in range(4):
            tick=time.perf_counter();model(torch.zeros((protocol['batch_size'],3,*protocol['patch']),dtype=torch.uint8)).mean().backward();model.zero_grad(set_to_none=True);times.append(time.perf_counter()-tick)
        report={'model':str(path),'parameters':sum(p.numel() for p in model.parameters()),'patch_forward_backward_seconds':times,'scope':'Contended1-thread CPU patch capacity only, not accuracy or isolated speed'}
        (OUT/(NAME+'-training-capacity-probe.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2));return
    split=json.loads(splitpath.read_text(encoding='utf-8'));sdd2=FullResolutionSdd2(split['fit'])
    rows=training_rows(original['split_items']['train']);hard=hardnegative_keys(json.loads((OUT/'v10-train-hardneg.json').read_text(encoding='utf-8'))['records'])
    require_matched_hardnegatives([r[4] for r in rows],hard)
    enc=list(model.encoder.parameters())+list(model.context.parameters())+list(model.context_projection.parameters());ref=list(model.refine.parameters())
    known={id(p) for p in enc+ref};decoder=[p for p in model.parameters() if id(p) not in known]
    optimizer=torch.optim.AdamW([{'params':enc,'lr':5e-5},{'params':decoder,'lr':1e-4},{'params':ref,'lr':3e-4}],weight_decay=1e-4)
    started=time.perf_counter();history=[];train_mode(model)
    for step in range(1,STEPS+1):
        image,target=training_batch(rows,sdd2,rng,hard,step)
        optimizer.zero_grad(set_to_none=True);loss=sensitivity_loss(model(image),target)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at{step}')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%25==0:
            row={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-started};history.append(row);print(json.dumps(row),flush=True)
            (OUT/(NAME+'-training-progress.json')).write_text(json.dumps({'protocol':protocol,'history':history},indent=2),encoding='utf-8')
        if step in protocol['save_at']:
            path=ROOT/'models/factory'/f'{NAME}-step{step}.onnx';torch.save(model.state_dict(),path.with_suffix('.pt'));export_serving(model,path);train_mode(model)
    report={'protocol':protocol,'history':history,'elapsed_seconds':time.perf_counter()-started,'calibration_images_opened':0,'comparison_images_opened':0,'official_sdd2_test_still_sealed':True}
    (OUT/(NAME+'-training.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
