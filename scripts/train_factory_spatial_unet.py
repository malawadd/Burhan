"""Train a spatial, end-to-end compact U-Net on real KolektorSDD training items.

Use the isolated Torch export environment. This file never opens comparison
images or masks. Calibration and deployment benchmarking run separately.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]


class SpatialUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1,
                                             groups=channels, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                    nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x) + x


class SpatialUNet(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        if pretrained is not None:
            backbone.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True))
        self.encoder = nn.ModuleList(list(backbone.features.children())[:7])
        self.coarse = nn.Conv2d(40, 12, 1)
        self.middle = nn.Conv2d(24, 12, 1)
        self.middle_spatial = nn.Sequential(SpatialUnit(12), SpatialUnit(12))
        self.fine = nn.Conv2d(16, 12, 1)
        self.fine_spatial = nn.Sequential(SpatialUnit(12), SpatialUnit(12))
        self.classifier = nn.Conv2d(12, 1, 1)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, bgr):
        x = (bgr.to(torch.float32)[:, [2, 1, 0]] / 255. - self.mean) / self.std
        fine = middle = None
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index == 0:
                fine = x
            elif index == 3:
                middle = x
        x = F.interpolate(self.coarse(x), size=middle.shape[-2:], mode='bilinear', align_corners=False)
        x = self.middle_spatial(F.relu(x + self.middle(middle)))
        x = F.interpolate(x, size=fine.shape[-2:], mode='bilinear', align_corners=False)
        return self.classifier(self.fine_spatial(F.relu(x + self.fine(fine))))


def export(model, path, shape=(1, 3, 1280, 512)):
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(shape, dtype=torch.uint8), str(path),
                          input_names=['bgr'], output_names=['logits'], opset_version=17,
                          dynamo=False, do_constant_folding=True)


def train_mode(model):
    model.train()
    # Patch batches must not replace pretrained population statistics with
    # brightness-biased positive-only patch statistics. Affine parameters train.
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def load_training(root, item_names):
    rows = []
    for item in item_names:
        for path in sorted((root / item).glob('*.jpg')):
            frame = cv2.imread(str(path))
            mask = cv2.imread(str(path.with_name(path.stem + '_label.bmp')), 0)
            if frame is None or mask is None:
                raise ValueError(f'Invalid real training image/mask: {path}')
            frame = cv2.resize(frame, (512, 1280), interpolation=cv2.INTER_AREA)
            mask = cv2.resize((mask > 0).astype(np.uint8), (512, 1280), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(mask)
            rows.append((frame, mask, ys, xs, str(path.relative_to(root))))
    return rows


def load_sdd2_training(root, split):
    rows=[]
    names=split.get('fit_images') or [Path(row['image']).name for row in split['fit']]
    for name in names:
        path=root/name
        frame=cv2.imread(str(path))
        mask=cv2.imread(str(path.with_name(path.stem+'_GT.png')),0)
        if frame is None or mask is None:
            raise ValueError(f'Invalid SDD2 training image/mask: {path}')
        ys,xs=np.nonzero(mask)
        # Cache native images (~230x630), not 2x-upscaled images for all 2331
        # views; resizing occurs only for a selected training patch source.
        rows.append((frame,(mask>0).astype(np.uint8),ys,xs,name))
    return rows


def batch(rows, rng, count=8, hardneg=None):
    positive = [row for row in rows if len(row[2])]
    normal = [row for row in rows if not len(row[2])]
    images, masks = [], []
    h, w = 320, 128
    for index in range(count):
        # Four positive-centred, two normal-view, two unbiased random patches.
        source = positive if index % 4 < 2 else normal if index % 4 == 2 else rows
        if index%4==2 and hardneg:
            source=[row for row in rows if hardneg.get(row[4])]
        frame, mask, ys, xs, name = source[int(rng.integers(len(source)))]
        if frame.shape[:2]!=(1280,512):
            frame=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_LINEAR)
            mask=cv2.resize(mask,(512,1280),interpolation=cv2.INTER_NEAREST)
            if source is positive:
                ys,xs=np.nonzero(mask)
        if source is positive:
            point = int(rng.integers(len(ys)))
            top = int(np.clip(ys[point] - rng.integers(32, h - 32), 0, 1280 - h))
            left = int(np.clip(xs[point] - rng.integers(16, w - 16), 0, 512 - w))
        elif index%4==2 and hardneg:
            centers=hardneg[name];point=centers[int(rng.integers(len(centers)))]
            top=int(np.clip(point['y']-h//2+rng.integers(-32,33),0,1280-h))
            left=int(np.clip(point['x']-w//2+rng.integers(-16,17),0,512-w))
        else:
            top, left = int(rng.integers(1280-h+1)), int(rng.integers(512-w+1))
        image = frame[top:top+h, left:left+w].copy()
        target = mask[top:top+h, left:left+w].copy()
        if rng.random() < .5:
            image, target = image[:, ::-1], target[:, ::-1]
        if rng.random() < .5:
            image, target = image[::-1], target[::-1]
        image = np.clip((image.astype(np.float32)-127.5) * rng.uniform(.8, 1.2) +
                        127.5 + rng.uniform(-20., 20.), 0, 255).astype(np.uint8)
        images.append(image.transpose(2, 0, 1))
        masks.append(target[None])
    return torch.from_numpy(np.ascontiguousarray(images)), torch.from_numpy(np.ascontiguousarray(masks)).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--save-at', type=int, nargs='+', default=[300, 600])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--name', default='spatial_unet_v9')
    parser.add_argument('--dataset-family',choices=['sdd1','sdd2'],default='sdd1')
    parser.add_argument('--split-json',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--hardneg-json',type=Path)
    parser.add_argument('--extended-score-grid',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(20261025)
    random.seed(20261025)
    rng = np.random.default_rng(20261025)
    directory = ROOT/'output/optimization/factory-round2'
    directory.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT/'models/factory/kolektor_surface_rf.joblib.json').read_text(encoding='utf-8'))
    weights = ROOT/'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
    expected = '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
    if hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
        raise ValueError('Official pretrained weight checksum mismatch')
    model = SpatialUNet(weights)
    if args.resume:
        model.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=True))
    protocol = {'seed': 20261025, 'candidate': args.name,
                'architecture': 'End-to-end trainable MobileNetV3Small blocks0:7, stride2/8/16 skip decoder, depthwise3x3 spatial residuals',
                'full_input_shape': [1,3,1280,512], 'output_shape':[1,1,640,256],
                'training_patch': [320,128], 'batch_size':8, 'steps':args.steps, 'save_at':args.save_at,
                'encoder_lr': .0002, 'decoder_lr': .001, 'optimizer':'AdamW weight_decay=.0001',
                'loss':'BCEWithLogits positive_weight=10 + per-positive-patch softDice',
                'sampling':'50% positive-centered, 25% normal-view, 25% unbiased training patch',
                'augmentation':'Train only matching horizontal/vertical flips, contrast .8:1.2, brightness -20:20',
                'batchnorm':'Frozen running statistics; affine and all convolution parameters train',
                'training_mask_projection':'Nearest resize into full input, then 2x2 max pool to stride-two output, preserving thin real annotation coverage; evaluation masks remain original resolution',
                'split_items':original['split_items'], 'training_only_items':original['split_items']['train'],
                'checkpoint_selection':'Calibration only, never comparison; localized recall then pixelIoU, normal FPR<=3/70',
                'threshold_grid':[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,.99,.995,.999],
                'minimum_area_grid':[1,2,4,8,16,32],
                'pretrained_sha256':expected, 'training_data':'Real original KolektorSDD; no generated validation or demonstration footage'}
    if args.resume:
        protocol['warmstart_checkpoint']=str(args.resume)
        protocol['warmstart_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
    if args.extended_score_grid:
        protocol['threshold_grid'].extend([.9999,.99999,.999999])
        protocol['score_grid_reason']='V9 calibration found overconfident normal locations; extended before this new candidate is trained or calibrated. No comparison observations used.'
    if args.hardneg_json:
        protocol['hardnegative_patch_mining']={'file':str(args.hardneg_json),'sha256':hashlib.sha256(args.hardneg_json.read_bytes()).hexdigest(),
                                             'sampling':'Replace25%normal-view patches with jittered train-only hard-location patches, matching true masks retained'}
    if args.dataset_family=='sdd2':
        if not args.split_json:
            raise ValueError('SDD2 requires a predeclared author-training-only split JSON')
        sdd2_partition=json.loads(args.split_json.read_text(encoding='utf-8'))
        protocol.update(training_data='Real official KolektorSDD2 author training images only, reserved training-calibration and official test excluded',
                        training_only_items=[],downstream_sdd1_item_split=original['split_items'],
                        sdd2_fit_view_count=len(sdd2_partition.get('fit_images') or sdd2_partition['fit']),
                        sdd2_split_json=str(args.split_json),sdd2_split_sha256=hashlib.sha256(args.split_json.read_bytes()).hexdigest())
    (directory/(args.name+'-protocol-before-fit.json')).write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.probe:
        path = ROOT/'models/factory'/(args.name+'-untrained-probe.onnx')
        export(model, path)
        value = torch.zeros((8,3,320,128),dtype=torch.uint8)
        train_mode(model)
        times=[]
        for _ in range(6):
            tick=time.perf_counter();result=model(value);result.mean().backward();model.zero_grad(set_to_none=True)
            times.append(time.perf_counter()-tick)
        report={'scope':'Untrained capacity probe; no accuracy claim','parameters':sum(p.numel() for p in model.parameters()),
                'patch_forward_backward_seconds':times,'warm_step_median':float(np.median(times[2:])),
                'onnx':str(path),'onnx_bytes':path.stat().st_size}
        (directory/(args.name+'-training-capacity.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2));return
    if args.dataset_family=='sdd2':
        split=json.loads(args.split_json.read_text(encoding='utf-8'))
        rows=load_sdd2_training(ROOT/'datasets/factory/KolektorSDD2/train',split)
    else:
        rows=load_training(ROOT/'datasets/factory/KolektorSDD',original['split_items']['train'])
    encoder_ids={id(p) for p in model.encoder.parameters()}
    optimizer=torch.optim.AdamW([{'params':list(model.encoder.parameters()),'lr':.0002},
                                {'params':[p for p in model.parameters() if id(p) not in encoder_ids],'lr':.001}],weight_decay=.0001)
    loss_function=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.))
    hardneg=None
    if args.hardneg_json:
        hardneg={row['image']:row['locations'] for row in json.loads(args.hardneg_json.read_text(encoding='utf-8'))['records']}
    started=time.perf_counter();history=[];train_mode(model)
    for step in range(1,args.steps+1):
        image,target=batch(rows,rng,hardneg=hardneg)
        target=F.max_pool2d(target,2)
        optimizer.zero_grad(set_to_none=True)
        logits=model(image)
        loss=loss_function(logits,target)
        probability=logits.sigmoid()
        has_target=target.flatten(1).sum(1)>0
        if has_target.any():
            selected_p,selected_t=probability[has_target],target[has_target]
            dice=1-((2*(selected_p*selected_t).flatten(1).sum(1)+1)/
                    (selected_p.flatten(1).sum(1)+selected_t.flatten(1).sum(1)+1)).mean()
            loss=loss+dice
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite training loss at step {step}; do not export this model')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step % 25 == 0:
            row={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
        if step in args.save_at:
            prefix=ROOT/'models/factory'/f'{args.name}-step{step}'
            torch.save(model.state_dict(),prefix.with_suffix('.pt'))
            export(model,prefix.with_suffix('.onnx'))
            train_mode(model)
        (directory/(args.name+'-training-progress.json')).write_text(json.dumps({'protocol':protocol,'history':history},indent=2),encoding='utf-8')
    report={'protocol':protocol,'history':history,'elapsed_seconds':time.perf_counter()-started,
            'real_training_views':len(rows),'training_positive_views':sum(bool(len(r[2])) for r in rows),
            'comparison_images_opened':0,'calibration_images_opened':0}
    (directory/(args.name+'-training.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'history','protocol'}},indent=2))


if __name__=='__main__':
    main()
