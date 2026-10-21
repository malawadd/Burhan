"""Attempt4: rotation/scale-covered real square patches, same phase-max serving."""
import cv2,numpy as np,torch
import train_factory_round3c as base
from train_factory_round3d import PackedPixelRefiner


def oriented_training_batch(rows,sdd2,rng,hard,step):
    positives=[row for row in rows if len(row[2])]
    difficult=[row for row in rows if hard.get(row[4])]
    images=[];targets=[]
    for index in range(4):
        if step%5==0:
            positive=index<2;pool=sdd2.positive if positive else sdd2.normal
            image,truth,ys,xs,name=sdd2.get(pool[int(rng.integers(len(pool)))]);role=0 if positive else 2
        else:
            role=[0,0,1,2][index];pool=positives if role==0 else difficult if role==1 else rows
            image,truth,ys,xs,name=pool[int(rng.integers(len(pool)))]
        side=int(rng.choice([256,384,512]))
        if role==0:
            point=int(rng.integers(len(ys)));cy,cx=float(ys[point]),float(xs[point])
            cy+=rng.uniform(-.3,.3)*side;cx+=rng.uniform(-.3,.3)*side
        elif role==1:
            point=hard[name][int(rng.integers(len(hard[name])))];cy,cx=point['y'],point['x']
            cy+=rng.uniform(-.125,.125)*side;cx+=rng.uniform(-.125,.125)*side
        else:
            cy=rng.uniform(side/2,1280-side/2);cx=rng.uniform(side/2,512-side/2) if side<512 else 256.
        top=int(np.clip(cy-side/2,0,1280-side));left=int(np.clip(cx-side/2,0,512-side))
        frame=cv2.resize(image[top:top+side,left:left+side],(320,320),interpolation=cv2.INTER_AREA)
        target=(cv2.resize(truth[top:top+side,left:left+side].astype(np.float32),(320,320),interpolation=cv2.INTER_AREA)>0).astype(np.uint8)
        turns=int(rng.integers(4));frame,target=np.rot90(frame,turns),np.rot90(target,turns)
        if rng.random()<.5:frame,target=frame[:,::-1],target[:,::-1]
        frame=np.clip((frame.astype(np.float32)-127.5)*rng.uniform(.25,1.5)+127.5+rng.uniform(-40,40),0,255)
        if rng.random()<.15:frame=cv2.GaussianBlur(frame,(3,3),float(rng.uniform(.3,.8)))
        images.append(np.clip(frame,0,255).astype(np.uint8).transpose(2,0,1));targets.append(target[None])
    return torch.from_numpy(np.ascontiguousarray(images)),torch.from_numpy(np.ascontiguousarray(targets)).float()


if __name__=='__main__':
    base.NAME='spatial_round3f_orientation_multiscale'
    base.ARCHITECTURE='Samequalifiedphase-supervised packedpixelrefiner+halfgridmaxlogitserving asattempt3C; trainingonlyorientation/scale/contrastcoveragechanged'
    base.FullResolutionRefiner=PackedPixelRefiner;base.SERVING_POOL=2
    base.WARMSTART_SOURCE=base.ROOT/'models/factory/spatial_round3e_phase_max_refine-step200.pt'
    base.WARMSTART_FREEZE=base.OUT/'winner-attempt3-phase-max-freeze-before-comparison.json'
    base.WARMSTART_ORIGIN='Immutableattempt3Ccalibration-selected200, notcomparison-selectedcheckpoint'
    base.SEED=20261024;base.STEPS=900;base.SAVE_AT=[300,600,900]
    base.training_batch=oriented_training_batch
    base.PROTOCOL_OVERRIDES={'batch_size':4,'patch':[320,320],
        'training_output_shape':[4,1,320,320],'full_source_supervision_shape':[1,1,1280,512],
        'sampling':'Every5thbatchSDD2innerfit2positive+2realnormal;other80%SDD1originaltrain2positive+1trainminedhard+1unbiased; real source squares256/384/512 resizedisotropically320',
        'augmentation':'Uniformmatched0/90/180/270degreequarterturns+horizontalflip;realtrainingcontrast.25:1.5,brightness+-40,15%mildblur; no generated evaluation damage',
        'mask_projection':'OriginalGTdirect512x1280binaryoccupancythenactualsquarecropandmatchedINTER_AREA>0to320; fullpatchloss; noevalGTdilation',
        'capacity_gate':'Accuracy-firstoriginalauditmayfollowall3calandglobalfreeze becauseunchangedservinggraphpreviously19.66/s; finalnewweights/policypairedprojectquiescentcapacityandruntime>=15mandatorybeforepromotion',
        'pipeline_calibration':'Immutablecompletedcheckpoints maycalibratewhilefixed900-stepCPU1fitcontinues; nofeedbackorschedulechanges; all3beforefreeze',
        'calibration_cache_identity':'ExactmodelSHA plusorderedcaloriginalimage/maskSHA256,nativegeometry,andclass; cachedscoresinvalidatedonsourcechanges; sameidentityacrossall3checkpoints',
        'train_orientation_audit_sha256':base.sha(base.OUT/'attempt4-train-orientation.json'),
        'train_orientation_audit':'attempt4-train-orientation.json, TRAIN-only beforefit; nopubliccomparisonpixelsused'}
    base.main()
