"""Round4 real-data supervised training; prepare first, exact root approval before NN.

The serving topology is the retained C200 phase-max model. This file never opens
SDD1 calibration/comparison or SDD2 reserved/test pixels while preparing/fitting.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from datetime import datetime,timezone
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'output/optimization/factory-round4-supervised'
NAME='supervised_round4_real_coverage'
PROTOCOL=OUT/'protocol-before-fit.json'
FIXTURE=OUT/'preparation-tests-v2-encoding.json'
C_SOURCE=ROOT/'models/factory/spatial_round3e_phase_max_refine-step200.pt'
INIT_WEIGHTS=ROOT/'models/factory/encoder_sources/mobilenet_v3_small-047dcff4.pth'
C_META=C_SOURCE.with_suffix('.onnx.json')
C_FREEZE=ROOT/'output/optimization/factory-round3/winner-attempt3-phase-max-freeze-before-comparison.json'
SDD2_SPLIT=ROOT/'datasets/factory/KolektorSDD2/train-inner-split-before-fit.json'
SDD2_SPLIT_SHA='ecee83db4becdac3133a2d4c0950ef383c511d143b5509aa8c1fdf61d445a47c'
MTD_CURATED_SHA='53cb4a0eac6ed52e26eb713933fcf9739c2111b576c666c6ad806a34a3280158'
MTD_SCREEN=ROOT/'datasets/factory/round4/magnetic-tile-author/cross-source-screen-v1.json'
MTD_SCREEN_SHA='39c258323c0ad2a60852c171d7ba9a9ce1c58aa9ebd1ac10cbe9a5a35e1ecb20'
HARD=ROOT/'output/optimization/factory-round3/v10-train-hardneg.json'
SEED=20261031
TRANSFER_STEPS=4096
FINE_STEPS=900
SAVE_AT=[300,600,900]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def projected(frame,mask,dataset):
    """Actual binary occupancy only; new domains retain native aspect ratio."""
    if frame is None or mask is None or frame.shape[:2]!=mask.shape:raise ValueError('Actual paired source geometry required')
    if dataset=='SDD1':size=(512,1280)
    else:size=(512,max(1,round(mask.shape[0]*512/mask.shape[1])))
    image=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
    target=(cv2.resize((mask>0).astype(np.float32),size,interpolation=cv2.INTER_AREA)>0).astype(np.uint8)
    return image,target

def crop_observed(image,target,top,height):
    """Zero-padding is input-only; target/loss validity excludes unobserved pixels."""
    if image.shape[:2]!=target.shape or image.shape[1]!=512 or top<0 or top>=len(target):raise ValueError('Invalid observed crop')
    actual=min(height,len(target)-top)
    frame=np.zeros((height,512,3),np.uint8);truth=np.zeros((height,512),np.uint8);valid=np.zeros_like(truth)
    frame[:actual]=image[top:top+actual];truth[:actual]=target[top:top+actual];valid[:actual]=1
    return frame,truth,valid

def coverage_crops(target,rng):
    """Every foreground pixel appears in an actual, nonoverlapping vertical tile."""
    if target.ndim!=2 or target.shape[1]!=512:raise ValueError('512-wide actual target required')
    positives=int(target.sum());tiles=[]
    if positives:
        for top in range(0,len(target),320):
            count=int(target[top:top+320].sum())
            if count:tiles.append({'top':top,'height':320,'foreground_pixels':count,'role':'complete-positive-tile'})
        if sum(r['foreground_pixels'] for r in tiles)!=positives:raise ValueError('Projected TRAIN foreground omitted')
    else:
        top=int(rng.integers(max(1,len(target)-320+1)))
        tiles.append({'top':top,'height':320,'foreground_pixels':0,'role':'normal-source-crop'})
    return tiles

def source_rows(mtd_manifest):
    """Manifest contract is deliberately explicit; no generic dataset globbing."""
    frozen=read(C_FREEZE)['winner']
    if sha(C_META)!=frozen['metadata_sha256'] or sha(C_SOURCE.with_suffix('.onnx'))!=frozen['model_sha256']:raise ValueError('Retained C200 model/metadata differs from original freeze')
    meta=read(C_META);rows=[]
    for item in meta['split_items']['train']:
        for image in sorted((ROOT/'datasets/factory/KolektorSDD'/item).glob('*.jpg')):
            rows.append({'id':'SDD1/'+item+'/'+image.name,'dataset':'SDD1','image':image.relative_to(ROOT).as_posix(),
                         'mask':image.with_name(image.stem+'_label.bmp').relative_to(ROOT).as_posix()})
    if len(rows)!=239:raise ValueError('Exact protected239 SDD1TRAIN views required')
    if sha(SDD2_SPLIT)!=SDD2_SPLIT_SHA:raise ValueError('Protected SDD2 FIT split changed')
    split=read(SDD2_SPLIT)
    if len(split['fit'])!=1865 or sum(r['defective'] for r in split['fit'])!=197:raise ValueError('Exact1865/197 SDD2FIT required')
    for row in split['fit']:
        image=ROOT/'datasets/factory/KolektorSDD2'/row['image'];mask=ROOT/'datasets/factory/KolektorSDD2'/row['mask']
        if sha(image)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('Frozen SDD2 FIT source drift')
        rows.append({'id':'SDD2/'+row['image'],'dataset':'SDD2','image':image.relative_to(ROOT).as_posix(),
                     'mask':mask.relative_to(ROOT).as_posix(),'expected_defective':row['defective']})
    if sha(mtd_manifest)!=MTD_CURATED_SHA:raise ValueError('Exact independently curated MTD manifest required')
    mt=read(mtd_manifest)
    if mt['all_images_TRAIN_only'] is not True or mt['all_original_masks_preserved'] is not True:raise ValueError('TRAIN-only original MTD masks required')
    # Populated after the independent author-source acquisition/QA contract is final.
    for row in mt['rows']:
        if row['split']!='train' or row['training_asset_valid'] is not True or row['mask_encoding']!='native_uint8_nonzero_support_TRAIN_v1':raise ValueError('Independently validated TRAIN-only MTD original/conversion required')
        image=ROOT/row['image'];mask=ROOT/row['mask']
        if sha(image)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('Acquired MTD source drift')
        rows.append({'id':'MTD/'+row['source_image'],'dataset':'MTD','image':row['image'],'mask':row['mask'],
                     'expected_defective':row['category']!='MT_Free','category':row['category']})
    if len(mt['rows'])!=1340 or len(mt['quarantined_rows'])!=4:raise ValueError('Exact1340 accepted MTD TRAIN originals and4 preserved label-disagreement quarantines required')
    if len({r['id'] for r in rows})!=3444:raise ValueError('Duplicate/missing accepted real TRAIN source')
    return rows,meta

def build_ledger(rows):
    rng=np.random.default_rng(SEED);ledger=[];counts={};coverage={}
    for source in rows:
        image=ROOT/source['image'];mask=ROOT/source['mask'];frame=cv2.imread(str(image));truth=cv2.imread(str(mask),0)
        if frame is None or truth is None or frame.shape[:2]!=truth.shape:raise ValueError('Invalid real TRAIN pair '+source['id'])
        if truth.dtype!=np.uint8:raise ValueError('Native grayscale byte annotation required')
        values,hist=np.unique(truth,return_counts=True);codes=values.tolist()
        defective=bool(np.any(truth))
        if 'expected_defective' in source and defective!=source['expected_defective']:raise ValueError('Source class/mask disagreement '+source['id'])
        source.update(image_sha256=sha(image),mask_sha256=sha(mask),native_shape=list(truth.shape),mask_codes=codes,
                      mask_code_histogram={str(int(v)):int(n) for v,n in zip(values,hist)},native_foreground_pixels=int(np.count_nonzero(truth)),defective=defective)
        _,target=projected(frame,truth,source['dataset']);tiles=coverage_crops(target,rng)
        source.update(projected_shape=list(target.shape),projected_foreground_pixels=int(target.sum()))
        ledger.extend([{'source_id':source['id'],**tile} for tile in tiles])
        count=counts.setdefault(source['dataset'],{'images':0,'positive':0,'normal':0,'mandatory_tiles':0,'foreground_pixels':0})
        count['images']+=1;count['positive']+=defective;count['normal']+=not defective;count['mandatory_tiles']+=len(tiles);count['foreground_pixels']+=int(target.sum())
        coverage[source['id']]={'projected_foreground_pixels':int(target.sum()),'ledger_foreground_pixels':sum(t['foreground_pixels'] for t in tiles)}
    if len(ledger)>TRANSFER_STEPS:raise ValueError('Complete source/foreground ledger exceeds fixed4096; report/review before fitting: '+str(len(ledger)))
    mandatory=len(ledger)
    positive={d:[r for r in ledger if r['foreground_pixels'] and r['source_id'].startswith(d+'/')] for d in counts}
    if not all(positive.values()):raise ValueError('Each supervised domain needs actual foreground')
    for index in range(TRANSFER_STEPS-mandatory):
        domain=['SDD1','SDD2','MTD'][index%3];pool=positive[domain]
        ledger.append(dict(pool[int(rng.integers(len(pool)))],role='source-balanced-positive-repeat'))
    rng.shuffle(ledger)
    return ledger,counts,coverage,mandatory

def sources_unchanged(rows):
    for row in rows:
        if sha(ROOT/row['image'])!=row['image_sha256'] or sha(ROOT/row['mask'])!=row['mask_sha256']:raise ValueError('Pinned TRAIN source changed '+row['id'])

class SourceCache:
    def __init__(self,rows):self.rows={r['id']:r for r in rows};self.cache=OrderedDict()
    def get(self,key):
        if key in self.cache:
            value=self.cache.pop(key);self.cache[key]=value;return value
        row=self.rows[key];image=ROOT/row['image'];mask=ROOT/row['mask']
        if sha(image)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('TRAIN source changed before read')
        value=projected(cv2.imread(str(image)),cv2.imread(str(mask),0),row['dataset']);self.cache[key]=value
        if len(self.cache)>24:self.cache.popitem(last=False)
        return value

def augment(frame,truth,valid,rng):
    if rng.random()<.5:frame,truth,valid=frame[:,::-1],truth[:,::-1],valid[:,::-1]
    if rng.random()<.5:frame,truth,valid=frame[::-1],truth[::-1],valid[::-1]
    frame=np.clip((frame.astype(np.float32)-127.5)*rng.uniform(.6,1.4)+127.5+rng.uniform(-25,25),0,255).astype(np.uint8)
    if rng.random()<.1:frame=cv2.GaussianBlur(frame,(3,3),float(rng.uniform(.3,.6)))
    return tuple(np.ascontiguousarray(x) for x in (frame,truth,valid))

def fine_ledger(rows,transfer):
    from factory_training_keys import hardnegative_keys
    hard=hardnegative_keys(read(HARD)['records']);rng=np.random.default_rng(SEED+1);cache=SourceCache(rows)
    original=[r for r in rows if r['dataset']=='SDD1'];positive=[r for r in original if r['defective']]
    difficult=[r for r in original if hard.get(r['id'][5:])]
    if not difficult:raise ValueError('Actual existing TRAIN hard locations required')
    new={d:[t for t in transfer if t['source_id'].startswith(d+'/')] for d in ['SDD2','MTD']};plan=[]
    for index in range(FINE_STEPS):
        slot=index%5
        if slot==4:
            domain=['SDD2','MTD'][(index//5)%2];pool=new[domain]
            plan.append(dict(pool[int(rng.integers(len(pool)))],role='new-real-domain-rehearsal'));continue
        pool=positive if slot<2 else difficult if slot==2 else original
        source=pool[int(rng.integers(len(pool)))];_,truth=cache.get(source['id'])
        if slot<2:
            ys=np.nonzero(truth)[0];cy=float(ys[int(rng.integers(len(ys)))])+rng.uniform(-192,192)
        elif slot==2:
            points=hard[source['id'][5:]];cy=points[int(rng.integers(len(points)))]['y']+rng.uniform(-80,80)
        else:cy=rng.uniform(320,960)
        top=int(np.clip(cy-320,0,640))
        plan.append({'source_id':source['id'],'top':top,'height':640,'foreground_pixels':int(truth[top:top+640].sum()),
                     'role':['SDD1-positive','SDD1-positive','SDD1-hard-location','SDD1-unbiased'][slot]})
    return plan

def supervised_loss(logits,target,valid):
    """Original dense sensitivity objective, restricted to actually observed pixels."""
    import torch
    from torch.nn import functional as F
    observed=valid>.5;positive=(target>.5)&observed;negative=(target<=.5)&observed
    if not observed.any():raise ValueError('No observed TRAIN pixels')
    pos=F.softplus(-logits)[positive];neg=F.softplus(logits)[negative]
    k=min(len(neg),max(1024,len(neg)//10))
    loss=2*(pos.mean() if len(pos) else logits.sum()*0)+2*(neg.topk(k,sorted=False).values.mean() if k else logits.sum()*0)
    if positive.any():
        probability=logits.sigmoid();tp=(probability*positive).sum();fp=(probability*negative).sum();fn=((1-probability)*positive).sum()
        loss=loss+1-(tp+1)/(tp+.3*fp+.7*fn+1)
    return loss

def prepare(mtd_manifest):
    if PROTOCOL.exists() or (OUT/'proposed-protocol-before-fit.json').exists():raise ValueError('Preserve protected ROUND4 preparation')
    fixture=read(FIXTURE)
    if fixture['all_checks_passed'] is not True or fixture['tests']!=9 or fixture['skipped']!=0:raise ValueError('All9 actual geometry/loss/guard/search tests required')
    for rel,h in fixture['source_pins'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Stale ROUND4 preparation tests')
    if sha(MTD_SCREEN)!=MTD_SCREEN_SHA:raise ValueError('Exact independent cross-source screen required')
    rows,meta=source_rows(mtd_manifest);ledger,counts,coverage,mandatory=build_ledger(rows);fine=fine_ledger(rows,ledger)
    write(OUT/'TRAIN-source-manifest.json',{'sources':rows,'counts':counts,'MTD_manifest_sha256':sha(mtd_manifest),'SDD2_split_sha256':sha(SDD2_SPLIT),
        'new_domain_physical_independence':'Unknown; MTD is TRAIN-only, no internal score claimed; SDD2 source/hash split is not verified physical-item independence'})
    write(OUT/'mandatory-TRAIN-foreground-coverage.json',{'all_projected_foreground_covered':all(r['projected_foreground_pixels']==r['ledger_foreground_pixels'] for r in coverage.values()),
        'sources':coverage,'mandatory_tiles':mandatory,'complete_NORMAL_pixels_coverage_claimed':False})
    write(OUT/'transfer-ledger-before-fit.json',{'seed':SEED,'steps':TRANSFER_STEPS,'mandatory_tiles':mandatory,'ledger':ledger})
    write(OUT/'fine-ledger-before-fit.json',{'seed':SEED+1,'steps':FINE_STEPS,'ledger':fine})
    paths=[Path(__file__),ROOT/'scripts/train_factory_round3d.py',ROOT/'scripts/train_factory_round3c.py',ROOT/'scripts/train_factory_round3.py',
           ROOT/'scripts/train_factory_spatial_unet.py',ROOT/'scripts/train_factory_round3b.py',ROOT/'scripts/factory_training_keys.py',
           ROOT/'scripts/calibrate_factory_stfpm_learn.py',ROOT/'scripts/factory_stfpm_common.py',ROOT/'scripts/factory_mask_metrics.py',
           ROOT/'scripts/evaluate_factory_round4.py',ROOT/'tests/test_factory_round4_supervised.py',ROOT/'edge_runtime/profiles/factory_encoder.py']
    protocol={'name':NAME,'seed':SEED,'transfer_steps':TRANSFER_STEPS,'fine_steps':FINE_STEPS,'save_at':SAVE_AT,'cpu_threads':1,'batch_size':1,
        'architecture':'Exact retained C200 PackedPixelRefiner then MaxPool2d2; no serving topology/normalization/output change',
        'warmstart_sha256':sha(C_SOURCE),'warmstart_onnx_sha256':sha(C_SOURCE.with_suffix('.onnx')),'warmstart_metadata_sha256':sha(C_META),'warmstart_freeze_sha256':sha(C_FREEZE),
        'constructor_pretrained_sha256':sha(INIT_WEIGHTS),'constructor_state':'Constructor uses pinned officialImageNet source required by existing class, then STRICT full C200 state overrides all tensors beforeoptimizer',
        'MTD_manifest_path':str(mtd_manifest.relative_to(ROOT)),'MTD_manifest_sha256':sha(mtd_manifest),'sources_sha256':sha(OUT/'TRAIN-source-manifest.json'),
        'preparation_tests_sha256':sha(FIXTURE),'independent_cross_source_screen_sha256':sha(MTD_SCREEN),
        'coverage_sha256':sha(OUT/'mandatory-TRAIN-foreground-coverage.json'),'ledger_sha256':sha(OUT/'transfer-ledger-before-fit.json'),
        'fine_ledger_sha256':sha(OUT/'fine-ledger-before-fit.json'),'hard_TRAIN_locations_sha256':sha(HARD),'mandatory_tiles':mandatory,'counts':counts,
        'split_items':meta['split_items'],'input_shape':[1,3,1280,512],'output_shape':[1,1,640,256],
        'transfer_geometry':'320x512 actual full-width tiles; SDD1 exact512x1280 inputgrid, newdomains isotropicwidth512; inputpad has lossvalid0; binaryany-overlapmaskprojection, noGTmorphology',
        'raw_mask_interpretation':'SDD1/SDD2 preserve existing C originaluint8mask>0 support (including2gray85SDD1pixels); MTD explicit curatedoursupport>0. Rawcodes/histograms retained, noannotationfile edits/thresholdchange',
        'fine_geometry':'640x512 actualSDD1fullwidth; positivepointverticaljitter+-192, hardTRAINcentres or unbiasedactualcrop; every5thstep fixednewdomainledgerrehearsal',
        'augmentation':'Matched horizontal/verticalflips; contrast.6:1.4 brightness+-25,10%3x3blur sigma.3:.6; no90degrees, no synthesis/noevaluationaugmentation',
        'loss':'2positiveBCE+2top10%negativeBCE(min1024)+TverskyFP.3/FN.7, onlyobservedpixels; logitsfloat32,targetfloat32,validbool',
        'optimizer':'AdamW fixed perphase: transfer encoder/context2e-5,decoder/refiner1e-4; fine encoder/context1e-5,decoder/refiner5e-5; weightdecay1e-4,frozenBN,clipnorm5,freshoptimizeratfine',
        'raw_high_grid':[.05,.1,.25,.5,.7,.9,.95,.98,.99,.995,.999,.9999,.99999,.999999],
        'area_grid':[1,2,4,8,16,32],'growth_grid':[None,.05,.1,.25,.5,.7,.8,.9,.95,.98,.99],
        'selection':'All3completed/checkpointactualnumericthenfixedoriginal80CAL; bothprospectivearms/all3outcomes/globalwinnerperarmfreeze beforeANYoriginal; samefullrawgrid/growth/weakest-positive rank; pixelprecision>=.65; noCAL/originalfeedbackintofit',
        'selection_arms':{'primary':{'normal_FP_budget':1},'secondary':{'normal_FP_budget':3}},'foreground_pixel_precision_guard':.65,
        'secondary_scope':'<=3/70targets fewer than historic5/69normalVIEWalarms, notbetter than C2001/69; bothoutcomesmandatory, nooriginal-driven arm selection',
        'numeric':'OwnlearnedTorchscore threefixedTRAIN/zero/255inputs versusactualGPUf16 scores maxabs<=.01, actualconfig/finite/bounded/nonconstant; failureexcludesCAL, all3terminalrequired',
        'performance':'Root actualfitted>=15completeprofile before exposedoriginalaudit; paired16FPS60s/fullcounters beforeacceptance; nohistoryspeedinheritance',
        'official_SDD2_test_still_sealed':True,'SDD2_reserved466_opened':False,'MTD_used_for_training_only':True,'calibration_or_comparison_pixels_used_for_fit':0,
        'source_pins':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    write(OUT/'proposed-protocol-before-fit.json',protocol)
    print(json.dumps({'mandatory_tiles':mandatory,'transfer_steps':TRANSFER_STEPS,'fine_steps':FINE_STEPS,'counts':counts,'proposal_sha256':sha(OUT/'proposed-protocol-before-fit.json')},indent=2))

def verify_plan():
    proposal=OUT/'proposed-protocol-before-fit.json';protocol=read(PROTOCOL)
    if protocol!=read(proposal):raise ValueError('Exact approved proposal required')
    approval=read(OUT/'parent-approved-fit-decision.json')
    if approval.get('approved_exact_round4_fit') is not True or approval['proposal_sha256']!=sha(proposal):raise ValueError('Root exact fit approval required')
    for rel,h in protocol['source_pins'].items():
        if sha(ROOT/rel)!=h:raise ValueError('ROUND4 source changed before execution')
    for name,field in [('TRAIN-source-manifest.json','sources_sha256'),('mandatory-TRAIN-foreground-coverage.json','coverage_sha256'),
                       ('transfer-ledger-before-fit.json','ledger_sha256'),('fine-ledger-before-fit.json','fine_ledger_sha256')]:
        if sha(OUT/name)!=protocol[field]:raise ValueError('Protected TRAIN plan changed')
    if (sha(C_SOURCE)!=protocol['warmstart_sha256'] or sha(C_META)!=protocol['warmstart_metadata_sha256'] or sha(HARD)!=protocol['hard_TRAIN_locations_sha256']
        or sha(C_SOURCE.with_suffix('.onnx'))!=protocol['warmstart_onnx_sha256'] or sha(C_FREEZE)!=protocol['warmstart_freeze_sha256']
        or sha(INIT_WEIGHTS)!=protocol['constructor_pretrained_sha256']):
        raise ValueError('Warmstart or TRAIN mining changed')
    if (sha(ROOT/protocol['MTD_manifest_path'])!=protocol['MTD_manifest_sha256'] or sha(FIXTURE)!=protocol['preparation_tests_sha256']
        or sha(MTD_SCREEN)!=protocol['independent_cross_source_screen_sha256']
        or protocol['selection_arms']!={'primary':{'normal_FP_budget':1},'secondary':{'normal_FP_budget':3}}):raise ValueError('Protected source/tests/dual policy declaration changed')
    rows=read(OUT/'TRAIN-source-manifest.json')['sources'];sources_unchanged(rows)
    return protocol,rows

def train():
    import torch
    from torch import nn
    from train_factory_round3d import PackedPixelRefiner
    from train_factory_spatial_unet import train_mode,export
    if (OUT/'training-started.json').exists():raise ValueError('Preserve any partial training; no silent optimizer restart')
    protocol,rows=verify_plan();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED);random.seed(SEED)
    cache=SourceCache(rows);model=PackedPixelRefiner(INIT_WEIGHTS)
    model.load_state_dict(torch.load(C_SOURCE,map_location='cpu',weights_only=True),strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('Source state nonfinite')
    fixed_buffers={n:t.clone() for n,t in model.named_buffers()};rng=np.random.default_rng(SEED+2)
    archive=OUT/'executed-sources-before-optimizer'
    for rel,h in protocol['source_pins'].items():
        path=archive/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/rel).read_bytes())
        if sha(path)!=h:raise ValueError('Source archive failure')
    write(archive/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'source_pins':protocol['source_pins']})
    write(OUT/'training-started.json',{'protocol_sha256':sha(PROTOCOL),'source_sha256':sha(C_SOURCE),'source_archive_sha256':sha(archive/'manifest.json'),
        'cpu_threads':torch.get_num_threads(),'phases':['transfer4096','fine900'],'scope':'TRAIN optimizer work; no inference FPS claim'})
    original=next(r for r in rows if r['dataset']=='SDD1');raw=cv2.imread(str(ROOT/original['image']))
    reference_input=np.stack([cv2.resize(raw,(512,1280),interpolation=cv2.INTER_AREA),np.zeros((1280,512,3),np.uint8),np.full((1280,512,3),255,np.uint8)])
    history=[];started=time.perf_counter()
    for phase in ['transfer','fine']:
        plan=read(OUT/(phase+'-ledger-before-fit.json'))['ledger'];expected=TRANSFER_STEPS if phase=='transfer' else FINE_STEPS
        if len(plan)!=expected:raise ValueError('Fixed complete TRAIN schedule required')
        enc=list(model.encoder.parameters())+list(model.context.parameters())+list(model.context_projection.parameters())
        encoder_ids={id(p) for p in enc};decoder=[p for p in model.parameters() if id(p) not in encoder_ids]
        optimizer=torch.optim.AdamW([{'params':enc,'lr':2e-5 if phase=='transfer' else 1e-5},
                                    {'params':decoder,'lr':1e-4 if phase=='transfer' else 5e-5}],weight_decay=1e-4)
        train_mode(model)
        for step,tile in enumerate(plan,1):
            image,target=cache.get(tile['source_id']);frame,truth,valid=crop_observed(image,target,tile['top'],tile['height'])
            if int(truth.sum())!=tile['foreground_pixels']:raise ValueError('Prepared foreground crop differs at execution')
            frame,truth,valid=augment(frame,truth,valid,rng)
            x=torch.from_numpy(frame.transpose(2,0,1)[None].copy());y=torch.from_numpy(truth[None,None].copy()).float();v=torch.from_numpy(valid[None,None].copy()).bool()
            optimizer.zero_grad(set_to_none=True);logits=model(x);loss=supervised_loss(logits,y,v)
            if not torch.isfinite(loss):raise ValueError('Nonfinite TRAIN loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite TRAIN gradient')
            nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if any(not torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('Nonfinite updated model state')
            if any(not torch.equal(t,fixed_buffers[n]) for n,t in model.named_buffers()):raise ValueError('Frozen BN/normalization state changed')
            if step%50==0 or step==expected:
                record={'phase':phase,'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-started,'source_dataset':tile['source_id'].split('/')[0]}
                history.append(record);print(json.dumps(record),flush=True)
                (OUT/'training-progress.json').write_text(json.dumps({'protocol_sha256':sha(PROTOCOL),'history':history},indent=2),encoding='utf8')
            if phase=='fine' and step in SAVE_AT:
                path=ROOT/f'models/factory/{NAME}-step{step}.onnx'
                if path.exists() or path.with_suffix('.pt').exists():raise ValueError('Preserve checkpoint bytes')
                torch.save(model.state_dict(),path.with_suffix('.pt'));torch.save(optimizer.state_dict(),OUT/f'checkpoint{step}-optimizer.pt')
                serving=nn.Sequential(model,nn.MaxPool2d(2));export(serving,path)
                with torch.inference_mode():
                    reference=serving(torch.from_numpy(reference_input.transpose(0,3,1,2).copy())).sigmoid().numpy()[:,0].copy()
                reference_path=OUT/f'checkpoint{step}-torch-reference.npz'
                with reference_path.open('xb') as stream:np.savez_compressed(stream,inputs_bgr=reference_input,scores=reference)
                baseline=read(C_META)
                keys=['feature_version','split_items','split_seed','supported_height_width_ratio','output_logits','input_width','input_height',
                      'input_shape','output_shape','model_grid_width','model_grid_height','score_meaning','architecture','minimum_component_area_units','serving_score_reduction']
                metadata={k:baseline[k] for k in keys}
                metadata.update(model_sha256=sha(path),profile_version=NAME,model_name='Experimental supervised real-data surface defect review',
                    training_protocol=protocol,round4_training_protocol_sha256=sha(PROTOCOL),official_sdd2_test_still_sealed=True,MTD_training_only=True,
                    dataset='KolektorSDD1 TRAIN + KolektorSDD2 innerFIT + author MagneticTile TRAIN-only',
                    dataset_urls=['https://www.vicos.si/resources/kolektorsdd/','https://www.vicos.si/resources/kolektorsdd2/','https://github.com/abin24/Magnetic-tile-defect-datasets.'],
                    trained_utc=datetime.now(timezone.utc).isoformat(),calibration_status='pending_for_new_weights',
                    pixel_threshold=baseline['pixel_threshold'],minimum_component_pixels=baseline['minimum_component_pixels'],hysteresis_growth_threshold=baseline['hysteresis_growth_threshold'],
                    threshold_selection='Initial C policy is a placeholder, not calibration evidence for new weights; select only from protected Round4 CAL',
                    pretraining_lineage={'C200_PT_sha256':protocol['warmstart_sha256'],'C200_ONNX_sha256':protocol['warmstart_onnx_sha256']},
                    license='Kolektor CC-BY-NC-SA4 research prototype plus author MTD research-use permission/citation; no commercial MTD grant declared',
                    manual_review_only=True,automatic_acceptance=False,automatic_rejection=False,field_validated=False)
                write(path.with_suffix('.onnx.json'),metadata)
                write(OUT/f'checkpoint{step}-training.json',{'step':step,'protocol_sha256':sha(PROTOCOL),'model_sha256':sha(path),'state_sha256':sha(path.with_suffix('.pt')),
                    'initial_metadata_sha256':sha(path.with_suffix('.onnx.json')),'optimizer_sha256':sha(OUT/f'checkpoint{step}-optimizer.pt'),'reference_sha256':sha(reference_path),
                    'all_state_finite':True,'frozen_BN_normalization_unchanged':True,'elapsed_seconds':time.perf_counter()-started})
                train_mode(model)
        if phase=='transfer':
            torch.save(model.state_dict(),OUT/'transfer4096-complete.pt')
            write(OUT/'transfer-complete.json',{'steps':TRANSFER_STEPS,'protocol_sha256':sha(PROTOCOL),'state_sha256':sha(OUT/'transfer4096-complete.pt'),'no_CAL_feedback':True})
    sources_unchanged(rows)
    write(OUT/'training-complete.json',{'protocol_sha256':sha(PROTOCOL),'transfer_steps':TRANSFER_STEPS,'fine_steps':FINE_STEPS,'saved_all_three':SAVE_AT,
        'all_TRAIN_sources_unchanged':True,'frozen_BN_normalization_unchanged':True,'elapsed_seconds':time.perf_counter()-started,'history':history,
        'calibration_or_comparison_pixels_opened':0,'official_SDD2_test_still_sealed':True})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','record-protocol','train'],required=True)
    parser.add_argument('--MTD-manifest',type=Path);parser.add_argument('--root-released',action='store_true');args=parser.parse_args();cv2.setNumThreads(1)
    if args.stage=='prepare':prepare(args.MTD_manifest.resolve());return
    if args.stage=='record-protocol':
        approval=read(OUT/'parent-approved-fit-decision.json');proposal=OUT/'proposed-protocol-before-fit.json'
        if approval.get('approved_exact_round4_fit') is not True or approval['proposal_sha256']!=sha(proposal):raise ValueError('Root exact fit approval required')
        write(PROTOCOL,read(proposal));verify_plan();return
    if not args.root_released:raise ValueError('Separate root training compute release required')
    train()

if __name__=='__main__':main()
