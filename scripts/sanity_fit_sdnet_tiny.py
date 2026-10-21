"""64-update eight-TRAIN-photo localization sanity fit; no generalization claim.

Only the preselected TRAIN originals are decoded. No CAL/development/owner pixels,
threshold selection, network downloads, model installation or promotion.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import hashlib,json,random,shutil,sys,time
import cv2,numpy as np,torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_sdnet_expansion as shared
from construction_ssd_background_loss import install
DATA=ROOT/'datasets/construction/expansion-public-v2/sdnet2025-corrected/manifest.json'
BASE=ROOT/'models/construction/expansion-sdnet-v1/ssdlite320/model-state.pt'
OUT=ROOT/'output/evaluation/construction-resumption-fastening-tiny-v1'
SEED=20261021
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):shared.write(path,value)
def render(row,predictions,title):
    image=cv2.imread(str(ROOT/row['image']))
    if predictions is None:
        for b in row['boxes']:
            color=(0,255,0)if b['class_id']==0 else(255,200,0)
            for poly in b['polygons']:cv2.polylines(image,[np.rint(np.array(poly).reshape(-1,2)).astype(np.int32)],True,color,2)
            x,y,w,h=map(round,b['bbox']);cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
    else:
        candidates=sorted([p for p in predictions if p['score']>=.25],key=lambda p:-p['score'])
        for p in candidates[:20]:
            x,y,w,h=map(round,p['bbox']);color=(0,180,255);cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
            cv2.putText(image,('L'if p['class_id']==0 else'A')+f"{p['score']:.2f}",(x,max(20,y)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
        title+=f' {len(candidates)} boxes (top20 shown)'
    cv2.rectangle(image,(0,0),(640,35),(0,0,0),-1);cv2.putText(image,title,(6,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return image
def main():
    if OUT.exists():raise ValueError('Preserve tinyfit experiment; use another version')
    OUT.mkdir(parents=True);cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    manifest=json.loads(DATA.read_text())
    if manifest['classes']!=shared.CLASSES or manifest['private_videos_used']:raise ValueError('Dataset contract changed')
    train=[r for r in manifest['records']if r['split']=='train'];order=lambda r:hashlib.sha256(('sdnet-tiny-learning-v1:'+r['source_sha256']).encode()).hexdigest()
    unseated=sorted([r for r in train if r['boxes']and {b['class_id']for b in r['boxes']}=={0}],key=order)[:2]
    absence=sorted([r for r in train if r['boxes']and {b['class_id']for b in r['boxes']}=={1}],key=order)[:2]
    fixed=sorted([r for r in train if not r['boxes']],key=order)[:4];selected=unseated+absence+fixed
    if len(selected)!=8 or len({r['source_sha256']for r in selected})!=8:raise ValueError('Need8exactuniqueTRAINphotos')
    initial=OUT/'initial-state.pt';shutil.copyfile(BASE,initial)
    if sha(initial)!=sha(BASE):raise ValueError('Pinnedinitialstate differs')
    protocol={'scope':'Eight TRAIN photos repeated64updates; learning/localization sanity only, not heldoutaccuracy','script_sha256':sha(__file__),'shared_helper_sha256':sha(shared.__file__),'background_loss_sha256':sha(ROOT/'scripts/construction_ssd_background_loss.py'),
        'data_manifest_sha256':sha(DATA),'initial_state_source':BASE.relative_to(ROOT).as_posix(),'pinned_initial_state_sha256':sha(initial),'initialization':'Existing SDNET200-update SSDLite320v1 allweights; no freshCOCOhead or networkdownload',
        'selection':'2pureclass0+2pureclass1+4emptyFixed correctedTRAIN records bySHA256(sdnet-tiny-learning-v1:source_sha256), frozen beforeinference','selected_records':selected,
        'steps':64,'batch_size':8,'sampling':'All8eachupdate, seeded shuffledorder','seed':SEED,'threads':1,'inference_image_size':320,'head_lr':.001,'late_features_lr':.0001,'weight_decay':.0005,
        'features':'Only backbonefeatures.1 andextra trainable; allheadstrainable; backboneBNrunningstatsfrozen','loss':'Backgroundawaremin3hardneg/image; normalizer>=batchsize',
        'evaluation':'Same8TRAINphotographs before/after at fixedconfidence .25/.5; correctclassboxIoU>=.5; finalfixed64updatecheckpoint','predicted_masks':False,'ground_truth_masks':'Real publisherpolygons shown, never represented as modelmasks',
        'CAL_pixels_used':False,'development_pixels_used':False,'owner_pixels_used':False,'model_installed':False,'promoted':False}
    write(OUT/'protocol-before-forward.json',protocol)
    for path in [Path(__file__),Path(shared.__file__),ROOT/'scripts/construction_ssd_background_loss.py']:shutil.copyfile(path,OUT/(path.stem+'-source-at-run.py'))
    for row in selected:
        if row['split']!='train' or sha(ROOT/row['image'])!=row['source_sha256']:raise ValueError('Selected TRAIN pixels changed')
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=3)
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=True)
    for name,p in model.backbone.named_parameters():p.requires_grad=name.startswith('features.1.')or name.startswith('extra.')
    install(model)
    write(OUT/'trainable-parameters.json',{'trainable':[n for n,p in model.named_parameters()if p.requires_grad],'frozen':[n for n,p in model.named_parameters()if not p.requires_grad],'actual_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()})
    before,_=shared.ssd_predictions(model,selected)
    cache=[(shared.image_tensor(r),shared.target(r),r['id'])for r in selected]
    optimizer=torch.optim.AdamW([{'params':model.head.parameters(),'lr':.001},{'params':[p for p in model.backbone.parameters()if p.requires_grad],'lr':.0001}],weight_decay=.0005)
    model.train();model.backbone.eval();rng=random.Random(SEED);history=[];schedule=[];started=time.perf_counter()
    for step in range(64):
        batch=rng.sample(cache,len(cache));schedule.append([r[2]for r in batch]);parts=model([r[0]for r in batch],[r[1]for r in batch]);loss=sum(parts.values())
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        optimizer.zero_grad();loss.backward();grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10))
        if not np.isfinite(grad_norm):raise ValueError('Nonfinite gradient')
        optimizer.step();history.append({'step':step+1,**{k:float(v.detach())for k,v in parts.items()},'total':float(loss.detach()),'unclipped_gradient_norm':grad_norm,'actual_threads':torch.get_num_threads(),**model.background_loss_diagnostic})
        if (step+1)%8==0:write(OUT/'training-history.json',{'history':history,'batch_ids':schedule});print('TINY_SDNET_STEP',step+1,'LOSS',float(loss.detach()),'SECONDS',time.perf_counter()-started,flush=True)
    elapsed=time.perf_counter()-started;model.eval();torch.save(model.state_dict(),OUT/'final-state.pt');after,_=shared.ssd_predictions(model,selected)
    results={'protocol_sha256':sha(OUT/'protocol-before-forward.json'),'initial_state_sha256':sha(initial),'final_state_sha256':sha(OUT/'final-state.pt'),'steps':64,'train_seconds':elapsed,'loss_first8_mean':float(np.mean([r['total']for r in history[:8]])),'loss_last8_mean':float(np.mean([r['total']for r in history[-8:]])),
        'fixed_points':[{'threshold':t,'before':shared.metrics(selected,before,t),'after':shared.metrics(selected,after,t)}for t in [.25,.5]],'validation_status':'TRAIN_learning_sanity_only','all8_source_ids':[r['source_ids']for r in selected],
        'mask_score':None,'mask_reason':'SSDLite predictsboxes, not masks; GTpolygons displayed only','no_CAL_development_owner_pixels':True,'not_installed_or_promoted':True}
    write(OUT/'predictions.json',{'selected_records':selected,'before':before,'after':after});write(OUT/'results.json',results)
    cards=[]
    for row,b,a in zip(selected,before,after):
        short='/'.join(row['source_ids']);panel=np.hstack([render(row,None,'GT '+short),render(row,b,'BEFORE >=.25'),render(row,a,'AFTER64 >=.25')])
        dest=OUT/'overlays'/(row['id']+'.jpg');dest.parent.mkdir(exist_ok=True);cv2.imwrite(str(dest),panel);cards.append(cv2.resize(panel,(960,320)))
    cv2.imwrite(str(OUT/'train-before-after-contact-sheet.jpg'),np.vstack(cards));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
