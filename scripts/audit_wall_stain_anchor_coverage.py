"""Audit original wall TRAIN targets against unchanged SSD320 anchor assignment.

Only one already selected SDNET TRAIN photo supplies feature-map geometry. Wall
TRAIN boxes are mathematical metadata; no wall CAL/development/owner pixels.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,sys
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large,_utils
from torchvision.ops import box_iou
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import train_sdnet_expansion as shared
from construction_ssd_matcher import UniqueForcedSSDMatcher
OUT=ROOT/'output/evaluation/construction-resumption-fastening-tiny-unique-v1'
DATA=ROOT/'datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json'
TINY=ROOT/'output/evaluation/construction-resumption-fastening-tiny-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cv_source=json.loads((TINY/'protocol-before-forward.json').read_text())['selected_records'][0]
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=3).eval()
    model.load_state_dict(torch.load(TINY/'initial-state.pt',weights_only=True,map_location='cpu'))
    with torch.no_grad():
        images,_=model.transform([shared.image_tensor(cv_source)],None);features=model.backbone(images.tensors)
        anchors=model.anchor_generator(images,list(features.values()))[0]
    manifest=json.loads(DATA.read_text());rows=[r for r in manifest['records']if r['split']=='train'and r['boxes']]
    unique=UniqueForcedSSDMatcher(.5);stock=_utils.SSDMatcher(.5);receipts=[]
    for row in rows:
        boxes=[]
        for b in row['boxes']:
            x,y,w,h=b['bbox'];boxes.append([x/2,y/2,(x+w)/2,(y+h)/2])
        quality=box_iou(torch.tensor(boxes,dtype=torch.float32),anchors);old=stock(quality);new=unique(quality)
        receipts.append({'id':row['id'],'source_image_sha256':row['source_sha256'],'capture_block_200':row['capture_block_200'],'source_boxes':row['boxes'],
            'original_positive_anchor_counts':[int((old==i).sum())for i in range(len(boxes))],'unique_positive_anchor_counts':[int((new==i).sum())for i in range(len(boxes))],
            'best_anchor_indices':quality.argmax(1).tolist(),'best_anchor_iou_per_gt':quality.max(1).values.tolist(),'unique_forced_anchor_ious':unique.assignment_diagnostics[-1]['forced_anchor_ious']})
    result={'script_sha256':sha(__file__),'matcher_sha256':sha(ROOT/'scripts/construction_ssd_matcher.py'),'dataset_sha256':sha(DATA),'scope':'All40originalwallTRAINpositives; annotationgeometryaudit, noadditionalwallfit or CAL/development/ownerpixels',
        'geometry_source_train_photo':cv_source['source_ids'],'geometry_source_sha256':cv_source['source_sha256'],'feature_map_shapes':{k:list(v.shape)for k,v in features.items()},'anchor_count':len(anchors),'anchor_tensor_sha256':hashlib.sha256(anchors.numpy().tobytes()).hexdigest(),
        'positive_photos':len(receipts),'target_boxes':sum(len(r['source_boxes'])for r in receipts),'original_unassigned_targets':sum(sum(n==0 for n in r['original_positive_anchor_counts'])for r in receipts),'original_affected_photos':sum(any(n==0 for n in r['original_positive_anchor_counts'])for r in receipts),'unique_unassigned_targets':sum(sum(n==0 for n in r['unique_positive_anchor_counts'])for r in receipts),
        'semantics':'Publisherleakage/stainappearance, not humiditymeasurement. Knownmixedlabels and siteoverlap limitations unchanged.','rows':receipts}
    (OUT/'wall-original-TRAIN-anchor-coverage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
