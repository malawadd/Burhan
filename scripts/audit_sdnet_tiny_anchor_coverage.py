"""Audit SSD assignment for the already selected tiny TRAIN images; no fit."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,sys
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import box_iou
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import train_sdnet_expansion as shared
OUT=ROOT/'output/evaluation/construction-resumption-fastening-tiny-v1'
def main():
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=3).eval()
    model.load_state_dict(torch.load(OUT/'final-state.pt',map_location='cpu',weights_only=True))
    data=json.loads((OUT/'predictions.json').read_text());rows=data['selected_records'];receipts=[]
    with torch.no_grad():
        for row in rows:
            if not row['boxes']:continue
            tensor=shared.image_tensor(row);target=shared.target(row)
            images,targets=model.transform([tensor],[target]);features=model.backbone(images.tensors);anchors=model.anchor_generator(images,list(features.values()))[0]
            quality=box_iou(targets[0]['boxes'],anchors);matches=model.proposal_matcher(quality);best=quality.argmax(1)
            counts=[int((matches==i).sum())for i in range(len(row['boxes']))]
            receipts.append({'source_ids':row['source_ids'],'gt_boxes_source640':[b['bbox']for b in row['boxes']],
                'transformed_boxes':targets[0]['boxes'].tolist(),'positive_anchor_counts_per_gt':counts,'best_anchor_indices_per_gt':best.tolist(),'best_anchor_iou_per_gt':quality.max(1).values.tolist(),
                'best_anchor_collision':len(set(best.tolist()))<len(best),'unassigned_gt_indices':[i for i,n in enumerate(counts)if n==0]})
    result={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((OUT/'final-state.pt').read_bytes()).hexdigest(),
        'scope':'Selected8TRAIN only; actual SSD320 targettransform, anchor generator andproposal_matcher; no gradientupdates','rows':receipts}
    (OUT/'anchor-coverage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
