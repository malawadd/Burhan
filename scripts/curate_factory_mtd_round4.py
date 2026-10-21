"""Additive MTD TRAIN label convention; preserve original PNGs and failed initial QA."""
import argparse, collections, hashlib, json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/factory/round4/magnetic-tile-author'
INITIAL=BASE/'train-asset-manifest.json'
PROTOCOL=BASE/'label-curation-protocol-v1-before-conversion.json'
OUTPUT=BASE/'curated-TRAIN-manifest-v1.json'
QUARANTINE={
 'MT_Uneven/Imgs/exp3_num_45042.jpg',
 'MT_Uneven/Imgs/exp4_num_124690.jpg',
 'MT_Uneven/Imgs/exp4_num_352527.jpg',
 'MT_Uneven/Imgs/exp4_num_45057.jpg'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
 with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')

def value():
 return {'name':'MTD_round4_additive_TRAIN_label_curation_v1',
  'initial_QA_path':INITIAL.relative_to(ROOT).as_posix(),'initial_QA_sha256':sha(INITIAL),
  'acquisition_manifest_sha256':sha(BASE/'original-acquisition-manifest.json'),
  'original_protocol_sha256':sha(BASE/'protocol-before-original-pixels.json'),
  'source_script_sha256':sha(__file__),
  'decision':'Root approved data-only TRAIN nonzero occupancy; no learned model or evaluation outcome involved',
  'label_convention':'Native uint8 single-channel PNG >0 is foreground support, 0 background. This is OUR deterministic TRAIN conversion, not a claimed author binary threshold. Gray boundary pixels are included; original files unchanged.',
  'reason':'Original native PNGs contain 0/255 interiors plus gray boundary values. Initial binary-only QA rejected all388 nonempty masks; preserve that first result.',
  'fixed_quarantined_positive_category_empty_masks':sorted(QUARANTINE),
  'quarantine_reason':'Publisher defect-category image with completely empty paired native mask; never use as a certified normal',
  'expected_counts':{'original_pairs':1344,'accepted_TRAIN':1340,'accepted_positive':388,'accepted_normal':952,'quarantined':4},
  'assignment':'All accepted assets TRAIN-only; no internal MTD validation or accuracy benchmark',
  'rights':'Author welcomes research with citation; no formal commercial license established',
  'physical_independence':'Unknown; multiple illumination views of same tile exist; filenames are not verified physical IDs',
  'NN_executed':False,'SDD2_reserved_or_TEST_opened':False}

def curate():
 protocol=read(PROTOCOL)
 if protocol!=value():raise ValueError('Protected curation protocol/source identities changed')
 initial=read(INITIAL);accepted=[];excluded=[];hist=np.zeros(256,np.int64)
 for old in initial['rows']:
  row=dict(old);image=ROOT/row['image'];mask=ROOT/row['mask']
  if sha(image)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('Original asset changed')
  a=cv2.imread(str(mask),cv2.IMREAD_UNCHANGED)
  if a is None or a.ndim!=2 or a.dtype!=np.uint8 or list(a.shape)!=row['native_mask_shape']:raise ValueError('Unexpected native mask schema')
  codes,counts=np.unique(a,return_counts=True)
  if codes.tolist()!=row['mask_codes'] or int(np.count_nonzero(a))!=row['foreground_pixels']:raise ValueError('Initial QA/native support mismatch')
  hist+=np.bincount(a.ravel(),minlength=256)
  row.update(initial_QA_issues=row.pop('QA_issues'),mask_encoding='native_uint8_nonzero_support_TRAIN_v1',
    mask_gray_boundary_pixels=int(np.count_nonzero((a>0)&(a<255))),mask_white_interior_pixels=int(np.count_nonzero(a==255)),
    binary_conversion='mask>0; original raw grayscale PNG unchanged',native_mask_code_histogram={str(int(k)):int(n) for k,n in zip(codes,counts)},
    training_asset_valid=True,QA_issues=[])
  empty=not a.any()
  if row['source_image'] in QUARANTINE:
   if not empty or row['category']=='MT_Free':raise ValueError('Fixed quarantine semantics changed')
   row.update(training_asset_valid=False,split='quarantine',QA_issues=['positive_category_empty_mask_do_not_train_as_normal'])
   excluded.append(row)
  else:
   if empty!=(row['category']=='MT_Free'):raise ValueError('Unreviewed label/category contradiction')
   if row['split']!='train':raise ValueError('All accepted source assignment TRAIN-only')
   accepted.append(row)
 counts={'original_pairs':len(initial['rows']),'accepted_TRAIN':len(accepted),'accepted_positive':sum(r['foreground_pixels']>0 for r in accepted),
         'accepted_normal':sum(r['foreground_pixels']==0 for r in accepted),'quarantined':len(excluded)}
 if counts!=protocol['expected_counts'] or {r['source_image'] for r in excluded}!=QUARANTINE:raise ValueError('Fixed acceptance counts differ')
 result={'curation_protocol_sha256':sha(PROTOCOL),'initial_QA_sha256':sha(INITIAL),'original_acquisition_sha256':sha(BASE/'original-acquisition-manifest.json'),
  'all_images_TRAIN_only':True,'counts':counts,'accepted_categories':dict(collections.Counter(r['category'] for r in accepted)),
  'mask_convention':protocol['label_convention'],'all_original_masks_preserved':True,'rows':accepted,'quarantined_rows':excluded,
  'all_native_mask_code_histogram':{str(i):int(v) for i,v in enumerate(hist) if v},
  'fixed_visual_QA':initial['fixed_visual_QA'],'physical_group_independence':protocol['physical_independence'],
  'rights_scope':protocol['rights'],'NN_executed':False,'MTD_accuracy_claim':False}
 write(OUTPUT,result)
 print(json.dumps({'path':OUTPUT.relative_to(ROOT).as_posix(),'sha256':sha(OUTPUT),'counts':counts},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['record-protocol','curate'],required=True);args=parser.parse_args();cv2.setNumThreads(1)
 if args.stage=='record-protocol':write(PROTOCOL,value());print(sha(PROTOCOL))
 else:curate()
