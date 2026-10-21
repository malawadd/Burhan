"""MTD cross-source audit: decode TRAIN only; protected CAL/comparison hashes only."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/factory/round4/magnetic-tile-author'
MTD=OUT/'curated-TRAIN-manifest-v1.json'
FIT=ROOT/'output/optimization/factory-round3/fit-admission-geometry-proposal.json'
CAL=ROOT/'output/optimization/factory-round3l9-learned-partitioned/checkpoint200-calibration.json'
OLD=ROOT/'output/optimization/factory-round3/after-localization-audit.json'
PROTOCOL=OUT/'cross-source-screen-protocol-v1-before-TRAIN-decodes.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def declaration():
 fit=read(FIT)['images'];mt=read(MTD)
 if len(fit)!=2104 or len(mt['rows'])!=1340:raise ValueError('Exact239+1865 TRAIN /1340MTD required')
 if sum(r['dataset']=='SDD1_TRAIN' for r in fit)!=239 or sum(r['dataset']=='SDD2_inner_FIT' for r in fit)!=1865:raise ValueError('Unexpected TRAIN selection')
 return {'source_pins':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),MTD,FIT,CAL,OLD]},
  'selection':fit,'method':'Exact source SHA all known partitions; native pixel SHA+63AC DCT32 pHash Hamming<=6 only on239 SDD1 TRAIN +1865 SDD2 FIT versus1340 MTD TRAIN. Flags are not verified duplicates.',
  'protected_pixel_policy':'CAL80 and exposed comparison80: existing metadata image SHA ONLY; no image or mask decode. No SDD2 reserved466 or official TEST access.',
  'threshold_hamming':6,'new_NN':False,'automatic_accept_or_reject':False}
def run():
 p=read(PROTOCOL)
 if p!=declaration():raise ValueError('Pinned screening identities changed')
 mt=read(MTD)['rows'];protected=[]
 for r in read(CAL)['calibration_source_identity']['ordered_sources']:protected.append({'partition':'SDD1_CAL','image':r['image'],'sha256':r['image_sha256']})
 for r in read(OLD)['records']:protected.append({'partition':'SDD1_exposed_comparison','image':r['path'],'sha256':r['sha256']})
 if len(protected)!=160:raise ValueError('Exact160 metadata-only protected views required')
 known=[{'partition':r['dataset'],'image':r['image'],'sha256':r['sha256']} for r in p['selection']]+protected
 exact=[{'MTD':m['source_image'],'known':r} for m in mt for r in known if m['image_sha256']==r['sha256']]
 fingerprints=[];flags=[];pixel_matches=[]
 for row in p['selection']:
  path=ROOT/row['image']
  if sha(path)!=row['sha256']:raise ValueError('Pinned TRAIN source changed')
  a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
  if a is None or a.dtype!=np.uint8:raise ValueError('TRAIN image decode schema')
  gray=a if a.ndim==2 else cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
  v=cv2.dct(cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()[1:]
  ph=sum(int(x)<<i for i,x in enumerate(v>np.median(v)))
  pixel=hashlib.sha256(a.tobytes()).hexdigest()
  fingerprints.append({'image':row['image'],'partition':row['dataset'],'sha256':row['sha256'],'native_shape':list(a.shape),'pixel_sha256':pixel,'phash63':f'{ph:016x}'})
  for m in mt:
   distance=(ph^int(m['native_gray_phash63'],16)).bit_count()
   if distance<=6:flags.append({'MTD':m['source_image'],'known_TRAIN':row['image'],'known_partition':row['dataset'],'hamming':distance,'review_status':'flag_only_unreviewed'})
   if list(a.shape)==m['native_image_shape'] and pixel==m['native_image_pixel_sha256']:pixel_matches.append({'MTD':m['source_image'],'known_TRAIN':row['image']})
 write(OUT/'cross-source-screen-v1.json',{'protocol_sha256':sha(PROTOCOL),'exact_file_matches':exact,'exact_native_pixel_matches_TRAIN_only':pixel_matches,
  'TRAIN_perceptual_flags':flags,'TRAIN_fingerprints':fingerprints,'counts':{'MTD_TRAIN':1340,'known_TRAIN':2104,'protected_metadata_only':160,'exact_matches':len(exact),'pixel_matches':len(pixel_matches),'perceptual_flags':len(flags)},
  'limitations':'No new CAL/comparison/reserved/test pixels opened. No protected perceptual-screen claim. A low hash distance flags review only; no matches does not prove physical/site independence.',
  'NN_executed':False,'datasets_modified':False})
 print(json.dumps({'counts':{'exact_matches':len(exact),'pixel_matches':len(pixel_matches),'perceptual_flags':len(flags)},'sha256':sha(OUT/'cross-source-screen-v1.json')},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['record-protocol','run'],required=True);args=a.parse_args();cv2.setNumThreads(1)
 if args.stage=='record-protocol':write(PROTOCOL,declaration());print(sha(PROTOCOL))
 else:run()
