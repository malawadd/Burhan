"""Read only R4 TRAIN label text and metadata; no image/NN/CAL-label decode."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
import argparse

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/construction/evcivil-round4-gain'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--split',choices=['train','calibration'],default='train');args=parser.parse_args()
    manifest=json.loads((BASE/'native-input-manifest.json').read_text(encoding='utf-8'))
    exposures=json.loads((BASE/'exposures.json').read_text(encoding='utf-8'))
    boxes={str((ROOT/r['image']).resolve()):r['boxes'] for r in exposures if r['split']==args.split}
    errors=[];counts=Counter();dimensions=Counter();minimum=[float('inf'),float('inf')];maximum_error=0.;verified=0
    for row in manifest:
        if row['split']!=args.split:continue
        path=ROOT/row['label'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['label_sha256']:errors.append('Label hash changed: '+str(path))
        width,height=row['shape_wh'];actual=[]
        for line in raw.decode().splitlines():
            if not line.strip():continue
            values=list(map(float,line.split()))
            if len(values)!=5 or not all(math.isfinite(v) for v in values):errors.append('Invalid normalized label: '+str(path));continue
            cls,x,y,w,h=values
            if cls not in [0,1] or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:errors.append('Invalid native normalized geometry: '+str(path))
            actual.append((int(cls),(x-w/2)*width,(y-h/2)*height,w*width,h*height))
            counts['boxes_'+str(int(cls))]+=1;minimum=[min(minimum[0],w*width),min(minimum[1],h*height)]
        expected=boxes.get(str((ROOT/row['image']).resolve()),[])
        if expected:
            wanted=[(b['class_id'],*b['bbox']) for b in expected];remaining=set(range(len(wanted)))
            if len(wanted)!=len(actual):errors.append('Publisher box/label count mismatch: '+str(path))
            # Match near-identical geometry, not lexicographic float ordering:
            # two source regions can share exact x but rounded labels differ
            # at1e-8, which must not swap their y/height association.
            for item in actual:
                choices=[(max(abs(a-b) for a,b in zip(item,wanted[i])),i) for i in remaining if item[0]==wanted[i][0]]
                delta,index=min(choices,default=(float('inf'),None))
                maximum_error=max(maximum_error,delta)
                if delta>1e-3:errors.append('Class/geometry mismatch: '+str(path))
                if index is not None:remaining.remove(index)
            verified+=1
        elif actual:errors.append('Unregistered positive labels: '+str(path))
        counts['images']+=1;counts['positive_images' if actual else 'negative_images']+=1
        counts['gained_images']+=row['gain']!=1;dimensions[str(row['shape_wh'])]+=1
    fit={r['raw_source_file_sha256'] for r in manifest if r['split']=='train'}
    cal={r['raw_source_file_sha256'] for r in manifest if r['split']=='calibration'}
    if fit&cal:errors.append('Raw source file identity overlap fitting/calibration')
    report={'scope':args.split.upper()+' label text and metadata only; no image decoding, heldout or NN',
            'passed':not errors,'errors':errors,'counts':dict(counts),'publisher_box_records_verified':verified,
            'maximum_native_coordinate_error_pixels':maximum_error,'minimum_training_label_wh_pixels':minimum,
            'native_dimensions':dict(dimensions),'raw_source_file_hash_overlap_fit_cal':len(fit&cal),
            'protocol_sha256':hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),
            'native_manifest_sha256':hashlib.sha256((BASE/'native-input-manifest.json').read_bytes()).hexdigest()}
    path=ROOT/('output/evaluation/construction-round4-independent-'+('calibration-' if args.split=='calibration' else '')+'label-contract.json')
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
