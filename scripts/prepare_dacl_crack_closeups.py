"""Separate TRAIN-only native closeups of the same40 acquired real crack photos."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import numpy as np
from PIL import Image, ImageDraw
import prepare_dacl_positive_contingency as preparation

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'datasets/construction/dacl10k/round3-positive-contingency'
OUT=ROOT/'datasets/construction/dacl10k/round3-crack-closeup-contingency'
EXPECTED='2dc962b359bba28ba8eb1871184c2c2c5d8aa841cf4540d393fccd150276c360'
GEOMETRY='TRAIN-only true-target-pixel-centered exact4:3 crop:1.5x anchor bbox context, native320x240 minimum/960x720 maximum; center nearest rasterized target pixel to source bbox center, row-major tie break, image-clamped; gray/bilinear640x480; source-clipped boxes projected continuously and semantic masks nearest. Same40 source photos, no new independent samples.'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_path=ORIGINAL/'asset-manifest.json'
    if sha(original_path)!=EXPECTED:raise RuntimeError('Original80 frozen manifest changed')
    original=json.loads(original_path.read_text(encoding='utf-8'))
    selected=[r for r in original['records'] if r['cohort']=='crack_family']
    if len(selected)!=40:raise RuntimeError('Expected same40 crack-family source photos')
    protocol={'original80_manifest_sha256':EXPECTED,'source_names':[r['source_name'] for r in selected],
              'geometry':GEOMETRY,'source_selection':'All40 crack-family photos from original80, same fixed order; no model output or new photo selection.',
              'scope':'Unused separately versioned training-only closeups, not calibration/heldout, new independent photos, synthetic damage or video evidence.',
              'license':'DACL10k original author CC BY-NC4.0 data terms'}
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=OUT/'protocol-before-derivation.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Closeup protocol changed')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    preparation.OUT=OUT
    records=[];preview=None
    for source in selected:
        name=source['source_name']
        source_image=ORIGINAL/source['original_image'];source_annotation=ORIGINAL/source['original_annotation']
        if sha(source_image)!=source['original_image_sha256'] or sha(source_annotation)!=source['original_annotation_sha256']:
            raise RuntimeError('Original acquired source changed')
        annotation=json.loads(source_annotation.read_text(encoding='utf-8'))
        polygons=annotation['polygons'];width,height=annotation['image_width'],annotation['image_height']
        targets=[(preparation.polygon_id(poly,index),index,poly) for index,poly in enumerate(polygons) if poly['label'] in {'Crack','ACrack'}]
        _,anchor_index,anchor=min(targets)
        mask=preparation.clipped_mask(anchor,width,height,(0,0,width,height))
        bbox=mask.getbbox()
        if bbox is None:raise RuntimeError('Source target has no rasterized pixels')
        ys,xs=np.nonzero(np.asarray(mask))
        cx,cy=(bbox[0]+bbox[2]-1)/2,(bbox[1]+bbox[3]-1)/2
        nearest=int(np.argmin((xs-cx)**2+(ys-cy)**2))
        cx,cy=float(xs[nearest])+.5,float(ys[nearest])+.5
        upper=min(240,width//4,height//3)
        if upper<80:raise RuntimeError('Source too small for promised native minimum')
        units=min(upper,max(80,math.ceil(max((bbox[2]-bbox[0])*1.5/4,(bbox[3]-bbox[1])*1.5/3))))
        cw,ch=4*units,3*units
        crop=(min(width-cw,max(0,math.floor(cx-cw/2))),min(height-ch,max(0,math.floor(cy-ch/2))),cw,ch)
        image=OUT/'original/images'/source_image.name;ann=OUT/'original/annotations'/source_annotation.name
        image.parent.mkdir(parents=True,exist_ok=True);ann.parent.mkdir(parents=True,exist_ok=True)
        if not image.exists():shutil.copyfile(source_image,image)
        if not ann.exists():shutil.copyfile(source_annotation,ann)
        if sha(image)!=source['original_image_sha256'] or sha(ann)!=source['original_annotation_sha256']:raise RuntimeError('Copied source identity mismatch')
        row={'name':name,'metadata':{'width':width,'height':height},'semseg_ground_truth_filled':{'polylines':polygons}}
        derived=preparation.derive(row,'crack_family',image,crop_override=crop,output_stem=name+'-closeup',geometry_description=GEOMETRY)
        records.append({'source_name':name,'source_view_id':name+'-closeup','cohort':'crack_family_closeup','source_split':'train',
                        'original_image':image.relative_to(OUT).as_posix(),'original_image_sha256':sha(image),
                        'original_annotation':ann.relative_to(OUT).as_posix(),'original_annotation_sha256':sha(ann),
                        'source_url':source['source_url'],'source_labels':source['source_labels'],'derived':derived})
        if preview is None:preview=records[-1]
        print(name,crop,len(derived['instances']),'retained boxes',flush=True)
    manifest={'protocol_sha256':sha(protocol_path),'original80_manifest_sha256':EXPECTED,'class_names':['spalling','crack'],
              'class_map':preparation.CLASS_MAP,'license':protocol['license'],'usage':protocol['scope'],
              'unique_original_photos':40,'additional_independent_photos':0,'records':records}
    path=OUT/'asset-manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    image=Image.open(OUT/preview['derived']['image']).convert('RGB');canvas=Image.new('RGB',(1280,520),(20,20,20))
    canvas.paste(image,(0,40));canvas.paste(image,(640,40));draw=ImageDraw.Draw(canvas)
    draw.text((8,10),'REAL TRAIN CLOSEUP / SAME SOURCE PHOTO',fill='white');draw.text((648,10),'PUBLISHER TARGET BOXES / NOT MODEL OUTPUT',fill='white')
    for item in preview['derived']['instances']:
        x1,y1,x2,y2=item['output_projected_source_bbox_xyxy'];color='yellow' if item['class_id']==1 else 'cyan'
        draw.rectangle((640+x1,40+y1,640+x2,40+y2),outline=color,width=2)
    (OUT/'qa').mkdir(exist_ok=True);canvas.save(OUT/'qa/first-closeup-gt.jpg',quality=90)
    print(json.dumps({'manifest':str(path),'sha256':sha(path),'views':40,'original80_unchanged':sha(original_path)==EXPECTED},indent=2))


if __name__=='__main__':main()
