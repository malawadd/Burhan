"""Import real fully annotated DACL crack/spall-negative images in matching grayscale."""
import hashlib
import json
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'datasets/construction/dacl10k'
BASE=ROOT/'datasets/construction/evcivil-round3'
EXPECTED='0413a8e3b596af25b72469a93e914956824d0be39bd4da5b78c7ca2d64cf402d'


if __name__=='__main__':
    cv2.setNumThreads(1);manifest=SOURCE/'round3-negative-manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Frozen background manifest hash changed')
    data=json.loads(manifest.read_text(encoding='utf8'));records=data['records']
    mapping={'negative_train':'train','negative_calibration':'calibration','negative_heldout':'heldout'};results=[];hashes=set()
    for row in records:
        split=mapping[row['partition']];image=SOURCE/row['image'];raw=image.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=row['image_sha256'] or digest in hashes:raise RuntimeError('Background hash mismatch/leak')
        hashes.add(digest)
        if set(row['labels']) & {'Crack','ACrack','Spall','Spalling'}:raise RuntimeError('Crack/spall-negative label contract violated')
        gray=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise RuntimeError('Could not decode genuine background image')
        height,width=gray.shape;ratio=346/260
        cw=min(width,round(height*ratio));ch=min(height,round(width/ratio));cx=(width-cw)//2;cy=(height-ch)//2
        gray=cv2.resize(gray[cy:cy+ch,cx:cx+cw],(346,260),interpolation=cv2.INTER_AREA)
        stem='dacl-negative-'+image.stem;target=BASE/'images'/split/(stem+'.png');label=BASE/'labels'/split/(stem+'.txt')
        target.parent.mkdir(exist_ok=True,parents=True);label.parent.mkdir(exist_ok=True,parents=True)
        cv2.imwrite(str(target),gray);label.write_text('',encoding='utf8')
        results.append(row|{'split':split,'derived_image':str(target.relative_to(ROOT)),'derived_gray_pixel_sha256':hashlib.sha256(gray.tobytes()).hexdigest(),
                           'source_crop_xywh':[cx,cy,cw,ch],'derived_shape_hw':[260,346],
                           'derivation':'Grayscale + fixed center crop to camera aspect and resize346x260, matching positive input geometry; original complete image is class-negative, no invented content',
                           'ground_truth_scope':'Complete publisher DACL polygons contain no crack/ACrack/spall; other damage may exist, so this is a two-class negative, not globally healthy concrete.'})
    report={'source_manifest':str(manifest.relative_to(ROOT)),'source_manifest_sha256':EXPECTED,'license':'DACL10k CC BY-NC 4.0; non-commercial restriction carries into trained artifact',
            'identity_scope':'Unique source-image hashes, official train/cal subset split and official validation heldout; physical bridge identities not documented.',
            'records':results}
    (BASE/'background-import.json').write_text(json.dumps(report,indent=2),encoding='utf8');print({s:sum(x['split']==s for x in results) for s in mapping.values()})
