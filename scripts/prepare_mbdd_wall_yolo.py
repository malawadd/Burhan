"""Prepare separate hashed YOLO inputs from independently reviewed wall data.

No neural operation. This does not change original source photos/annotations,
old18 diagnostics, existing construction split or model registry.
"""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction/finishes-expansion/mbdd-wall-five-class-v1'
OUT=DATA/'prepared-yolo'
CLASSES=['crack','leakage','abscission','corrosion','bulge']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path=DATA/'asset-manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    review_path=DATA/'independent-data-review.json'
    if not review_path.exists():raise RuntimeError('Independent source/label/visual data review is required before copying fit inputs')
    review=json.loads(review_path.read_text(encoding='utf-8'))
    if review.get('source_manifest_sha256')!=sha(manifest_path) or not review.get('approved_for_train_probe',False):
        raise RuntimeError('Independent data review does not approve this exact manifest for TRAIN probe')
    records=[]
    for row in manifest['records']:
        if row['status']!='eligible_pending_independent_review':continue
        source=DATA/row['image'];annotation=DATA/row['annotation']
        if sha(source)!=row['image_sha256'] or sha(annotation)!=row['annotation_sha256']:raise RuntimeError('Original wall source changed')
        split=row['split'];name=Path(row['source_name']).stem;w,h=row['frame_size_wh']
        image=OUT/'images'/split/(name+'.jpg');label=OUT/'labels'/split/(name+'.txt')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        if image.exists() and sha(image)!=row['image_sha256']:raise RuntimeError('Prepared wall source copy differs')
        if not image.exists():shutil.copyfile(source,image)
        lines=[]
        for box in row['boxes']:
            x1,y1,x2,y2=box['source_xyxy'];index=CLASSES.index(box['label'])
            if index!=box['class_id']:raise RuntimeError('Publisher class mapping changed')
            lines.append(f'{index} {(x1+x2)/2/w:.12f} {(y1+y2)/2/h:.12f} {(x2-x1)/w:.12f} {(y2-y1)/h:.12f}')
        encoded=('\n'.join(lines)+'\n').encode()
        if label.exists() and label.read_bytes()!=encoded:raise RuntimeError('Prepared wall label differs')
        label.write_bytes(encoded)
        records.append({**row,'prepared_image':image.relative_to(ROOT).as_posix(),'prepared_image_sha256':sha(image),
                        'prepared_label':label.relative_to(ROOT).as_posix(),'prepared_label_sha256':sha(label)})
    counts={split:{name:sum(any(b['label']==name for b in r['boxes']) for r in records if r['split']==split) for name in CLASSES}
            for split in ['train','calibration']}
    if any(counts['train'][name]<20 or counts['calibration'][name]<5 for name in CLASSES):
        raise RuntimeError('Post-quarantine source subset too small; do not silently backfill')
    OUT.mkdir(parents=True,exist_ok=True)
    yaml='path: '+OUT.as_posix()+'\ntrain: images/train\nval: images/calibration\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES))
    target=OUT/'dataset.yaml'
    if target.exists() and target.read_text(encoding='utf-8')!=yaml:raise RuntimeError('Prepared dataset specification changed')
    target.write_text(yaml,encoding='utf-8')
    report={'source_manifest_sha256':sha(manifest_path),'review_sha256':sha(review_path),'preparer_sha256':sha(Path(__file__)),
            'classes':CLASSES,'counts_positive_photos':counts,'records':records,'dataset_yaml_sha256':sha(target),
            'exposure_policy':'off; original source RGB photos copied byte-exactly; no gain/darkening or annotation ROI',
            'annotation_geometry':'Original publisher continuous XYXY mapped to normalized YOLO XYWH at original native dimensions; no 1-based shift',
            'negative_scope':'Positive wall imagery with other-class absences. No certified healthy-wall FPR dataset.',
            'status':'Prepared source inputs for TRAIN-only probe; no model fit authorized by this file'}
    result=OUT/'input-manifest.json';payload=(json.dumps(report,indent=2)+'\n').encode()
    if result.exists() and result.read_bytes()!=payload:raise RuntimeError('Prepared wall input manifest changed')
    result.write_bytes(payload)
    print(json.dumps({'manifest':str(result),'sha256':sha(result),'counts':counts,'photos':len(records)},indent=2))


if __name__=='__main__':main()
