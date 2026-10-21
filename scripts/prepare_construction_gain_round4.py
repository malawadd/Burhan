"""Prepare one shared camera-gain policy for real fit and same-source calibration."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from edge_runtime.profiles.construction_preprocessing import camera_exposure
OLD=ROOT/'datasets/construction/evcivil-round4'
NEW=ROOT/'datasets/construction/evcivil-round4-gain'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1);NEW.mkdir(parents=True,exist_ok=True)
    existing_protocol=NEW/'data-protocol.json'
    if existing_protocol.exists():
        frozen=json.loads(existing_protocol.read_text(encoding='utf8'))
        if frozen['preprocessing_source_sha256']!=sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py') or frozen['source_data_protocol_sha256']!=sha(OLD/'data-protocol.json'):
            raise RuntimeError('Refusing to mutate inputs after their source or shared policy changed')
        existing_manifest=NEW/'native-input-manifest.json'
        if sha(existing_manifest)!=frozen['native_input_manifest_sha256']:
            raise RuntimeError('Prepared native-input manifest changed')
        for row in json.loads(existing_manifest.read_text(encoding='utf8')):
            if sha(ROOT/row['raw_source_image'])!=row['raw_source_file_sha256'] or sha(ROOT/row['image'])!=row['image_sha256']:
                raise RuntimeError('Refusing to overwrite changed source or prepared pixels')
    source_yaml=yaml.safe_load((OLD/'dataset.yaml').read_text(encoding='utf8'))
    mapping={};records=[];counts={};previewed=set()
    for split in ['train','calibration']:
        directories=source_yaml['train']if split=='train'else[source_yaml['val']]
        if isinstance(directories,str):directories=[directories]
        sources=sorted({p.resolve()for folder in directories for p in Path(folder).rglob('*')if p.suffix.lower()in{'.png','.jpg','.jpeg'}})
        expected=832 if split=='train' else 122
        if len(sources)!=expected:raise RuntimeError('Unexpected source count; refusing silently different fit/calibration')
        image_dir=NEW/f'images/{split}';label_dir=NEW/f'labels/{split}'
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        counts[split]={'images':len(sources),'gain_applied':0}
        for source in sources:
            label=Path(source.as_posix().replace('/images/','/labels/')).with_suffix('.txt')
            if not source.is_relative_to(ROOT)or not label.is_file():raise RuntimeError('Source identity/labels missing')
            relative=str(source.relative_to(ROOT));suffix=hashlib.sha256(relative.encode('utf8')).hexdigest()[:10]
            target=image_dir/f'{source.stem}-{suffix}.png';label_target=label_dir/f'{source.stem}-{suffix}.txt'
            raw=cv2.imread(str(source));corrected,gain=camera_exposure(raw,'low_light_gain')
            if corrected.shape!=raw.shape:raise RuntimeError('Camera policy changed spatial geometry')
            cv2.imwrite(str(target),corrected);shutil.copyfile(label,label_target)
            decoded=cv2.imread(str(target))
            if not np.array_equal(decoded,corrected):raise RuntimeError('Preprocessing PNG is not exact nativeBGR model input')
            row={'split':split,'raw_source_image':relative,'raw_source_file_sha256':sha(source),
                 'raw_native_bgr_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                 'image':str(target.relative_to(ROOT)),'image_sha256':sha(target),
                 'label':str(label_target.relative_to(ROOT)),'label_sha256':sha(label_target),
                 'gain':gain,'native_model_input_bgr_sha256':hashlib.sha256(corrected.tobytes()).hexdigest(),
                 'shape_wh':[raw.shape[1],raw.shape[0]]}
            mapping[source]=row;records.append(row)
            counts[split]['gain_applied']+=gain!=1
            if gain!=1 and split not in previewed:
                previewed.add(split)
                panel=np.hstack([raw,corrected]);panel=cv2.copyMakeBorder(panel,55,0,0,0,cv2.BORDER_CONSTANT,value=(23,23,23))
                cv2.putText(panel,f'{split}: fixed FIRST low-light input, actualgain{gain:.2f}x',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(panel,'RAW SOURCE',(7,44),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(panel,'ACTUAL NATIVE MODEL INPUT',(raw.shape[1]+7,44),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                cv2.imwrite(str(NEW/f'{split}-first-low-light-input-preview.jpg'),panel)
    for split in ['train','calibration']:
        if len(list((NEW/f'images/{split}').glob('*.png')))!=counts[split]['images']:raise RuntimeError('Stale prepared input files')
    source_yaml['train']=(NEW/'images/train').as_posix();source_yaml['val']=(NEW/'images/calibration').as_posix()
    (NEW/'dataset.yaml').write_text(yaml.safe_dump(source_yaml,sort_keys=False),encoding='utf8')
    exposures=json.loads((OLD/'exposures.json').read_text(encoding='utf8'))
    for exposure in exposures:
        old_path=(ROOT/exposure['image']).resolve()
        if old_path in mapping:
            transformed=mapping[old_path]
            exposure.update(raw_source_image=exposure['image'],image=transformed['image'],
                            native_model_input_bgr_sha256=transformed['native_model_input_bgr_sha256'],model_input_gain=transformed['gain'])
    (NEW/'exposures.json').write_text(json.dumps(exposures,indent=2),encoding='utf8')
    background=json.loads((OLD/'background-import.json').read_text(encoding='utf8'))
    for record in background['records']:
        original=(ROOT/record['derived_image']).resolve()
        if original in mapping:
            record.update(raw_source_derived_image=record['derived_image'],derived_image=mapping[original]['image'],
                          native_model_input_bgr_sha256=mapping[original]['native_model_input_bgr_sha256'],model_input_gain=mapping[original]['gain'])
    (NEW/'background-import.json').write_text(json.dumps(background,indent=2),encoding='utf8')
    shutil.copyfile(OLD/'night-holdout-protocol.json',NEW/'night-holdout-protocol.json')
    (NEW/'native-input-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    protocol={'schema_version':1,'status':'PREPARED ONLY; no new fit started or model/training protocol yet',
              'future_training_input_size':640,'policy':'low_light_gain',
              'exact_rule':'Native grayp99<64 -> allBGR*min(8,128/max(p99,1)), clip0..255, uint8truncation; otherwise identical pixels',
              'preprocessing_source_sha256':sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py'),
              'source_data_protocol_sha256':sha(OLD/'data-protocol.json'),'counts':counts,
              'native_input_manifest_sha256':sha(NEW/'native-input-manifest.json'),'exposures_sha256':sha(NEW/'exposures.json'),
              'scope':'Same source IDs and untouched GT; raw originalcal files unchanged. DerivedgainVAL inputs allow proper checkpoint selection. Finalserving/cal applysamefunction tooriginalraw beforeletterbox.',
              'no_heldout_predictions':True,'heldout_source_path_preserved':source_yaml['test']}
    path=NEW/'data-protocol.json'
    if path.exists()and json.loads(path.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Frozen gain-data protocol changed')
    path.write_text(json.dumps(protocol,indent=2),encoding='utf8');print(json.dumps(protocol,indent=2),flush=True)


if __name__=='__main__':main()
