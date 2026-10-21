"""Shared verified native-input preparation; never infer or change source pixels."""
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')


class NativeDataset:
    def __init__(self,base,target,expected_data_sha,preparation_plan=None):
        self.base=base.resolve();self.target=target.resolve()
        if not self.target.is_relative_to((ROOT/'datasets/construction').resolve()) or self.target==self.base or self.target.exists():
            raise RuntimeError('Need a distinct new dataset target inside construction workspace')
        if sha(self.base/'data-protocol.json')!=expected_data_sha:raise RuntimeError('Base prepared-data identity changed')
        self.parent=json.loads((self.base/'data-protocol.json').read_text(encoding='utf8'))
        if sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py')!=self.parent['preprocessing_source_sha256']:
            raise RuntimeError('Shared camera policy changed')
        native=self.base/'native-input-manifest.json'
        if sha(native)!=self.parent['native_input_manifest_sha256']:raise RuntimeError('Base native-input manifest changed')
        self.original=json.loads(native.read_text(encoding='utf8'))
        for row in self.original:
            for key in ['image','label']:
                path=(ROOT/row[key]).resolve()
                if not path.is_relative_to(self.base) or sha(path)!=row[key+'_sha256']:
                    raise RuntimeError('Original prepared input/label identity changed')
        self.target.mkdir()
        if preparation_plan is not None:write_json(self.target/'preparation-plan.json',preparation_plan)
        for row in self.original:
            for key in ['image','label']:
                destination=ROOT/self.remap(row[key]);destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/row[key],destination)
                if sha(destination)!=row[key+'_sha256']:raise RuntimeError('Byte-identical base copy failed')
        self.records=self.remap(self.original)
        self.exposures=self.remap(json.loads((self.base/'exposures.json').read_text(encoding='utf8')))
        self.cal_raw_bgr={r['raw_native_bgr_sha256'] for r in self.original if r['split']=='calibration'}
        self.held_gray={r['sha256_pixels'] for r in self.exposures if r['split']=='heldout' and 'sha256_pixels' in r}

    def remap(self,value):
        if isinstance(value,dict):return {key:self.remap(item) for key,item in value.items()}
        if isinstance(value,list):return [self.remap(item) for item in value]
        if isinstance(value,str):
            for prefix in [str(self.base.relative_to(ROOT)),self.base.relative_to(ROOT).as_posix()]:
                if value.startswith(prefix+'\\') or value.startswith(prefix+'/'):
                    return str(self.target.relative_to(ROOT)/Path(value).relative_to(Path(prefix)))
        return value

    def add(self,source,source_sha,boxes,name,scope,source_metadata,expected_gain=None,expected_gray_sha=None):
        source=source.resolve()
        if not source.is_relative_to(ROOT) or sha(source)!=source_sha:raise RuntimeError('Accepted raw source identity changed')
        raw=cv2.imread(str(source))
        if raw is None:raise RuntimeError('Undecodable accepted source')
        height,width=raw.shape[:2];gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
        raw_sha=hashlib.sha256(raw.tobytes()).hexdigest();gray_sha=hashlib.sha256(gray.tobytes()).hexdigest()
        if raw_sha in self.cal_raw_bgr or gray_sha in self.held_gray:raise RuntimeError('Exact added/evaluation pixel overlap')
        if expected_gray_sha is not None and gray_sha!=expected_gray_sha:raise RuntimeError('Accepted camera gray identity changed')
        corrected,gain=camera_exposure(raw,'low_light_gain')
        if expected_gain is not None and gain!=expected_gain:raise RuntimeError('Accepted image gain changed')
        lines=[]
        for item in boxes:
            if item['label'] not in {'spalling','crack'}:raise RuntimeError('Unsupported native annotation class')
            x,y,w,h=map(float,item['bbox']);right=min(width,x+w);bottom=min(height,y+h);x=max(0,x);y=max(0,y)
            if right<=x or bottom<=y:raise RuntimeError('No positive native source box')
            values=[(x+right)/2/width,(y+bottom)/2/height,(right-x)/width,(bottom-y)/height]
            lines.append(str({'spalling':0,'crack':1}[item['label']])+' '+' '.join(f'{v:.16g}' for v in values))
        if not lines:raise RuntimeError('Positive-only new sources cannot become invented negative images')
        stem=name+'-'+source_sha[:10];image=self.target/'images/train'/f'{stem}.png';label=self.target/'labels/train'/f'{stem}.txt'
        if image.exists() or label.exists():raise RuntimeError('Added source identity collision')
        cv2.imwrite(str(image),corrected);label.write_text('\n'.join(lines)+'\n',encoding='utf8')
        if not np.array_equal(cv2.imread(str(image)),corrected):raise RuntimeError('Native prepared PNG differs from actual model input')
        record={'split':'train','raw_source_image':str(source.relative_to(ROOT)),'raw_source_file_sha256':source_sha,
            'raw_native_bgr_sha256':raw_sha,'image':str(image.relative_to(ROOT)),'image_sha256':sha(image),
            'label':str(label.relative_to(ROOT)),'label_sha256':sha(label),'gain':gain,
            'native_model_input_bgr_sha256':hashlib.sha256(corrected.tobytes()).hexdigest(),'shape_wh':[width,height],
            'annotation_scope':scope,**source_metadata}
        self.records.append(record)
        self.exposures.append({'split':'train','image':record['image'],'sha256_pixels':gray_sha,'boxes':boxes,
            'annotation_scope':scope,'input_exposure_gain':gain,**source_metadata})

    def finish(self,protocol,training_count):
        counts={split:sum(r['split']==split for r in self.records) for split in ['train','calibration']}
        if counts!={'train':training_count,'calibration':122}:raise RuntimeError('Exact combined split count differs')
        for split,count in counts.items():
            if len(list((self.target/f'images/{split}').glob('*.png')))!=count or len(list((self.target/f'labels/{split}').glob('*.txt')))!=count:
                raise RuntimeError('Unexpected missing/stale native output files')
        write_json(self.target/'native-input-manifest.json',self.records);write_json(self.target/'exposures.json',self.exposures)
        write_json(self.target/'background-import.json',self.remap(json.loads((self.base/'background-import.json').read_text(encoding='utf8'))))
        shutil.copy2(self.base/'night-holdout-protocol.json',self.target/'night-holdout-protocol.json')
        dataset=yaml.safe_load((self.base/'dataset.yaml').read_text(encoding='utf8'))
        dataset.update(path=self.target.as_posix(),train=(self.target/'images/train').as_posix(),val=(self.target/'images/calibration').as_posix())
        (self.target/'dataset.yaml').write_text(yaml.safe_dump(dataset),encoding='utf8')
        protocol.update(policy='low_light_gain',preprocessing_source_sha256=sha(ROOT/'edge_runtime/profiles/construction_preprocessing.py'),
            native_input_manifest_sha256=sha(self.target/'native-input-manifest.json'),exposures_sha256=sha(self.target/'exposures.json'),
            counts={split:{'images':counts[split],'gain_applied':sum(r['split']==split and r['gain']>1 for r in self.records)} for split in counts})
        write_json(self.target/'data-protocol.json',protocol)
        return {'data_root':str(self.target.relative_to(ROOT)),'data_protocol_sha256':sha(self.target/'data-protocol.json'),
            'native_input_manifest_sha256':protocol['native_input_manifest_sha256'],'exposures_sha256':protocol['exposures_sha256'],'counts':protocol['counts'],'fit_started':False}
