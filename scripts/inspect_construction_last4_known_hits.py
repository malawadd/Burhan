"""Metadata-only post-hoc explanation of already frozen CAL matches. No NN/images.

This never selects weights or operating points. Any later illustrative source
must be explicitly labelled post-hoc, complete-video development evidence.
"""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/evaluation/construction-round3'
NAMES={'last4':'evcivil-yolov8n-r4-last4-640-gain',
       'best_epoch1':'evcivil-yolov8n-r4-real-lowlight-640-gain',
       'prior_head':'evcivil-yolov8n-r4-head-640-gain'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs={name:BASE/stem/'calibration-runtime-records.json' for name,stem in NAMES.items()}
    records={name:json.loads(path.read_text(encoding='utf-8')) for name,path in inputs.items()}
    matches={name:{p['matched_annotation_id']:p for row in rows for p in row['predictions'] if p.get('matched_annotation_id')}
             for name,rows in records.items()}
    sources=json.loads((ROOT/'datasets/construction/evcivil-round3/exposures.json').read_text(encoding='utf-8'))
    indexed={(row['global_defect_id'],row['original_frame_index']):row for row in sources if row['split']=='calibration'}
    wanted=[]
    for row in records['last4']:
        for expected in row['expected']:
            if expected['id'] not in matches['last4']:continue
            thin=expected['label']=='crack' and min(expected['bbox'][2:])<=30 and max(expected['bbox'][2:])>=2*min(expected['bbox'][2:])
            if expected['label']!='spalling' and not thin:continue
            source=indexed[(row['source'],row['original_frame_index'])]
            wanted.append({'source':row['source'],'source_group':source['source_group'],'original_frame_index':row['original_frame_index'],
                           'original_camera_timestamp_us':source['original_timestamp_us'],'original_frame_period_us':source['source_frame_period_us'],
                           'annotation_timestamp_us':expected['annotation_timestamp_us'],'annotation_id':expected['id'],'publisher_box':expected['bbox'],
                           'class':expected['label'],'geometric_thin_box':thin,'raw_calibration_image':row['image'],
                           'model_matches':{name:matches[name].get(expected['id']) for name in NAMES}})
    points={name:ROOT/'models/construction'/(stem+'.operating-point.json') for name,stem in NAMES.items()}
    result={'scope':'POST-HOC descriptive audit of already frozen reused CAL matches; not independent accuracy or evidence of generalization. Metadata only; no images decoded, no NN, no source/cut changes.',
            'source_files':{name:{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)} for name,path in inputs.items()},
            'operating_points':{name:{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'class_confidence':json.loads(path.read_text())['class_confidence']} for name,path in points.items()},
            'matches':wanted,'future_illustrative_rule':'Any new source chosen from these hits must be marked post-hoc known-hit illustration, full original frame/PTS timeline and all unmatched/missed outcomes; no old metric/gate change.'}
    out=ROOT/'output/evaluation/construction-last4-candidate/known-calibration-hits-metadata.json'
    if out.exists():raise RuntimeError('Preserve existing known-hit metadata audit')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
