"""Lightweight real-data/hash/geometry QA, with explicitly GT-only crop previews."""
from pathlib import Path
import argparse
import hashlib
import json
import math
from collections import Counter
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/construction/dacl10k'
BASE = DATA / 'round3-positive-contingency'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closeups',action='store_true')
    args=parser.parse_args()
    if args.closeups:
        BASE=DATA/'round3-crack-closeup-contingency'
    manifest_path = BASE / 'asset-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    existing = json.loads((DATA / 'round3-asset-manifest.json').read_text(encoding='utf-8'))
    excluded_names = {r['source_name'] for r in existing['records']}
    excluded_hashes = {r['image_sha256'] for r in existing['records']}
    seen_names, seen_hashes, counts, instances, vanished = set(), set(), Counter(), Counter(), 0
    first = {}
    for row in manifest['records']:
        name, source_sha = row['source_name'], row['original_image_sha256']
        assert row['source_split'] == 'train'
        assert name not in excluded_names and source_sha not in excluded_hashes
        assert name not in seen_names and source_sha not in seen_hashes
        seen_names.add(name); seen_hashes.add(source_sha)
        counts[row['cohort']] += 1
        for file_key, hash_key in [('original_image','original_image_sha256'), ('original_annotation','original_annotation_sha256')]:
            assert sha(BASE / row[file_key]) == row[hash_key]
        derived = row['derived']
        for file_key, hash_key in [('image','image_sha256'), ('label','label_sha256'), ('derivation','derivation_sha256')]:
            assert sha(BASE / derived[file_key]) == derived[hash_key]
        image = Image.open(BASE / derived['image'])
        assert image.size == (640,480) and image.mode == 'RGB'
        red, green, blue = image.split()
        assert red.tobytes() == green.tobytes() == blue.tobytes()
        if args.closeups:
            left,top,cw,ch=derived['crop_xywh']
            assert cw*3==ch*4 and 320<=cw<=960 and 240<=ch<=720
            assert left>=0 and top>=0 and left+cw<=derived['source_size'][0] and top+ch<=derived['source_size'][1]
        lines = (BASE / derived['label']).read_text(encoding='utf-8').splitlines()
        assert len(lines) == len(derived['instances']) and lines
        for line, instance in zip(lines, derived['instances']):
            values = [float(x) for x in line.split()]
            assert len(values) == 5 and int(values[0]) in {0,1}
            assert all(math.isfinite(x) for x in values)
            assert 0 <= values[1] <= 1 and 0 <= values[2] <= 1 and 0 < values[3] <= 1 and 0 < values[4] <= 1
            assert int(values[0]) == instance['class_id']
            instances[instance['source_label']] += 1
            vanished += instance['output_nearest_mask_bbox_xyxy'] is None
        first.setdefault(row['cohort'], row)
    assert counts == ({'crack_family_closeup':40} if args.closeups else {'crack_family':40,'spalling':40})
    qa_dir = BASE / 'qa'
    qa_dir.mkdir(exist_ok=True)
    previews = []
    for cohort, row in first.items():
        derived = row['derived']
        source = Image.open(BASE / derived['image']).convert('RGB')
        panel = Image.new('RGB',(1280,520),(20,20,20))
        panel.paste(source,(0,40)); panel.paste(source,(640,40))
        draw = ImageDraw.Draw(panel)
        draw.text((8,10), 'REAL DERIVED TRAIN CROP',fill='white')
        draw.text((648,10), 'PUBLISHER-POLYGON BOXES / NOT MODEL OUTPUT',fill='white')
        for item in derived['instances']:
            x1,y1,x2,y2 = item['output_projected_source_bbox_xyxy']
            color = (255,190,20) if item['class_id']==1 else (20,220,255)
            draw.rectangle((640+x1,40+y1,640+x2,40+y2),outline=color,width=2)
            draw.text((640+x1,max(40,40+y1-12)),item['source_label'],fill=color)
        path=qa_dir/f'{cohort}-first-derived-gt.jpg'
        panel.save(path,quality=90)
        previews.append(path.relative_to(BASE).as_posix())
    report={'scope':'Actual downloaded original data and derived TRAIN targets only. No model inference or performance evidence.',
            'asset_manifest_sha256':sha(manifest_path),'counts':dict(counts),'retained_polygon_boxes_by_source_class':dict(instances),
            'retained_source_boxes_whose_nearest_resized_mask_vanishes':vanished,
            'exclusions':'All95 existing names/hashes plus selected-source byte duplicates; officialTRAIN only. Physical bridge independence unverified.',
            'checks':'Original/annotation/derived hashes, unique source images/hashes, grayRGB640x480, finite normalized positive boxes, class mapping and complete label count.',
            'gt_only_previews':previews,'status':'UNUSED CONTINGENCY; no active training/calibration/heldout changes'}
    (qa_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
