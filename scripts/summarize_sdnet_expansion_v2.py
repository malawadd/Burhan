"""Write the completed SDNET v2 report from frozen research and served results."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/training/construction-expansion-sdnet-v2-performance'


def overlap(a,b):
    x,y=max(a[0],b[0]),max(a[1],b[1]);xx,yy=min(a[0]+a[2],b[0]+b[2]),min(a[1]+a[3],b[1]+b[3])
    intersection=max(0,xx-x)*max(0,yy-y)
    return intersection/max(1e-9,a[2]*a[3]+b[2]*b[3]-intersection)


def percentage(value):return '—' if value is None else f'{100*value:.2f}%'


def main():
    result_path=BASE/'results.json';served_path=BASE/'served-development.json'
    result=json.loads(result_path.read_text());served=json.loads(served_path.read_text())
    dataset=json.loads((ROOT/'datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json').read_text())
    rows={r['image']:r for r in dataset['records']}
    classes=dataset['classes'];ious={label:[]for label in classes};fixed_boxes={label:0 for label in classes}
    for row in served['records']:
        truth=rows[row['image']]
        for cls,label in enumerate(classes):
            boxes=[b['bbox']for b in truth['boxes']if b['class_id']==cls];used=set()
            predictions=sorted([p for p in row['predictions']if p['label']==label],key=lambda p:-p['score'])
            if truth['publisher_label']=='Fixed':fixed_boxes[label]+=len(predictions)
            for prediction in predictions:
                best=max([(overlap(prediction['bbox'],b),i)for i,b in enumerate(boxes)if i not in used],default=(0,None))
                if best[0]>=.5:used.add(best[1]);ious[label].append(best[0])
    lines=['# Extended fastening fit: measured results','',
        'Date: 2026-10-21. This is a local research comparison on previously exposed development photographs. **No category is released.**','',
        'The final model completes eight logical epochs at 512px, batch 4, freeze first 4 layers: the first epoch used one thread and contiguous layout; seven remaining epochs used four threads and channels_last. Saved EMA model and optimizer values restored exactly. RNG, sampler and uninterrupted raw-model trajectories are not restored; this is a distinct candidate. See the [full experiment record](construction-expansion-sdnet-benchmarks.md).','',
        '## Before and after at their previously frozen calibration points','',
        '| Candidate / category | TP | Annotation-unmatched predictions | FN | Annotation-relative precision | Recall |','|---|---:|---:|---:|---:|---:|']
    for name,path in [('YOLO320 two-epoch',ROOT/'output/training/construction-expansion-sdnet-v1/yolov8n-results.json'),
                      ('SSDLite320 200-step',ROOT/'output/training/construction-expansion-sdnet-v1/ssdlite320-results.json')]:
        report=json.loads(path.read_text())['after_test_calibration_selected']['classes']
        for label in classes:
            s=report[label];precision=s['precision']if s['tp']+s['fp']else None
            lines.append(f"| {name} / {label} | {s['tp']} | {s['fp']} | {s['fn']} | {percentage(precision)} | {percentage(s['recall'])} |")
    for label in classes:
        s=served['classes'][label]
        lines.append(f"| Extended YOLO512 actual served / {label} | {s['true_positives']} | {s['false_positives']} | {s['false_negatives']} | {percentage(s['precision'])} | {percentage(s['recall'])} |")
    lines+=['','The 98 photographs are the same exposed v1 TEST group, now explicitly **reused development**. Thresholds were selected on CAL before final development inference. The first two rows per category are earlier candidate outputs; “before” does not imply that an earlier fastening model was reliable. Available annotations are not verified cross-class exhaustive, so unmatched predictions are not proven physical false alarms.','',
        '## Actual localization and publisher Fixed-only views','',
        '| Category | Frozen threshold | Mean IoU of matched served boxes | Flagged Fixed photographs | Boxes on Fixed photographs |','|---|---:|---:|---:|---:|']
    thresholds=served['model']['class_confidence']
    for label in classes:
        s=served['classes'][label];average=sum(ious[label])/len(ious[label])if ious[label]else None
        lines.append(f"| {label} | {thresholds[label]:.3f} | {'—' if average is None else f'{average:.3f}'} | {s['flagged_fixed_negative_images']} / {s['fixed_negative_images']} | {fixed_boxes[label]} |")
    lines+=['','Matches require correct-category box IoU ≥0.5. This detector produces boxes; actual-mask experiments have separate region/pixel reports. Only 30 publisher Fixed photographs are present, with unknown physical-site independence; normal-view performance remains exploratory.','',
        '## Training and execution evidence','',
        f"- Resume fit duration: {result['resume_seconds']:.1f}s for seven resumed epochs, including framework setup/final validation. Initial epoch timing is retained separately.",
        '- The original first-epoch checkpoint, before-resume protocol, model/optimizer restoration, actual layout/thread receipts, RNG limitation note, CAL cuts and CPU-ONNX parity are retained in the experiment directories.',
        '- The actual deployed adapter report binds model, manifest, dataset and adapter hashes; its per-photograph predictions are retained. Concurrent observations are not isolated inference qualifications.',
        '- All inference uses the laptop. No owner recordings entered TRAIN/CAL or threshold selection. Public training photographs remain identified as photographs.',
        '- The required independent 50-positive/50-negative per-category validation is absent. This report establishes neither production fastening correctness nor torque, concealed embedment or code compliance.','',
        'Machine artifacts:','',
        '- `output/training/construction-expansion-sdnet-v2-performance/results.json`',
        '- `output/training/construction-expansion-sdnet-v2-performance/served-development.json`',
        '- `models/construction/expansion-sdnet-v2-performance/yolov8n/manifest.json`','']
    target=ROOT/'docs/construction-expansion-sdnet-v2-results.md';target.write_text('\n'.join(lines),encoding='utf8')
    print(target)


if __name__=='__main__':main()
