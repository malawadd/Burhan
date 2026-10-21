"""Evaluate the real CPU serving adapter on frozen PHELE TEST photos."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from edge_runtime.profiles.construction_task_detector import ConstructionTaskDetector

OUT = ROOT/"output/evaluation/construction-expansion-phele-v1"


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    area = max(0, right-x)*max(0, bottom-y)
    return area/max(1e-12, a[2]*a[3]+b[2]*b[3]-area)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["yolov8n", "ssdlite320"], required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    path = ROOT/f"models/construction/expansion-phele-v1/{args.model}/manifest.json"
    adapter = ConstructionTaskDetector(path)
    manifest = json.loads((OUT/"training-manifest.json").read_text())
    rows = [row for row in manifest["records"] if row["split"] == "test"]
    warmup = cv2.imread(str(ROOT/next(row["image"] for row in manifest["records"] if row["split"] == "train")))
    for _ in range(3):
        adapter.infer(warmup)
    classes = manifest["classes"]
    stats = {name: {"tp": 0, "fp": 0, "fn": 0, "positive_images": 0, "negative_images": 0, "flagged_negative_images": 0, "matched_ious": []} for name in classes}
    records, times = [], []
    for row in rows:
        image = cv2.imread(str(ROOT/row["image"]))
        started = time.perf_counter()
        detections = adapter.infer(image)
        times.append(time.perf_counter()-started)
        for class_id, label in enumerate(classes):
            truth = [box["bbox"] for box in row["boxes"] if box["class_id"] == class_id]
            chosen = sorted([item for item in detections if item["label"] == label], key=lambda item: -item["score"])
            stat = stats[label]
            stat["positive_images" if truth else "negative_images"] += 1
            stat["flagged_negative_images"] += int(not truth and bool(chosen))
            matched = set()
            for item in chosen:
                candidates = [(iou(item["bbox"], box), index) for index, box in enumerate(truth) if index not in matched]
                overlap, index = max(candidates, default=(0, None))
                item["matched_iou"] = overlap
                item["matched_publisher_box"] = index if overlap >= .5 else None
                if overlap >= .5:
                    matched.add(index)
                    stat["tp"] += 1
                    stat["matched_ious"].append(overlap)
                else:
                    stat["fp"] += 1
            stat["fn"] += len(truth)-len(matched)
        records.append({"member": row["original_member"], "image": row["image"], "publisher_boxes": row["boxes"], "predictions": detections})
    for stat in stats.values():
        stat["precision"] = stat["tp"]/max(1, stat["tp"]+stat["fp"])
        stat["recall"] = stat["tp"]/max(1, stat["tp"]+stat["fn"])
        overlaps = stat.pop("matched_ious")
        stat["mean_matched_iou"] = float(np.mean(overlaps)) if overlaps else None
    output = {"architecture": args.model, "model": adapter.describe(), "test_images": len(rows), "iou_threshold": .5,
              "classes": stats, "predictions": records, "timing": {"warmup_train_calls_discarded": 3,
              "batch": 1, "includes_preprocessing_inference_nms": True, "includes_disk_decode": False,
              "seconds": sum(times), "throughput_fps": len(times)/sum(times), "p50_ms": float(np.percentile(times, 50)*1000),
              "p95_ms": float(np.percentile(times, 95)*1000), "contention": "Concurrent training may be active; this is not isolated-runtime acceptance"},
              "release_pass": False, "scope": "Real frozen photo diagnostic using shipped CPU adapter; physical site independence and 50+50 category release not established"}
    target = OUT/f"{args.model}-served-box-evaluation.json"
    target.write_text(json.dumps(output, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"report": str(target), "classes": stats, "timing": output["timing"]}), flush=True)


if __name__ == "__main__":
    main()
