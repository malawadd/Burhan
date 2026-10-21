"""Evaluate actual deployed CPU adapters on frozen photo groups, never tune cuts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_task_detector import ConstructionTaskDetector, discover_models


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-x)*max(0, bottom-y)
    return intersection/max(1e-9, a[2]*a[3]+b[2]*b[3]-intersection)


def evaluate(manifest_path, dataset_path, out):
    model = ConstructionTaskDetector(manifest_path)
    dataset = json.loads(dataset_path.read_text())
    if dataset.get("private_videos_used") or dataset.get("private_owner_videos_used"):
        raise ValueError("Public evaluation must exclude private owner footage")
    classes = model.spec["classes"]
    if dataset["classes"] != classes:
        raise ValueError("Model and frozen dataset classes differ")
    stats = {label: {"true_positives": 0, "false_positives": 0, "false_negatives": 0,
                     "positive_images": 0, "negative_images": 0, "flagged_negative_images": 0,
                     "fixed_negative_images": 0, "flagged_fixed_negative_images": 0,
                     "pixel_tp": 0, "pixel_fp": 0, "pixel_fn": 0} for label in classes}
    rows = [row for row in dataset["records"] if row["split"] == "test"]
    times, records = [], []
    for row in rows:
        image_path = ROOT / row["image"]
        if sha(image_path) != row["source_sha256"]:
            raise ValueError("Frozen source image bytes changed")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Cannot decode frozen source image")
        started = time.perf_counter()
        predicted = model.infer(image)
        times.append((time.perf_counter()-started)*1000)
        mask_truth = None
        if row.get("mask"):
            mask_path = ROOT / row["mask"]
            if sha(mask_path) != row["mask_sha256"]:
                raise ValueError("Frozen source polygon mask changed")
            mask_truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        summarized = []
        for index, label in enumerate(classes):
            truth = [b["bbox"] for b in row["boxes"] if b["class_id"] == index]
            chosen = sorted([p for p in predicted if p["label"] == label], key=lambda p: -p["score"])
            s = stats[label]
            s["positive_images" if truth else "negative_images"] += 1
            if not truth and chosen:
                s["flagged_negative_images"] += 1
            if row.get("publisher_label") == "Fixed":
                s["fixed_negative_images"] += 1
                s["flagged_fixed_negative_images"] += int(bool(chosen))
            matched = set()
            for p in chosen:
                best = max(((iou(p["bbox"], b), j) for j, b in enumerate(truth) if j not in matched), default=(0, -1))
                if best[0] >= .5:
                    s["true_positives"] += 1
                    matched.add(best[1])
                else:
                    s["false_positives"] += 1
                summarized.append({key: value for key, value in p.items() if key != "mask"})
            s["false_negatives"] += len(truth)-len(matched)
            if mask_truth is not None and model.spec["architecture"] in {"lraspp", "yolov8n_seg"}:
                predicted_mask = np.zeros(image.shape[:2], bool)
                for p in chosen:
                    if "mask" in p:
                        predicted_mask |= p["mask"]
                expected = mask_truth == index+1
                s["pixel_tp"] += int(np.count_nonzero(predicted_mask & expected))
                s["pixel_fp"] += int(np.count_nonzero(predicted_mask & ~expected))
                s["pixel_fn"] += int(np.count_nonzero(~predicted_mask & expected))
        records.append({"image": row["image"], "source_sha256": row["source_sha256"], "predictions": summarized})
    for s in stats.values():
        tp, fp, fn = (s[key] for key in ("true_positives", "false_positives", "false_negatives"))
        s.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None)
        if s["pixel_tp"]+s["pixel_fn"]:
            s["pixel_precision"] = s["pixel_tp"]/max(1, s["pixel_tp"]+s["pixel_fp"])
            s["pixel_recall"] = s["pixel_tp"]/(s["pixel_tp"]+s["pixel_fn"])
            s["pixel_iou"] = s["pixel_tp"]/(s["pixel_tp"]+s["pixel_fp"]+s["pixel_fn"])
    report = {"scope": "Served one-thread CPU-ONNX predictions at already-frozen thresholds on exposed frozen photo TEST/development groups. No selection, independent site/video or healthy-wall claim. Box matches IoU>=.5; pixel metrics separately use actual source polygon masks when available. Region-mask IoU metrics remain in dedicated mask comparison.",
        "model": model.describe(), "manifest_sha256": sha(manifest_path), "dataset_manifest_sha256": sha(dataset_path),
        "adapter_sha256": sha(ROOT / "edge_runtime/profiles/construction_task_detector.py"),
        "images": len(rows), "classes": stats, "timing_observation": {"calls": len(times),
        "p50_ms": float(np.percentile(times, 50)) if times else None,
        "p95_ms": float(np.percentile(times, 95)) if times else None,
        "concurrency": "Other training may be active; not an isolated speed qualification"}, "records": records}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report = evaluate(args.manifest.resolve(), args.dataset.resolve(), args.output.resolve())
    print(json.dumps({"images": report["images"], "classes": report["classes"]}, indent=2))


if __name__ == "__main__":
    main()
