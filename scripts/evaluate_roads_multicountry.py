"""Fixed, public provider-test diagnostic of the multi-country road model."""
from __future__ import annotations

import json
import hashlib
import argparse
import platform
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.roads import RoadsProfile, _iou
from scripts.evaluate_roads_real import truth

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["multicountry", "fiveclass"], default="fiveclass")
    parser.add_argument("--engines", default="onnxruntime,opencv")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    dataset = ROOT / "datasets/roads-rdd2022-test-balanced"
    manifest = json.loads((dataset / "manifest.json").read_text())
    output = ROOT / "output/evaluation" / f"roads-real-{args.variant}"
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for engine in args.engines.split(","):
        # Keep this historical reproduction fixed even if the application default
        # adopts a faster backend or spatial size later.
        profile = RoadsProfile({"demo": False, "model_variant": args.variant, "engine": engine,
                                "input_size": 640, "cpu_threads": 2, "cpu_spinning": True,
                                "confirm_frames": 1})
        warmup = np.zeros((640, 640, 3), np.uint8)
        for i in range(3):
            profile.process(warmup, FrameContext(i, float(i), "warmup", "evaluation"))
        rows, timings = [], []
        totals, classes, by_partition = Counter(), Counter(), {}
        original_totals = Counter()
        invalid_annotations = 0
        for index, row in enumerate(manifest["images"]):
            image_path = dataset / "images" / row["file_name"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Public evaluation image hash changed")
            image = cv2.imread(str(image_path))
            original_h, original_w = image.shape[:2]
            if max(image.shape[:2]) > 1920:
                ratio = 1920 / max(image.shape[:2])
                image = cv2.resize(image, (max(1, round(original_w * ratio)), max(1, round(original_h * ratio))), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
            actual = []
            for bbox, category in zip(row["objects"]["bbox"], row["objects"]["categories"]):
                x, y, bw, bh = bbox
                if (not all(np.isfinite(bbox)) or category not in (0, 1, 2, 3) or bw <= 0 or bh <= 0
                        or x < 0 or y < 0 or x + bw > original_w + 1 or y + bh > original_h + 1):
                    invalid_annotations += 1
                    continue
                scaled = [x * w / original_w, y * h / original_h, bw * w / original_w, bh * h / original_h]
                actual.append({"bbox": scaled, "class_id": category})
                classes[category] += 1
            started = time.perf_counter()
            result = profile.process(image, FrameContext(index, time.perf_counter(), row["file_name"], "provider-test"))
            elapsed = (time.perf_counter() - started) * 1000
            timings.append(elapsed)
            pred = [d for d in result.domain_state["tracks"] if d["class_id"] < 4]
            available = list(actual)
            tp = 0
            for detection in sorted(pred, key=lambda d: -d["model_score"]):
                same = [a for a in available if a["class_id"] == detection["class_id"]]
                best = max(same, key=lambda a: _iou(a["bbox"], detection["bbox"]), default=None)
                if best is not None and _iou(best["bbox"], detection["bbox"]) >= .5:
                    tp += 1
                    available.remove(best)
            counts = {"true_positive": tp, "false_positive": len(pred) - tp, "false_negative": len(available)}
            original_xml = dataset / "original-annotations" / (Path(row["file_name"]).stem + ".xml")
            original_truth = truth(original_xml) if original_xml.exists() else None
            original_counts = None
            if original_truth is not None:
                original_scaled = [{"class_id": a["class_id"], "bbox": [a["bbox"][0] * w / original_w,
                                    a["bbox"][1] * h / original_h, a["bbox"][2] * w / original_w,
                                    a["bbox"][3] * h / original_h]} for a in original_truth]
                remaining = list(original_scaled)
                original_tp = 0
                for detection in sorted(pred, key=lambda d: -d["model_score"]):
                    same = [a for a in remaining if a["class_id"] == detection["class_id"]]
                    best = max(same, key=lambda a: _iou(a["bbox"], detection["bbox"]), default=None)
                    if best is not None and _iou(best["bbox"], detection["bbox"]) >= .5:
                        original_tp += 1
                        remaining.remove(best)
                original_counts = {"true_positive": original_tp, "false_positive": len(pred) - original_tp,
                                   "false_negative": len(remaining), "ground_truth_boxes": len(original_scaled)}
                original_totals.update(original_counts)
            totals.update(counts)
            by_partition.setdefault(row["partition"], Counter()).update(counts)
            rows.append({"image": row["file_name"], "partition": row["partition"], "original_shape_hw": [original_h, original_w],
                         "input_shape_hw": [h, w], "ground_truth": actual, "predictions": pred, **counts,
                         "original_ground_truth": original_truth, "original_counts": original_counts,
                         "repair_observations": [d for d in result.domain_state["tracks"] if d["class_id"] == 4],
                         "process_ms": round(elapsed, 2), "inference_ms": result.metrics["model_inference_ms"]})
            if engine == "onnxruntime":
                cv2.imwrite(str(output / row["file_name"]), result.annotated_frame)
        tp, fp, fn = (totals[k] for k in ("true_positive", "false_positive", "false_negative"))
        otp, ofp, ofn = (original_totals[k] for k in ("true_positive", "false_positive", "false_negative"))
        reports[engine] = {"model": profile.describe(), "counts": {**totals, "images": len(rows),
                          "ground_truth_boxes": sum(classes.values()), "ground_truth_class_ids": dict(classes),
                          "negative_images": sum(not r["ground_truth"] for r in rows),
                          "invalid_annotation_count": invalid_annotations, "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / (tp + fn) if tp + fn else None}, "by_partition": by_partition,
                          "original_voc_counts": {**original_totals, "precision": otp / (otp + ofp) if otp + ofp else None,
                                                  "recall": otp / (otp + ofn) if otp + ofn else None},
                          "timing_ms": {"p50": round(float(np.percentile(timings, 50)), 2), "p95": round(float(np.percentile(timings, 95)), 2),
                                        "p99": round(float(np.percentile(timings, 99)), 2), "max": round(max(timings), 2),
                                        "mean_process_fps": round(1000 / float(np.mean(timings)), 2)}, "images": rows}
        print(engine, json.dumps({"counts": reports[engine]["counts"], "original_voc_counts": reports[engine]["original_voc_counts"], "timing_ms": reports[engine]["timing_ms"]}), flush=True)
    report = {"platform": platform.platform(), "opencv_version": cv2.__version__, "cpu_threads": 2,
              "selection": manifest["selection"], "data_revision": manifest["revision"], "data_split": manifest["split"],
              "limitations": "28 stratified convenience cases; not representative, no AP. Five-class author does not publish training membership: possible training overlap, not independent accuracy. Provider test split establishes no holdout for that separate author. Route independence not audited. Replacement followed poor India-only model results and mirror-label defects; not official challenge/field accuracy.",
              "threshold": .25, "class_aware_iou": .5, "annotation_contract": manifest["annotation_contract"],
              "network_test": "Python socket.connect blocked throughout; local weights/images only",
              "timing_scope": "3 warmups then per-image profile.process; source image read/1920-cap and browser/capture excluded; model letterbox/inference/decoding/annotation included",
              "engines": reports}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden during evaluation")):
        main()
