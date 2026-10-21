"""Real public-image CPU smoke plus tiny class-aware IoU evaluation; no network."""
from __future__ import annotations

import json
import platform
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.roads import RoadsProfile, _iou

ROOT = Path(__file__).resolve().parents[1]
CODES = ["D00", "D10", "D20", "D40"]


def truth(path):
    tree = ET.parse(path)
    boxes = []
    for obj in tree.findall("object"):
        name = obj.findtext("name")
        if name not in CODES:
            continue
        box = obj.find("bndbox")
        # Pascal VOC uses one-based inclusive coordinates.
        x1, y1 = float(box.findtext("xmin")) - 1, float(box.findtext("ymin")) - 1
        x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
        boxes.append({"class_id": CODES.index(name), "bbox": (x1, y1, x2 - x1, y2 - y1)})
    return boxes


def main():
    cv2.setNumThreads(2)
    output = ROOT / "output/evaluation/roads-real"
    output.mkdir(parents=True, exist_ok=True)
    profile = RoadsProfile({"demo": False, "model_variant": "india", "engine": "opencv", "confirm_frames": 1})
    warm = cv2.imread(str(ROOT / "datasets/roads-rdd2022-samples/01.jpg"))
    for i in range(3):
        profile.process(warm, FrameContext(i, float(i), "warmup", "evaluation"))
    rows, times, smoke = [], [], []
    for group in ("roads-rdd2022-samples", "roads-rdd2022-czech-subset"):
        paths = sorted((ROOT / "datasets" / group).glob("*.jpg")) if group.endswith("samples") else sorted((ROOT / "datasets" / group / "images").glob("*.jpg"))
        for index, path in enumerate(paths):
            frame = cv2.imread(str(path))
            started = time.perf_counter()
            result = profile.process(frame, FrameContext(index, time.perf_counter(), path.name, group))
            elapsed = (time.perf_counter() - started) * 1000
            times.append(elapsed)
            pred = result.domain_state["tracks"]
            row = {"image": path.name, "group": group, "process_ms": round(elapsed, 2),
                   "inference_ms": result.metrics["model_inference_ms"], "predictions": pred}
            if group.endswith("subset"):
                actual = truth(path.parents[1] / "annotations" / (path.stem + ".xml"))
                available = list(actual)
                tp = 0
                for detection in sorted(pred, key=lambda d: -d["model_score"]):
                    same = [a for a in available if a["class_id"] == detection["class_id"]]
                    best = max(same, key=lambda a: _iou(a["bbox"], detection["bbox"]), default=None)
                    if best is not None and _iou(best["bbox"], detection["bbox"]) >= .5:
                        tp += 1
                        available.remove(best)
                row.update(ground_truth=actual, true_positive=tp, false_positive=len(pred) - tp,
                           false_negative=len(available))
                rows.append(row)
            else:
                smoke.append(row)
            cv2.imwrite(str(output / f"{group}-{path.name}"), result.annotated_frame)
    totals = Counter()
    for row in rows:
        for name in ("true_positive", "false_positive", "false_negative"):
            totals[name] += row[name]
    tp, fp, fn = (totals[n] for n in ("true_positive", "false_positive", "false_negative"))
    class_counts = Counter(CODES[a["class_id"]] for row in rows for a in row["ground_truth"])
    report = {"model": profile.describe(), "opencv_version": cv2.__version__, "platform": platform.platform(),
              "cpu_threads": 2, "warmup_frames": 3, "threshold": .25, "class_aware_iou": .5,
              "network_test": "Python socket.connect blocked throughout evaluation; assets already installed",
              "selection": "First 16 lexicographic Czech train images, separate country from author-reported India training; not representative. Author samples are inference smoke without annotations.",
              "evaluation_limit": "No independent training audit; no adjacent-route independence guarantee; not a benchmark or field qualification. No AP computed.",
              "timing_scope": "Profile process including resize, model forward, decoding and annotation; excludes file load/capture/render. Combined public sample workload.",
              "timing_ms": {"p50": round(float(np.percentile(times, 50)), 2), "p95": round(float(np.percentile(times, 95)), 2),
                            "p99": round(float(np.percentile(times, 99)), 2), "max": round(max(times), 2),
                            "mean_process_fps": round(1000 / float(np.mean(times)), 2)},
              "labeled_counts": {"images": len(rows), **totals, "precision": tp / (tp + fp) if tp + fp else None,
                                 "recall": tp / (tp + fn) if tp + fn else None,
                                 "ground_truth_classes": dict(class_counts),
                                 "ground_truth_boxes": sum(len(r["ground_truth"]) for r in rows),
                                 "negative_images": sum(not r["ground_truth"] for r in rows)},
              "labeled_images": rows, "unlabeled_smoke": smoke}
    (output / "report-india.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"timing_ms": report["timing_ms"], "labeled_counts": report["labeled_counts"],
                      "smoke_images": len(smoke), "report": str(output / "report-india.json")}, indent=2))


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("No network allowed during road inference")):
        main()
