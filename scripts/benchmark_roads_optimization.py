"""Compare unchanged-weight road exports on the same fixed public diagnostic.

Labels and scores never select thresholds. Config selection using this diagnostic
is explicitly development-biased and does not create a held-out accuracy result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
import sys
import time
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
    parser.add_argument("--configs", default="640:2,640:4,320:2,320:4,416:4")
    parser.add_argument("--openvino-site", help="Optional isolated export-env site-packages for provisional engine comparison; app numpy/OpenCV already imported")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--name", default="roads-optimization")
    args = parser.parse_args()
    if args.openvino_site:
        sys.path.append(str(Path(args.openvino_site).resolve()))
    cv2.setNumThreads(2)
    dataset = ROOT / "datasets/roads-rdd2022-test-balanced"
    manifest = json.loads((dataset / "manifest.json").read_text())
    output = ROOT / "output/evaluation" / args.name
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in manifest["images"]:
        path = dataset / "images" / row["file_name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Diagnostic image changed")
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        if max(h, w) > 1920:
            image = cv2.resize(image, (round(w * 1920 / max(h, w)), round(h * 1920 / max(h, w))), interpolation=cv2.INTER_AREA)
        rh, rw = image.shape[:2]
        xml = dataset / "original-annotations" / f"{path.stem}.xml"
        actual = [{"class_id": a["class_id"], "bbox": [a["bbox"][0] * rw / w, a["bbox"][1] * rh / h,
                                                               a["bbox"][2] * rw / w, a["bbox"][3] * rh / h]} for a in truth(xml)]
        cases.append((row, image, actual, hashlib.sha256(xml.read_bytes()).hexdigest()))
    reports = {}
    for config in args.configs.split(","):
        pieces = config.split(":")
        size, threads = map(int, pieces[:2])
        engine = pieces[2] if len(pieces) >= 3 else "onnxruntime"
        device = pieces[3] if len(pieces) == 4 else "CPU"
        label = f"{size}px-{threads}threads-{engine}-{device}"
        initialized = time.perf_counter()
        profile = RoadsProfile({"demo": False, "model_variant": "fiveclass", "input_size": size,
                                "cpu_threads": threads, "cpu_spinning": size == 640 and threads == 2,
                                "engine": engine, "openvino_device": device,
                                "confidence": .25, "nms_iou": .45, "confirm_frames": 1})
        initialization_ms = (time.perf_counter() - initialized) * 1000
        # Discard fixed warmups; source loading and session construction excluded.
        for index in range(5):
            profile.process(cases[0][1], FrameContext(index, time.perf_counter(), "warmup", label))
        timings, inference, rows, counts = [], [], [], Counter()
        class_counts = {i: Counter() for i in range(4)}
        for repeat in range(args.repeats):
            for index, (row, image, actual, xml_sha) in enumerate(cases):
                started = time.perf_counter()
                result = profile.process(image, FrameContext(index + repeat * len(cases), started, row["file_name"], label))
                elapsed = (time.perf_counter() - started) * 1000
                timings.append(elapsed)
                inference.append(result.metrics["model_inference_ms"])
                if repeat:
                    continue  # Accuracy images counted once, never repeats.
                predictions = [d for d in result.domain_state["tracks"] if d["class_id"] < 4]
                available, tp = list(actual), 0
                for detection in sorted(predictions, key=lambda d: -d["model_score"]):
                    same = [a for a in available if a["class_id"] == detection["class_id"]]
                    best = max(same, key=lambda a: _iou(a["bbox"], detection["bbox"]), default=None)
                    if best is not None and _iou(best["bbox"], detection["bbox"]) >= .5:
                        tp += 1
                        class_counts[detection["class_id"]]["true_positive"] += 1
                        available.remove(best)
                    else:
                        class_counts[detection["class_id"]]["false_positive"] += 1
                for missed in available:
                    class_counts[missed["class_id"]]["false_negative"] += 1
                current = {"true_positive": tp, "false_positive": len(predictions) - tp,
                           "false_negative": len(available), "ground_truth_boxes": len(actual),
                           "repair_observations": result.metrics["repair_observation_count"]}
                counts.update(current)
                rows.append({"file_name": row["file_name"], "sha256": row["sha256"], "original_xml_sha256": xml_sha,
                             "original_truth": actual, "predictions": predictions,
                             "repairs": [d for d in result.domain_state["tracks"] if d["class_id"] == 4],
                             "first_process_ms": elapsed, "first_inference_ms": result.metrics["model_inference_ms"], **current})
                if row["file_name"] in {"China_MotorBike_001578.jpg", "China_MotorBike_001563.jpg"}:
                    cv2.imwrite(str(output / f"{label}-{row['file_name']}"), result.annotated_frame)
        tp, fp, fn = (counts[k] for k in ("true_positive", "false_positive", "false_negative"))
        module_name = {"openvino": "openvino", "onnxruntime": "onnxruntime", "opencv": "cv2"}[engine]
        report = {"model": profile.describe(), "initialization_ms": initialization_ms,
                  "engine_module_file": str(Path(sys.modules[module_name].__file__).resolve()),
                  "counts": {**counts, "images": len(cases),
                   "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None},
                  "class_counts": class_counts,
                  "timing_ms": {"samples": len(timings), "p50": float(np.percentile(timings, 50)),
                    "p95": float(np.percentile(timings, 95)), "p99": float(np.percentile(timings, 99)),
                    "max": max(timings), "mean_process_fps": 1000 / float(np.mean(timings)),
                    "inference_p50": float(np.percentile(inference, 50))}, "images": rows,
                  "process_samples_ms": timings, "inference_samples_ms": inference}
        reports[label] = report
        print(label, json.dumps({"counts": report["counts"], "timing_ms": report["timing_ms"]}), flush=True)
        # Persist each completed candidate if a subsequent comparison fails.
        summary = {"platform": platform.platform(), "opencv": cv2.__version__, "numpy": np.__version__,
                   "python_executable": sys.executable, "openvino_site_override": args.openvino_site, "threshold": .25,
                   "matching": "class-aware greedy IoU >= .5 against original VOC four damage labels; repair excluded",
                   "selection_bias": "320/416/640 and thread options chosen before scoring; final preset selected using these28 cases. No threshold fitting. Development diagnostic has possible training overlap, not independent accuracy.",
                   "timing_scope": "5 warmups; full process including letterbox/FP32 inference on explicitly recorded device/decode/annotation; image read/session construction/capture/browser excluded. Accuracy once per28images; timing repetitions not new samples.",
                   "network": "socket.connect denied", "repeats": args.repeats, "reports": reports}
        (output / "report.json").write_text(json.dumps(summary, indent=2))
        del profile


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("No network during model benchmark")):
        main()
