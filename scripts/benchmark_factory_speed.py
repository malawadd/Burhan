"""Paired frozen-holdout CPU benchmark; no fitting or threshold selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.factory import FactoryProfile
from edge_runtime.profiles.factory_defect import extract_features, extract_features_fast
from train_factory_surface import samples, matrix


def timing(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": max(values),
            "mean": float(np.mean(values)), "serial_capacity_fps": 1000 / float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/factory/kolektor_surface_rf.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/factory/KolektorSDD"))
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/factory-speed-comparison.json"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verify-pixels", action="store_true")
    parser.add_argument("--after-backend", choices=["auto", "native_cascade", "threshold_cascade"], default="auto")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    profiles = {name: FactoryProfile({"defect_model_path": str(args.model), "defect_backend": backend,
                                    "confirmation_frames": 1})
                for name, backend in (("before", "sklearn"), ("after", args.after_backend))}
    metadata = profiles["before"].describe()["model_metadata"]
    rows = samples(args.dataset, metadata["split_items"]["test"])
    decoded = [(path, cv2.imread(str(path)), label) for path, _, label in rows]
    # Identical warmup data and excluded timings for both backends.
    for name, profile in profiles.items():
        for n in range(2):
            profile.process(decoded[0][1], FrameContext(n, time.perf_counter(), unit_id=f"warm-{n}"))
        profile.reset()
    records = []
    for repeat in range(args.repeat):
        for index, (path, frame, label) in enumerate(decoded):
            record = {"path": str(path.relative_to(args.dataset)), "defective": label, "repeat": repeat,
                      "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            order = ("before", "after") if (index + repeat) % 2 == 0 else ("after", "before")
            for name in order:
                context = FrameContext(index, time.perf_counter(), source_id="frozen_item_holdout",
                                       session_id=f"speed-{name}", unit_id=f"{repeat}/{path.name}/{index}")
                started = time.perf_counter()
                result = profiles[name].process(frame, context)
                elapsed = (time.perf_counter() - started) * 1000
                record[name] = {"status": result.status, "process_ms": elapsed, "metrics": result.metrics,
                                "findings": [{"bbox": f.bbox, "score": f.score} for f in result.findings]}
            if args.verify_pixels:
                original_features, _ = extract_features(frame, metadata["input_width"])
                fast_features, _ = extract_features_fast(frame, metadata["input_width"])
                record["feature_value_disagreements"] = int(np.count_nonzero(original_features != fast_features))
                models = {name: profile._defect_model for name, profile in profiles.items()}
                scores = {name: model.predict(frame)[0] for name, model in models.items()}
                threshold = metadata["pixel_threshold"]
                binary = scores["before"] >= threshold
                record["pixel_threshold_disagreements"] = int(np.count_nonzero(binary != (scores["after"] >= threshold)))
                record["candidate_score_max_difference"] = float(np.max(np.abs(scores["before"][binary] - scores["after"][binary]))) if binary.any() else 0.
            records.append(record)
            if (index + 1) % 20 == 0:
                print(f"Completed {index + 1}/{len(decoded)} images, repeat {repeat + 1}", flush=True)
    report = {"model_sha256": profiles["after"].describe()["model_sha256"],
              "metadata_sha256": hashlib.sha256(args.model.with_suffix(args.model.suffix + ".json").read_bytes()).hexdigest(),
              "images": len(decoded), "repeats": args.repeat, "opencv": cv2.__version__, "python": platform.python_version(),
              "opencv_threads": cv2.getNumThreads(), "forest_baseline_workers": 2,
              "frozen_threshold": metadata["pixel_threshold"], "frozen_component_area": metadata["minimum_component_pixels"],
              "split_items": metadata["split_items"]["test"],
              "scope": "Paired alternating full FactoryProfile.process over predecoded frozen 80 item-heldout images; two warmups/backend excluded. File read/capture/queue/UI/event persistence/actuation excluded.",
              "after_backend": profiles["after"].describe()["model_backend"],
              "native_code_sha256": profiles["after"].describe()["native_code_sha256"],
              "selection": "Exact-threshold early rejection and integer-equivalent native input-quality reduction on unchanged forest/feature grid/threshold; no calibration or test optimization grid, no fitting.",
              "backends": {}, "decision_disagreements": sum(r["before"]["status"] != r["after"]["status"] for r in records),
              "finding_box_disagreements": sum([f["bbox"] for f in r["before"]["findings"]] != [f["bbox"] for f in r["after"]["findings"]] for r in records),
              "quality_metric_disagreements": sum(any(r["before"]["metrics"][k] != r["after"]["metrics"][k] for k in ("laplacian_variance", "grayscale_std")) for r in records),
              "records": records}
    for name in profiles:
        accepted = [r for r in records if r[name]["status"] in {"fault", "clear"}]
        report["backends"][name] = {"process_ms": timing([r[name]["process_ms"] for r in records]),
                                  "accepted": len(accepted), "abstained": len(records) - len(accepted),
                                  "image_metrics": matrix([r["defective"] for r in accepted], [r[name]["status"] == "fault" for r in accepted])}
    report["p95_speedup"] = report["backends"]["before"]["process_ms"]["p95"] / report["backends"]["after"]["process_ms"]["p95"]
    if args.verify_pixels:
        report["feature_value_disagreements"] = sum(r["feature_value_disagreements"] for r in records)
        report["pixel_threshold_disagreements"] = sum(r["pixel_threshold_disagreements"] for r in records)
        report["candidate_score_max_difference"] = max(r["candidate_score_max_difference"] for r in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
