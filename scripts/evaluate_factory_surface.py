"""Evaluate the complete factory profile on the frozen model's test items."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.factory import FactoryProfile
from train_factory_surface import samples, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/factory/kolektor_surface_rf.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/factory/KolektorSDD"))
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/factory-profile-real-input.json"))
    args = parser.parse_args()
    cv2.setNumThreads(2)
    profile = FactoryProfile({"defect_model_path": str(args.model), "confirmation_frames": 1})
    metadata = profile.describe()["model_metadata"]
    rows = samples(args.dataset, metadata["split_items"]["test"])
    records = []
    saved_cases = set()
    for n, (path, _, defective) in enumerate(rows):
        frame = cv2.imread(str(path))
        context = FrameContext(n, time.perf_counter(), source_id="public_item_holdout", session_id="factory-evaluation",
                               unit_id=str(path.relative_to(args.dataset)))
        start = time.perf_counter()
        result = profile.process(frame, context)
        ms = (time.perf_counter() - start) * 1000
        records.append({"path": str(path.relative_to(args.dataset)), "defective": defective,
                        "status": result.status, "findings": len(result.findings), "process_ms": ms,
                        "emit_event": result.emit_event, "metrics": result.metrics})
        case = "true-positive" if defective and result.status in {"fault","review"} else "miss" if defective else "normal" if result.status in {"clear","no_candidate"} else "false-positive"
        if case not in saved_cases:
            image_path = args.output.parent / f"factory-heldout-{case}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), result.annotated_frame)
            saved_cases.add(case)
    accepted = [r for r in records if r["status"] in {"fault","review","clear","no_candidate"}]
    times = [r["process_ms"] for r in records]
    report = {"model_sha256": profile.describe()["model_sha256"], "images": len(records), "accepted": len(accepted),
              "abstained_defective": sum(r["defective"] for r in records if r not in accepted),
              "abstained_normal": sum(not r["defective"] for r in records if r not in accepted),
              "accepted_image_metrics": matrix([r["defective"] for r in accepted], [r["status"] in {"fault","review"} for r in accepted]),
              "process_ms": {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
                             "p99": float(np.percentile(times, 99)), "max": max(times)},
              "scope": "Full FactoryProfile.process on original held-out item images with independent per-image unit IDs; excludes file decode, capture, queues, UI, disk evidence and actuator",
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
