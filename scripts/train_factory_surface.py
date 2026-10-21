"""Download/fit/evaluate a small offline OpenCV-feature Random Forest on KolektorSDD.

No torch, CUDA, cloud inference, or test-set threshold selection.
Run from project root: python scripts/train_factory_surface.py --download
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier

from edge_runtime.profiles.factory_defect import extract_features, FEATURE_VERSION

URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip"
SHA256 = "65dc621693418585de9c4467d1340ea7958a6181816f0dc2883a1e8b61f9d4dc"
BYTES = 101831129


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(root, download):
    archive = root.parent / "KolektorSDD.zip"
    if not archive.is_file() and download:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=60) as src, temporary.open("wb") as dst:
            while block := src.read(1024 * 1024):
                dst.write(block)
        temporary.replace(archive)
    if archive.is_file():
        if archive.stat().st_size != BYTES or checksum(archive) != SHA256:
            raise ValueError("KolektorSDD archive failed the pinned size/SHA256 check")
        if not list(root.glob("kos*/*.jpg")):
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                if any(not (root / name).resolve().is_relative_to(root.resolve()) for name in z.namelist()):
                    raise ValueError("Unsafe path in archive")
                z.extractall(root)
    if len(list(root.glob("kos*/*.jpg"))) != 399:
        raise ValueError(f"Expected 399 images at {root}. Supply dataset or use --download.")
    return archive


def samples(root, item_names):
    rows = []
    for item in item_names:
        for image in sorted((root / item).glob("*.jpg")):
            mask = image.with_name(image.stem + "_label.bmp")
            target = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if target is None:
                raise ValueError(f"Missing label: {mask}")
            rows.append((image, target, bool(target.any())))
    return rows


def matrix(labels, predictions):
    labels, predictions = np.array(labels, dtype=bool), np.array(predictions, dtype=bool)
    tp = int(np.count_nonzero(labels & predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    tn = int(np.count_nonzero(~labels & ~predictions))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fn + fp), "accuracy": (tp + tn) / max(1, len(labels))}


def has_region(score, threshold, area):
    n, _, stats, _ = cv2.connectedComponentsWithStats((score >= threshold).astype(np.uint8), connectivity=8)
    return n > 1 and bool((stats[1:, cv2.CC_STAT_AREA] >= area).any())


def predict_scores(model, rows, width):
    records = []
    for path, _, positive in rows:
        frame = cv2.imread(str(path))
        start = time.perf_counter()
        features, shape = extract_features(frame, width)
        idx = int(np.flatnonzero(model.classes_ == 1)[0])
        scores = model.predict_proba(features)[:, idx].reshape(shape)
        ms = (time.perf_counter() - start) * 1000
        records.append((path, positive, scores, ms))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/factory/KolektorSDD"))
    parser.add_argument("--model", type=Path, default=Path("models/factory/kolektor_surface_rf.joblib"))
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/factory-real-input.json"))
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20261020)
    rng = np.random.default_rng(20261020)
    started = time.perf_counter()
    archive = acquire(args.dataset, args.download)
    names = sorted(p.name for p in args.dataset.iterdir() if p.is_dir() and p.name.startswith("kos"))
    if len(names) != 50:
        raise ValueError("Expected 50 physical item folders")
    shuffled = [names[i] for i in rng.permutation(len(names))]
    split = {"train": sorted(shuffled[:30]), "calibration": sorted(shuffled[30:40]), "test": sorted(shuffled[40:])}
    rows = {key: samples(args.dataset, values) for key, values in split.items()}
    train_x, train_y = [], []
    width = 256
    for path, mask, _ in rows["train"]:
        features, shape = extract_features(cv2.imread(str(path)), width)
        target = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        # Real masks at the feature-grid resolution; no defect labels from test/calibration enter training.
        positive = np.flatnonzero(target.ravel())
        negative = np.flatnonzero(~target.ravel())
        selected_pos = rng.choice(positive, min(400, len(positive)), replace=False) if len(positive) else positive
        selected_neg = rng.choice(negative, min(2000, len(negative)), replace=False)
        indices = np.concatenate([selected_neg, np.tile(selected_pos, 2)])
        train_x.append(features[indices])
        train_y.append(target.ravel()[indices].astype(np.int32))
    x, y = np.concatenate(train_x), np.concatenate(train_y)
    model = RandomForestClassifier(n_estimators=32, max_depth=12, min_samples_leaf=4,
                                   n_jobs=2, random_state=20261020)
    model.fit(x, y)
    print(f"Trained on {len(x)} sampled feature rows from {len(rows['train'])} images", flush=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model, compress=3)
    calibration = predict_scores(model, rows["calibration"], width)
    choices = []
    for threshold in (.5, .7, .8, .9, .95, .98, .99, .995):
        maximum_areas = []
        for record in calibration:
            _, _, stats, _ = cv2.connectedComponentsWithStats((record[2] >= threshold).astype(np.uint8), connectivity=8)
            maximum_areas.append(int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0)
        for area in (4, 8, 16, 32, 64, 128, 256, 512):
            score = matrix([r[1] for r in calibration], [a >= area for a in maximum_areas])
            choices.append((score["f1"], score["precision"], score["recall"], threshold, area, score))
    _, _, _, threshold, area, calibration_metrics = max(choices)
    print(f"Calibration-only thresholds: vote={threshold} component={area}; {calibration_metrics}", flush=True)
    metadata = {"model_name": "Kolektor surface Random Forest with OpenCV features", "feature_version": FEATURE_VERSION,
                "trained_utc": datetime.now(timezone.utc).isoformat(), "input_width": width,
                "model_sha256": checksum(args.model), "pixel_threshold": threshold,
                "minimum_component_pixels": area, "supported_height_width_ratio": [2.3, 2.7],
                "trees": 32, "maximum_tree_depth": 12, "dataset": "KolektorSDD", "scikit_learn": sklearn.__version__,
                "normal_samples_per_training_image": 2000, "positive_repeat_count": 2,
                "dataset_url": "https://www.vicos.si/resources/kolektorsdd/",
                "dataset_archive_url": URL, "dataset_archive_sha256": SHA256,
                "license": "Dataset and derived artifact: CC-BY-NC-SA-4.0; research/non-commercial prototype",
                "split_seed": 20261020, "split_items": split,
                "threshold_selection": "Maximum image F1 on calibration items only; then precision, recall, threshold, area",
                "field_validated": False, "score_meaning": "Mean decision-tree leaf defect fraction; not calibrated probability",
                "trust": "Locally trained artifact only. Joblib is pickle based; do not load untrusted files."}
    args.model.with_suffix(args.model.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    test = predict_scores(model, rows["test"], width)
    metrics = matrix([r[1] for r in test], [has_region(r[2], threshold, area) for r in test])
    # Rank-based image AUROC from max pixel vote score, separate from the thresholded component decision.
    positives = [float(r[2].max()) for r in test if r[1]]
    negatives = [float(r[2].max()) for r in test if not r[1]]
    auc = sum((p > n) + .5 * (p == n) for p in positives for n in negatives) / max(1, len(positives) * len(negatives))
    times = [r[3] for r in test]
    report = {"metadata": metadata, "model_bytes": args.model.stat().st_size,
              "archive_bytes": archive.stat().st_size if archive.exists() else None,
              "sampled_training_rows": len(x), "training_row_counts": dict(Counter(map(int, y))),
              "split_counts": {key: {"images": len(values), "defective": sum(r[2] for r in values),
                                     "normal": sum(not r[2] for r in values)} for key, values in rows.items()},
              "calibration": calibration_metrics, "test": {**metrics, "max_vote_image_auroc": auc},
              "test_process_ms": {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
                                  "p99": float(np.percentile(times, 99)), "max": max(times)},
              "seconds_total": time.perf_counter() - started, "opencv": cv2.__version__,
              "timing_scope": "Serial image feature extraction + all pixel forest votes; excludes image disk read, connected components, runtime and UI",
              "evaluation_scope": "Custom seeded physical-item holdout, not official 3-fold benchmark; no field validation",
              "test_predictions": [{"path": str(path.relative_to(args.dataset)), "defective": label,
                                    "predicted_defect": has_region(scores, threshold, area),
                                    "max_vote": float(scores.max()), "process_ms": ms} for path, label, scores, ms in test]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test": report["test"], "test_process_ms": report["test_process_ms"],
                      "model_bytes": report["model_bytes"], "report": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
