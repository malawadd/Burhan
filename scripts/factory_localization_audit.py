"""Independent original-mask audit, including stronger checks than one-pixel hits.

This is accuracy-only. It intentionally does not call its execution times speed
benchmarks. The exposed SDD1 comparison remains a development comparison.
"""
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


def retained_mask(scores: np.ndarray, threshold: float, area: int, shape: tuple[int, int], growth_threshold=None) -> np.ndarray:
    """Match runtime's area filter and stable largest-20 retained-component cap."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((scores >= threshold).astype(np.uint8), connectivity=8)
    candidates = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= area) + 1
    if len(candidates) > 20:
        candidates = candidates[np.argsort(-stats[candidates, cv2.CC_STAT_AREA], kind="stable")[:20]]
    keep = np.zeros(count, dtype=np.uint8)
    keep[candidates] = 1
    retained=keep[labels]
    if growth_threshold is not None:
        if not 0<=growth_threshold<=threshold:
            raise ValueError('Invalid independent-audit hysteresis threshold')
        low_count,low_labels=cv2.connectedComponents((scores>=growth_threshold).astype(np.uint8),connectivity=8)
        grow_ids=set(np.unique(low_labels[retained.astype(bool)]).tolist())-{0}
        low_keep=np.zeros(low_count,dtype=np.uint8)
        for identifier in grow_ids:low_keep[identifier]=1
        retained=low_keep[low_labels]
    return cv2.resize(retained, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def mask_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    pred, gt = prediction.astype(bool), truth.astype(bool)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and annotation must have identical original-resolution geometry")
    intersection = int(np.count_nonzero(pred & gt))
    pred_n, gt_n = int(pred.sum()), int(gt.sum())
    p_count, p_labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    g_count, g_labels = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    # Contingency-table matching scores every predicted region, including false locations.
    table = np.bincount((g_labels.astype(np.int64) * p_count + p_labels).ravel(), minlength=g_count * p_count).reshape(g_count, p_count)
    g_area, p_area = table.sum(axis=1), table.sum(axis=0)
    overlap = table[1:, 1:]
    union = g_area[1:, None] + p_area[None, 1:] - overlap
    ious = overlap / np.maximum(1, union)
    region_matches = {}
    for cutoff in (.1, .25, .5):
        edges = [(float(ious[g, p]), g, p) for g, p in zip(*np.nonzero(ious >= cutoff))]
        matched_g, matched_p = set(), set()
        for _, g, p in sorted(edges, reverse=True):
            if g not in matched_g and p not in matched_p:
                matched_g.add(g)
                matched_p.add(p)
        region_matches[str(cutoff)] = len(matched_g)
    return {
        "flagged": pred_n > 0, "localized_any_pixel": intersection > 0,
        "pixel_tp": intersection, "pixel_predicted": pred_n, "pixel_target": gt_n,
        "pixel_recall": intersection / gt_n if gt_n else None,
        "pixel_precision": intersection / pred_n if pred_n else 0.0,
        "iou": intersection / (gt_n + pred_n - intersection) if gt_n + pred_n else None,
        "dice": 2 * intersection / (gt_n + pred_n) if gt_n + pred_n else None,
        "gt_regions": g_count - 1, "predicted_regions": p_count - 1,
        "region_matches_at_iou": region_matches,
        "localized_10pct_coverage": bool(gt_n and intersection / gt_n >= .1),
        "localized_50pct_coverage": bool(gt_n and intersection / gt_n >= .5),
    }


def runtime_box_metrics(findings, truth: np.ndarray) -> dict:
    """Supplement mask metrics with actual emitted boxes, in original coordinates.

    Target boxes enclose publisher-mask connected components. This is explicitly
    a derived box proxy; it never substitutes for original-mask localization.
    """
    height, width = truth.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(truth.astype(np.uint8), connectivity=8)
    target_boxes = stats[1:, :4].astype(int).tolist()
    predicted = []
    for finding in findings:
        if finding.bbox is None:
            continue
        box = np.asarray(finding.bbox, dtype=float)
        if box.shape != (4,) or not np.all(np.isfinite(box)):
            raise ValueError("Runtime finding has invalid bbox coordinates")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError("Runtime bbox extends beyond original analysis image")
        predicted.append({"bbox_xywh": box.tolist(), "score": finding.score, "kind": finding.kind})
    edges = []
    for g, (gx, gy, gw, gh) in enumerate(target_boxes):
        for p, finding in enumerate(predicted):
            px, py, pw, ph = finding["bbox_xywh"]
            intersection = max(0, min(gx + gw, px + pw) - max(gx, px)) * max(0, min(gy + gh, py + ph) - max(gy, py))
            iou = intersection / (gw * gh + pw * ph - intersection)
            edges.append((iou, g, p))
    matches = {}
    for cutoff in (.1, .25, .5):
        used_gt, used_pred = set(), set()
        for iou, g, p in sorted(edges, reverse=True):
            if iou >= cutoff and g not in used_gt and p not in used_pred:
                used_gt.add(g)
                used_pred.add(p)
        matches[str(cutoff)] = len(used_gt)
    return {"runtime_box_proxy": {
        "scope": "Actual emitted runtime boxes versus connected-component boxes derived from original publisher masks; supplementary to pixel metrics",
        "gt_boxes_xywh": target_boxes, "runtime_findings": predicted,
        "gt_regions": count - 1, "predicted_regions": len(predicted), "matches_at_iou": matches,
    }}


def aggregate(records: list[dict]) -> dict:
    positive = [row for row in records if row["defective"]]
    normal = [row for row in records if not row["defective"]]
    tp = sum(row["pixel_tp"] for row in records)
    pred_n = sum(row["pixel_predicted"] for row in records)
    gt_n = sum(row["pixel_target"] for row in records)
    regions = sum(row["gt_regions"] for row in records)
    pred_regions = sum(row["predicted_regions"] for row in records)
    result = {
        "views": len(records), "defective_views": len(positive), "normal_views": len(normal),
        "abstentions": sum(row.get("abstained", False) for row in records),
        "any_pixel_localized_views": sum(row["localized_any_pixel"] for row in positive),
        "any_pixel_localized_view_recall": sum(row["localized_any_pixel"] for row in positive) / max(1, len(positive)),
        "localized_view_recall_iou_0_1": sum((row["iou"] or 0) >= .1 for row in positive) / max(1, len(positive)),
        "localized_view_recall_10pct_coverage": sum(row["localized_10pct_coverage"] for row in positive) / max(1, len(positive)),
        "localized_view_recall_50pct_coverage": sum(row["localized_50pct_coverage"] for row in positive) / max(1, len(positive)),
        "normal_false_alarm_views": sum(row["flagged"] for row in normal),
        "normal_false_alarm_rate_all_views": sum(row["flagged"] for row in normal) / max(1, len(normal)),
        "false_location_defective_views": sum(row["flagged"] and not row["localized_any_pixel"] for row in positive),
        "foreground_pixel_recall": tp / max(1, gt_n), "foreground_pixel_precision": tp / max(1, pred_n),
        "micro_pixel_iou": tp / max(1, gt_n + pred_n - tp),
        "macro_defective_view_iou": float(np.mean([row["iou"] for row in positive])) if positive else None,
        "gt_regions": regions, "predicted_regions": pred_regions,
        "region_recall": {cutoff: sum(row["region_matches_at_iou"][cutoff] for row in records) / max(1, regions) for cutoff in ("0.1", "0.25", "0.5")},
        "region_precision": {cutoff: sum(row["region_matches_at_iou"][cutoff] for row in records) / max(1, pred_regions) for cutoff in ("0.1", "0.25", "0.5")},
    }
    if records and all("runtime_box_proxy" in row for row in records):
        boxes = [row["runtime_box_proxy"] for row in records]
        gt_boxes = sum(row["gt_regions"] for row in boxes)
        predicted_boxes = sum(row["predicted_regions"] for row in boxes)
        matches = {cutoff: sum(row["matches_at_iou"][cutoff] for row in boxes) for cutoff in ("0.1", "0.25", "0.5")}
        result["runtime_box_proxy"] = {
            "gt_regions": gt_boxes, "predicted_regions": predicted_boxes, "matches_at_iou": matches,
            "recall": {cutoff: value / max(1, gt_boxes) for cutoff, value in matches.items()},
            "precision": {cutoff: value / max(1, predicted_boxes) for cutoff, value in matches.items()},
        }
    return result


def save_visual_evidence(directory: Path, path: Path, frame: np.ndarray, truth: np.ndarray,
                         prediction: np.ndarray, metrics: dict) -> dict:
    """Save every defective view and normal false alarm, with separate real masks."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{path.parent.name}-{path.stem}"
    target = directory / f"{stem}-localization.jpg"
    mask_path = directory / f"{stem}-predicted-mask.png"
    gt_path = directory / f"{stem}-publisher-mask.png"
    gt, pred = truth.astype(bool), prediction.astype(bool)
    ground, model, difference = frame.copy(), frame.copy(), frame.copy()
    ground[gt] = (.25 * ground[gt] + .75 * np.array([255, 210, 0])).astype(np.uint8)
    model[pred] = (.25 * model[pred] + .75 * np.array([0, 180, 255])).astype(np.uint8)
    for mask, color in ((gt & pred, [70, 220, 70]), (gt & ~pred, [60, 60, 255]), (pred & ~gt, [0, 180, 255])):
        difference[mask] = (.15 * difference[mask] + .85 * np.array(color)).astype(np.uint8)
    h, w = frame.shape[:2]
    display_h = min(h, 900)
    display_w = round(w * display_h / h)
    panels = [cv2.resize(value, (display_w, display_h), interpolation=cv2.INTER_AREA)
              for value in (frame, ground, model, difference)]
    body = np.hstack(panels)
    canvas = np.full((display_h + 125, display_w * 4, 3), 22, np.uint8)
    canvas[85:85 + display_h] = body
    cv2.putText(canvas, f"{stem} | original real image / original-resolution masks", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1)
    labels = ["RAW INPUT", "PUBLISHER GT (cyan)", "MODEL MASK (amber)", "TP green / FN red / FP amber"]
    for i, label in enumerate(labels):
        cv2.putText(canvas, label, (i * display_w + 8, 67), cv2.FONT_HERSHEY_SIMPLEX, .44, (235, 235, 235), 1)
    cv2.putText(canvas, f"Pixel recall {metrics['pixel_recall'] or 0:.3f} | precision {metrics['pixel_precision']:.3f} | IoU {metrics['iou'] or 0:.3f}; displayed masks scaled, exact PNGs saved separately", (10, display_h + 111), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1)
    for dest, value in ((target, canvas), (mask_path, pred.astype(np.uint8) * 255), (gt_path, gt.astype(np.uint8) * 255)):
        if not cv2.imwrite(str(dest), value):
            raise IOError(f"Cannot write localization evidence {dest}")
    return {"visualization": str(target), "predicted_mask": str(mask_path), "publisher_mask": str(gt_path),
            "selection": "All defective views and all flagged normal views; no favorable-case filtering"}


def main() -> None:
    from edge_runtime.contracts import FrameContext
    from edge_runtime.profiles.factory import FactoryProfile
    from train_factory_surface import samples
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "calibration", "test"), default="test")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/factory/KolektorSDD")
    parser.add_argument("--visuals-dir", type=Path, help="Save all defective views and normal false alarms with separate original-resolution masks")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    profile = FactoryProfile({"defect_model_path": str(args.model.resolve()), "defect_backend": "auto", "confirmation_frames": 1})
    description = profile.describe()
    model = profile._defect_model
    group_names = description["model_metadata"]["split_items"][args.partition]
    captured = {}
    original = model.predict
    def capture(frame):
        answer = original(frame)
        captured["scores"] = answer[0]
        return answer
    model.predict = capture
    records = []
    for index, (path, mask, defective) in enumerate(samples(args.dataset, group_names)):
        frame = cv2.imread(str(path))
        captured.clear()
        result = profile.process(frame, FrameContext(index, time.perf_counter(), unit_id=str(path)))
        abstained = result.status not in {"fault", "review", "clear", "no_candidate"}
        prediction = np.zeros(mask.shape, dtype=bool) if abstained else retained_mask(captured["scores"], model.pixel_threshold, model.area_threshold, mask.shape,model.metadata.get('hysteresis_growth_threshold'))
        metrics = mask_metrics(prediction, mask > 0)
        box_metrics = runtime_box_metrics(result.findings, mask > 0)
        evidence = save_visual_evidence(args.visuals_dir, path, frame, mask > 0, prediction, metrics) if args.visuals_dir and (defective or metrics["flagged"]) else None
        if not abstained:
            assert metrics["flagged"] == (result.status in {"fault", "review"}), "Audit must match actual runtime finding policy"
        records.append({"path": str(path.relative_to(args.dataset)), "item": path.parent.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "mask_sha256": hashlib.sha256(path.with_name(path.stem + "_label.bmp").read_bytes()).hexdigest(),
                        "defective": defective, "abstained": abstained, "status": result.status, **metrics, **box_metrics,
                        "mask_shape_hw": list(prediction.shape),
                        "prediction_mask_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                        **({"visual_evidence": evidence} if evidence else {})})
    report = {"scope": "Accuracy-only original-mask audit; SDD1 test is exposed development data, not a blind holdout",
              "threshold_selection": "Uses frozen model policy; no selection is performed by this script",
              "localization_protocol": "Any-pixel hit retained for comparability; >=10%/>=50% annotated pixel coverage and one-to-one IoU>=0.1/0.25/0.5 region matches expose weak hits and oversized predictions. No background true negatives in pixel accuracy.",
              "supplementary_box_protocol": "v2 adds actual runtime bbox coordinates/one-to-one box IoU against publisher-mask component boxes; primary mask metrics unchanged",
              "partition": args.partition, "physical_items": group_names, "model": description,
              "metrics": aggregate(records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
