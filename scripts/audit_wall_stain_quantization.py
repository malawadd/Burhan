"""Explain frozen-point IoU count drift between float and outward-integer boxes."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_task_detector import ConstructionTaskDetector
from scripts.audit_construction_task_serving import iou

OUT = ROOT/"output/evaluation/construction-expansion-wall-stain-v1/verification-v1"


def matches(predictions, truth):
    used, result = set(), []
    for prediction in predictions:
        overlap, index = max(((iou(prediction, box), index) for index, box in enumerate(truth) if index not in used), default=(0, -1))
        accepted = overlap >= .5
        if accepted:
            used.add(index)
        result.append({"best_iou": overlap, "truth_index": index, "matched": accepted})
    return result, len(used)


def main():
    cv2.setNumThreads(1)
    manifest_path = ROOT/"models/construction/expansion-wall-stain-v1/ssdlite320/manifest.json"
    adapter = ConstructionTaskDetector(manifest_path)
    if adapter.spec["confidence"] != .25:
        raise ValueError("Frozen point changed")
    data = json.loads((ROOT/"datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json").read_text())
    float_tp, integer_tp, total_predictions, total_truth = 0, 0, 0, 0
    differences = []
    for row in [row for row in data["records"] if row["split"] == "test"]:
        path = ROOT/row["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError("Frozen pixels changed")
        image = cv2.imread(str(path))
        if image.shape[:2] != (640, 640):
            raise ValueError("This audit requires native 640-square prepared frames")
        rgb = cv2.cvtColor(cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)/255
        outputs = dict(zip(adapter.output_names, adapter.session.run(None, {adapter.input_name: tensor})))
        selected = np.flatnonzero((outputs["labels"] == 1) & (outputs["scores"] >= .25) & (outputs["scores"] <= 1))
        selected = selected[np.argsort(outputs["scores"][selected])[::-1][:300]]
        float_boxes, integer_boxes, scores = [], [], []
        for index in selected:
            xyxy = np.clip(outputs["boxes"][index]*2, 0, 640)
            x1, y1, x2, y2 = map(float, xyxy)
            if min(x2-x1, y2-y1) < 1:
                continue
            float_boxes.append([x1, y1, x2-x1, y2-y1])
            ix1, iy1 = np.floor(xyxy[:2]).astype(int)
            ix2, iy2 = np.ceil(xyxy[2:]).astype(int)
            integer_boxes.append([int(ix1), int(iy1), int(ix2-ix1), int(iy2-iy1)])
            scores.append(float(outputs["scores"][index]))
        truth = [box["bbox"] for box in row["boxes"]]
        floating, ft = matches(float_boxes, truth)
        integers, it = matches(integer_boxes, truth)
        float_tp += ft
        integer_tp += it
        total_predictions += len(float_boxes)
        total_truth += len(truth)
        for index, (fm, im) in enumerate(zip(floating, integers)):
            if fm["matched"] != im["matched"] or fm["matched"] and fm["truth_index"] != im["truth_index"]:
                differences.append({"image": row["image"], "source_sha256": row["source_sha256"], "prediction_score": scores[index],
                    "float_bbox": float_boxes[index], "integer_bbox": integer_boxes[index], "float_match": fm, "integer_match": im,
                    "ground_truth_boxes": truth})
    result = {"model_sha256": adapter.spec["model_sha256"], "confidence_unchanged": .25, "box_match_iou": .5,
        "float_box_counts": {"tp": float_tp, "fp": total_predictions-float_tp, "fn": total_truth-float_tp},
        "outward_integer_box_counts": {"tp": integer_tp, "fp": total_predictions-integer_tp, "fn": total_truth-integer_tp},
        "differences": differences, "scope": "Same frozen ONNX predictions and fixed thresholds; only box coordinate quantization differs. No threshold or model selection."}
    target = OUT/"quantization-drift-audit.json"
    target.write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
