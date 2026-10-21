"""Versioned wall-stain notes and predeclared fixed-.5 counts, no new inference."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_construction_task_serving import iou

OUT = ROOT/"output/evaluation/construction-expansion-wall-stain-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = OUT/"ssdlite320-results.json"
    digest = sha(original)
    result = json.loads(original.read_text())
    serving = json.loads((OUT/"verification-v1/served-frozen-point-025.json").read_text())
    dataset = json.loads((ROOT/"datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json").read_text())
    truth = {row["image"]: [box["bbox"] for box in row["boxes"]] for row in dataset["records"] if row["split"] == "test"}
    counts = {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "positive_images": 0, "negative_images": 0, "flagged_negative_images": 0}
    for row in serving["records"]:
        expected = truth[row["image"]]
        chosen = sorted([item for item in row["predictions"] if item["score"] >= .5], key=lambda item: -item["score"])
        counts["positive_images" if expected else "negative_images"] += 1
        counts["flagged_negative_images"] += int(not expected and bool(chosen))
        matched = set()
        for prediction in chosen:
            overlap, index = max(((iou(prediction["bbox"], box), index) for index, box in enumerate(expected) if index not in matched), default=(0, -1))
            if overlap >= .5:
                matched.add(index)
                counts["true_positives"] += 1
            else:
                counts["false_positives"] += 1
        counts["false_negatives"] += len(expected)-len(matched)
    counts["precision"] = counts["true_positives"]/max(1, counts["true_positives"]+counts["false_positives"])
    counts["recall"] = counts["true_positives"]/max(1, counts["true_positives"]+counts["false_negatives"])
    note = {"frozen_result": original.relative_to(ROOT).as_posix(), "frozen_result_sha256": digest,
            "original_release_blocker": result["release_blocker"],
            "corrected_wall_specific_blocker": "Wall/building physical-site identity is unverified; image grouping does not establish site independence; 47 positive and 51 other-defect target-negative TEST photos are not qualified healthy-wall validation, and the measured precision/recall targets fail.",
            "correction_scope": "The physical-connection wording was inherited from a generic fastener helper. This supplemental note corrects prose only, without modifying frozen results, data, model, threshold or evaluation.",
            "serving_confidence_unchanged": .25, "serving_selected_point": serving["classes"],
            "predeclared_comparison_point": {"confidence": .5, "basis": "Fixed .5 before/after comparisons already exist in the frozen original result; this is not a newly selected operating point.",
                "method": "Filter the already saved actual .25 serving detections at .5; no new inference or threshold selection", "metrics": counts},
            "release_pass": False}
    if sha(original) != digest:
        raise ValueError("Frozen result changed during supplemental audit")
    path = OUT/"verification-v1/supplemental-notes-v1.json"
    if path.exists():
        raise ValueError("Preserve prior supplemental notes")
    path.write_text(json.dumps(note, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"frozen_result_sha256": digest, "fixed_05_serving": counts}), flush=True)


if __name__ == "__main__":
    main()
