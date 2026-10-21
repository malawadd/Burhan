"""Render every frozen PHELE TEST photograph using saved predictions, no tuning."""
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"output/evaluation/construction-expansion-phele-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["yolov8n", "ssdlite320"], default="yolov8n")
    architecture = parser.parse_args().model
    report = json.loads((OUT/f"{architecture}-results.json").read_text())
    manifest = json.loads((OUT/"training-manifest.json").read_text())
    rows = [row for row in manifest["records"] if row["split"] == "test"]
    if architecture == "yolov8n":
        predictions = json.loads((OUT/"yolov8n-predictions.json").read_text())
    else:
        serving = json.loads((OUT/"ssdlite320-served-box-evaluation.json").read_text())
        baseline = report["before_test_fixed_point_0_5"]
        if baseline["tp"]+baseline["fp"] != 0:
            raise ValueError("Cannot reconstruct nonempty unsaved baseline predictions")
        # The measured baseline has zero predictions at this exact point.
        predictions = {"test_members": [r["member"] for r in serving["predictions"]], "before_test": [[] for _ in rows],
                       "after_test": [[{"class_id": manifest["classes"].index(item["label"]), "bbox": item["bbox"], "score": item["score"]}
                                       for item in r["predictions"]] for r in serving["predictions"]]}
    if [row["original_member"] for row in rows] != predictions["test_members"]:
        raise ValueError("Frozen TEST prediction order changed")
    classes = manifest["classes"]
    destination = OUT/("test-photo-comparisons" if architecture == "yolov8n" else "ssdlite320-test-photo-comparisons")
    destination.mkdir(parents=True, exist_ok=True)
    threshold = report["after_calibration_selected"]["threshold"]
    lines = ["# PHELE: all frozen TEST photo comparisons", "", "These are actual publisher photographs, not videos or generated scenes. Green boxes are publisher ground truth; red boxes are model predictions. Before is the fresh semantic head at 0.5. After uses the CAL-only chosen point shown in each panel. Physical site independence is unverified; these images are not a qualified deployment test.", ""]
    for index, row in enumerate(rows):
        source = cv2.imread(str(ROOT/row["image"]))
        panels = []
        for title, boxes, point, color in [("Original photograph", [], 0, (255, 255, 255)),
                ("Publisher boxes", row["boxes"], 0, (60, 210, 60)),
                ("Before fresh head @0.50", predictions["before_test"][index], .5, (40, 70, 240)),
                (f"After final fit @{threshold:.2f}", predictions["after_test"][index], threshold, (40, 70, 240))]:
            image = source.copy()
            for box in boxes:
                if box.get("score", 1) < point:
                    continue
                x, y, width, height = map(round, box["bbox"])
                cv2.rectangle(image, (x, y), (x+width, y+height), color, 2)
                label = classes[box["class_id"]]
                if "score" in box:
                    label += f" {box['score']:.2f}"
                cv2.putText(image, label, (max(0, x), max(15, y-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            panel = np.full((355, 320, 3), 24, np.uint8)
            panel[:320] = cv2.resize(image, (320, 320))
            cv2.putText(panel, title, (5, 341), cv2.FONT_HERSHEY_SIMPLEX, .43, (245, 245, 245), 1)
            panels.append(panel)
        name = Path(row["original_member"]).stem
        path = destination/(name+".png")
        cv2.imwrite(str(path), np.hstack(panels))
        lines.extend([f"## {name}", "", f"![Original, publisher annotation, initialized head, final model]({name}.png)", ""])
    (destination/"README.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print(f"Rendered all {len(rows)} actual TEST photos to {destination}")


if __name__ == "__main__":
    main()
