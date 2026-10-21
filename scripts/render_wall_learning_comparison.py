"""Render fixed CAL positives/negatives, with source boxes and both predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "calibration-predictions.json").read_text())
    manifest = json.loads((ROOT / "datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json").read_text())
    rows = {r["id"]: r for r in manifest["records"] if r["split"] == "calibration"}
    ids = sorted(report["ids"], key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    selected = [i for i in ids if rows[i]["boxes"]][:3] + [i for i in ids if not rows[i]["boxes"]][:3]
    panels = []
    for id_ in selected:
        row = rows[id_]
        image = cv2.imread(str(ROOT / row["image"]))
        idx = report["ids"].index(id_)
        columns = []
        for title, annotations, truth in [("Publisher stain boxes", row["boxes"], True),
            ("Before / fixed 0.5", report["before"][idx], False),
            ("Corrected loss / fixed 0.5", report["after"][idx], False)]:
            view = image.copy()
            for b in annotations:
                if not truth and b["score"] < .5:
                    continue
                x, y, w, h = [round(v) for v in b["bbox"]]
                cv2.rectangle(view, (x,y), (x+w,y+h), (70,230,70) if truth else (40,180,255), 2)
                if not truth:
                    cv2.putText(view, f"{b['score']:.2f}", (x,max(15,y-3)), cv2.FONT_HERSHEY_SIMPLEX, .5, (40,180,255), 1)
            view = cv2.resize(view, (400,400))
            cv2.rectangle(view, (0,0), (400,45), (15,15,15), -1)
            cv2.putText(view, title, (8,18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
            cv2.putText(view, id_, (8,36), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            columns.append(view)
        panels.append(np.hstack(columns))
    cv2.imwrite(str(args.run / "fixed-calibration-comparisons.jpg"), np.vstack(panels))
    (args.run / "visual-selection.json").write_text(json.dumps({"selection": "First3 positive and3 negative CAL originals by SHA256(id), no prediction selection", "ids": selected, "point": .5}, indent=2))


if __name__ == "__main__":
    main()
