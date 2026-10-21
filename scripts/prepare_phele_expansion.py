"""Audit PHELE taxonomy, select original photographs and freeze diagnostic splits."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/construction/expansion-public-v1/phele"
CLASSES = ["peeling_paint", "mold_appearance"]
SOURCE_IDS = {5: 0, 9: 1}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def select():
    inventory = {row["name"]: row for row in json.loads((DATA/"zip-inventory.json").read_text())}
    class_names = (DATA/"original/PHELE/classes.txt").read_text().splitlines()
    assert class_names[5] == "Peeling paint" and class_names[9] == "Mold"
    rows, totals = [], Counter()
    for split in ("train", "val", "test"):
        positives, negatives = [], []
        for path in sorted((DATA/"original/PHELE"/split/"labels").glob("*.txt")):
            targets, all_labels = [], []
            for line in path.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("Expected real YOLO bounding boxes, not masks: "+str(path))
                original_id = int(fields[0])
                geometry = list(map(float, fields[1:]))
                if not 0 <= original_id < len(class_names) or not all(0 <= value <= 1 for value in geometry) or min(geometry[2:]) <= 0:
                    raise ValueError("Invalid source label")
                all_labels.append(original_id)
                totals[class_names[original_id]] += 1
                if original_id in SOURCE_IDS:
                    targets.append({"source_class_id": original_id, "class_id": SOURCE_IDS[original_id], "source_yolo": geometry})
            member = f"PHELE/{split}/images/{path.stem}.jpg"
            if member not in inventory:
                raise ValueError("Source image absent")
            row = {"member": member, "source_label": path.relative_to(ROOT).as_posix(), "publisher_split": split,
                   "source_label_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "split": "cal" if split == "val" else split, "boxes": targets, "source_all_classes": all_labels,
                   "source_scene_identity": "not_provided_by_publisher", "negative_scope": "target_labels_absent_not_verified_healthy_wall" if not targets else None}
            (positives if targets else negatives).append(row)
        # Deterministic filename-hash selection, before observing image content/model.
        negatives.sort(key=lambda row: hashlib.sha256(row["member"].encode()).hexdigest())
        rows.extend(positives+negatives[:20 if split == "train" else 5])
    audit = {"classes": CLASSES, "source_classes": class_names, "actual_source_class_5": "Peeling paint", "actual_source_class_9": "Mold",
             "publisher_record_definition": "Record describes Mold/dampness; actual archive class is Mold. Use mold_appearance only; no damp-only, species or moisture-quantity claim.",
             "source_box_counts_all_800_labels": dict(totals), "selection": rows,
             "selected_split_counts": dict(Counter(row["split"] for row in rows)),
             "no_owner_videos": True, "whole_archive_sha256_verified": False, "annotation_type": "bounding_boxes_only_no_masks",
             "scope": "Public original PHELE photos; excluded separate RGB-T dataset. Publisher split retained pending duplicate audit; no source/site identity is supplied."}
    write(DATA/"selection-audit-before-image-acquisition.json", audit)
    write(DATA/"selected-image-members.json", [row["member"] for row in rows])
    print(json.dumps({"selected": len(rows), "split_counts": audit["selected_split_counts"], "box_counts": dict(totals)}))


def prepare():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    selected = json.loads((DATA/"selection-audit-before-image-acquisition.json").read_text())
    rows, fingerprints, digests = [], [], []
    for original in selected["selection"]:
        path = DATA/"original"/original["member"]
        raw = cv2.imread(str(path))
        if raw is None:
            raise ValueError("Missing/unreadable real photo: "+str(path))
        gray = cv2.cvtColor(cv2.resize(raw, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        dct = cv2.dct(gray)[:8, :8]
        fingerprints.append(dct.flatten() > np.median(dct.flatten()[1:]))
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        height, width = raw.shape[:2]
        rows.append(dict(original, original_sha256=digests[-1], original_dimensions=[width, height]))
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(rows)):
        for j in range(i):
            if digests[i] == digests[j] or np.count_nonzero(fingerprints[i] != fingerprints[j]) <= 4:
                parent[find(i)] = find(j)
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    kept, removed = [], []
    priority = {"test": 0, "cal": 1, "train": 2}
    for indices in groups.values():
        order = sorted(indices, key=lambda i: (priority[rows[i]["split"]], rows[i]["member"]))
        keep = order[0]
        rows[keep]["image_duplicate_group"] = hashlib.sha256("|".join(sorted(digests[i] for i in indices)).encode()).hexdigest()
        kept.append(rows[keep])
        removed.extend({"member": rows[i]["member"], "reason": "exact_or_phash_distance_le4_duplicate_group", "retained_member": rows[keep]["member"]} for i in order[1:])
    destination = DATA/"prepared"
    for row in kept:
        raw = cv2.imread(str(DATA/"original"/row["member"]))
        height, width = raw.shape[:2]
        factor = 640/max(height, width)
        resized = cv2.resize(raw, (round(width*factor), round(height*factor)), interpolation=cv2.INTER_AREA)
        pad_x, pad_y = (640-resized.shape[1])//2, (640-resized.shape[0])//2
        raw = np.full((640, 640, 3), 114, np.uint8)
        raw[pad_y:pad_y+resized.shape[0], pad_x:pad_x+resized.shape[1]] = resized
        for box in row["boxes"]:
            cx, cy, bw, bh = box["source_yolo"]
            box["prepared_yolo"] = [(cx*width*factor+pad_x)/640, (cy*height*factor+pad_y)/640, bw*width*factor/640, bh*height*factor/640]
        name = Path(row["member"]).stem
        image = destination/row["split"]/"images"/(name+".jpg")
        label = destination/row["split"]/"labels"/(name+".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image), raw, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise ValueError("Image write failed")
        label.write_text("".join(str(b["class_id"])+" "+" ".join(map(str, b["prepared_yolo"]))+"\n" for b in row["boxes"]))
        row.update(prepared_image=image.relative_to(ROOT).as_posix(), prepared_label=label.relative_to(ROOT).as_posix(),
                   prepared_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), prepared_label_sha256=hashlib.sha256(label.read_bytes()).hexdigest())
    manifest = {"classes": CLASSES, "records": sorted(kept, key=lambda row: (row["split"], row["member"])), "duplicates_removed": removed,
                "split_counts": dict(Counter(row["split"] for row in kept)), "private_owner_videos_used": False,
                "site_independence_verified": False, "model_release_status": "research_only",
                "split_policy": "Publisher splits, exact/perceptual duplicate groups removed with TEST then CAL retention priority before training. Physical site identity unavailable.",
                "test_role": "Once-only photo diagnostic after final checkpoint/threshold freeze; not independent site accuracy",
                "source_selection_sha256": hashlib.sha256((DATA/"selection-audit-before-image-acquisition.json").read_bytes()).hexdigest()}
    write(DATA/"prepared-manifest.json", manifest)
    (destination/"dataset.yaml").write_text("path: "+destination.resolve().as_posix()+"\ntrain: train/images\nval: cal/images\ntest: test/images\nnames:\n  0: peeling_paint\n  1: mold_appearance\n", encoding="utf8")
    print(json.dumps({"counts": manifest["split_counts"], "duplicates_removed": len(removed)}))


def contact_sheet():
    import cv2
    import numpy as np
    manifest = json.loads((DATA/"prepared-manifest.json").read_text())
    candidates = [row for row in manifest["records"] if row["split"] == "train" and row["boxes"]]
    candidates.sort(key=lambda row: (not any(box["class_id"] == 1 for box in row["boxes"]), row["member"]))
    cells = []
    for row in candidates[:32]:
        image = cv2.imread(str(ROOT/row["prepared_image"]))
        for box in row["boxes"]:
            cx, cy, width, height = box["prepared_yolo"]
            cv2.rectangle(image, (round((cx-width/2)*640), round((cy-height/2)*640)), (round((cx+width/2)*640), round((cy+height/2)*640)), (0, 210, 255) if box["class_id"] else (60, 210, 60), 3)
        image = cv2.resize(image, (250, 250))
        cell = np.full((280, 250, 3), 24, np.uint8)
        cell[:250] = image
        cv2.putText(cell, Path(row["member"]).stem, (4, 269), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1)
        cells.append(cell)
    while len(cells)%4:
        cells.append(np.zeros((280, 250, 3), np.uint8))
    sheet = np.vstack([np.hstack(cells[i:i+4]) for i in range(0, len(cells), 4)])
    target = ROOT/"output/evaluation/construction-expansion-phele-v1/train-only-source-audit.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), sheet)
    write(target.with_suffix(".json"), {"scope": "TRAIN-only source/annotation visual audit; yellowMold greenPeeling; not model predictions", "members": [row["member"] for row in candidates[:32]]})
    print(target)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "prepare", "contact_sheet"])
    args = parser.parse_args()
    {"select": select, "prepare": prepare, "contact_sheet": contact_sheet}[args.stage]()
