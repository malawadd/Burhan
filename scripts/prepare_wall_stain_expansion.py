"""Add source-faithful leakage/stain research data, excluding protected diagnostics.

Existing audited MBDD TRAIN/CAL photographs retain their roles. A new fixed
120-photo image test is selected before acquiring/viewing those photos. Physical
site independence remains unknown; other-defect negatives are not healthy walls.
"""
from __future__ import annotations

import concurrent.futures
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_mbdd_wall_training as source

OUT = ROOT / "datasets/construction/expansion-public-v1/wall-stain"
BASE = ROOT / "datasets/construction/finishes-expansion"
SEED = "wall-stain-expansion-20261021-v1:"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    data = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != data:
        raise ValueError("Preserve frozen wall-stain data: " + str(path))
    path.write_text(data, encoding="utf-8")


def valid(row):
    w, h = row["frame_size_wh"]
    return w > 0 and h > 0 and all(np.isfinite(b["source_xyxy"]).all() and
        0 <= b["source_xyxy"][0] < b["source_xyxy"][2] <= w and
        0 <= b["source_xyxy"][1] < b["source_xyxy"][3] <= h for b in row["boxes"])


def main():
    cv2.setNumThreads(1)
    inv_path = BASE / "source-records/mbdd2025-publisher-inventory.json"
    old_path = BASE / "mbdd-wall-five-class-v1/asset-manifest.json"
    diag_path = BASE / "mbdd2025-wall-photos/asset-manifest.json"
    annotation_path = BASE / "mbdd-wall-five-class-v1/original-annotation-range.bin"
    old_protocol = json.loads((old_path.parent / "protocol-before-source-labels-pixels.json").read_text())
    inventory = json.loads(inv_path.read_text())
    old = json.loads(old_path.read_text())
    diagnostic = json.loads(diag_path.read_text())
    forbidden_blocks = {source.block_id(row["annotation_member"]) for row in diagnostic["records"]}
    exposed = {row["source_name"] for row in old["records"]}
    entries = {entry["name"]: entry for entry in inventory["entries"]}
    offsets = sorted({entry["header_offset"] for entry in entries.values()})
    annotations = annotation_path.read_bytes()
    if sha(annotation_path) != "2c8f63ac5569e8fc7f495e6c2de76a648457dd7e071441d502340a853fc31a73":
        raise ValueError("Frozen original annotation range changed")
    candidates = []
    for entry in entries.values():
        if "/Annotations/" not in entry["name"] or not entry["name"].endswith(".xml"):
            continue
        raw = source.unpack(entry, annotations, old_protocol["annotation_range"]["start"])
        row = source.parsed(entry, raw)
        if row["capture_block_200"] in forbidden_blocks or row["source_name"] in exposed or not valid(row) or not row["boxes"]:
            continue
        row["image_member"] = "MBDD2025/JPEGImages/" + row["source_name"]
        if row["image_member"] in entries:
            candidates.append(row)
    order = lambda row: hashlib.sha256((SEED + row["source_name"]).encode()).hexdigest()
    positive = sorted([r for r in candidates if any(b["label"] == "leakage" for b in r["boxes"])], key=order)[:60]
    negative = sorted([r for r in candidates if all(b["label"] != "leakage" for b in r["boxes"])], key=order)[:60]
    if len(positive) != 60 or len(negative) != 60:
        raise ValueError("Insufficient fixed image-test positives/other-defect negatives")
    selection = [{**row, "split": "test"} for row in positive+negative]
    protocol = {"source": "https://zenodo.org/records/15622584", "license": "CC BY 4.0",
        "script_sha256": sha(Path(__file__)), "source_inventory_sha256": sha(inv_path),
        "old_manifest_sha256": sha(old_path), "protected_diagnostics_sha256": sha(diag_path),
        "annotation_range_sha256": sha(annotation_path), "seed": SEED,
        "source_mapping": {"leakage": "damp_stained_appearance"},
        "semantics": "Publisher leakage means visible stain/leakage appearance; not active water, humidity quantity or cause.",
        "train_cal_policy": "Retain prior eligible valid TRAIN/CAL photo roles; only source leakage boxes are positives.",
        "test_policy": "First60 leakage-positive plus60 other-defect target-negative unused photos by fixed source-name hash; no owner/protected diagnostic blocks.",
        "independence": "Image-group research only; old training/calibration exposure known; physical sites unknown and same block may cross image-test boundary. No healthy-wall FPR claim.",
        "private_owner_videos_used": False, "selection": selection}
    write_once(OUT / "protocol-before-new-photo-acquisition.json", protocol)
    url = inventory["publisher"]["files"][0]["links"]["self"]

    def acquire(row):
        entry = entries[row["image_member"]]
        target = OUT / "original" / row["source_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raw = target.read_bytes()
        else:
            start = entry["header_offset"]
            end = min(offset for offset in offsets if offset > start)-1
            raw = source.unpack(entry, source.byte_range(url, start, end), start)
            target.write_bytes(raw)
        import zlib
        if len(raw) != entry["bytes"] or f"{zlib.crc32(raw):08x}" != entry["crc32"]:
            raise ValueError("Wall stain original CRC/size mismatch")
        return {**row, "original": target.relative_to(ROOT).as_posix(), "original_sha256": sha(target)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        acquired = []
        for row in executor.map(acquire, selection):
            acquired.append(row)
            if len(acquired) % 20 == 0:
                print("CRC_VERIFIED_NEW_WALL_PHOTOS", len(acquired), flush=True)
    rows = []
    for row in old["records"]:
        if row.get("status") != "eligible_pending_independent_review" or not valid(row) or row["capture_block_200"] in forbidden_blocks:
            continue
        path = old_path.parent / row["image"]
        rows.append({**row, "original": path.relative_to(ROOT).as_posix(), "original_sha256": sha(path)})
    rows += acquired
    # Quarantine new tests that resemble old TRAIN/CAL views. Do not backfill by
    # inspecting model results. This is not proof of physical site independence.
    hashes = {}
    for row in rows:
        image = cv2.imread(str(ROOT / row["original"]))
        if image is None or list(image.shape[1::-1]) != row["frame_size_wh"]:
            raise ValueError("Source image/XML geometry mismatch")
        hashes[row["source_name"]] = source.visual_hash(image)
    quarantined = []
    for row in acquired:
        p = hashes[row["source_name"]]
        for other in rows:
            if other["split"] == "test":
                continue
            distance = sum(a != b for a, b in zip(p, hashes[other["source_name"]]))
            if row["original_sha256"] == other["original_sha256"] or distance <= 6:
                quarantined.append({"test": row["source_name"], "exposed": other["source_name"], "phash_distance": distance})
                break
    excluded = {row["test"] for row in quarantined}
    records = []
    for row in rows:
        if row["source_name"] in excluded:
            continue
        image = cv2.imread(str(ROOT / row["original"]))
        h, w = image.shape[:2]
        ratio = min(640/w, 640/h)
        nw, nh = round(w*ratio), round(h*ratio)
        px, py = (640-nw)//2, (640-nh)//2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[py:py+nh, px:px+nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        target = OUT / "prepared/images" / row["split"] / row["source_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = []
        for box in row["boxes"]:
            if box["label"] != "leakage":
                continue
            x1, y1, x2, y2 = box["source_xyxy"]
            boxes.append({"class_id": 0, "bbox": [x1*nw/w+px, y1*nh/h+py, (x2-x1)*nw/w, (y2-y1)*nh/h], "source_label": "leakage"})
        label_path = OUT / "prepared/labels" / row["split"] / (target.stem+".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(f"0 {(b['bbox'][0]+b['bbox'][2]/2)/640:.8f} {(b['bbox'][1]+b['bbox'][3]/2)/640:.8f} {b['bbox'][2]/640:.8f} {b['bbox'][3]/640:.8f}\n" for b in boxes))
        records.append({"id": target.stem, "image": target.relative_to(ROOT).as_posix(), "source_sha256": sha(target),
            "split": row["split"], "boxes": boxes, "original": row["original"], "original_sha256": row["original_sha256"],
            "capture_block_200": row["capture_block_200"], "source_size_wh": [w, h]})
    yaml = OUT / "prepared/dataset.yaml"
    yaml.write_text(f"path: {yaml.parent.resolve().as_posix()}\ntrain: images/train\nval: images/calibration\nnames:\n  0: damp_stained_appearance\n")
    manifest = {"classes": ["damp_stained_appearance"], "source": protocol["source"], "license": protocol["license"],
        "records": records, "split_counts": dict(Counter(row["split"] for row in records)),
        "positive_images": dict(Counter(row["split"] for row in records if row["boxes"])),
        "quarantined_tests": quarantined, "protocol_sha256": sha(OUT / "protocol-before-new-photo-acquisition.json"),
        "private_videos_used": False, "independence": protocol["independence"], "validation_status": "research_only"}
    write_once(OUT / "prepared/manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("split_counts", "positive_images", "validation_status")}), flush=True)


if __name__ == "__main__":
    main()
