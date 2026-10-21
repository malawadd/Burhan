"""Freeze a source-image-separated inner split of official SDD2 TRAIN only."""
from pathlib import Path
import hashlib
import json
import random
import re
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/factory/KolektorSDD2"
SEED = 20261020


def main():
    cv2.setNumThreads(1)
    rows = []
    groups = {}
    for path in sorted((DATASET / "train").glob("*.png")):
        if not re.fullmatch(r"\d+\.png", path.name):
            continue
        label = path.with_name(path.stem + "_GT.png")
        annotation = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE)
        if annotation is None:
            raise RuntimeError(f"Missing/corrupt training mask {label}")
        image_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {"image": f"train/{path.name}", "mask": f"train/{label.name}",
               "defective": bool(annotation.any()), "image_sha256": image_sha,
               "mask_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
               "annotation_shape": list(annotation.shape)}
        rows.append(row)
        groups.setdefault(image_sha, []).append(row)
    if len(rows) != 2331 or sum(row["defective"] for row in rows) != 246:
        raise RuntimeError("Unexpected official canonical training inventory; inspect before splitting")
    for group in groups.values():
        if len({row["mask_sha256"] for row in group}) != 1:
            raise RuntimeError("Identical images have conflicting labels; resolve before splitting")
    rng = random.Random(SEED)
    selected = set()
    for defective in (False, True):
        keys = sorted(key for key, group in groups.items() if group[0]["defective"] == defective)
        rng.shuffle(keys)
        selected.update(keys[:round(len(keys) * .2)])
    fit = [row for row in rows if row["image_sha256"] not in selected]
    calibration = [row for row in rows if row["image_sha256"] in selected]
    duplicates = []
    for name, canonical in [("10301 (copy).png", "10301.png"), ("10301_GT (copy).png", "10301_GT.png")]:
        copy = DATASET / "train" / name
        if copy.exists():
            first = hashlib.sha256(copy.read_bytes()).hexdigest()
            second = hashlib.sha256((copy.parent / canonical).read_bytes()).hexdigest()
            if first != second:
                raise RuntimeError("Expected release copy is no longer identical")
            duplicates.append({"excluded": f"train/{name}", "identical_to": f"train/{canonical}", "sha256": first})
    def counts(partition):
        return {"images": len(partition), "defective": sum(row["defective"] for row in partition),
                "normal": sum(not row["defective"] for row in partition)}
    report = {"dataset": "KolektorSDD2", "seed": SEED,
              "source_manifest_sha256": hashlib.sha256((DATASET / "source-manifest.json").read_bytes()).hexdigest(),
              "protocol": "Canonical numeric PNG files from official TRAIN only. Group identical image SHA-256s, sort then seeded shuffle independently for each class; round20%ofuniquegroups allocated to inner calibration. Remaining groups fit. No official test contents opened.",
              "physical_item_independence": "Unknown: original release supplies no verified physical-item/batch IDs. Source-image/hash disjointness is not item disjointness.",
              "official_test_opened": False, "excluded_identical_copy_files": duplicates,
              "counts": {"fit": counts(fit), "calibration": counts(calibration)},
              "fit": fit, "calibration": calibration}
    path = DATASET / "train-inner-split-before-fit.json"
    if path.exists():
        # A split is immutable; rerunning verifies the same canonical records.
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in ("fit", "calibration", "seed", "protocol"):
            if previous[key] != report[key]:
                raise RuntimeError("Existing frozen split differs; never silently replace it")
    else:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "counts": report["counts"], "excluded_copy_files": duplicates}, indent=2))


if __name__ == "__main__":
    main()
