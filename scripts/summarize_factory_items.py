"""Supplementary offline item grouping of frozen view audits; no inference."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    raw = path.read_bytes()
    audit = json.loads(raw)
    records = audit["records"]
    items = []
    for item in audit["physical_items"]:
        rows = [r for r in records if r["item"] == item]
        defects = [r for r in rows if r["defective"]]
        localized = [r for r in defects if not r["abstained"] and r["iou"] >= .1]
        items.append({"item": item, "views": len(rows), "defective_views": len(defects),
                      "meaningfully_localized_defective_views": len(localized),
                      "any_flag": any(r["flagged"] for r in rows),
                      "any_meaningful_defect_localization": bool(localized),
                      "all_defective_views_localized": bool(defects) and len(localized) == len(defects),
                      "abstained_views": sum(r["abstained"] for r in rows)})
    normal = [r for r in items if not r["defective_views"]]
    defective = [r for r in items if r["defective_views"]]
    return {"source": str(path.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "model_sha256": audit["model"]["model_sha256"],
            "items": items, "defective_items": len(defective), "fully_normal_items": len(normal),
            "defective_items_with_any_flag": sum(r["any_flag"] for r in defective),
            "defective_items_with_meaningful_localization": sum(r["any_meaningful_defect_localization"] for r in defective),
            "defective_items_with_all_defective_views_localized": sum(r["all_defective_views_localized"] for r in defective),
            "fully_normal_item_alarm_rate": sum(r["any_flag"] for r in normal) / len(normal) if normal else None,
            "scope": "Posthoc offline grouping, not runtime item coverage or an acceptance rule. Any flag can be in the wrong view/location. Primary view/mask metrics remain unchanged; no fully normal items means item false-alarm rate is unmeasured."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing analysis; choose a new output version")
    result = [summarize(path) for path in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result:
        print(json.dumps({k: v for k, v in row.items() if k != "items"}))
