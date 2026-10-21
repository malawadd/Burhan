"""Install the pinned photo research specialist without changing a station default."""
import hashlib
import json
import pprint
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(relative, digest):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Changed or outside evidence: {relative}")


def main():
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    proposal = json.loads((ROOT / "output/evaluation/construction-finishes-integration-proposal.json").read_text(encoding="utf-8"))
    verify(proposal["actual_report"], proposal["actual_report_sha256"])
    variants = proposal["model_variants"]
    if list(variants) != ["tile_surface_anomaly_research"]:
        raise ValueError("Unexpected research variant")
    spec = {**variants["tile_surface_anomaly_research"], "experimental": True,
            "diagnostic_label": "RESEARCH - WEAK TILE LOCALIZATION"}
    verify("models/construction/" + spec["file"], spec["sha256"])
    verify("edge_runtime/profiles/construction_tile_adapter.py", spec["wrapper_sha256"])
    verify("edge_runtime/profiles/construction_tile.py", spec["fitted_source_sha256"])
    if not spec["diagnostic_only"] or spec["classes"] != ["tile_surface_anomaly"]:
        raise ValueError("Research scope changed")
    source_path = ROOT / "edge_runtime/profiles/construction_model.py"
    source = source_path.read_text(encoding="utf-8")
    existing = DAMAGE_VARIANTS.get("tile_surface_anomaly_research")
    if existing is not None and existing != spec:
        raise ValueError("Refusing changed registration")
    if existing is None:
        marker = "DAMAGE_VARIANTS = {\n"
        if source.count(marker) != 1:
            raise ValueError("Unexpected registry structure")
        entry = "    'tile_surface_anomaly_research': " + pprint.pformat(spec, width=110).replace("\n", "\n    ") + ",\n"
        source = source.replace(marker, marker + entry, 1)
        compile(source, str(source_path), "exec")
    catalog_path = ROOT / "config/construction-capabilities.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    known = {row["id"]: row for row in catalog["categories"]}
    for row in proposal["category_proposals"]:
        if row["id"] in known and known[row["id"]] != row:
            raise ValueError("Refusing changed capability")
        if row["id"] not in known:
            catalog["categories"].append(row)
    preset_path = ROOT / "config/public-presets.json"
    presets = json.loads(preset_path.read_text(encoding="utf-8"))
    known = {row["label"]: row for row in presets}
    if len(proposal["photo_presets"]) != 6:
        raise ValueError("All six fixed cases are required")
    for item in proposal["photo_presets"]:
        verify(item["source"], item["source_sha256"])
        verify(item["comparison_image"], item["comparison_image_sha256"])
        row = {"label": item["label"], "profile": "construction", "path": item["source"],
               "input_fps": 2, "deadline_ms": 1500, "preview_fps": 4, "video_mode": "every_frame",
               "purpose": "photo_evidence", "media_type": "still_photo",
               "source_sha256": item["source_sha256"], "comparison_image": item["comparison_image"],
               "independent_accuracy_credit": False, "profile_options": item["profile_options"],
               "note": "Real manufacturing PHOTO research; generic surface anomaly, not a semantic defect subtype or installed ceramic inspection. All117 photo test results retained: crack region IoU.50 0/17; good3/33 flags. Raw-threshold pixel metrics are separate from serving area filtering. No video or15FPS claim."}
        if row["label"] in known and known[row["label"]] != row:
            raise ValueError("Refusing changed preset")
        if row["label"] not in known:
            presets.append(row)
    source_path.write_text(source, encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    preset_path.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Registered tile research, six real PHOTO cases and explicit unsupported finish checks; defaults preserved.")


if __name__ == "__main__":
    main()
