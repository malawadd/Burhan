"""Expose the six completed fixed-pair videos, always as diagnostics."""
import hashlib
import json
from pathlib import Path
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Outside evidence")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_exact(existing, row, key):
    previous = next((r for r in existing if r[key] == row[key]), None)
    if previous is not None and previous != row:
        raise ValueError("Refusing to overwrite changed diagnostic")
    if previous is None:
        existing.append(row)


def main():
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    folder = ROOT / "output/evaluation/construction-pixel-videos"
    pairs = json.loads((folder / "paired-summary.json").read_text(encoding="utf-8"))
    videos_path = ROOT / "config/construction-videos.json"
    presets_path = ROOT / "config/public-presets.json"
    videos = json.loads(videos_path.read_text(encoding="utf-8"))
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    variants = {}
    for tag, cases in pairs.items():
        if [r["sequence"] for r in cases] != ["crack_89", "crack_129", "ros_crack_21"]:
            raise ValueError("Complete fixed cases required")
        variant = "pixel_fixed50_" + tag.lower()
        for case in cases:
            if sha(case["video"]) != case["video_sha256"] or sha(case["report"]) != case["report_sha256"]:
                raise ValueError("Changed completed evidence")
            report = json.loads((ROOT / case["report"]).read_text(encoding="utf-8"))
            if (report["fixed_probability_threshold"] != .5 or report["independent_accuracy_credit"]
                    or report["selection_protocol_sha256"] != "4cf2ff2193c0c2b80504c357cee9eb780388e73922727ee69520e88bbca22ae9"):
                raise ValueError("Fixed-pair provenance differs")
            model = report["model"]
            model_path = Path(model["model_path"]).resolve()
            relative = model_path.relative_to(ROOT / "models/construction").as_posix()
            if sha(model_path) != case["model_sha256"]:
                raise ValueError("Model changed")
            metadata = json.loads(model_path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
            variants[variant] = {"file": relative, "sha256": case["model_sha256"], "decoder": "lite_semseg",
                "classes": ["crack"], "capabilities": ["surface_crack"], "input_size": 384, "confidence": .5,
                "exposure_policy": "low_light_gain", "diagnostic_only": True, "experimental": True,
                "source": metadata["source"], "license": metadata["license"],
                "display_name": f"{tag.title()} fixed .50 crack mask · FAILED research",
                "diagnostic_label": "RESEARCH - FIXED .50 / NOT QUALIFIED",
                "validation_scope": "Fixed original TUT versus FINAL300, exposed/calibration videos. All-three fitted calibration guards failed. No spalling support, no video pixel GT or independent accuracy credit."}
            counts = report["summary"]
            label = f"Crack mask diagnostic · {tag.title()} / {case['sequence']} / fixed .50"
            note = (f"Real complete camera recording: {counts['localized_matches']} box matches, "
                    f"{counts['missed_defects']} missed, {counts['false_alerts']} unmatched. "
                    "Yellow raw mask / blue served mask do not assert true pixels. Failed research; spalling unsupported; no independent accuracy credit.")
            replay = report["source"]["lossless_replay_input"]["path"]
            options = {"mode": "damage", "damage_variant": variant, "stage": "structure", "engine": "openvino",
                       "openvino_device": "GPU", "cpu_threads": 1, "mask_rendering": "masks",
                       "exposure_policy": "low_light_gain", "confidence": .5, "damage_classes": ["crack"], "confirm_frames": 1}
            append_exact(presets, {"label": label, "profile": "construction", "path": replay, "input_fps": 15,
                         "deadline_ms": 500, "preview_fps": 8, "video_mode": "every_frame", "purpose": "failure_diagnostic",
                         "profile_options": options, "note": note, "independent_accuracy_credit": False}, "label")
            append_exact(videos, {"id": variant + "-" + case["sequence"], "label": label, "path": case["video"],
                         "evidence_path": case["report"], "purpose": "failure_diagnostic", "issue_summary": note, "note": note,
                         "source_url": "https://doi.org/10.11583/DTU.25924228", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                         "attribution": "Gamage et al., ev-CIVIL, DTU, CC BY4.0. Publisher annotations are external to model input.",
                         "source_preset_label": label, "video_sha256": case["video_sha256"], "report_sha256": case["report_sha256"],
                         "independent_accuracy_credit": False}, "id")
    registry_path = ROOT / "edge_runtime/profiles/construction_model.py"
    source = registry_path.read_text(encoding="utf-8")
    entries = ""
    for key, spec in variants.items():
        if key in DAMAGE_VARIANTS and DAMAGE_VARIANTS[key] != spec:
            raise ValueError("Changed fixed variant")
        if key not in DAMAGE_VARIANTS:
            entries += "    " + repr(key) + ": " + pprint.pformat(spec, width=110).replace("\n", "\n    ") + ",\n"
    if source.count("DAMAGE_VARIANTS = {\n") != 1:
        raise ValueError("Unexpected registry")
    source = source.replace("DAMAGE_VARIANTS = {\n", "DAMAGE_VARIANTS = {\n" + entries, 1)
    compile(source, str(registry_path), "exec")
    registry_path.write_text(source, encoding="utf-8")
    for path, value in [(videos_path, videos), (presets_path, presets)]:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Registered six real fixed-pair mask videos and matching raw-input diagnostics; defaults preserved.")


if __name__ == "__main__":
    main()
