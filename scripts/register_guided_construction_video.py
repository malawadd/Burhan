"""Register a completed real comparison and original source as local diagnostics."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT / "output/evaluation").resolve()):
        parser.error("Completed comparison must be inside output/evaluation")
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    video = directory / "original-vs-research.mp4"
    if not video.is_file() or report["source_frames"] != report["output_frames"] or len(report["frames"]) != report["source_frames"]:
        raise ValueError("Comparison does not cover every genuine source frame")
    frames = []
    for row in report["frames"]:
        frames.append({"source_frame_index": row["source_frame_index"], "video_time_s": row["source_time_ms"]/1000,
            "frame_size_wh": [row["shape"][1], row["shape"][0]], "evaluated": False, "expected": [],
            "predictions": [{"id": f"p{index}", "label": item["label"], "score": item["score"], "bbox": item["bbox"],
                "evaluation_status": "unscored", "matched_annotation_id": None, "iou": None}
                for index, item in enumerate(row["observations"])]})
    evidence = {"title": args.label, "issue_summary": "Owner footage inspected with a newly trained research model; predictions remain unscored.",
        "annotation_scope": "No independent frame-level ground truth. Every candidate is unscored; no localized-match, recall, false-alarm or code-acceptance claim.",
        "source": {"original_camera_frames": report["source_frames"], "path": report["source"], "sha256": report["source_sha256"]},
        "summary": {"annotated_frames": 0, "localized_matches": None, "missed_defects": None, "false_alerts": None,
                    "unsupported_annotations": 0}, "frames": frames,
        "next_check": "Inspect the actual fixing/finish close up against the approved reference; these public-data models may miss this installation or raise unrelated candidates."}
    evidence_path = directory / "recording-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    task = report["profile"]["inspection_task"]
    metadata_path = report["profile"]["model_provenance"]["model_metadata_path"]
    preset_label = args.label + " / original source"
    row = {"id": args.id, "label": args.label, "path": video.relative_to(ROOT).as_posix(),
        "evidence_path": evidence_path.relative_to(ROOT).as_posix(), "purpose": "failure_diagnostic",
        "independent_accuracy_credit": False, "private_local_input": True,
        "issue_summary": evidence["issue_summary"], "note": evidence["annotation_scope"],
        "attribution": "Owner-provided local video; no redistribution license inferred. Never fitted, calibrated or uploaded.",
        "source_preset_label": preset_label}
    registry = ROOT / "config/construction-videos.json"
    rows = json.loads(registry.read_text(encoding="utf-8"))
    if any(item["id"] == args.id for item in rows):
        raise ValueError("Recording ID already registered; preserve it")
    presets_path = ROOT / "config/public-presets.json"
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    if any(item["label"] == preset_label for item in presets):
        raise ValueError("Original source preset already registered")
    presets.append({"label": preset_label, "profile": "construction", "path": report["source"], "source_kind": "file",
        "input_fps": 30, "preview_fps": 30, "deadline_ms": 1000, "video_mode": "realtime",
        "purpose": "failure_diagnostic", "private_local_input": True, "independent_accuracy_credit": False,
        "profile_options": {"inspection_task": task, "stage": "all", "research_model_manifest": metadata_path},
        "note": evidence["annotation_scope"]})
    rows.append(row)
    registry.write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
    presets_path.write_text(json.dumps(presets, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"recording": args.id, "frames": len(frames), "candidate_frames": report["candidate_frames"], "scored": False}))


if __name__ == "__main__":
    main()
