"""Replace only six display assets after verifying unchanged scored records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/evaluation/construction-pixel-videos/display-v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt = OUT / "registration.json"
    if receipt.exists():
        raise ValueError("Preserve completed display registration")
    manifest = OUT / "all-six-presentation-manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if len(rows) != 6:
        raise ValueError("All six required")
    path = ROOT / "config/construction-videos.json"
    before_bytes = path.read_bytes()
    config = json.loads(before_bytes)
    by_id = {row["id"]: row for row in config}
    frame_count = 0
    for row in rows:
        old = by_id[row["id"]]
        if old["purpose"] != "failure_diagnostic":
            raise ValueError("Research scope changed")
        old_report = ROOT / old["evidence_path"]
        new_report = ROOT / row["report"]
        if sha(old_report) != old["report_sha256"] or sha(new_report) != row["report_sha256"]:
            raise ValueError("Scoring source changed")
        previous = json.loads(old_report.read_text(encoding="utf-8"))
        current = json.loads(new_report.read_text(encoding="utf-8"))
        derivative = current["presentation_derivative"]
        if derivative["model_inference_calls"] != 0 or derivative["original_report_sha256"] != old["report_sha256"]:
            raise ValueError("Not a cached display derivative")
        comparable = {key: value for key, value in current.items() if key != "presentation_derivative"}
        comparable["derived_video"] = previous["derived_video"]
        if comparable != previous or sha(ROOT / row["video"]) != row["video_sha256"]:
            raise ValueError("Model, scored records, policy or display identity differs")
        frame_count += len(previous["frames"])
        old.update(path=row["video"], evidence_path=row["report"], video_sha256=row["video_sha256"],
                   report_sha256=row["report_sha256"],
                   presentation="Display v2: same actual masks and publisher boxes on the recorded gain-adjusted native input; unchanged RAW pane and complete counts. No new NN.")
    if frame_count != 700:
        raise ValueError("Complete original frame records required")
    backup = OUT / "construction-videos-before-display-registration.json"
    if backup.exists():
        raise ValueError("Preserve previous registration input")
    backup.write_bytes(before_bytes)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt.write_text(json.dumps({"all_checks_passed": True, "model_inference_calls": 0,
        "frames_unchanged": frame_count, "manifest_sha256": sha(manifest),
        "previous_config_sha256": hashlib.sha256(before_bytes).hexdigest(), "new_config_sha256": sha(path),
        "scope": "Presentation-only registration; all original videos, masks and reports retained."}, indent=2) + "\n", encoding="utf-8")
    print("Six display-v2 recordings registered; all700 scored records unchanged; zero NN calls.")


if __name__ == "__main__":
    main()
