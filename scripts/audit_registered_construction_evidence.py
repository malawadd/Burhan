"""Read-only local-service checks of completed recordings; no model inference."""
import json
import argparse
from pathlib import Path
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8765"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/evaluation/construction-round3/registered-fourteen-comparison-http-audit.json")
    parser.add_argument("--include-posthoc", action="store_true")
    parser.add_argument("--include-cubit", action="store_true", help="Also require all five author-model diagnostics")
    args = parser.parse_args()
    with urllib.request.urlopen(BASE + "/api/construction/videos", timeout=30) as response:
        inventory = json.load(response)
    config = json.loads((ROOT / "config/construction-videos.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in config}
    prefixes = ("pixel_fixed50_", "evcivil_r4_")
    if args.include_posthoc:
        prefixes += ("posthoc-",)
    if args.include_cubit:
        prefixes += ("cubit-author-v2-",)
    selected = [row for row in inventory if row["id"].startswith(prefixes)]
    expected = 14 + (4 if args.include_posthoc else 0) + (5 if args.include_cubit else 0)
    if len(selected) != expected:
        raise ValueError(f"All {expected} requested complete comparison recordings required")
    records = []
    for row in selected:
        if not row["available"] or not row["evidence_available"] or row["purpose"] != "failure_diagnostic":
            raise ValueError("Completed diagnostic missing or promoted")
        with urllib.request.urlopen(urllib.request.Request(BASE + row["url"], method="HEAD"), timeout=30) as response:
            video_status = response.status
        with urllib.request.urlopen(BASE + row["evidence_url"], timeout=30) as response:
            evidence = json.load(response)
            evidence_status = response.status
        source = json.loads((ROOT / by_id[row["id"]]["evidence_path"]).read_text(encoding="utf-8"))
        if evidence != source or source["independent_accuracy_credit"]:
            raise ValueError("API evidence differs or has independent accuracy credit")
        records.append({"id": row["id"], "video_head_status": video_status,
                        "evidence_status": evidence_status, "source_json_equal": True,
                        "frames": len(evidence["frames"]), "summary": evidence["summary"]})
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
              "scope": "Local HTTP availability/schema/source-JSON equality only. No NN, new accuracy, playback decode or speed qualification.",
              "records": records}
    target = args.output
    if target.exists():
        raise ValueError("Preserve the first integration verification")
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"All{expected} completed comparison videos HEAD200 and evidence200; full source JSON equality verified.")


if __name__ == "__main__":
    main()
