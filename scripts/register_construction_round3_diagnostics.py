"""Idempotently add six explicitly failed-calibration cases; no model execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    proposal = json.loads((ROOT / "output/evaluation/construction-round3/diagnostic-integration-proposal.json").read_text(encoding="utf-8"))
    for row in proposal["recording_rows"]:
        for key, digest in (("path", "video_sha256"), ("evidence_path", "report_sha256")):
            path = (ROOT / row[key]).resolve()
            if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != row[digest]:
                raise ValueError(f"Pinned diagnostic artifact differs: {row['id']} / {key}")
    additions = {"construction-videos.json": ("id", proposal["recording_rows"]),
                 "public-presets.json": ("label", proposal["public_preset_rows"])}
    updates = []
    for filename, (identity, rows) in additions.items():
        path = ROOT / "config" / filename
        current = json.loads(path.read_text(encoding="utf-8"))
        by_key = {row[identity]: row for row in current}
        for row in rows:
            row = dict(row)
            if row.get("purpose") != "failure_diagnostic":
                raise ValueError("Failed candidates must remain explicit diagnostics")
            row["independent_accuracy_credit"] = False
            row["note"] += " Publisher-labelled surface material loss; substrate and structural cause are not confirmed."
            if "profile_options" in row:
                row["profile_options"] = {**row["profile_options"], "stage": "structure"}
            existing = by_key.get(row[identity])
            if existing is not None and existing != row:
                raise ValueError(f"Refusing to overwrite an existing changed row: {row[identity]}")
            if existing is None:
                current.append(row)
        updates.append((path, current))
    for path, rows in updates:
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Registered six pinned diagnostic recordings and six raw-source presets; no primary category changed.")


if __name__ == "__main__":
    main()
