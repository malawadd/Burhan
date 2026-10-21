"""Read existing timed traces; no inference, retiming, filtering or model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    rows = report["diagnostic_trace"]
    fps = float(report["requested_fps"])
    deadline = float(report["state"]["config"]["deadline_ms"])
    late = [r for r in rows if r["host_to_profile_return_ms"] > deadline]
    first = rows[0]["captured_monotonic"] if rows else 0.0
    return {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "requested_fps": fps,
        "deadline_ms": deadline,
        "traced_profile_calls": len(rows),
        "profile_calls_exceeding_input_period": sum(r["profile_ms"] > 1000 / fps for r in rows),
        "late_traced_calls": len(late),
        "late_with_profile_call_alone_within_deadline": sum(r["profile_ms"] <= deadline for r in late),
        "maximum_queue_before_profile_ms": max((r["queue_before_profile_ms"] for r in rows), default=None),
        "late_calls": [
            {"sequence": r["sequence"], "source_receipt_seconds_from_first": r["captured_monotonic"] - first,
             "queue_ms": r["queue_before_profile_ms"], "profile_ms": r["profile_ms"],
             "receipt_to_profile_return_ms": r["host_to_profile_return_ms"],
             "neural_stage_ms": r.get("neural_stage_ms")}
            for r in late
        ],
        "limits": "Diagnostic clock ends at profile return. Official counters and decision latency stay in source report. Queue contribution is observed, not a causal attribution to a specific host process or GPU operation. No late call is removed or retimed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing analysis; use a distinct output version")
    result = {"scope": __doc__, "reports": [summarize(path) for path in args.reports]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["reports"]:
        print(json.dumps({k: v for k, v in row.items() if k != "late_calls"}))


if __name__ == "__main__":
    main()
