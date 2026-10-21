"""TRAIN-only localization diagnosis for a completed wall fit (not validation)."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_wall_stain_learning as experiment
import train_sdnet_expansion as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run / "train-localization-diagnostic.json"
    if output.exists():
        raise ValueError("Preserve previous diagnostic")
    manifest = json.loads(experiment.DATA.read_text())
    rows = [r for r in manifest["records"] if r["split"] == "train"]
    for row in rows:
        if experiment.sha(ROOT / row["image"]) != row["source_sha256"]:
            raise ValueError("Training photo changed")
    protocol = {"scope": "TRAIN learning only; not calibration, TEST or owner evaluation",
        "model_state_sha256": experiment.sha(args.run / "model-state.pt"),
        "data_sha256": experiment.sha(experiment.DATA), "script_sha256": experiment.sha(__file__),
        "constructor_sha256": experiment.sha(experiment.__file__), "helper_sha256": experiment.sha(shared.__file__),
        "train_ids": [r["id"] for r in rows], "threads": 1, "selection": "All181original TRAIN photos, no new choice"}
    shared.write(args.run / "protocol-before-train-diagnostic.json", protocol)
    shared.CLASSES = manifest["classes"]
    model = experiment.build(args.run / "model-state.pt")
    predictions, speed = shared.ssd_predictions(model, rows)
    shared.write(output, {"protocol": protocol, "predictions": predictions,
        "fixed_points": [shared.metrics(rows,predictions,t) for t in [.1,.15,.2,.25,.35,.5]],
        "timing": speed, "timing_scope": "contended forward timing, not runtime qualification"})
    print([(r["threshold"],r["tp"],r["fp"],r["fn"]) for r in json.loads(output.read_text())["fixed_points"]])


if __name__ == "__main__":
    main()
