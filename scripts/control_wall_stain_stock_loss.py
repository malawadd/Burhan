"""Contemporaneous original-loss control for the 200-update wall experiment.

Same current helper, construction, seeds, optimizer and TRAIN batch schedule.
CAL only; no TEST/owner pixels. Preserve all previous checkpoints and reports.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_wall_stain_learning as experiment
import train_sdnet_expansion as shared


def main():
    out = ROOT / "output/evaluation/construction-wall-learning/stock-loss-200-control-v1"
    if out.exists():
        raise ValueError("Preserve experiment")
    out.mkdir(parents=True)
    manifest = json.loads(experiment.DATA.read_text())
    train = [r for r in manifest["records"] if r["split"] == "train"]
    cal = [r for r in manifest["records"] if r["split"] == "calibration"]
    for row in train + cal:
        if experiment.sha(ROOT / row["image"]) != row["source_sha256"]:
            raise ValueError("Frozen photo changed")
    protocol = {"purpose": "Matched original-loss control, not a third candidate selected on TEST",
        "dataset_sha256": experiment.sha(experiment.DATA), "script_sha256": experiment.sha(__file__),
        "constructor_sha256": experiment.sha(experiment.__file__), "shared_helper_sha256": experiment.sha(shared.__file__),
        "source_sha256": experiment.sha(experiment.SOURCE), "seed": 20261021, "threads": 1,
        "steps": 200, "batch": 8, "lr": .001, "weight_decay": .0005,
        "test_pixels_used": False, "owner_pixels_used": False, "backbone_frozen": True,
        "loss": "Original installed torchvision SSD.compute_loss, no background correction"}
    shared.write(out / "protocol-before-run.json", protocol)
    for path in [Path(__file__), Path(experiment.__file__), Path(shared.__file__)]:
        shutil.copyfile(path, out / (path.stem + "-source-at-run.py"))
    torch.set_num_threads(1)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    random.seed(20261021)
    shared.CLASSES = manifest["classes"]
    model = experiment.build()
    cache = [(shared.image_tensor(r), shared.target(r), r["id"]) for r in train]
    rng = random.Random(20261021)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0005)
    model.train()
    model.backbone.eval()
    history, schedule = [], []
    started = time.perf_counter()
    for step in range(200):
        batch = rng.sample(cache, 8)
        schedule.append([r[2] for r in batch])
        loss_parts = model([r[0] for r in batch], [r[1] for r in batch])
        loss = sum(loss_parts.values())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        history.append({"step": step+1, **{k: float(v.detach()) for k,v in loss_parts.items()}})
        if (step+1) % 25 == 0:
            print("STOCK_CONTROL", step+1, "LOSS", float(loss.detach()), "SECONDS", time.perf_counter()-started, flush=True)
    elapsed = time.perf_counter()-started
    model.eval()
    torch.save(model.state_dict(), out / "model-state.pt")
    predictions, speed = shared.ssd_predictions(model, cal)
    corrected_dir = out.parent / "background-loss-200-v1"
    corrected = json.loads((corrected_dir / "calibration-predictions.json").read_text())
    corrected_history = json.loads((corrected_dir / "training-history.json").read_text())
    if corrected_history["batch_ids"] != schedule or corrected["ids"] != [r["id"] for r in cal]:
        raise ValueError("Control inputs/schedule differ")
    historical = torch.load(experiment.BASE, weights_only=True, map_location="cpu")
    max_weight_error = max(float((value-historical[name]).abs().max()) for name,value in model.state_dict().items())
    shared.write(out / "calibration-predictions.json", {"ids": corrected["ids"], "predictions": predictions})
    shared.write(out / "training-history.json", {"history": history, "batch_ids": schedule})
    shared.write(out / "results.json", {"protocol": protocol, "train_seconds": elapsed, "calibration_speed": speed,
        "identical_train_batch_schedule": True, "historical_state_max_absolute_error": max_weight_error,
        "calibration_fixed_points": [{"threshold": t, "stock_control": shared.metrics(cal,predictions,t),
            "corrected_loss": shared.metrics(cal,corrected["after"],t)} for t in [.05,.1,.15,.2,.25,.35,.5,.65,.8,.9]],
        "status": "Calibration diagnosis only, no independent accuracy or release"})
    print("COMPLETE", out, "HISTORICAL_WEIGHT_MAX_ERROR", max_weight_error, flush=True)


if __name__ == "__main__":
    main()
