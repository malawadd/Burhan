"""Controlled real-photo SSD background-loss experiment; never reads TEST pixels.

Default run reproduces the original fresh COCO initialization, 200 updates,
uniform TRAIN batches, frozen backbone and optimizer. Only the loss differs.
Calibration is reported for diagnosis; no automatic promotion or owner tuning.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_sdnet_expansion as shared
from construction_ssd_background_loss import install

DATA = ROOT / "datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json"
BASE = ROOT / "models/construction/expansion-wall-stain-v1/ssdlite320/model-state.pt"
SOURCE = ROOT / "models/construction/expansion-initializers/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(state=None):
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    model.load_state_dict(torch.load(SOURCE, weights_only=True, map_location="cpu"))
    channels = [branch[-1].in_channels for branch in model.head.classification_head.module_list]
    model.head.classification_head = SSDLiteClassificationHead(channels, model.anchor_generator.num_anchors_per_location(),
        2, partial(nn.BatchNorm2d, eps=.001, momentum=.03))
    if state:
        model.load_state_dict(torch.load(state, weights_only=True, map_location="cpu"))
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="background-loss-200-v1")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--finetune-features", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    if not args.run.replace("-", "").isalnum() or not 1 <= args.steps <= 3000:
        raise ValueError("Invalid bounded experiment")
    out = ROOT / "output/evaluation/construction-wall-learning" / args.run
    if out.exists():
        raise ValueError("Preserve existing experiment; use a new run name")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    random.seed(20261021)
    manifest = json.loads(args.data.read_text())
    if manifest["classes"] != ["damp_stained_appearance"] or manifest["private_videos_used"]:
        raise ValueError("Dataset contract changed")
    shared.CLASSES = manifest["classes"]
    train = [r for r in manifest["records"] if r["split"] == "train"]
    cal = [r for r in manifest["records"] if r["split"] == "calibration"]
    # TEST rows are counted for audit only; never decoded or used for selection.
    for row in train + cal:
        if sha(ROOT / row["image"]) != row["source_sha256"]:
            raise ValueError("Frozen image changed")
    protocol = {"data_sha256": sha(args.data), "script_sha256": sha(__file__),
        "loss_sha256": sha(ROOT / "scripts/construction_ssd_background_loss.py"),
        "source_sha256": sha(SOURCE), "baseline_state_sha256": sha(BASE),
        "initial_state_sha256": sha(args.initial_state) if args.initial_state else None,
        "steps": args.steps, "batch": 8, "seed": 20261021, "threads": 1,
        "train_count": len(train), "train_positive": sum(bool(r["boxes"]) for r in train),
        "cal_count": len(cal), "test_pixels_used": False, "owner_pixels_used": False,
        "changes": ["At least 3 hard background anchors per empty image", "normalizer >= batch size"],
        "finetune_features": args.finetune_features, "balanced": args.balanced,
        "inference_changed": False, "selection": "Fixed final checkpoint; no TEST or owner selection"}
    shared.write(out / "protocol-before-run.json", protocol)
    shutil.copyfile(__file__, out / "trainer-source-at-run.py")
    shutil.copyfile(ROOT / "scripts/construction_ssd_background_loss.py", out / "loss-source-at-run.py")
    original = build(BASE)
    before_cal, _ = shared.ssd_predictions(original, cal)
    negatives = [r for r in train if not r["boxes"]][:8]
    tensors = [shared.image_tensor(r) for r in negatives]
    targets = [shared.target(r) for r in negatives]
    original.train()
    original.backbone.eval()
    negative_before = original(tensors, targets)
    sum(negative_before.values()).backward()
    old_gradient = sum(float(p.grad.abs().sum()) for p in original.parameters() if p.grad is not None)
    original.zero_grad()
    install(original)
    # Reset identical head BN state for fair negative-only forward comparison.
    original.load_state_dict(torch.load(BASE, weights_only=True, map_location="cpu"))
    original.train()
    original.backbone.eval()
    negative_after = original(tensors, targets)
    sum(negative_after.values()).backward()
    new_gradient = sum(float(p.grad.abs().sum()) for p in original.parameters() if p.grad is not None)
    negative_diagnostic = {"real_train_ids": [r["id"] for r in negatives],
        "original_losses": {k: float(v.detach()) for k, v in negative_before.items()},
        "corrected_losses": {k: float(v.detach()) for k, v in negative_after.items()},
        "original_gradient_abs_sum": old_gradient, "corrected_gradient_abs_sum": new_gradient,
        "corrected_sampling": original.background_loss_diagnostic}
    shared.write(out / "negative-learning-diagnostic.json", negative_diagnostic)
    print("NEGATIVE_DIAGNOSTIC", json.dumps(negative_diagnostic), flush=True)
    del original
    # Match the original fresh-head RNG sequence independently of diagnostics.
    torch.manual_seed(20261021)
    model = install(build(args.initial_state))
    if args.finetune_features:
        for name, parameter in model.backbone.named_parameters():
            parameter.requires_grad = name.startswith("features.1.") or name.startswith("extra.")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": .001 if not args.initial_state else .0002},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": .00005}], weight_decay=.0005)
    cache = [(shared.image_tensor(r), shared.target(r), r["id"]) for r in train]
    positive_cache = [r for r in cache if len(r[1]["labels"])]
    negative_cache = [r for r in cache if not len(r[1]["labels"])]
    rng = random.Random(20261021)
    model.train()
    model.backbone.eval()  # freeze running statistics, including any finetuned layers
    history, schedule = [], []
    started = time.perf_counter()
    for step in range(args.steps):
        batch = (rng.sample(positive_cache, 4) + rng.sample(negative_cache, 4)) if args.balanced else rng.sample(cache, 8)
        schedule.append([item[2] for item in batch])
        loss_parts = model([r[0] for r in batch], [r[1] for r in batch])
        loss = sum(loss_parts.values())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        history.append({"step": step + 1, **{k: float(v.detach()) for k, v in loss_parts.items()},
            "positives": sum(bool(len(r[1]["labels"])) for r in batch),
            **model.background_loss_diagnostic})
        if (step + 1) % 25 == 0:
            print("WALL_LEARNING", step+1, "LOSS", float(loss.detach()), "SECONDS", time.perf_counter()-started, flush=True)
    elapsed = time.perf_counter() - started
    model.eval()
    torch.save(model.state_dict(), out / "model-state.pt")
    after_cal, speed = shared.ssd_predictions(model, cal)
    shared.write(out / "calibration-predictions.json", {"before": before_cal, "after": after_cal, "ids": [r["id"] for r in cal]})
    shared.write(out / "training-history.json", {"history": history, "batch_ids": schedule})
    result = {"protocol": protocol, "training_seconds": elapsed, "checkpoint_sha256": sha(out / "model-state.pt"),
        "negative_diagnostic": negative_diagnostic, "calibration_speed": speed,
        "calibration_fixed_points": [{"threshold": t, "before": shared.metrics(cal, before_cal, t),
            "after": shared.metrics(cal, after_cal, t)} for t in [.05, .1, .15, .2, .25, .35, .5, .65, .8, .9]],
        "status": "Training/calibration diagnostic, not independent accuracy or release evidence"}
    shared.write(out / "results.json", result)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
