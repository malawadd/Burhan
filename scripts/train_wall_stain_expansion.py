"""Bounded visible-stain SSDLite fit; source leakage is appearance, not moisture."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
from pathlib import Path
import random
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

DATA = ROOT / "datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json"
OUT = ROOT / "output/evaluation/construction-expansion-wall-stain-v1"
MODELS = ROOT / "models/construction/expansion-wall-stain-v1/ssdlite320"
CLASSES = ["damp_stained_appearance"]
STEPS = 200


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    random.seed(20261021)
    manifest = json.loads(DATA.read_text())
    if manifest["classes"] != CLASSES or manifest["private_videos_used"]:
        raise ValueError("Wall-stain dataset contract changed")
    for row in manifest["records"]:
        if sha(ROOT / row["image"]) != row["source_sha256"]:
            raise ValueError("Prepared wall image changed")
    shared.CLASSES = CLASSES
    shared.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset_manifest_sha256": sha(DATA), "trainer_sha256": sha(Path(__file__)),
        "shared_helper_sha256": sha(Path(shared.__file__)), "architecture": "ssdlite320", "steps": STEPS,
        "batch": 8, "cpu_threads": torch.get_num_threads(), "seed": 20261021,
        "model_selection": "Fixed final checkpoint; threshold on CAL only, TEST evaluated once before/after.",
        "interpretation": "Visible stain/leakage appearance, no moisture measurement or active-leak claim.",
        "negatives": "Other-defect target-negative photographs, not healthy-wall false positive rate.",
        "site_independence": "Unknown; exploratory photo evidence only", "private_owner_videos_used": False}
    protocol_path = OUT / "protocol-before-training.json"
    if protocol_path.exists():
        raise ValueError("This frozen experiment already exists; create a new version")
    shared.write(protocol_path, protocol)
    source = ROOT / "models/construction/expansion-initializers/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
    if sha(source) != "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2":
        raise ValueError("Official initializer SHA changed")
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    model.load_state_dict(torch.load(source, weights_only=True, map_location="cpu"))
    channels = [branch[-1].in_channels for branch in model.head.classification_head.module_list]
    from functools import partial
    model.head.classification_head = SSDLiteClassificationHead(channels, model.anchor_generator.num_anchors_per_location(), 2,
                                                              partial(nn.BatchNorm2d, eps=.001, momentum=.03))
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    roles = {role: [row for row in manifest["records"] if row["split"] == role] for role in ("train", "calibration", "test")}
    before, _ = shared.ssd_predictions(model, roles["test"])
    cache = [(shared.image_tensor(row), shared.target(row)) for row in roles["train"]]
    rng = random.Random(20261021)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.001, weight_decay=.0005)
    model.train()
    model.backbone.eval()
    started = time.perf_counter()
    losses = []
    for step in range(STEPS):
        batch = rng.sample(cache, 8)
        loss = sum(model([item[0] for item in batch], [item[1] for item in batch]).values())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite wall-stain training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 25 == 0:
            print("WALL_STAIN_STEP", step+1, "LOSS", losses[-1], "SECONDS", time.perf_counter()-started, flush=True)
    training_seconds = time.perf_counter()-started
    model.eval()
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "model-state.pt")
    cal, speed = shared.ssd_predictions(model, roles["calibration"])
    test, _ = shared.ssd_predictions(model, roles["test"])
    result, threshold = shared.summarize("ssdlite320", before, None, cal, test, roles["calibration"], roles["test"], speed)
    result.update({"protocol": protocol, "split_counts": manifest["split_counts"], "positive_images": manifest["positive_images"],
                   "train_seconds": training_seconds, "loss_first_10": float(np.mean(losses[:10])), "loss_last_10": float(np.mean(losses[-10:])),
                   "source_initialization_sha256": sha(source)})
    shared.write(OUT / "ssdlite320-results.json", result)
    torch.onnx.export(shared.SSDExport(model), torch.zeros(1, 3, 320, 320), str(MODELS / "model.onnx"), input_names=["images"],
        output_names=["boxes", "labels", "scores"], opset_version=17, dynamo=False, external_data=False,
        dynamic_axes={"boxes": {0: "detections"}, "labels": {0: "detections"}, "scores": {0: "detections"}})
    spec = {"architecture": "ssdlite320", "input_size": 320, "model_file": "model.onnx", "model_sha256": sha(MODELS / "model.onnx"),
        "classes": CLASSES, "confidence": threshold, "preprocessing": "letterbox_rgb_0_1", "inspection_task": "walls_finishes",
        "title": "Wall stain/leakage appearance · SSDLite research", "source": manifest["source"],
        "license": "MBDD2025 CC BY4.0; TorchVision BSD3Clause; COCO initializer conditions apply",
        "validation_status": "research_only", "validation_summary": "Source leakage/stain appearance; no independent site/video qualification or healthy-wall false-alarm evidence",
        "evaluation_report": str((OUT / "ssdlite320-results.json").resolve()), "private_videos_used": False}
    shared.write(MODELS / "manifest.json", spec)
    print(json.dumps({"threshold": threshold, "after": result["after_test_calibration_selected"], "train_seconds": training_seconds}), flush=True)


if __name__ == "__main__":
    main()
