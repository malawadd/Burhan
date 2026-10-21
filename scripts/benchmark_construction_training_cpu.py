"""TRAIN-only CPU format/thread timing; no optimizer or model-quality selection.

All models are disposable fresh clones; SDNET v2 checkpoints are never opened,
modified, interrupted or evaluated. This is not deployment FPS qualification.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import statistics
import time

import cv2
import numpy as np
import psutil
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"output/optimization/construction-expansion-training-cpu"
MANIFEST = ROOT/"datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json"
SOURCE = ROOT/"models/construction/expansion-initializers/yolov8n.pt"
EXPECTED = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def context():
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "nice"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if "train_sdnet_expansion_v2.py" in command and process.info["pid"] != os.getpid():
                active.append({"pid": process.info["pid"], "name": process.info["name"], "priority": process.info["nice"],
                               "command": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {"active_sdnet_v2_processes": active, "benchmark_priority": psutil.Process().nice(),
            "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
            "scope": "Contended TRAIN compute probe; no isolated deployment speed claim"}


def main():
    if (OUT/"protocol-before-forward.json").exists():
        raise ValueError("Preserve original benchmark; do not silently overwrite")
    if sha(SOURCE) != EXPECTED:
        raise ValueError("Official initializer changed")
    manifest = json.loads(MANIFEST.read_text())
    rows = [row for row in manifest["records"] if row["split"] == "train"][:4]
    images, boxes, indices, classes = [], [], [], []
    for batch_index, row in enumerate(rows):
        path = ROOT/row["image"]
        if sha(path) != row["source_sha256"]:
            raise ValueError("Frozen TRAIN photo changed")
        image = cv2.imread(str(path))
        if image.shape[:2] != (640, 640):
            raise ValueError("Expected fixed 640-square source")
        rgb = cv2.cvtColor(cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        images.append(rgb.transpose(2, 0, 1))
        for box in row["boxes"]:
            x, y, width, height = box["bbox"]
            boxes.append([(x+width/2)/640, (y+height/2)/640, width/640, height/640])
            indices.append(batch_index)
            classes.append([box["class_id"]])
    batch = {"img": torch.from_numpy(np.ascontiguousarray(np.stack(images))).float()/255,
             "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "batch_idx": torch.tensor(indices), "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1)}
    configuration = [{"threads": threads, "memory_format": memory_format} for threads in (1, 2, 4) for memory_format in ("channels_first", "channels_last")]
    protocol = {"script_sha256": sha(Path(__file__)), "source_pt_sha256": EXPECTED, "dataset_manifest_sha256": sha(MANIFEST),
                "train_inputs": [{"id": row["id"], "image": row["image"], "sha256": row["source_sha256"]} for row in rows],
                "batch": 4, "image_size": 512, "actual_box_count": len(boxes), "freeze_first_blocks": 4, "always_frozen": "DFL",
                "seed": 20261021, "configurations": configuration, "warmup_steps": 2, "timed_steps": 3,
                "timed_work": "zero_grad excluded; forward, detection loss and backward included; no optimizer or data decoding",
                "optimizer_updates": 0, "calibration_or_test_pixels_used": False, "private_owner_videos_used": False,
                "quality_selection": False, "context": context()}
    write("protocol-before-forward.json", protocol)
    torch.manual_seed(20261021)
    source = YOLO(str(SOURCE))
    architecture = copy.deepcopy(source.model.yaml)
    architecture["nc"] = 2
    architecture_path = OUT/"fresh-two-class-architecture.yaml"
    architecture_path.write_text(yaml.safe_dump(architecture), encoding="utf8")
    base = YOLO(str(architecture_path)).load(str(SOURCE)).model.float()
    base.names = dict(enumerate(manifest["classes"]))
    base.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    frozen = [f"model.{index}." for index in range(4)]+[".dfl"]
    for name, parameter in base.named_parameters():
        parameter.requires_grad = not any(prefix in name for prefix in frozen)
    initial_sha = parameter_sha(base)
    trainable = {name: list(parameter.shape) for name, parameter in base.named_parameters() if parameter.requires_grad}
    write("initialization.json", {"parameter_sha256": initial_sha, "trainable_shapes": trainable,
                                   "frozen_parameter_count": sum(parameter.numel() for parameter in base.parameters() if not parameter.requires_grad),
                                   "total_parameters": sum(parameter.numel() for parameter in base.parameters())})
    results, baseline_gradient, baseline_loss = [], None, None
    for config in configuration:
        torch.set_num_threads(config["threads"])
        model = copy.deepcopy(base)
        model.train()
        for name, module in model.named_modules():
            if any(prefix in name for prefix in frozen) and isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        fmt = torch.channels_last if config["memory_format"] == "channels_last" else torch.contiguous_format
        model.to(memory_format=fmt)
        if parameter_sha(model) != initial_sha:
            raise ValueError("Memory-layout conversion changed model values")
        current = dict(batch, img=batch["img"].contiguous(memory_format=fmt))
        elapsed, losses = [], []
        for step in range(5):
            model.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss, _ = model(current)
            loss = loss.sum()
            loss.backward()
            duration = time.perf_counter()-started
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite compute-probe loss")
            if step >= 2:
                elapsed.append(duration)
                losses.append(float(loss.detach()))
        gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.requires_grad}
        if set(gradients) != set(trainable) or any(value is None or not torch.isfinite(value).all() for value in gradients.values()):
            raise ValueError("Gradient coverage/finite-work check failed")
        gradient_shapes = {name: list(value.shape) for name, value in gradients.items()}
        if gradient_shapes != trainable:
            raise ValueError("Gradient shapes differ from actual trainable parameters")
        vector = torch.cat([gradients[name].detach().contiguous().flatten() for name in sorted(gradients)])
        if baseline_gradient is None:
            baseline_gradient = vector.clone()
            baseline_loss = losses[-1]
        gradient_relative_error = float(torch.linalg.vector_norm(vector-baseline_gradient)/torch.linalg.vector_norm(baseline_gradient).clamp_min(1e-12))
        loss_relative_error = abs(losses[-1]-baseline_loss)/max(1e-12, abs(baseline_loss))
        unchanged = parameter_sha(model) == initial_sha
        row = {**config, "actual_torch_threads": torch.get_num_threads(), "input_strides": list(current["img"].stride()),
               "input_channels_last": current["img"].is_contiguous(memory_format=torch.channels_last), "timed_seconds": elapsed,
               "median_seconds_per_step": statistics.median(elapsed), "losses": losses, "gradient_parameter_count": len(gradients),
               "gradient_shapes_match": gradient_shapes == trainable, "gradient_relative_l2_error_vs_baseline": gradient_relative_error,
               "loss_relative_error_vs_baseline": loss_relative_error, "gradient_work_comparable": gradient_relative_error < .01 and loss_relative_error < .001,
               "parameters_unchanged": unchanged, "optimizer_updates": 0, "process_rss_bytes": psutil.Process().memory_info().rss}
        if not unchanged:
            raise ValueError("Disposable probe parameters changed without an optimizer")
        results.append(row)
        write("results-in-progress.json", {"protocol": protocol, "results": results})
        print(json.dumps(row), flush=True)
        del model, gradients, vector
    baseline = results[0]["median_seconds_per_step"]
    eligible = [row for row in results if row["gradient_work_comparable"] and row["parameters_unchanged"]]
    winner = min(eligible, key=lambda row: row["median_seconds_per_step"])
    report = {"protocol": protocol, "results": results, "best_compute_configuration": winner,
              "speedup_vs_1thread_channels_first": baseline/winner["median_seconds_per_step"],
              "at_least_2x": baseline/winner["median_seconds_per_step"] >= 2,
              "scope": "TRAIN-only forward/backward compute and gradient checks; no optimizer updates, model selection or deployment performance qualification"}
    write("results.json", report)
    print("SUMMARY", json.dumps({"best": {key: winner[key] for key in ("threads", "memory_format", "median_seconds_per_step")},
                                 "speedup": report["speedup_vs_1thread_channels_first"], "at_least_2x": report["at_least_2x"]}), flush=True)


if __name__ == "__main__":
    main()
