"""Pinned DetectionBench multi-country YOLOv8n setup and isolated ONNX export.

Run this with the optional .venv-roads-export interpreter after installing torch,
torchvision, ultralytics, onnx and onnxscript there. These are build tools, not
runtime dependencies. Network is used only here, not for inspection.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = "acf216a43d796f7910fc854f662c1e9cfa03fc01"
SOURCE_HASH = "d1c3536d8e4a873b7ff251d1cb81c0047100f06c86f92e08596de2d618618a3e"
DATA_REV = "72b01f94c2e5bfe430861afd73995f2d412b8855"


def download(url, target, expected=None, maximum=32 * 1024 * 1024):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        data = target.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Unexpected oversized asset")
    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Source SHA256 mismatch: {target.name}")
    target.write_bytes(data)
    return digest


def dataset():
    """Four SHA-ranked test images per capture partition, chosen before scoring."""
    rows = []
    destination = ROOT / "datasets/roads-rdd2022-test-balanced"
    base = f"https://huggingface.co/datasets/dronefreak/RDD2022/resolve/{DATA_REV}"
    for shard in ("000", "001"):
        path = destination / f"source-metadata-{shard}.jsonl"
        download(f"{base}/data/images/test/shard_{shard}/metadata.jsonl", path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["shard"] = shard
            rows.append(row)
    groups = sorted(set(r["file_name"].rsplit("_", 1)[0] for r in rows))
    selected = []
    for group in groups:
        candidates = [r for r in rows if r["file_name"].rsplit("_", 1)[0] == group]
        chosen = sorted(candidates, key=lambda r: hashlib.sha256(r["file_name"].encode()).hexdigest())[:4]
        for row in chosen:
            url = f"{base}/data/images/test/shard_{row['shard']}/{row['file_name']}"
            sha = download(url, destination / "images" / row["file_name"])
            row.update(url=url, sha256=sha, partition=group)
            selected.append(row)
            print("Test image:", row["file_name"], flush=True)
    download(f"{base}/README.md", destination / "UPSTREAM-README.md")
    manifest = {"source": "https://huggingface.co/datasets/dronefreak/RDD2022", "revision": DATA_REV,
                "split": "provider test (not official CRDDC test)",
                "selection": "First four SHA256(filename)-ranked test images per seven capture partitions; fixed before replacement-model scoring; no confidence/label filtering",
                "provenance": "Provider reports these as held out from its training; training membership and route independence not independently audited",
                "license": "Mirror declares CC BY-SA 4.0; original Figshare metadata declares CC BY 4.0. Preserve attribution and mirror notice.",
                "annotation_contract": "objects.bbox is original-image absolute COCO XYWH, zero-based; category ids 0 D00,1 D10,2 D20,3 D40",
                "attribution": "Arya et al., RDD2022: A multi-national image dataset for automatic Road Damage Detection; reformatted by dronefreak/DetectionBench",
                "images": selected}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/roads-yolo-settings"))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO

    directory = ROOT / "models/roads"
    url = f"https://huggingface.co/dronefreak/rdd2022-yolov8n/resolve/{REV}/best.pt"
    source = directory / "yolov8n-rdd2022-multicountry.pt"
    download(url, source, SOURCE_HASH)
    download(f"https://huggingface.co/dronefreak/rdd2022-yolov8n/resolve/{REV}/README.md", directory / "MULTICOUNTRY-UPSTREAM-README.md")
    torch.set_num_threads(2)
    model = YOLO(str(source))
    expected = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
    names = [model.names[i] for i in range(4)]
    if names != expected:
        raise ValueError(f"Unexpected model class order: {names}")
    exported = Path(model.export(format="onnx", imgsz=640, dynamic=False, simplify=False,
                                 opset=17, half=False, device="cpu", nms=False))
    target = directory / "yolov8n-rdd2022-multicountry.onnx"
    if exported != target:
        target.write_bytes(exported.read_bytes())
    checked = onnx.load(str(target))
    onnx.checker.check_model(checked)
    manifest = {"model_id": "dronefreak/rdd2022-yolov8n", "model_url": url, "revision": REV,
                "source_sha256": SOURCE_HASH, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "file": str(target.relative_to(ROOT)).replace("\\", "/"), "bytes": target.stat().st_size,
                "classes": names, "license": "AGPL-3.0 (explicit model card); proprietary distribution requires applicable alternate license",
                "export": {"format": "onnx", "imgsz": 640, "dynamic": False, "simplify": False, "opset": 17,
                           "half": False, "device": "cpu", "nms": False},
                "tool_versions": {"python": platform.python_version(), "torch": torch.__version__,
                                  "ultralytics": ultralytics.__version__, "onnx": onnx.__version__},
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (directory / "multicountry-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
