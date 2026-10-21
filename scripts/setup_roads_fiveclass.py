"""Export the RDD2022 model that distinguishes potholes from repaired surfaces."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

from scripts.setup_roads_multicountry import ROOT, download

REVISION = "59b19d21f6ee0c5ad44a6455b36204a0e283b34b"
SOURCE_HASH = "ccf91e3199d81ee761f4a3a012da752939c9d72e9b1eaa242e508db3163270a0"


def main():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/roads-yolo-settings"))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO
    directory = ROOT / "models/roads"
    url = f"https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022/resolve/{REVISION}/yolo12s_RDD2022_best.pt"
    source = directory / "yolo12s-rdd2022-fiveclass.pt"
    download(url, source, SOURCE_HASH)
    download(f"https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022/resolve/{REVISION}/README.md", directory / "FIVECLASS-UPSTREAM-README.md")
    license_url = "https://raw.githubusercontent.com/sunsmarterjie/yolov12/2abab7153a065fb2925e8088e9ca2b19016ab7d6/LICENSE"
    download(license_url, directory / "YOLOV12-UPSTREAM-AGPL-3.0.txt")
    torch.set_num_threads(2)
    model = YOLO(str(source))
    names = [model.names[i] for i in range(5)]
    if names != ["D00", "D10", "D20", "D40", "Repair"]:
        raise ValueError(f"Unexpected class order: {names}")
    path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=False, opset=17,
                        half=False, device="cpu", nms=False)
    checked = onnx.load(path)
    onnx.checker.check_model(checked)
    manifest = {"model_id": "rezzzq/yolo12s-road-damage-rdd2022", "revision": REVISION, "model_url": url,
                "source_sha256": SOURCE_HASH, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "file": str(path), "bytes": Path(path).stat().st_size, "classes": names,
                "license": "Author declares MIT; derived from Ultralytics/YOLOv12 AGPL code/pretraining. Proprietary distribution not cleared.",
                "license_source": {"url": license_url, "revision": "2abab7153a065fb2925e8088e9ca2b19016ab7d6"},
                "export": {"imgsz": 640, "dynamic": False, "simplify": False, "opset": 17,
                           "half": False, "device": "cpu", "nms": False},
                "tool_versions": {"python": platform.python_version(), "torch": torch.__version__,
                                  "ultralytics": ultralytics.__version__, "onnx": onnx.__version__},
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (directory / "fiveclass-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
