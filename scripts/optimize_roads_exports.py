"""Export fixed-size CPU candidates from the unchanged pinned five-class weights.

No training, threshold fitting, test-image calibration or network access occurs.
Run in .venv-roads-export, not the application environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="320,416")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/roads-yolo-settings"))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO
    directory = ROOT / "models/roads"
    baseline = json.loads((directory / "fiveclass-manifest.json").read_text())
    source = directory / "yolo12s-rdd2022-fiveclass.pt"
    if hashlib.sha256(source.read_bytes()).hexdigest() != baseline["source_sha256"]:
        raise ValueError("Pinned source weights changed")
    torch.set_num_threads(2)
    for size in map(int, args.sizes.split(",")):
        if size not in (320, 416, 512):
            raise ValueError("Only explicitly supported candidate sizes: 320,416,512")
        model = YOLO(str(source))
        generated = Path(model.export(format="onnx", imgsz=size, dynamic=False, simplify=False,
                                      opset=17, half=False, device="cpu", nms=False))
        target = directory / f"yolo12s-rdd2022-fiveclass-{size}.onnx"
        # Ultralytics overwrites the source-named export; restore the 640 export
        # after every candidate using its preserved bytes.
        target.write_bytes(generated.read_bytes())
        generated.write_bytes((directory / "yolo12s-rdd2022-fiveclass-baseline640.onnx").read_bytes())
        checked = onnx.load(target)
        onnx.checker.check_model(checked)
        manifest = {**baseline, "file": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "bytes": target.stat().st_size, "optimization": "same weights; smaller fixed spatial input; no quantization or fitting",
                    "export": {**baseline["export"], "imgsz": size},
                    "tool_versions": {"python": platform.python_version(), "torch": torch.__version__,
                                      "ultralytics": ultralytics.__version__, "onnx": onnx.__version__}}
        (directory / f"fiveclass-{size}-manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Exported {size}: {manifest['sha256']}", flush=True)


if __name__ == "__main__":
    directory = ROOT / "models/roads"
    baseline_path = directory / "yolo12s-rdd2022-fiveclass.onnx"
    preserved = directory / "yolo12s-rdd2022-fiveclass-baseline640.onnx"
    original = baseline_path.read_bytes()
    expected = json.loads((directory / "fiveclass-manifest.json").read_text())["sha256"]
    if hashlib.sha256(original).hexdigest() != expected:
        raise ValueError("Baseline export hash mismatch; refusing to overwrite")
    preserved.write_bytes(original)
    try:
        main()
    finally:
        baseline_path.write_bytes(original)
