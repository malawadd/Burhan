"""Normal-app EdgeRuntime stress on clocked, varying real public road images.

This is software replay throughput, not vehicle/line coverage or new accuracy.
Model initialization is reported separately; the first processed frames remain
in timing/deadline counts so cold inference is not hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import cv2

import edge_runtime.runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--deadline", type=float, default=150)
    parser.add_argument("--engine", choices=["openvino", "onnxruntime"], default="openvino")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--output", type=Path, default=ROOT / "output/optimization/roads-runtime-gpu12.json")
    args = parser.parse_args()
    if args.engine == "onnxruntime" and args.device != "CPU":
        parser.error("ONNX Runtime comparison uses --device CPU")
    dataset = ROOT / "datasets/roads-rdd2022-test-balanced"
    manifest = json.loads((dataset / "manifest.json").read_text())
    images, hashes = [], []
    for row in manifest["images"]:
        path = dataset / "images" / row["file_name"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError("Diagnostic public image hash changed")
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        if max(h, w) > 1920:
            image = cv2.resize(image, (round(w * 1920 / max(h, w)), round(h * 1920 / max(h, w))), interpolation=cv2.INTER_AREA)
        images.append(image)
        hashes.append({"file_name": row["file_name"], "sha256": digest})

    class ClockedPublicReplay:
        backend = "predecoded_varying28_public_image_software_clock"

        def __init__(self, *_):
            self.sequence = 0

        def open(self):
            self.started = time.perf_counter()

        def read(self):
            if time.perf_counter() - self.started >= args.seconds:
                return None, None
            image = images[self.sequence % len(images)].copy()
            self.sequence += 1
            return image, (self.sequence - 1) * 1000 / args.fps

        def close(self):
            pass

    with tempfile.TemporaryDirectory(prefix="roads-runtime-benchmark-") as directory:
        edge = runtime.EdgeRuntime(directory)
        try:
            options = {"demo": False, "backend": "model", "model_variant": "fiveclass",
                       "engine": args.engine, "input_size": args.input_size, "cpu_threads": 4,
                       "openvino_device": args.device, "confidence": .25, "nms_iou": .45,
                       "confirm_frames": 2}
            with patch.object(runtime, "LocalSource", ClockedPublicReplay):
                started = time.perf_counter()
                edge.start(runtime.RuntimeConfig(profile="roads", source="file:predecoded-public-road-replay",
                           input_fps=args.fps, deadline_ms=args.deadline, save_evidence=False,
                           preview_fps=6, profile_options=options))
                station_start_ms = (time.perf_counter() - started) * 1000
                deadline = time.monotonic() + args.seconds + 30
                while time.monotonic() < deadline:
                    state = edge.snapshot()
                    if state["error"]:
                        raise RuntimeError(state["error"])
                    if state["state"] == "eof":
                        break
                    time.sleep(.02)
                else:
                    raise RuntimeError("Runtime did not drain within the bounded deadline")
            import openvino as ov
            report = {"scope": "30-second default, software-clock replay of varying28 public real images; not moving footage, object-tracking accuracy, independent accuracy, vehicle speed or physical coverage",
                      "normal_app_interpreter": str(Path(sys.executable).resolve()),
                      "openvino_module_file": str(Path(ov.__file__).resolve()), "openvino_version": ov.get_version(),
                      "input_acquisition": "Public28 hashes verified, images decoded/1920-capped before timing; copy per capture; runtime performs actual capture pacing, inference, preview and local event storage",
                      "input_images": hashes, "station_start_ms": station_start_ms,
                      "cold_first_inference": "Included in processed/late/error/timing counts; no discarded runtime warmup",
                      "requested_seconds": args.seconds, "requested_fps": args.fps,
                      "deadline_ms": args.deadline, "network": "socket.connect denied", "state": state}
        finally:
            edge.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output), "state": state["state"], "metrics": state["metrics"],
                      "model": state["description"], "media": state["media"]}, indent=2), flush=True)


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden in local GPU runtime benchmark")):
        main()
