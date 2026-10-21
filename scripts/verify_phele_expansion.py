"""Verify PHELE model hashes, ONNX structure and native/CPU parity on TRAIN."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import functools
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"output/evaluation/construction-expansion-phele-v1"
MODELS = ROOT/"models/construction/expansion-phele-v1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)


def verify(architecture):
    directory = MODELS/architecture
    manifest = json.loads((directory/"manifest.json").read_text())
    path = directory/"model.onnx"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["model_sha256"]:
        raise ValueError("Export hash mismatch")
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)
    if any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in graph.graph.initializer):
        raise ValueError("Unbound external model tensors")
    data = json.loads((OUT/"training-manifest.json").read_text())
    row = next(r for r in data["records"] if r["split"] == "train")
    image = cv2.imread(str(ROOT/row["image"]))
    size = manifest["input_size"]
    interpolation = cv2.INTER_LINEAR if architecture == "yolov8n" else cv2.INTER_AREA
    rgb = cv2.cvtColor(cv2.resize(image, (size, size), interpolation=interpolation), cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)/255
    if architecture == "yolov8n":
        from ultralytics import YOLO
        model = YOLO(str(OUT/"yolo-phele-fit/weights/last.pt")).model.float().eval()
        with torch.no_grad():
            reference = {"output0": model(torch.from_numpy(tensor))[0].numpy()}
    else:
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
        channels = [branch[-1].in_channels for branch in model.head.classification_head.module_list]
        model.head.classification_head = SSDLiteClassificationHead(channels, model.anchor_generator.num_anchors_per_location(), 3,
            functools.partial(torch.nn.BatchNorm2d, eps=.001, momentum=.03))
        model.load_state_dict(torch.load(directory/"model-state.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            reference = {name: value.numpy() for name, value in model([torch.from_numpy(tensor[0])])[0].items()}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = dict(zip([item.name for item in session.get_outputs()], session.run(None, {session.get_inputs()[0].name: tensor})))
    checks = {}
    for name, expected in reference.items():
        actual = outputs[name]
        shape = expected.shape == actual.shape
        checks[name] = {"shape": list(actual.shape), "same_shape": shape,
                        "max_absolute_error": float(np.abs(expected-actual).max()) if shape and expected.size else None,
                        "passed": bool(shape and np.allclose(expected, actual, atol=.002, rtol=.0001))}
    result = {"architecture": architecture, "model_sha256": digest, "onnx_checker_pass": True, "external_data": False,
              "training_image": row["image"], "training_image_sha256": row["source_sha256"], "input_size": size,
              "torch_threads": torch.get_num_threads(), "ort_intra_threads": 1, "ort_inter_threads": 1,
              "providers": session.get_providers(), "tolerance_absolute": .002, "tolerance_relative": .0001,
              "outputs": checks, "passed": all(item["passed"] for item in checks.values()),
              "scope": "Numerical export parity only, not model accuracy or FPS qualification"}
    (OUT/f"{architecture}-cpu-onnx-parity.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise ValueError("CPU model export parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["yolov8n", "ssdlite320"], required=True)
    verify(parser.parse_args().model)
