"""Fresh evaluation reconstruction and TRAIN-only parity for wall-stain SSD."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import functools
import hashlib
import json

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"models/construction/expansion-wall-stain-v1/ssdlite320"
DATA = ROOT/"datasets/construction/expansion-public-v1/wall-stain/prepared/manifest.json"
OUT = ROOT/"output/evaluation/construction-expansion-wall-stain-v1/verification-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if (OUT/"protocol-before-parity.json").exists():
        raise ValueError("Preserve original parity verification; create a new version")
    spec = json.loads((DEST/"manifest.json").read_text())
    data = json.loads(DATA.read_text())
    if spec["classes"] != ["damp_stained_appearance"] or spec["confidence"] != .25 or spec["classes"] != data["classes"]:
        raise ValueError("Frozen class/confidence changed")
    model_path = DEST/"model.onnx"
    if sha(model_path) != spec["model_sha256"]:
        raise ValueError("ONNX hash mismatch")
    rows = [row for row in data["records"] if row["split"] == "train"][:3]
    protocol = {"script_sha256": sha(Path(__file__)), "model_sha256": sha(model_path), "saved_state_sha256": sha(DEST/"model-state.pt"),
                "dataset_manifest_sha256": sha(DATA), "manifest_sha256": sha(DEST/"manifest.json"),
                "inputs": [{"image": row["image"], "sha256": row["source_sha256"]} for row in rows],
                "selection": "First three frozen TRAIN records, before this verification forward",
                "mode": "Fresh model reconstruction followed by explicit eval(); no reused export wrapper model",
                "channels": "2: background and publisher leakage appearance", "confidence_unchanged": .25,
                "torch_threads": 1, "ort_intra_threads": 1, "ort_inter_threads": 1,
                "accuracy_or_threshold_selection": False, "owner_or_test_pixels_used": False}
    write("protocol-before-parity.json", protocol)
    graph = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(graph)
    external = [tensor.name for tensor in graph.graph.initializer if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data]
    if external:
        raise ValueError("Unbound external ONNX tensors")
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    channels = [branch[-1].in_channels for branch in model.head.classification_head.module_list]
    model.head.classification_head = SSDLiteClassificationHead(channels, model.anchor_generator.num_anchors_per_location(), 2,
        functools.partial(torch.nn.BatchNorm2d, eps=.001, momentum=.03))
    model.load_state_dict(torch.load(DEST/"model-state.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval()
    if any(module.training for module in model.modules()):
        raise ValueError("Fresh evaluation model contains training-mode child modules")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    records = []
    for row in rows:
        path = ROOT/row["image"]
        if sha(path) != row["source_sha256"]:
            raise ValueError("Frozen TRAIN pixels changed")
        image = cv2.imread(str(path))
        rgb = cv2.cvtColor(cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)/255
        with torch.no_grad():
            reference = {name: value.numpy() for name, value in model([torch.from_numpy(tensor[0])])[0].items()}
        outputs = dict(zip([item.name for item in session.get_outputs()], session.run(None, {session.get_inputs()[0].name: tensor})))
        checks = {}
        for name in ("boxes", "labels", "scores"):
            expected, actual = reference[name], outputs[name]
            same_shape = expected.shape == actual.shape
            passed = same_shape and (np.array_equal(expected, actual) if name == "labels" else np.allclose(expected, actual, atol=.002, rtol=.0001))
            checks[name] = {"shape": list(actual.shape), "same_shape": same_shape,
                            "max_absolute_error": float(np.abs(expected-actual).max()) if same_shape and expected.size else None,
                            "passed": bool(passed)}
        records.append({"image": row["image"], "outputs": checks, "passed": all(check["passed"] for check in checks.values())})
    report = {"protocol": protocol, "onnx_checker_pass": True, "external_data": False,
              "providers": session.get_providers(), "fresh_model_and_children_training": False,
              "absolute_tolerance": .002, "relative_tolerance": .0001, "records": records,
              "passed": all(row["passed"] for row in records),
              "scope": "Numerical export verification only; no accuracy selection, training, or FPS qualification"}
    write("parity.json", report)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise ValueError("Wall-stain export parity failed")


if __name__ == "__main__":
    main()
