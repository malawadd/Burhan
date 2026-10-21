"""Pinned local CPU adapters for newly trained construction research models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "construction"


def read_manifest(path):
    if not isinstance(path, (str, Path)):
        raise ValueError("Research manifest must be a local path")
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(MODEL_ROOT.resolve()):
        raise ValueError("Construction research manifest must be inside models/construction")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Research manifest cannot be read as JSON") from error
    if not isinstance(spec, dict):
        raise ValueError("Research model manifest must be an object")
    if not isinstance(spec.get("architecture"), str) or spec["architecture"] not in {"yolov8n", "yolov8n_seg", "ssdlite320", "lraspp"}:
        raise ValueError("Unsupported construction research architecture")
    size = spec.get("input_size")
    if type(size) is not int or not 128 <= size <= 1024 or size % 32:
        raise ValueError("Research input_size must be a multiple of 32 in 128..1024")
    classes = spec.get("classes")
    if not isinstance(classes, list) or not classes or any(not isinstance(x, str) or not x for x in classes) or len(set(classes)) != len(classes):
        raise ValueError("Research classes must be a unique nonempty string list")
    model_file = spec.get("model_file")
    if not isinstance(model_file, str) or not model_file:
        raise ValueError("Research model_file must be a relative filename")
    model = (path.parent / model_file).resolve()
    if Path(model_file).is_absolute() or not model.is_relative_to(path.parent):
        raise ValueError("Research model must remain inside its manifest directory")
    expected = spec.get("model_sha256")
    try:
        actual = hashlib.sha256(model.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("Research ONNX file is unavailable") from error
    if not isinstance(expected, str) or len(expected) != 64 or actual != expected:
        raise ValueError("Research ONNX SHA256 mismatch")
    preprocessing = spec.get("preprocessing", "letterbox_rgb_0_1")
    if not isinstance(preprocessing, str) or preprocessing not in {"letterbox_rgb_0_1", "letterbox_rgb_imagenet"}:
        raise ValueError("Unsupported research preprocessing")
    class_confidence = spec.get("class_confidence", {})
    if not isinstance(class_confidence, dict):
        raise ValueError("Research class_confidence must be an object")
    thresholds = dict(class_confidence)
    default = spec.get("confidence", .5)
    for label in classes:
        value = thresholds.get(label, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Research confidence must be finite in 0..1")
        thresholds[label] = float(value)
    return path, model, spec, thresholds


def discover_models():
    """Only verified locally installed expansion manifests; no network activity."""
    rows = []
    for path in sorted(MODEL_ROOT.glob("expansion-*/**/*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or "model_file" not in raw or "architecture" not in raw:
                continue
            manifest, _, spec, _ = read_manifest(path)
        except (ValueError, TypeError, OSError, KeyError):
            continue
        rows.append({"manifest": manifest.relative_to(ROOT).as_posix(),
                     "title": f"{spec.get('title', 'Local research model')} · {spec['architecture']} {spec['input_size']}px",
                     "inspection_task": spec.get("inspection_task"), "classes": spec["classes"],
                     "status": "research_only", "model_sha256": spec["model_sha256"],
                     "reason": spec.get("validation_summary", "No independent category release is established")})
    return rows


class ConstructionTaskDetector:
    def __init__(self, manifest):
        import onnxruntime as ort
        self.path, model, self.spec, self.thresholds = read_manifest(manifest)
        self.input_size = self.spec["input_size"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        try:
            model_bytes = model.read_bytes()
        except OSError as error:
            raise ValueError("Research ONNX file is unavailable") from error
        if hashlib.sha256(model_bytes).hexdigest() != self.spec["model_sha256"]:
            raise ValueError("Research ONNX SHA256 changed before initialization")
        try:
            # Bind inference to the actual verified bytes. A path-based second
            # load can read a replaced graph or unhashed external weight files.
            self.session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
        except Exception as error:
            raise ValueError("Research ONNX must be valid and self-contained; verified model could not be initialized") from error
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, self.input_size, self.input_size]:
            raise ValueError("Research ONNX must have one fixed float32 NCHW image input")
        self.input_name = inputs[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.nms_iou = .45

    def describe(self):
        return {"model_name": self.spec.get("title", self.spec["architecture"]),
                "model_path": str(self.path.parent / self.spec["model_file"]),
                "model_metadata_path": str(self.path), "model_sha256": self.spec["model_sha256"],
                "classes": self.spec["classes"], "input_size": self.input_size,
                "class_confidence": self.thresholds, "preprocessing": self.spec.get("preprocessing", "letterbox_rgb_0_1"),
                "source": self.spec.get("source"), "license": self.spec.get("license"),
                "engine": "onnxruntime", "device": "CPU", "cpu_threads": 1,
                "validation_status": "research_only", "released_categories": [],
                "validation_summary": self.spec.get("validation_summary", "Independent real-video validation not passed")}

    def infer(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("Research adapter requires a uint8 BGR image")
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
        px, py = (self.input_size-nw)//2, (self.input_size-nh)//2
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        canvas[py:py+nh, px:px+nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        tensor = canvas[:, :, ::-1].astype(np.float32)/255.
        if self.spec.get("preprocessing") == "letterbox_rgb_imagenet":
            tensor = (tensor-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        outputs = self.session.run(None, {self.input_name: tensor})
        if any(not np.isfinite(value).all() for value in outputs):
            raise ValueError("Research model returned non-finite output")
        if self.spec["architecture"] == "lraspp":
            return self._segmentation(outputs[0], (h, w), (nw, nh), (px, py))
        segmented = self.spec["architecture"] == "yolov8n_seg"
        if self.spec["architecture"] in {"yolov8n", "yolov8n_seg"}:
            pred = outputs[0]
            count = len(self.spec["classes"])
            expected = 4+count+(32 if segmented else 0)
            if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[1] != expected:
                raise ValueError("Unexpected YOLO research output shape")
            if segmented and (len(outputs) != 2 or outputs[1].shape != (1, 32, self.input_size//4, self.input_size//4)):
                raise ValueError("Unexpected YOLO research prototype shape")
            pred = pred[0].T
            indices = pred[:, 4:4+count].argmax(axis=1)
            scores = pred[np.arange(len(pred)), indices+4]
            centers, sizes = pred[:, :2], pred[:, 2:4]
            boxes = np.column_stack((centers-sizes/2, centers+sizes/2))
        else:
            named = dict(zip(self.output_names, outputs))
            if not {"boxes", "labels", "scores"}.issubset(named):
                raise ValueError("SSDLite research outputs must be named boxes,labels,scores")
            boxes, indices, scores = named["boxes"].reshape(-1, 4), named["labels"].reshape(-1).astype(int)-1, named["scores"].reshape(-1)
            if len(boxes) != len(indices) or len(indices) != len(scores):
                raise ValueError("Inconsistent SSDLite detection outputs")
        detections = []
        for class_id, label in enumerate(self.spec["classes"]):
            candidates = np.flatnonzero((indices == class_id) & (scores >= self.thresholds[label]) & (scores <= 1))
            if not len(candidates):
                continue
            # Bound decoded work before NMS; fixed threshold applies before ranking.
            candidates = candidates[np.argsort(scores[candidates])[::-1][:300]]
            native = boxes[candidates].copy()
            native[:, [0, 2]] = np.clip((native[:, [0, 2]]-px)/(nw/w), 0, w)
            native[:, [1, 3]] = np.clip((native[:, [1, 3]]-py)/(nh/h), 0, h)
            xywh = np.column_stack((native[:, :2], native[:, 2:]-native[:, :2]))
            valid = (xywh[:, 2] >= 1) & (xywh[:, 3] >= 1)
            xywh, selected = xywh[valid], candidates[valid]
            if not len(selected):
                continue
            # TorchVision's exported SSDLite already performs its trained NMS.
            # Applying the YOLO NMS threshold again changes served predictions.
            keep = (np.arange(len(selected)) if self.spec["architecture"] == "ssdlite320" else
                    cv2.dnn.NMSBoxes(xywh.tolist(), scores[selected].astype(float).tolist(), 0., self.nms_iou))
            for index in np.asarray(keep).reshape(-1):
                bbox = xywh[index]
                # Round outside extents consistently, retaining positive native area.
                x1, y1 = np.floor(bbox[:2]).astype(int)
                x2, y2 = np.ceil(bbox[:2]+bbox[2:]).astype(int)
                item = {"label": label, "bbox": (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        "score": float(scores[selected[index]]), "localization": "box_only"}
                if segmented:
                    proto = outputs[1][0]
                    logits = pred[selected[index], 4+count:] @ proto.reshape(32, -1)
                    # Match retina/process_mask_native geometry: remove padding
                    # on the prototype grid and resize directly to native size.
                    ph, pw = proto.shape[1:]
                    gain = min(ph/h, pw/w)
                    pad_x, pad_y = (pw-w*gain)/2, (ph-h*gain)/2
                    top, left = round(pad_y-.1), round(pad_x-.1)
                    bottom, right = round(ph-pad_y+.1), round(pw-pad_x+.1)
                    native_logits = cv2.resize(logits.reshape(ph, pw)[top:bottom, left:right], (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = np.zeros((h, w), bool)
                    mask[y1:y2, x1:x2] = native_logits[y1:y2, x1:x2] > 0
                    if np.count_nonzero(mask) < 8:
                        continue
                    item.update(mask=mask, localization="native_prototype_mask")
                detections.append(item)
        return detections

    def _segmentation(self, logits, dimensions, resized, padding):
        if logits.ndim != 4 or logits.shape[:2] != (1, len(self.spec["classes"])+1):
            raise ValueError("LRASPP must output background plus the manifest classes")
        # Background is channel zero. Upsample logits before unletterboxing.
        logits = logits[0]
        logits = np.stack([cv2.resize(channel, (self.input_size, self.input_size)) for channel in logits])
        stable = logits-logits.max(axis=0, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        h, w = dimensions
        nw, nh = resized
        px, py = padding
        native_probabilities = np.stack([cv2.resize(probabilities[index, py:py+nh, px:px+nw], (w, h)) for index in range(len(probabilities))])
        winners = native_probabilities.argmax(axis=0)
        pred = []
        for index, label in enumerate(self.spec["classes"], 1):
            probability = native_probabilities[index]
            native = (probability >= self.thresholds[label]) & (winners == index)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(native.astype(np.uint8), 8)
            for component in range(1, count):
                x, y, cw, ch, area = stats[component]
                if area < 8:
                    continue
                mask = labels == component
                pred.append({"label": label, "bbox": (int(x), int(y), int(cw), int(ch)),
                             "score": float(probability[mask].mean()), "mask": mask})
        return pred
