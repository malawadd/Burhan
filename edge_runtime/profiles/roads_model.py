"""Verified local road-model contracts, CPU inference and candidate tracking."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from edge_runtime.contracts import Finding, FrameContext, InspectionResult

MODEL_SHA256 = "a93a8e1735aaf6cd502d13be54684b6335eb8cca1eec270d053e43550531e1f9"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "roads" / "yolo11n-rdd2022-india.onnx"
MULTICOUNTRY_PATH = MODEL_PATH.with_name("yolov8n-rdd2022-multicountry.onnx")
FIVECLASS_PATH = MODEL_PATH.with_name("yolo12s-rdd2022-fiveclass.onnx")
CLASS_NAMES = ("longitudinal crack", "transverse crack", "alligator crack", "pothole")


def decode(output, scale, padding, shape, threshold=0.25, nms_iou=0.45, num_classes=4, input_size=640):
    """Decode exported XYWH + explicit class scores; reject unknown contracts."""
    output = np.asarray(output)
    anchors = sum((input_size // stride) ** 2 for stride in (8, 16, 32))
    if input_size not in (320, 416, 512, 640) or num_classes not in (4, 5) or output.shape != (1, 4 + num_classes, anchors):
        raise ValueError(f"Unsupported road model output shape {output.shape}; expected (1,{4 + num_classes},{anchors})")
    rows = output[0].T
    rows = rows[np.isfinite(rows).all(axis=1)]
    labels = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), labels + 4]
    height, width = shape
    detections = []
    for label in range(num_classes):
        selected = np.where((labels == label) & (scores >= threshold))[0]
        boxes, values = [], []
        for i in selected:
            cx, cy, w, h = rows[i, :4]
            x1 = max(0, min(width, round((float(cx - w / 2) - padding[0]) / scale)))
            y1 = max(0, min(height, round((float(cy - h / 2) - padding[1]) / scale)))
            x2 = max(0, min(width, round((float(cx + w / 2) - padding[0]) / scale)))
            y2 = max(0, min(height, round((float(cy + h / 2) - padding[1]) / scale)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                values.append(float(scores[i]))
        if boxes:
            keep = cv2.dnn.NMSBoxes(boxes, values, threshold, nms_iou)
            for i in np.asarray(keep).reshape(-1):
                detections.append({"bbox": tuple(boxes[int(i)]), "score": values[int(i)], "class_id": label})
    return sorted(detections, key=lambda d: -d["score"])[:100]


class RoadDamageModel:
    def __init__(self, options=None):
        options = dict(options or {})
        self.variant = options.get("model_variant", "fiveclass")
        self.input_size = options.get("input_size", 640)
        self.cpu_threads = options.get("cpu_threads", 4)
        self.cpu_spinning = options.get("cpu_spinning", False)
        if type(self.input_size) is not int or self.input_size not in (320, 416, 512, 640):
            raise ValueError("road input_size must be 320, 416, 512 or 640")
        if type(self.cpu_threads) is not int or not 1 <= self.cpu_threads <= 8:
            raise ValueError("road cpu_threads must be an integer from 1 to 8")
        if type(self.cpu_spinning) is not bool:
            raise ValueError("road cpu_spinning must be a boolean")
        if self.variant != "fiveclass" and self.input_size != 640:
            raise ValueError("Smaller road inputs require the explicit fiveclass export")
        if self.variant not in {"multicountry", "india", "fiveclass"}:
            raise ValueError("road model_variant must be multicountry, fiveclass or india")
        manifest = {}
        if self.variant in {"multicountry", "fiveclass"}:
            default_path = MULTICOUNTRY_PATH if self.variant == "multicountry" else FIVECLASS_PATH
            suffix = f"-{self.input_size}" if self.variant == "fiveclass" and self.input_size != 640 else ""
            if suffix:
                default_path = default_path.with_name(f"yolo12s-rdd2022-fiveclass{suffix}.onnx")
            manifest_path = default_path.with_name(f"{self.variant}{suffix}-manifest.json")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            default_sha = manifest.get("sha256")
        else:
            default_path, default_sha = MODEL_PATH, MODEL_SHA256
        self.manifest = manifest
        self.class_names = CLASS_NAMES + ("repaired surface",) if self.variant == "fiveclass" else CLASS_NAMES
        if self.variant == "multicountry":
            self.class_names = CLASS_NAMES[:3] + ("surface anomaly (training label disputed)",)
        self.threshold = float(options.get("confidence", 0.25))
        self.nms_iou = float(options.get("nms_iou", 0.45))
        self.confirm_frames = int(options.get("confirm_frames", 2))
        self.track_timeout = float(options.get("track_timeout_seconds", 1.5))
        if (not math.isfinite(self.threshold) or not 0 < self.threshold < 1
                or not math.isfinite(self.nms_iou) or not 0 < self.nms_iou < 1
                or not 1 <= self.confirm_frames <= 30 or not math.isfinite(self.track_timeout)
                or not 0 < self.track_timeout <= 60):
            raise ValueError("Invalid road model confidence, NMS, confirmation or timeout")
        self.path = Path(options.get("model_path", default_path)).expanduser().resolve()
        if not self.path.is_file():
            raise ValueError("Road ONNX model missing. See docs/modules/roads.md for isolated model setup/export")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        expected = options.get("model_sha256", default_sha)
        if not expected:
            raise ValueError("Road model manifest missing; use documented setup/export or provide model_sha256")
        if self.sha256 != expected:
            raise ValueError("Road model SHA256 mismatch; use the documented exact model/export")
        self.engine = options.get("engine", "onnxruntime")
        self.openvino_device = options.get("openvino_device", "CPU")
        if self.openvino_device not in {"CPU", "GPU"}:
            raise ValueError("road openvino_device must be CPU or GPU")
        if self.engine != "openvino" and self.openvino_device != "CPU":
            raise ValueError("openvino_device GPU requires engine openvino")
        if self.engine == "opencv":
            self.runtime_version = cv2.__version__
            self.net = cv2.dnn.readNetFromONNX(str(self.path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif self.engine == "onnxruntime":
            import onnxruntime as ort
            self.runtime_version = ort.__version__
            settings = ort.SessionOptions()
            settings.intra_op_num_threads = self.cpu_threads
            settings.inter_op_num_threads = 1
            settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Release CPU between calls; three stations need not busy-spin cores.
            settings.add_session_config_entry("session.intra_op.allow_spinning", "1" if self.cpu_spinning else "0")
            self.session = ort.InferenceSession(str(self.path), sess_options=settings,
                                                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            if self.session.get_inputs()[0].shape != [1, 3, self.input_size, self.input_size]:
                raise ValueError("Road input_size does not match the fixed ONNX model contract")
        elif self.engine == "openvino":
            try:
                import openvino as ov
            except ImportError as error:
                raise ValueError("Optional OpenVINO CPU engine is not installed; see docs/roads-optimization.md") from error
            core = ov.Core()
            self.runtime_version = ov.get_version()
            if self.openvino_device not in core.available_devices:
                raise ValueError(f"Requested OpenVINO device {self.openvino_device} is unavailable; select CPU explicitly")
            model = core.read_model(str(self.path))
            if list(model.input().shape) != [1, 3, self.input_size, self.input_size]:
                raise ValueError("Road input_size does not match the fixed ONNX model contract")
            configuration = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
            if self.openvino_device == "CPU":
                configuration["INFERENCE_NUM_THREADS"] = self.cpu_threads
                configuration["NUM_STREAMS"] = 1
            self.compiled = core.compile_model(model, self.openvino_device, configuration)
            self.request = self.compiled.create_infer_request()
            self.resolved_device = str(self.compiled.get_property("EXECUTION_DEVICES"))
        else:
            raise ValueError("Road engine must be opencv, onnxruntime or openvino")
        self._next_id = 1
        self.reset()

    def reset(self):
        self._tracks = []
        self._identity = None
        self._last_time = None
        self._last_sequence = None

    def describe(self):
        model_name = {"multicountry": "YOLOv8n RDD2022 multi-country", "india": "YOLO11n RDD2022 India", "fiveclass": "YOLO12s RDD2022 five-class"}[self.variant]
        device = self.openvino_device if self.engine == "openvino" else "CPU"
        return {"name": "Local road damage model review", "method": f"{model_name} / {self.engine} {device}",
                "sha256": self.sha256, "model_path": str(self.path), "classes": list(self.class_names),
                "model_variant": self.variant, "engine": self.engine,
                "engine_version": self.runtime_version,
                "device": device,
                "execution_device": getattr(self, "resolved_device", device),
                "source_sha256": self.manifest.get("source_sha256"), "export_tool_versions": self.manifest.get("tool_versions"),
                "validation_scope": "Five-class author does not publish training membership; public-image compatibility diagnostic may overlap training, not independent accuracy" if self.variant == "fiveclass" else "Research option; not field-qualified",
                "input_shape": [1, 3, self.input_size, self.input_size], "cpu_threads": self.cpu_threads,
                "cpu_spinning": self.cpu_spinning,
                "optimization": "same FP32 weights; fixed spatial input; no quantization", "confidence_threshold": self.threshold,
                "confirm_frames": self.confirm_frames, "experimental": True,
                "model_source": {"multicountry": "https://huggingface.co/dronefreak/rdd2022-yolov8n", "india": "https://github.com/VivekVagale/road-hazard", "fiveclass": "https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022"}[self.variant],
                "license_status": "Multi-country YOLOv8n explicitly AGPL-3.0; India/fiveclass authors declare MIT but inherit Ultralytics/YOLO pretrained derivation; proprietary use not cleared",
                "limitations": "RDD2022 training domain; uncalibrated scores and unverified geographic generalization. No GPS, depth or engineering severity."}

    def process(self, frame: np.ndarray, context: FrameContext):
        from edge_runtime.profiles.roads import _iou
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or min(frame.shape[:2]) < 32):
            raise ValueError("Road model requires a uint8 BGR image at least 32px")
        if not math.isfinite(context.captured_monotonic):
            raise ValueError("captured_monotonic must be finite")
        identity = (context.source_id, context.session_id, frame.shape[:2])
        if (identity != self._identity or self._last_time is not None and context.captured_monotonic < self._last_time
                or self._last_sequence is not None and context.sequence <= self._last_sequence):
            self.reset()
        self._identity = identity
        self._last_time, self._last_sequence = context.captured_monotonic, context.sequence
        self._tracks = [t for t in self._tracks if context.captured_monotonic - t["time"] <= self.track_timeout]
        h, w = frame.shape[:2]
        size = self.input_size
        scale = min(size / h, size / w)
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
        left, top = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        tensor = cv2.dnn.blobFromImage(canvas, 1 / 255., (size, size), swapRB=True, crop=False)
        started = time.perf_counter()
        if self.engine == "opencv":
            self.net.setInput(tensor)
            output = self.net.forward()
        elif self.engine == "openvino":
            self.request.infer({0: tensor})
            output = self.request.get_output_tensor(0).data
        else:
            output = self.session.run(None, {self.input_name: tensor})[0]
        inference_ms = (time.perf_counter() - started) * 1000
        detections = decode(output, scale, (left, top), (h, w), self.threshold, self.nms_iou, len(self.class_names), size)
        annotated = frame.copy()
        findings, visible, events = [], [], []
        available = list(self._tracks)
        for detection in detections:
            bbox, label = detection["bbox"], detection["class_id"]
            same = [t for t in available if t["class_id"] == label]
            track = max(same, key=lambda t: _iou(t["bbox"], bbox), default=None)
            if track is not None and _iou(track["bbox"], bbox) >= 0.15:
                available.remove(track)
                track["hits"] += 1
            else:
                track = {"id": self._next_id, "hits": 1, "emitted": False, "class_id": label}
                self._next_id += 1
                self._tracks.append(track)
            track.update(bbox=bbox, time=context.captured_monotonic)
            confirmed = track["hits"] >= self.confirm_frames or track["emitted"]
            if confirmed and not track["emitted"]:
                track["emitted"] = True
                events.append(track["id"])
            name = self.class_names[label]
            score = detection["score"]
            findings.append(Finding("road_surface_repair" if label == 4 else "road_damage_candidate", name,
                                    bbox, round(score, 4), f"yolo_rdd2022_{self.variant}", "info"))
            visible.append({"id": track["id"], "class_id": label, "label": name, "bbox": list(bbox),
                            "model_score": round(score, 4), "confirmed": confirmed, "hits": track["hits"]})
            x, y, bw, bh = bbox
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (30, 180, 255), 2)
            cv2.putText(annotated, f"{name} {score:.2f} REVIEW", (x, max(18, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 180, 255), 1)
        cv2.putText(annotated, "LOCAL MODEL: human review required", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 180, 255), 2)
        confirmed_count = sum(t["confirmed"] for t in visible)
        # Keep memory bounded even if a direct caller supplies a frozen timestamp.
        self._tracks = sorted(self._tracks, key=lambda t: (t["time"], t["id"]), reverse=True)[:512]
        return InspectionResult(
            status="review" if confirmed_count else "pending" if detections else "no_detection",
            title="Road damage model: candidates for review" if detections else "Road damage model: no detections (not a safety assurance)",
            findings=findings, annotated_frame=annotated,
            metrics={"candidate_count": len(detections), "confirmed_count": confirmed_count,
                     "damage_candidate_count": sum(d["class_id"] < 4 for d in detections),
                     "repair_observation_count": sum(d["class_id"] == 4 for d in detections),
                     "model_inference_ms": round(inference_ms, 2), "experimental": True},
            guidance="Review candidate boxes. Model scores are not calibrated probabilities. RDD2022 training domain; misses and false alarms are possible. No depth or engineering severity.",
            action="request_review" if confirmed_count else "observe",
            emit_event=bool(events), event_key=f"roads-model:{context.session_id}:{context.source_id}:{','.join(map(str, events))}" if events else None,
            domain_state={"experimental": True, "backend": "model", "model_sha256": self.sha256,
                          "model_variant": self.variant, "engine": self.engine,
                          "engine_version": self.runtime_version,
                          "device": self.openvino_device if self.engine == "openvino" else "CPU",
                          "execution_device": getattr(self, "resolved_device", "CPU"),
                          "input_size": self.input_size, "cpu_threads": self.cpu_threads,
                          "tracks": visible, "new_track_ids": events, "scope": "model candidate review; no GPS/depth/severity"})
