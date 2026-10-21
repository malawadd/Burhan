"""Local capture/inference supervision. No network dependency in the processing path."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import cv2
import numpy as np

from .buffer import FrameBuffer
from .contracts import FrameContext
from .sources import LocalSource, create_profile, demo_module
from .storage import EventStore
from .media import MediaWorker


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class RuntimeConfig:
    profile: str = "factory"
    source: str = "demo"
    input_fps: float = 8
    width: int = 960
    height: int = 540
    deadline_ms: float = 350
    save_evidence: bool = False
    triggered: bool = False
    profile_options: dict = field(default_factory=dict)
    source_hold_seconds: float = 30
    preview_fps: float = 6
    video_mode: str = "realtime"


@dataclass
class Packet:
    frame: np.ndarray
    context: FrameContext
    deadline_monotonic: float | None = None


class EdgeRuntime:
    WORKER_PREPARATION_TIMEOUT_SECONDS=15.
    def __init__(self, data_dir="runtime-data"):
        cv2.setNumThreads(2)  # bound CPU contention on the development laptop
        self.store = EventStore(Path(data_dir))
        self.store.start()
        self._lock = threading.RLock()
        self._control = threading.Lock()
        self._profile_lock = threading.Lock()
        self._stop = threading.Event()
        self._capture_done = threading.Event()
        self._worker_ready=threading.Event()
        self._worker_error=None
        self._threads: list[threading.Thread] = []
        self.config = RuntimeConfig()
        self.profile = None
        self.media = None
        self.inspection_media = None
        self._guided_preview = False
        self._inspection_jpeg = None
        self._inspection_result = None
        self._inspection_payloads = {}
        self._inspection_cache = {}
        self._inspection_cache_bytes = 0
        self._preview_times = deque(maxlen=2000)
        self.video = {}
        self.buffer = FrameBuffer()
        self.state = "stopped"
        self.error = None
        self.session_id = None
        self.description = {}
        self.last_result = None
        self._raw_frame = None
        self._last_context = None
        self._jpeg = None
        self._preview_sequence = None
        self._preview_generated = 0.
        self._last_seen = 0.
        self._started = 0.
        self._ended = None
        self._triggers: queue.Queue = queue.Queue(maxsize=32)
        self._latencies = deque(maxlen=2000)
        self._inference_times = deque(maxlen=2000)
        self._inspected_latencies = deque(maxlen=2000)
        self._model_inference_times = deque(maxlen=2000)
        self._seen_events = set()
        self._missed_units = set()
        self._completed_units = set()
        self._trigger_times = {}
        self._counts = {"captured": 0, "processed": 0, "dropped": 0, "late": 0, "source_skipped": 0,
                        "inspected_frames": 0, "guidance_frames": 0, "model_inference_frames": 0,
                        "inspection_errors": 0, "missed_units": 0, "coverage_gaps": 0, "async_completions": 0}

    def start(self, config: RuntimeConfig):
        if config.profile not in {"factory", "freight", "roads", "construction"}:
            raise ValueError("Unknown profile")
        if config.video_mode not in {"realtime", "every_frame"}:
            raise ValueError("Video mode must be realtime or every_frame")
        if config.profile == "construction" and (config.source == "demo" or config.source.startswith("pdf:")):
            raise ValueError("Construction requires real local camera, video or image input")
        if config.profile == "construction":
            from .construction_catalog import load_catalog
            catalog = load_catalog(verify_models=False)
            stage = config.profile_options.get("stage", "site_safety")
            if not isinstance(stage, str) or stage not in {"all", *(item["id"] for item in catalog["stages"])}:
                raise ValueError("Unknown construction stage; select a researched stage or all")
        if not math.isfinite(config.input_fps) or not 1 <= config.input_fps <= 30:
            raise ValueError("Input FPS must be between 1 and 30")
        if not math.isfinite(config.deadline_ms) or not 20 <= config.deadline_ms <= 5000:
            raise ValueError("Deadline must be between 20 and 5000 ms")
        if not math.isfinite(config.source_hold_seconds) or not 1 <= config.source_hold_seconds <= 300:
            raise ValueError("Source hold time must be between 1 and 300 seconds")
        if not math.isfinite(config.preview_fps) or not 0 <= config.preview_fps <= 30:
            raise ValueError("Preview FPS must be between 0 and 30")
        with self._control:
            self._stop_current()
            profile_config = dict(config.profile_options)
            profile_config["demo"] = config.source == "demo"
            if config.profile == "factory" and config.source == "demo":
                profile_config["reference"] = demo_module("factory").demo_frame(0, config.width, config.height)
            profile = create_profile(config.profile, profile_config)
            with self._lock:
                self.config = config
                self.profile = profile
                self.description = profile.describe()
                self.description.setdefault("profile_version", {"factory": "reference_difference_v1", "freight": "document_geometry_v1", "roads": "experimental_dark_regions_v1", "construction": "construction_real_v1"}[config.profile])
                self.description["configuration_sha256"] = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
                if getattr(profile, "reference", None) is not None:
                    self.description["reference_sha256"] = hashlib.sha256(profile.reference.tobytes()).hexdigest()
                self.session_id = str(uuid.uuid4())
                self.state, self.error = "starting", None
                self.last_result, self._jpeg, self._raw_frame = None, None, None
                self._guided_preview = config.profile == "construction" and bool(config.profile_options.get("inspection_task"))
                self._inspection_jpeg, self._inspection_result = None, None
                self._inspection_payloads.clear()
                self._inspection_cache.clear()
                self._inspection_cache_bytes = 0
                self._preview_times.clear()
                self.video = {}
                self._last_context = None
                self._preview_sequence = None
                self._preview_generated = 0.
                self._started = time.perf_counter()
                self._ended = None
                self._last_seen = 0
                self._latencies.clear()
                self._inference_times.clear()
                self._inspected_latencies.clear()
                self._model_inference_times.clear()
                self._seen_events.clear()
                self._missed_units.clear()
                self._completed_units.clear()
                self._trigger_times.clear()
                self._triggers = queue.Queue(maxsize=32)
                self._counts = dict.fromkeys(self._counts, 0)
                self.buffer = FrameBuffer(4, "fifo" if config.profile == "factory" else "latest")
                self._stop = threading.Event()
                self._capture_done = threading.Event()
                self._worker_ready=threading.Event()
                self._worker_error=None
                session = self.session_id
                self.media = MediaWorker(lambda jpeg, sequence: self._set_preview(session, jpeg, sequence),
                                         self._encoded_evidence, config.preview_fps)
                self.inspection_media = (MediaWorker(lambda jpeg, sequence: self._set_inspection(session, jpeg, sequence),
                                                     self._encoded_evidence, 30)
                                         if self._guided_preview else None)
            inference_thread=threading.Thread(target=self._infer,name="edge-inference",daemon=True)
            self._threads=[inference_thread]
            inference_thread.start()
            ready=self._worker_ready.wait(timeout=self.WORKER_PREPARATION_TIMEOUT_SECONDS)
            if not ready or self._worker_error is not None:
                message=self._worker_error or 'Inspection worker preparation exceeded the 15-second startup timeout'
                self._stop.set();self._capture_done.set()
                self.buffer.close()
                self.media.close(timeout=3)
                if self.inspection_media is not None:
                    self.inspection_media.close(timeout=3)
                inference_thread.join(timeout=.05)
                with self._lock:
                    self.error=message;self.state='startup_error';self._ended=time.perf_counter()
                self._event('session_start_failed',{'error':message,'capture_started':False})
                raise RuntimeError(message)
            # Preparation cost is reported in initialization/startup telemetry.
            # The acquisition clock starts only after the worker is ready.
            with self._lock:self._started=time.perf_counter()
            capture_thread=threading.Thread(target=self._capture,name="edge-capture",daemon=True)
            self._threads.append(capture_thread)
            capture_thread.start()
            self._event("session_started", {"config": asdict(config), "opencv": cv2.__version__,
                                            "input_provenance": "self_authored_synthetic" if config.source == "demo" else "local_camera" if config.source.startswith("camera:") else "local_document" if config.source.startswith(("image:", "images:", "pdf:")) else "local_file_replay"})

    def _stop_current(self):
        self._stop.set()
        with self._lock:
            self.state = "stopping"
        for packet in self.buffer.close():
            self._drop(packet, "stopped_with_pending_work")
        for thread in self._threads:
            thread.join(timeout=6)
        if any(thread.is_alive() for thread in self._threads):
            with self._lock:
                self.error = "Source is still closing; wait before restarting the station"
            raise RuntimeError("Source is still closing; wait before restarting the station")
        closer = getattr(self.profile, "close", None)
        if closer is not None and closer(timeout=3) is False:
            with self._lock:
                self.error = "Document worker is still closing; wait before restarting the station"
            raise RuntimeError(self.error)
        # Inspection ownership has ended. An auxiliary JPEG backlog must not hold
        # an accepted production trigger in a pending state after inspection stops.
        for unit, accepted_at in list(self._trigger_times.items()):
            self._miss(Packet(None, FrameContext(-1, accepted_at, self.config.source,
                                                self.session_id or "none", unit_id=unit)), "station_stopped")
        if self.media is not None and not self.media.close(timeout=3):
            with self._lock:
                self.error = "Media worker is still closing; wait before restarting the station"
            raise RuntimeError(self.error)
        if self.inspection_media is not None and not self.inspection_media.close(timeout=3):
            with self._lock:
                self.error = "Media worker is still closing; wait before restarting the station"
            raise RuntimeError(self.error)
        self._threads = []
        with self._lock:
            self.state = "stopped"
            self._ended = time.perf_counter()
            # A timed-out native worker may drain before a later stop retry. Its
            # temporary closing warning is then resolved; retain unrelated source
            # or inspection failures so an operator can still diagnose the session.
            if self.error in {
                "Source is still closing; wait before restarting the station",
                "Document worker is still closing; wait before restarting the station",
                "Media worker is still closing; wait before restarting the station",
            }:
                self.error = None

    def stop(self):
        with self._control:
            self._stop_current()

    def close(self):
        self.stop()
        self.store.close()

    def _event(self, kind, payload, jpeg=None):
        self.store.submit(self.config.profile, self.session_id or "none", kind, json_safe(payload), jpeg)

    def _set_preview(self, session, jpeg, sequence):
        with self._lock:
            if session == self.session_id:
                self._jpeg = jpeg
                self._preview_sequence = sequence
                self._preview_generated = time.perf_counter()
                self._preview_times.append(self._preview_generated)

    def _set_inspection(self, session, jpeg, sequence):
        """Publish an immutable image/result pair, never boxes on another frame."""
        with self._lock:
            payload = self._inspection_payloads.get(sequence)
            if session == self.session_id and payload is not None:
                self._inspection_jpeg = jpeg
                self._inspection_result = payload
                self._inspection_cache[sequence] = (jpeg, payload)
                self._inspection_cache_bytes += len(jpeg)
                while len(self._inspection_cache) > 32 or (self._inspection_cache_bytes > 8*1024*1024 and len(self._inspection_cache) > 1):
                    first = next(iter(self._inspection_cache))
                    old, _ = self._inspection_cache.pop(first)
                    self._inspection_cache_bytes -= len(old)
                self._inspection_payloads = {key: value for key, value in self._inspection_payloads.items() if key > sequence}

    def _encoded_evidence(self, profile, session, payload, jpeg):
        kind = "document_completed" if payload.get("async_source_completion") else "finding"
        self.store.submit(profile, session, kind, payload, jpeg)

    def _finding(self, payload, frame):
        if self.config.save_evidence and frame is not None:
            if self.media.evidence(frame, self.config.profile, self.session_id, payload):
                return
            payload = dict(payload)
            payload["evidence_image_error"] = "Bounded media queue rejected image; metadata retained"
        kind = "document_completed" if payload.get("async_source_completion") else "finding"
        self._event(kind, payload)

    def _miss(self, packet, reason):
        unit = packet.context.unit_id
        if not unit:
            return
        with self._lock:
            if unit in self._missed_units or unit in self._completed_units:
                return
            self._missed_units.add(unit)
            self._trigger_times.pop(unit, None)
            self._counts["missed_units"] += 1
        self._event("inspection_missed", {"unit_id": unit, "sequence": packet.context.sequence, "reason": reason})

    def _drop(self, packet, reason):
        with self._lock:
            self._counts["dropped"] += 1
            if self.config.profile in {"roads", "construction"}:
                self._counts["coverage_gaps"] += 1
        self._miss(packet, reason)

    def _expire_triggers(self, reason=None):
        now = time.perf_counter()
        with self._lock:
            pending = list(self._trigger_times.items())
        for unit, accepted in pending:
            if reason or now >= accepted + self.config.deadline_ms / 1000:
                self._miss(Packet(None, FrameContext(-1, accepted, self.config.source,
                    self.session_id or "none", unit_id=unit)), reason or "trigger_deadline_expired_without_result")

    def _finish_source(self):
        finalizer = getattr(self.profile, "finish_source", None)
        if finalizer is not None and self._last_context is not None:
            deadline = time.perf_counter() + 45
            while not self._stop.is_set():
                try:
                    with self._profile_lock:
                        result = finalizer(timeout=.2)
                        pending = getattr(self.profile, "source_completion_state", lambda: {})().get("pending_ocr", False)
                except Exception as exc:
                    with self._lock:
                        self._counts["inspection_errors"] += 1
                        self.error = f"Source finalization failed: {type(exc).__name__}: {exc}"
                    self._event("source_incomplete", {"reason": self.error})
                    break
                if result is not None:
                    # A document job completes independently of the frame-guidance deadline.
                    # Reuse its original context; never count this as a fresh captured frame.
                    context = self._last_context
                    payload = self._serialize_result(result, context, dict(self.description), None, None, None)
                    payload["async_source_completion"] = True
                    payload["timing_scope"] = "asynchronous document completion; excluded from guidance-frame latency percentiles"
                    if result.annotated_frame is not None:
                        self.media.preview(result.annotated_frame, context.sequence)
                    with self._lock:
                        self.last_result = payload
                        self._counts["async_completions"] += 1
                    if result.emit_event and result.event_key not in self._seen_events:
                        self._seen_events.add(result.event_key)
                        self._finding(payload, result.annotated_frame)
                    break
                if not pending:
                    break
                if time.perf_counter() >= deadline:
                    with self._lock:
                        self.error = "Source ended while final OCR remained incomplete after 45 seconds"
                    self._event("source_incomplete", {"reason": self.error})
                    break
        self._expire_triggers("source_ended_before_final_unit_result")
        with self._lock:
            if self.state == "draining":
                self.state = "eof"
            self._ended = time.perf_counter()

    def _serialize_result(self, result, context, metadata, latency_ms, infer_ms, late):
        payload = json_safe({"status": result.status, "title": result.title,
            "guidance": result.guidance, "action": result.action,
            "findings": [asdict(f) for f in result.findings],
            "metrics": result.metrics, "domain_state": result.domain_state,
            "sequence": context.sequence, "unit_id": context.unit_id,
            "session_id": context.session_id,
            "source_id": context.source_id, "captured_monotonic": context.captured_monotonic,
            "provenance": "synthetic_demo" if self.config.source == "demo" else "local_camera" if self.config.source.startswith("camera:") else "local_document" if self.config.source.startswith(("image:", "images:", "pdf:")) else "file_replay",
            "source_time_ms": context.source_time_ms,
            "source_frame_index": context.source_frame_index,
            "source_timestamp_semantics": context.source_timestamp_semantics,
            "coordinate_space": "analysis_input_frame",
            "source_dimensions": context.source_dimensions, "analysis_dimensions": context.analysis_dimensions,
            "host_receipt_to_decision_ms": None if latency_ms is None else round(latency_ms, 2),
            "inference_ms": None if infer_ms is None else round(infer_ms, 2), "late": late,
            "runtime_version": "0.1.0", "profile_version": metadata.get("profile_version"),
            "configuration_sha256": metadata.get("configuration_sha256"),
            "reference_sha256": metadata.get("reference_sha256")})
        payload["model"] = {k: metadata[k] for k in ("model_sha256", "model_path", "model_name", "model_license", "model_version", "model", "model_backend", "requested_backend", "backend_fallback_reason", "native_code_sha256", "engine", "engine_version", "device", "execution_device", "input_shape", "cpu_threads", "cpu_spinning", "performance_preset", "mode", "crack_variant", "damage_variant", "classes", "class_confidence", "selected_stage", "confirm_frames", "nms_iou", "preprocessing", "ppe_preprocess", "supported_capabilities", "roi", "mask_rendering", "mask_scope", "show_context_objects", "overlay_scope", "confidence_threshold", "model_source", "license_status", "validation_scope", "safety_validation", "manual_review_only", "decision_authority", "model_metadata_sha256", "decision_threshold", "minimum_component_area", "region_decision_threshold", "model_initialization", "hysteresis_growth_threshold") if k in metadata}
        if isinstance(metadata.get("selected_damage_classes"), list):
            payload["model"]["selected_damage_classes"] = list(metadata["selected_damage_classes"])
        if isinstance(metadata.get("diagnostic_only"), bool):
            payload["model"]["diagnostic_only"] = metadata["diagnostic_only"]
        if isinstance(metadata.get("exposure_policy"), str):
            payload["model"]["exposure_policy"] = metadata["exposure_policy"]
        if isinstance(metadata.get("supported_height_width_ratio"), list):
            payload["model"]["supported_height_width_ratio"] = list(metadata["supported_height_width_ratio"])
        if isinstance(metadata.get("admission_variant"), dict):
            payload["model"]["admission_variant"] = deepcopy(metadata["admission_variant"])
        if metadata.get("mode") == "ppe" and "preprocessing" in metadata:
            payload["model"]["ppe_preprocess"] = metadata["preprocessing"]
        if "sha256" in metadata:
            payload["model"]["model_sha256"] = metadata["sha256"]
        if metadata.get("mode") == "guided_construction":
            payload["model"]["inspection_task"] = metadata.get("inspection_task")
            payload["model"]["inspection_kind"] = metadata.get("inspection_kind")
            if isinstance(metadata.get("model_provenance"), dict):
                # Keep the actual weight hash/thresholds with persisted findings,
                # not only in the station's transient description.
                payload["model"]["guided_model_provenance"] = deepcopy(metadata["model_provenance"])
        return payload

    def _capture(self):
        source = LocalSource(self.config.source, self.config.profile, self.config.input_fps,
                             self.config.width, self.config.height, self.config.source_hold_seconds)
        source.video_mode = self.config.video_mode
        sequence, failures = 0, 0
        next_frame = time.perf_counter()
        active_trigger = None
        try:
            source.open()
            # Opening a camera/codec can take seconds. It must not create a
            # backlog in the new session's capture schedule.
            next_frame = time.perf_counter()
            video = dict(getattr(source, "video_properties", {}))
            rate = video.get("source_fps")
            capture_rate = min(self.config.input_fps, rate) if self.config.video_mode == "realtime" and rate else self.config.input_fps
            interval = 1/capture_rate
            previous_skipped = 0
            capture_properties = json_safe(getattr(source, "capture_properties", {}))
            with self._lock:
                self.state = "running"
                self.description["capture_backend"] = source.backend
                self.description["capture_properties"] = capture_properties
                self.description["timestamp_semantics"] = "host_receipt_after_read; camera exposure latency unmeasured"
                self.video = video
            while not self._stop.is_set():
                frame, source_ms = source.read()
                captured = time.perf_counter()
                video = dict(getattr(source, "video_properties", {}))
                skipped = int(getattr(source, "source_skipped", 0))
                with self._lock:
                    self.video = video
                    difference = max(0, skipped-previous_skipped)
                    self._counts["source_skipped"] += difference
                    self._counts["coverage_gaps"] += difference
                previous_skipped = skipped
                if frame is None:
                    if self.config.source.startswith(("file:", "images:", "pdf:")):
                        with self._lock:
                            self.state = "draining"
                        self._event("source_eof", {"frames": sequence})
                        break
                    failures += 1
                    if failures >= 3:
                        raise RuntimeError("Camera produced no frames; station inspection unavailable")
                    self._stop.wait(.2)
                    continue
                failures = 0
                sequence += 1
                current_properties = getattr(source, "capture_properties", {})
                if current_properties != capture_properties:
                    capture_properties = json_safe(current_properties)
                    with self._lock:
                        self.description["capture_properties"] = capture_properties
                with self._lock:
                    self._counts["captured"] += 1
                    self._raw_frame = frame
                    self._last_seen = captured
                if self.config.profile == "construction" and getattr(self, "_guided_preview", False):
                    self.media.preview(frame, sequence)
                unit = None
                deadline = None
                if self.config.profile == "factory" and self.config.triggered:
                    if active_trigger is not None:
                        current_unit, accepted_at = active_trigger
                        if current_unit in self._completed_units or current_unit in self._missed_units:
                            active_trigger = None
                        elif captured >= accepted_at + self.config.deadline_ms / 1000:
                            self._miss(Packet(frame, FrameContext(sequence, captured, self.config.source,
                                        self.session_id, source_ms, current_unit)), "trigger_deadline_expired")
                            active_trigger = None
                    if active_trigger is None:
                        try:
                            active_trigger = self._triggers.get_nowait()
                        except queue.Empty:
                            pass
                    if active_trigger is None:
                        # Preview stays live while waiting for an explicit local trigger.
                        self.media.preview(frame, sequence)
                        next_frame += interval
                        self._stop.wait(max(0., next_frame - time.perf_counter()))
                        continue
                    unit, accepted_at = active_trigger
                    deadline = accepted_at + self.config.deadline_ms / 1000
                context = FrameContext(sequence, captured, self.config.source, self.session_id,
                                       source_ms, unit,
                                       tuple(video["original_dimensions"]) if video.get("original_dimensions") else None,
                                       tuple(video["analysis_dimensions"]) if video.get("analysis_dimensions") else None,
                                       video.get("source_frame_index"), video.get("timestamp_semantics"))
                packet = Packet(frame, context, deadline)
                for dropped in self.buffer.put(packet):
                    self._drop(dropped, "inference_queue_overflow")
                next_frame += interval
                # No burst of stale frames after a slow source or CPU stall.
                if next_frame < time.perf_counter() - interval:
                    next_frame = time.perf_counter()
                self._stop.wait(max(0., next_frame - time.perf_counter()))
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "source_error"
            self._event("source_error", {"error": self.error})
        finally:
            try:
                source.close()
            finally:
                self._capture_done.set()

    def _infer(self):
        try:
            with self._profile_lock:
                prepare=getattr(self.profile,'prepare_worker',None)
                if callable(prepare):
                    prepare()
                    with self._lock:self.description.update(self.profile.describe())
        except Exception as exc:
            self._worker_error=f'Inspection worker preparation failed: {type(exc).__name__}: {exc}'
            self._stop.set()
            return
        finally:
            self._worker_ready.set()
        while not self._stop.is_set():
            self._expire_triggers()
            packet = self.buffer.get()
            if packet is None:
                if self._stop.is_set():
                    break
                if self._capture_done.is_set():
                    self._finish_source()
                    break
                continue
            started = time.perf_counter()
            if packet.context.unit_id and (packet.context.unit_id in self._completed_units or
                                           packet.context.unit_id in self._missed_units):
                self._drop(packet, "unit_already_terminal")
                continue
            age_ms = (started-packet.context.captured_monotonic)*1000
            absolute_deadline = packet.deadline_monotonic or (packet.context.captured_monotonic + self.config.deadline_ms / 1000)
            if started > absolute_deadline:
                with self._lock:
                    self._counts["late"] += 1
                self._miss(packet, "frame_stale_before_inference")
                self._drop(packet, "frame_stale_before_inference")
                continue
            try:
                with self._profile_lock:
                    method_snapshot = dict(self.description)
                    self._last_context = packet.context
                    result = self.profile.process(packet.frame, packet.context)
                finished = time.perf_counter()
                infer_ms = (finished-started)*1000
                latency_ms = (finished-packet.context.captured_monotonic)*1000
                late = finished > absolute_deadline
                payload = self._serialize_result(result, packet.context, method_snapshot, latency_ms, infer_ms, late)
                if late:
                    with self._lock:
                        self._counts["late"] += 1
                    self._miss(packet, "decision_after_deadline")
                    payload["status"] = "late"
                    payload["guidance"] = "Late decision: no timely clear result. " + payload["guidance"]
                elif packet.context.unit_id:
                    if result.status in {"needs_input", "rejected", "inconclusive"}:
                        self._miss(packet, "trigger_has_no_final_inspection_result")
                    elif result.status not in {"pending", "calibrating"}:
                        with self._lock:
                            self._completed_units.add(packet.context.unit_id)
                            self._trigger_times.pop(packet.context.unit_id, None)
                annotated = result.annotated_frame if result.annotated_frame is not None else packet.frame
                if self._guided_preview:
                    with self._lock:
                        self._inspection_payloads[packet.context.sequence] = deepcopy(payload)
                        # Encoding can lag inference. Bound metadata along with the image queue.
                        if len(self._inspection_payloads) > 8:
                            del self._inspection_payloads[min(self._inspection_payloads)]
                    self.inspection_media.preview(annotated, packet.context.sequence)
                else:
                    self.media.preview(annotated, packet.context.sequence)
                with self._lock:
                    self._counts["processed"] += 1
                    performed = bool(result.metrics.get("inspection_performed", True))
                    self._counts["inspected_frames"] += int(performed)
                    self._counts["guidance_frames"] += int(not performed)
                    self._counts["model_inference_frames"] += int(bool(result.metrics.get("model_inference_ran", False)))
                    if performed:
                        self._inspected_latencies.append(latency_ms)
                    if result.metrics.get("model_inference_ran", False):
                        self._model_inference_times.append(infer_ms)
                    self._latencies.append(latency_ms)
                    self._inference_times.append(infer_ms)
                    self.last_result = payload
                key = result.event_key or f"{result.status}:{packet.context.sequence}"
                if result.emit_event and not late and key not in self._seen_events:
                    self._seen_events.add(key)
                    self._finding(payload, annotated)
                # Keep auxiliary memory bounded even in unattended sessions.
                if len(self._seen_events) > 2000:
                    self._seen_events.clear()
                if len(self._completed_units) > 5000:
                    self._completed_units.clear()
                if len(self._missed_units) > 5000:
                    self._missed_units.clear()
            except Exception as exc:
                with self._lock:
                    self._counts["inspection_errors"] += 1
                    self.error = f"{type(exc).__name__}: {exc}"
                self._miss(packet, "inspection_exception")
                self._event("inspection_error", {"sequence": packet.context.sequence, "error": self.error})

    def action(self, action: str):
        with self._control:
            if self.profile is None:
                raise ValueError("Start a station first")
            if action == "trigger":
                if self.config.profile != "factory" or not self.config.triggered:
                    raise ValueError("Enable factory triggered mode first")
                if self.state not in {"starting", "running"} or self._stop.is_set():
                    raise ValueError("Start the factory station before accepting triggers")
                unit = "trigger-" + uuid.uuid4().hex[:12]
                try:
                    accepted_at = time.perf_counter()
                    with self._lock:
                        self._trigger_times[unit] = accepted_at
                    self._triggers.put_nowait((unit, accepted_at))
                except queue.Full:
                    with self._lock:
                        self._trigger_times.pop(unit, None)
                    raise ValueError("Trigger queue full; trigger was not accepted")
                return {"accepted": True, "unit_id": unit}
            with self._profile_lock:
                if action == "reset":
                    self.profile.reset()
                    self._seen_events.clear()
                elif action == "capture" and hasattr(self.profile, "request_capture"):
                    if self.state != "running":
                        raise ValueError("Start the camera or replay source before requesting a capture")
                    self.profile.request_capture()
                elif action == "calibrate" and hasattr(self.profile, "calibrate"):
                    if self.state != "running" or time.perf_counter()-self._last_seen > 2:
                        raise ValueError("A current, running source is required for reference approval")
                    with self._lock:
                        frame = None if self._raw_frame is None else self._raw_frame.copy()
                    if frame is None:
                        raise ValueError("No current frame to calibrate")
                    calibration = self.profile.calibrate(frame)
                    if isinstance(calibration, dict) and not calibration.get("ok", True):
                        raise ValueError(calibration.get("error", "Reference calibration failed"))
                else:
                    raise ValueError("Unsupported action for current profile")
                description = self.profile.describe()
                if getattr(self.profile, "reference", None) is not None:
                    description["reference_sha256"] = hashlib.sha256(self.profile.reference.tobytes()).hexdigest()
                with self._lock:
                    # Capture can publish actual driver/frame properties while an
                    # action refreshes the profile. Preserve the latest snapshot.
                    method_metadata = {key: self.description.get(key) for key in ("profile_version", "configuration_sha256", "capture_backend", "capture_properties", "timestamp_semantics")}
                    self.description = description
                    self.description.update(method_metadata)
            self._event("operator_action", {"action": action})
            return {"accepted": True, "action": action, "description": self.description}

    def preview(self):
        with self._lock:
            return self._jpeg

    def preview_frame(self):
        """Read the JPEG and its identity atomically for browser deduplication."""
        with self._lock:
            return self._jpeg, self._preview_sequence, self.session_id

    def inspection_preview(self, sequence=None, session_id=None):
        with self._lock:
            if self._inspection_jpeg is None or self._inspection_result is None:
                return None
            if session_id is not None and session_id != self.session_id:
                raise ValueError("Inspection session changed; refresh the paired inspection result")
            if sequence is not None:
                cached = self._inspection_cache.get(sequence)
                if cached is None:
                    raise ValueError("Inspected frame changed; refresh the paired inspection result")
                return cached[0]
            return self._inspection_jpeg

    def snapshot(self):
        with self._lock:
            now = time.perf_counter()
            elapsed = max(.001, (self._ended or now)-self._started) if self._started else 0
            times = np.asarray(self._latencies, dtype=np.float64)
            inference_times = np.asarray(self._inference_times, dtype=np.float64)
            inspected_times = np.asarray(self._inspected_latencies, dtype=np.float64)
            model_times = np.asarray(self._model_inference_times, dtype=np.float64)
            metrics = dict(self._counts)
            metrics.update({"uptime_s": round(elapsed, 1), "queue_depth": len(self.buffer),
                            "capture_fps": round(self._counts["captured"]/elapsed, 1) if elapsed else 0,
                            "processed_fps": round(self._counts["processed"]/elapsed, 1) if elapsed else 0,
                            "frame_age_ms": round((now-self._last_seen)*1000, 1) if self._last_seen else None,
                            "inference_ms": round(float(np.median(self._inference_times)), 1) if self._inference_times else 0,
                            "max_ms": round(float(times.max()), 1) if times.size else 0})
            metrics["inspected_fps"] = round(self._counts.get("inspected_frames", self._counts["processed"])/elapsed, 2) if elapsed else 0
            metrics["guidance_fps"] = round(self._counts.get("guidance_frames", 0)/elapsed, 2) if elapsed else 0
            metrics["model_inference_fps"] = round(self._counts.get("model_inference_frames", 0)/elapsed, 2) if elapsed else 0
            metrics["preview_fps"] = (round((len(self._preview_times)-1) / (self._preview_times[-1]-self._preview_times[0]), 2)
                                      if len(self._preview_times) > 1 and self._preview_times[-1] > self._preview_times[0] else 0)
            metrics["evidence_age_ms"] = (round(max(0, now-self._inspection_result["captured_monotonic"])*1000, 1)
                                          if self._inspection_result else None)
            for percentile in (50, 95, 99):
                metrics[f"p{percentile}_ms"] = round(float(np.percentile(times, percentile)), 1) if times.size else 0
                metrics[f"inference_p{percentile}_ms"] = round(float(np.percentile(inference_times, percentile)), 1) if inference_times.size else 0
                metrics[f"inspection_latency_p{percentile}_ms"] = round(float(np.percentile(inspected_times, percentile)), 1) if inspected_times.size else None
                metrics[f"model_inference_p{percentile}_ms"] = round(float(np.percentile(model_times, percentile)), 1) if model_times.size else None
            return json_safe({"running": self.state in {"starting", "running", "draining"} and not self._stop.is_set(),
                              "profile": self.config.profile, "source": self.config.source,
                              "state": self.state, "error": self.error, "session_id": self.session_id,
                              "config": asdict(self.config), "description": self.description,
                              "metrics": metrics, "last_result": self.last_result,
                              "inspection_result": self._inspection_result,
                              "inspection_cache": {"frames": len(self._inspection_cache), "jpeg_bytes": self._inspection_cache_bytes,
                                                   "frame_limit": 32, "jpeg_byte_limit": 8*1024*1024},
                              "preview_kind": "live_unannotated" if self._guided_preview else "inspected_frame",
                              "storage": self.store.status(),
                              "media": self.media.status() if self.media else {},
                              "inspection_media": self.inspection_media.status() if self.inspection_media else {},
                              "video": self.video,
                              "preview_sequence": self._preview_sequence,
                              "preview_age_ms": round((now-self._preview_generated)*1000,1) if self._preview_generated else None,
                              "cloud": {"enabled": False, "raw_media_sync": False,
                                        "status": "Not configured; local operation independent"},
                              "recording": self.config.save_evidence,
                              "item_coverage": "manual local triggers" if self.config.triggered else "untriggered; every-item coverage unverified"})
