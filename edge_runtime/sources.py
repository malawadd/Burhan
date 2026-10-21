"""Local-only camera, video and generated demo sources. No cloud calls."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import math
import os
import time

import cv2
import numpy as np


def demo_module(profile: str):
    if profile not in {"factory", "freight", "roads", "construction"}:
        raise ValueError("Unsupported profile")
    return import_module(f"edge_runtime.profiles.{profile}")


def create_profile(name: str, config: dict | None = None):
    names = {"factory": "FactoryProfile", "freight": "FreightProfile", "roads": "RoadsProfile", "construction": "ConstructionProfile"}
    return getattr(demo_module(name), names[name])(config or {})


class LocalSource:
    def __init__(self, source: str, profile: str, fps: float, width=960, height=540, hold_seconds=3):
        self.source, self.profile = source, profile
        self.fps, self.width, self.height = fps, width, height
        self.capture = None
        self.index = 0
        self.last_source_ms = None
        self.backend = "generated"
        self.capture_properties = {}
        self.video_mode = "realtime"
        self.video_properties = {}
        self.source_skipped = 0
        self._video_started = None
        self._video_first_timestamp = None
        self._video_timestamp_origin = None
        self._last_video_timestamp = None
        self._paths = []
        self._still = None
        self._pdf = None
        self._page_index = -1
        self._hold_frames = max(1, round(fps * hold_seconds))

    def open(self):
        if self.source == "demo":
            return
        if self.source.startswith(("image:", "images:", "pdf:")):
            kind, raw = self.source.split(":", 1)
            path = Path(raw).expanduser().resolve()
            if kind == "images":
                if not path.is_dir():
                    raise ValueError("Image folder does not exist")
                self._paths = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})[:10000]
                if not self._paths:
                    raise ValueError("Image folder contains no supported images")
            elif kind == "image":
                if not path.is_file():
                    raise ValueError("Local image does not exist")
                self._paths = [path]
            else:
                if not path.is_file():
                    raise ValueError("Local PDF does not exist")
                import pypdfium2 as pdfium
                self._pdf = pdfium.PdfDocument(str(path))
                if not len(self._pdf):
                    raise ValueError("PDF contains no pages")
            self.backend = "local_pdfium" if kind == "pdf" else "local_image"
            return
        if self.source.startswith("camera:"):
            index = int(self.source.split(":", 1)[1])
            # MSMF is the verified working backend on this laptop; DSHOW failed to open.
            backend = cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_ANY
            self.capture = cv2.VideoCapture(index, backend)
            errors = {}
            def request(name, prop, value):
                try:
                    return bool(self.capture.set(prop, value))
                except (cv2.error, TypeError, ValueError) as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    return False
            def reported(name, prop):
                try:
                    value = float(self.capture.get(prop))
                    return value if math.isfinite(value) and value > 0 else None
                except (cv2.error, TypeError, ValueError) as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    return None
            width_ok = request("width", cv2.CAP_PROP_FRAME_WIDTH, self.width)
            height_ok = request("height", cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            fps_ok = request("fps", cv2.CAP_PROP_FPS, self.fps)
            buffer_ok = request("buffer", cv2.CAP_PROP_BUFFERSIZE, 1)
            driver_width = reported("width", cv2.CAP_PROP_FRAME_WIDTH)
            driver_height = reported("height", cv2.CAP_PROP_FRAME_HEIGHT)
            dimensions = [round(driver_width), round(driver_height)] if driver_width and driver_height else None
            self.capture_properties = {
                "fps_requested": self.fps, "fps_request_accepted": fps_ok,
                "driver_fps": reported("fps", cv2.CAP_PROP_FPS),
                "requested_dimensions": [self.width, self.height],
                "width_request_accepted": width_ok, "height_request_accepted": height_ok,
                "driver_dimensions": dimensions, "effective_dimensions": dimensions,
                "dimensions_source": "driver_reported",
                "buffer_size_requested": 1, "buffer_request_accepted": buffer_ok,
                "driver_buffer_size": reported("buffer", cv2.CAP_PROP_BUFFERSIZE),
                "driver_fps_is_measured": False, "exposure_latency_ms": None,
                "property_errors": errors,
            }
        elif self.source.startswith("file:"):
            path = Path(self.source[5:]).expanduser().resolve()
            if not path.is_file():
                raise ValueError("Local replay file does not exist")
            self.capture = cv2.VideoCapture(str(path))
        else:
            raise ValueError("Use demo, camera:0, file:<video>, image:<image>, images:<folder>, or pdf:<document>")
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError("Cannot open source; check camera privacy settings or replay codec")
        self.backend = self.capture.getBackendName()
        if self.source.startswith("file:"):
            rate = self.capture.get(cv2.CAP_PROP_FPS)
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            valid_rate = math.isfinite(rate) and 0 < rate <= 1000
            self.video_properties = {"source_fps": rate if valid_rate else None,
                                     "frame_count": int(count) if math.isfinite(count) and count > 0 else None,
                                     "duration_ms": count*1000/rate if valid_rate and math.isfinite(count) and count > 0 else None,
                                     "mode": self.video_mode, "source_skipped": 0,
                                     "playback_clock": "nominal_source_fps" if valid_rate else "requested_input_fps_fallback",
                                     "inference_coverage": "bounded runtime queue; decoded frames can still be dropped before inspection",
                                     "timestamp_semantics": "container timestamp; exposure time unknown"}

    def read(self) -> tuple[np.ndarray | None, float | None]:
        if self.source == "demo":
            frame = demo_module(self.profile).demo_frame(self.index, self.width, self.height)
            source_ms = self.index * 1000 / self.fps
            self.index += 1
            return frame, source_ms
        if self.source.startswith(("image:", "images:", "pdf:")):
            page_index = 0 if self.source.startswith("image:") else self.index // self._hold_frames
            count = len(self._pdf) if self._pdf is not None else len(self._paths)
            if page_index >= count:
                return None, None
            if page_index != self._page_index:
                if self._pdf is not None:
                    page = self._pdf[page_index]
                    try:
                        scale = min(2., 1800 / max(page.get_size()))
                        bitmap = page.render(scale=scale)
                        try:
                            array = bitmap.to_numpy()
                            # PDFium's default render format is BGR (or BGRA).
                            image = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR) if array.shape[2] == 4 else array.copy()
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
                else:
                    # imdecode handles Unicode Windows paths unlike imread on some builds.
                    encoded = np.fromfile(self._paths[page_index], np.uint8)
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Unsupported or corrupt image: {self._paths[page_index].name}")
                h, w = image.shape[:2]
                if max(h, w) > 1920:
                    image = cv2.resize(image, (max(1, round(w*1920/max(h,w))), max(1, round(h*1920/max(h,w)))), interpolation=cv2.INTER_AREA)
                self._still = image
                self._page_index = page_index
            source_ms = self.index * 1000 / self.fps
            self.index += 1
            return self._still.copy(), source_ms
        if self.source.startswith("file:") and self.video_mode == "realtime" and self.video_properties.get("source_fps"):
            now = time.perf_counter()
            if self._video_started is None:
                self._video_started = now
            target = int((now-self._video_started)*self.video_properties["source_fps"])
            while self.index < target:
                if not self.capture.grab():
                    self.video_properties = {**self.video_properties, "source_skipped": self.source_skipped}
                    return None, None
                self.index += 1
                self.source_skipped += 1
        ok, frame = self.capture.read()
        if not ok:
            return None, None
        if self.source.startswith("camera:"):
            dimensions = [int(frame.shape[1]), int(frame.shape[0])]
            if (self.capture_properties.get("effective_dimensions") != dimensions or
                    self.capture_properties.get("dimensions_source") != "observed_frame"):
                # Copy-on-write: runtime/API snapshots can retain prior metadata safely.
                self.capture_properties = {**self.capture_properties,
                                           "effective_dimensions": dimensions,
                                           "dimensions_source": "observed_frame"}
        source_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC) if self.source.startswith("file:") else None
        self.index += 1
        if self.source.startswith("file:"):
            timestamp_valid = (source_ms is not None and math.isfinite(source_ms) and source_ms >= 0
                               and (self._last_video_timestamp is None or source_ms > self._last_video_timestamp))
            if not timestamp_valid:
                rate = self.video_properties.get("source_fps")
                source_ms = ((self._video_timestamp_origin or 0.) + (self.index - 1) * 1000 / rate) if rate else None
                semantics = "nominal frame-index timestamp fallback; exposure time unknown"
            else:
                semantics = "container timestamp; exposure time unknown"
            if source_ms is not None:
                self._last_video_timestamp = source_ms
                rate = self.video_properties.get("source_fps")
                if self._video_timestamp_origin is None and rate:
                    self._video_timestamp_origin = source_ms - (self.index - 1) * 1000 / rate
            original = (int(frame.shape[1]),int(frame.shape[0]))
            if max(original) > 1920:
                ratio = 1920/max(original)
                frame = cv2.resize(frame,(max(1,round(original[0]*ratio)),max(1,round(original[1]*ratio))),interpolation=cv2.INTER_AREA)
            analysis = (int(frame.shape[1]),int(frame.shape[0]))
            if self._video_first_timestamp is None and source_ms is not None:
                self._video_first_timestamp = source_ms
            wall_ms = (time.perf_counter()-self._video_started)*1000 if self._video_started is not None else None
            lag_ms = max(0.,wall_ms-(source_ms-self._video_first_timestamp)) if wall_ms is not None and source_ms is not None else None
            self.video_properties = {**self.video_properties, "position_ms": source_ms,
                                     "source_wall_elapsed_ms": wall_ms, "replay_lag_ms": lag_ms,
                                     "timestamp_semantics": semantics,
                                     "original_dimensions": list(original), "analysis_dimensions": list(analysis),
                                     "scale_to_source": [original[0]/analysis[0],original[1]/analysis[1]],
                                     "source_frame_index": self.index-1, "source_skipped": self.source_skipped}
        return frame, source_ms

    def close(self):
        if self.capture is not None:
            self.capture.release()
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None


def probe_camera(index=0) -> dict:
    """Read a few frames without saving or returning image contents."""
    source = LocalSource(f"camera:{index}", "factory", 12, 640, 480)
    started = time.perf_counter()
    try:
        source.open()
        frames = []
        for _ in range(3):
            frame, _ = source.read()
            if frame is not None:
                frames.append(frame)
        return {"opened": bool(frames), "frames_read": len(frames),
                "dimensions": list(frames[-1].shape[:2][::-1]) if frames else None,
                "backend": source.backend, "capture_properties": source.capture_properties, "elapsed_ms": round((time.perf_counter()-started)*1000, 1),
                "saved_media": False}
    except Exception as exc:
        return {"opened": False, "error": str(exc), "saved_media": False}
    finally:
        source.close()
