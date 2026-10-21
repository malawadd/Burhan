"""Offline road damage model with a separate synthetic heuristic baseline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from edge_runtime.contracts import Finding, FrameContext, InspectionResult


@dataclass
class _Track:
    identifier: int
    bbox: tuple[int, int, int, int]
    last_seen: float
    last_sequence: int
    hits: int = 1
    emitted: bool = False


def _iou(a: tuple, b: tuple) -> float:
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - x) * max(0, bottom - y)
    return intersection / max(1, a[2] * a[3] + b[2] * b[3] - intersection)


class DarkRegionRoadsProfile:
    """Find compact dark candidate regions inside a manually calibrated road ROI.

    Scores measure darkness contrast, never a probability of road damage.
    Tracking assumes overlapping boxes in consecutive frames. No ego-motion,
    camera calibration, geographical deduplication or severity model is implied.
    """

    def __init__(self, config: dict | None = None):
        options = dict(config or {})
        self.roi = options.get("roi", [[0.39, 0.36], [0.61, 0.36], [0.94, 0.96], [0.06, 0.96]])
        points = np.asarray(self.roi, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            raise ValueError("roads roi must contain four normalized [x, y] points")
        self.contrast = float(options.get("min_contrast", 32.0))
        self.min_area = float(options.get("min_area_fraction", 0.0015))
        self.max_area = float(options.get("max_area_fraction", 0.065))
        self.min_focus = float(options.get("min_focus", 16.0))
        self.confirm_frames = max(1, int(options.get("confirm_frames", 3)))
        self.track_timeout = float(options.get("track_timeout_seconds", 1.0))
        self.match_iou = float(options.get("match_iou", 0.12))
        if self.track_timeout <= 0 or not 0 < self.match_iou <= 1:
            raise ValueError("track_timeout_seconds must be positive; match_iou must be in (0, 1]")
        if not 0 < self.min_area < self.max_area < 1:
            raise ValueError("area fractions must satisfy 0 < minimum < maximum < 1")
        self._next_id = 1
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._identity = None
        self._last_time = None
        self._last_sequence = None
        self._shape = None

    def describe(self) -> dict:
        return {
            "name": "Experimental road candidate review",
            "method": "OpenCV dark-region and shape heuristic with overlap tracking",
            "experimental": True,
            "requires": "Fixed camera, dry visible road, manually calibrated normalized ROI",
            "limitations": "Candidates are not verified potholes. No GPS, depth or engineering severity. Shadows can trigger alerts.",
            "roi": self.roi,
            "confirm_frames": self.confirm_frames,
            "track_timeout_seconds": self.track_timeout,
        }

    def process(self, frame: np.ndarray, context: FrameContext) -> InspectionResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("roads requires a uint8 BGR frame")
        if min(frame.shape[:2]) < 32:
            raise ValueError("roads frame dimensions must be at least 32 pixels")
        if not math.isfinite(context.captured_monotonic):
            raise ValueError("captured_monotonic must be finite")
        identity = (context.source_id, context.session_id)
        if (self._identity != identity or self._shape != frame.shape[:2]
                or (self._last_time is not None and context.captured_monotonic < self._last_time)
                or (self._last_sequence is not None and context.sequence <= self._last_sequence)):
            self.reset()
        self._identity, self._shape = identity, frame.shape[:2]
        self._last_time, self._last_sequence = context.captured_monotonic, context.sequence
        now = context.captured_monotonic
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.track_timeout]

        height, width = frame.shape[:2]
        polygon = np.round(np.asarray(self.roi) * [width - 1, height - 1]).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        roi_pixels = int(cv2.countNonZero(mask))
        if roi_pixels < 64:
            raise ValueError("roads ROI must cover a non-degenerate road area")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.median(gray[mask > 0]))
        focus = float(cv2.Laplacian(gray, cv2.CV_64F)[mask > 0].var())
        annotated = frame.copy()
        cv2.polylines(annotated, [polygon], True, (220, 170, 60), 2)
        cv2.putText(annotated, "EXPERIMENTAL: dark regions, not verified potholes", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.57, (30, 200, 255), 2)
        metrics = {"roi_brightness": round(brightness, 2), "roi_focus": round(focus, 2),
                   "experimental": True, "roi_pixels": roi_pixels}
        rejection = "underexposed" if brightness < 35 else "overexposed" if brightness > 235 else "blurred or textureless" if focus < self.min_focus else None
        if rejection:
            for track in self._tracks:
                if not track.emitted:
                    track.hits = 0
            return InspectionResult(
                status="rejected", title="Experimental road review: unusable frame",
                findings=[Finding("frame_quality", rejection, method="opencv_quality")],
                metrics=metrics, action="request_retake",
                guidance="Retake a focused, evenly lit view of the calibrated road area.",
                annotated_frame=annotated, domain_state={"experimental": True, "rejection": rejection})

        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        dark = np.where((smooth < brightness - self.contrast) & (mask > 0), 255, 0).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            aspect = w / max(1, h)
            # Large/elongated shadows, narrow markings and ROI boundary bands
            # are excluded. Compact shadows can still pass; no semantic claim.
            if not self.min_area <= area / roi_pixels <= self.max_area:
                continue
            if not 0.40 <= aspect <= 2.8 or area / max(1, w * h) < 0.30 or area / max(1, hull_area) < 0.50:
                continue
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            boundary = cv2.dilate(contour_mask, np.ones((5, 5), dtype=np.uint8))
            if np.any((boundary > 0) & (mask == 0)):
                continue
            contrast_score = (brightness - float(np.mean(gray[contour_mask > 0]))) / 255.0
            candidates.append(((int(x), int(y), int(w), int(h)), round(max(0.0, contrast_score), 3)))
        candidates.sort(key=lambda item: (item[0][1], item[0][0]))

        available = list(self._tracks)
        visible = []
        new_events = []
        findings = []
        for bbox, score in candidates:
            best = max(available, key=lambda track: _iou(track.bbox, bbox), default=None)
            if best is not None and _iou(best.bbox, bbox) >= self.match_iou:
                available.remove(best)
                best.hits = best.hits + 1 if context.sequence - best.last_sequence == 1 else 1
                best.bbox, best.last_seen, best.last_sequence = bbox, now, context.sequence
                track = best
            else:
                track = _Track(self._next_id, bbox, now, context.sequence)
                self._next_id += 1
                self._tracks.append(track)
            confirmed = track.hits >= self.confirm_frames or track.emitted
            if confirmed and not track.emitted:
                track.emitted = True
                new_events.append(track.identifier)
            label = f"Experimental dark candidate #{track.identifier}: {'review' if confirmed else 'pending'}"
            findings.append(Finding("road_dark_candidate", label, bbox=bbox, score=score,
                                    method="opencv_dark_region_heuristic", severity="info"))
            visible.append({"id": track.identifier, "bbox": list(bbox), "confirmed": confirmed,
                            "hits": track.hits, "contrast_score": score})
            x, y, w, h = bbox
            color = (0, 170, 255) if confirmed else (120, 200, 120)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"candidate {track.identifier} {'REVIEW' if confirmed else 'pending'}", (x, max(45, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
        confirmed_count = sum(item["confirmed"] for item in visible)
        metrics.update({"candidate_count": len(candidates), "confirmed_count": confirmed_count,
                        "active_track_count": len(self._tracks)})
        event_key = None
        if new_events:
            event_key = f"roads:{context.session_id}:{context.source_id}:" + ",".join(map(str, new_events))
        return InspectionResult(
            status="review" if confirmed_count else "pending" if candidates else "clear",
            title="Experimental road review: candidate regions" if candidates else "Experimental road review: no dark candidates",
            findings=findings, metrics=metrics,
            guidance="Human review required; dark paint, shadows, repairs and wet surfaces can mimic distress. No damage-free assurance.",
            action="request_review" if confirmed_count else "observe", annotated_frame=annotated,
            event_key=event_key, emit_event=bool(new_events),
            domain_state={"experimental": True, "tracks": visible, "new_track_ids": new_events,
                          "scope": "fixed-camera dark candidate heuristic; no GPS, depth or severity"})


def demo_frame(index: int, width: int = 960, height: int = 540) -> np.ndarray:
    """Deterministic synthetic asphalt with an irregular dark region moving down.

    Frames are generated here; process() never receives index or scene labels.
    An elongated shadow is included to exercise limited shape rejection.
    """
    rng = np.random.default_rng(114)
    texture = np.clip(145 + rng.normal(0, 4, (height, width)), 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame, (0, 0), (width - 1, int(height * 0.28)), (170, 184, 188), -1)
    cv2.line(frame, (int(width * 0.40), int(height * 0.35)), (int(width * 0.10), height - 1), (215, 215, 215), 5)
    cv2.line(frame, (int(width * 0.60), int(height * 0.35)), (int(width * 0.90), height - 1), (215, 215, 215), 5)
    # Short loop deliberately allows the candidate to leave/re-enter the ROI.
    progress = index % 100
    cx, cy = int(width * 0.5), int(height * (0.47 + 0.006 * progress))
    scale = min(width / 960.0, height / 540.0)
    points = np.array([[-41, -15], [-24, -25], [7, -24], [38, -11], [45, 7], [19, 23], [-16, 25], [-43, 10]], dtype=np.float32)
    points = np.round(points * scale + [cx, cy]).astype(np.int32)
    cv2.fillPoly(frame, [points], (49, 49, 49))
    cv2.polylines(frame, [points], True, (75, 75, 75), max(1, int(3 * scale)))
    cv2.ellipse(frame, (int(width * 0.5), int(height * 0.78)), (int(width * 0.17), max(2, int(height * 0.009))),
                -3, 0, 360, (80, 80, 80), -1)
    return frame


class RoadsProfile:
    """Select the trained local detector for real sources; preserve synthetic mode."""
    def __init__(self, config: dict | None = None):
        from edge_runtime.profiles.roads_model import RoadDamageModel
        options = dict(config or {})
        backend = options.get("backend", "heuristic" if options.get("demo", True) else "model")
        if backend not in {"heuristic", "model"}:
            raise ValueError("roads backend must be model or heuristic")
        self.backend = backend
        self._profile = RoadDamageModel(options) if backend == "model" else DarkRegionRoadsProfile(options)

    def process(self, frame, context):
        return self._profile.process(frame, context)

    def reset(self):
        return self._profile.reset()

    def describe(self):
        return {"profile_version": "roads-local-v4", "backend": self.backend, **self._profile.describe()}
