"""Fixed-view, explicitly calibrated visual product-fault baseline.

This module inspects pixels. Demo scene indices are used only by ``demo_frame``.
There is no generic machine diagnosis or electrical-function inference.
"""
from __future__ import annotations

import math
from copy import deepcopy
import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge_runtime.contracts import Finding, FrameContext, InspectionResult


def demo_frame(index: int, width: int = 960, height: int = 540) -> np.ndarray:
    """Self-authored synthetic board: clean, omission, scratch, displaced chip.

    Sixty frames per condition. Call index=0 for the approved DEMO reference.
    No text or hidden metadata encodes the condition in the returned frame.
    """
    scene = np.full((540, 960, 3), (31, 35, 39), np.uint8)
    cv2.rectangle(scene, (115, 75), (845, 465), (64, 112, 51), -1)
    cv2.rectangle(scene, (115, 75), (845, 465), (125, 170, 110), 3)
    rng = np.random.default_rng(194)
    for _ in range(40):
        x, y = int(rng.integers(145, 805)), int(rng.integers(100, 435))
        cv2.line(scene, (x, y), (x + 22, y), (60, 158, 155), 2)
        cv2.circle(scene, (x, y), 3, (80, 190, 180), 1)
    for x, y in [(140, 100), (820, 100), (140, 440), (820, 440)]:
        cv2.circle(scene, (x, y), 11, (210, 215, 205), -1)
        cv2.circle(scene, (x, y), 5, (20, 25, 20), -1)
    cv2.putText(scene, "INSPECTFLOW DEMO PCB / REV A", (175, 127),
                cv2.FONT_HERSHEY_SIMPLEX, .57, (205, 225, 200), 1, cv2.LINE_AA)
    condition = (max(0, int(index)) // 60) % 4
    for n, (x, y, w, h) in enumerate([(205, 180, 100, 65), (400, 170, 115, 85),
                                      (620, 185, 100, 65), (315, 330, 130, 65)]):
        for yy in range(y + 7, y + h - 2, 10):
            cv2.line(scene, (x - 9, yy), (x + w + 9, yy), (175, 188, 190), 4)
        if condition == 1 and n == 2:
            continue
        dx = 24 if condition == 3 and n == 1 else 0
        cv2.rectangle(scene, (x + dx, y), (x + dx + w, y + h), (25, 27, 31), -1)
        cv2.rectangle(scene, (x + dx, y), (x + dx + w, y + h), (80, 84, 88), 2)
        cv2.circle(scene, (x + dx + 9, y + 9), 3, (185, 190, 190), -1)
        cv2.putText(scene, f"U{n + 1}", (x + dx + 23, y + h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (165, 170, 175), 1, cv2.LINE_AA)
    for n in range(8):
        x = 540 + n * 25
        cv2.rectangle(scene, (x, 353), (x + 11, 382), (185, 190, 190), -1)
        cv2.rectangle(scene, (x, 361), (x + 11, 374), (32, 46, 115), -1)
    if condition == 2:
        cv2.line(scene, (555, 290), (735, 330), (235, 238, 240), 5, cv2.LINE_AA)
    if (width, height) != (960, 540):
        scene = cv2.resize(scene, (width, height), interpolation=cv2.INTER_AREA)
    return scene


class FactoryProfile:
    """Reference differencing with affine registration and finite confirmation."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.demo = bool(self.config.get("demo", False))
        self.reference: np.ndarray | None = None
        self._reference_gray: np.ndarray | None = None
        self._reference_keypoints = None
        self._reference_descriptors = None
        self._calibration_error = "Explicit calibration with an approved good item is required."
        self._orb = cv2.ORB_create(nfeatures=1400, fastThreshold=12)
        self._defect_model = None
        self.reset()
        if self.config.get("defect_model_path"):
            from .factory_defect import SurfaceDefectModel
            self._defect_model = SurfaceDefectModel(self.config["defect_model_path"],
                                                   self.config.get("defect_backend", "auto"))
        elif self.config.get("reference_path"):
            ref = cv2.imread(str(Path(self.config["reference_path"])))
            self.calibrate(ref)
        elif isinstance(self.config.get("reference"), np.ndarray):
            self.calibrate(self.config["reference"])
        elif self.demo:
            self.calibrate(demo_frame(0))

    def describe(self) -> dict[str, Any]:
        if self._defect_model is not None:
            return {"id": "factory", "name": "Factory surface-defect inspection",
                    "mode": "local_trained_surface_model",
                    "profile_version": self._defect_model.metadata.get("profile_version", "kolektor_rf_v2_cascade"),
                    "model_backend": self._defect_model.backend,
                    "requested_backend": self._defect_model.requested_backend,
                    "backend_fallback_reason": self._defect_model.backend_fallback_reason,
                    "native_code_sha256": self._defect_model.native.sha256 if self._defect_model.native else None,
                    "calibrated": True, "model_path": str(self._defect_model.path),
                    "model_sha256": self._defect_model.sha256,
                    "model_metadata_sha256":self._defect_model.metadata_sha256,
                    "decision_threshold":self._defect_model.pixel_threshold,
                    "minimum_component_area":self._defect_model.area_threshold,
                    "region_decision_threshold":getattr(self._defect_model,'region_threshold',None),
                    "model_metadata": self._defect_model.metadata,
                    "model_initialization":getattr(self._defect_model.neural,'initialization_report',None) or getattr(self,'_quality_initialization',None),
                    "hysteresis_growth_threshold":self._defect_model.metadata.get('hysteresis_growth_threshold'),
                    "supported_height_width_ratio":list(self._defect_model.metadata.get('supported_height_width_ratio',[.25,4.0])),
                    "admission_variant":deepcopy(self._defect_model.metadata.get('admission_variant')),
                    "supported_scope": "Kolektor-like grayscale commutator surfaces; not arbitrary factory scenes",
                    "score_meaning": self._defect_model.metadata.get("score_meaning", "Mean tree leaf defect fraction; not calibrated probability"),
                    "confirmation_frames": max(1, int(self.config.get("confirmation_frames", 3))),
                    "manual_review_only":bool(self._defect_model.metadata.get('manual_review_only',False)),
                    "decision_authority":"human_review_only" if self._defect_model.metadata.get('manual_review_only',False) else "prototype_candidate_detection",
                    "triggered_unit_semantics":"Completed observation only; no machine acceptance/rejection decision",
                    "calibration_required_for_camera": False, "field_validated": False}
        return {"id": "factory", "name": "Factory visual fault inspection",
                "mode": "synthetic_demo" if self.demo else "explicit_reference",
                "calibrated": self.reference is not None,
                "supported_scope": "One fixed-view visible product family; affine camera jitter only",
                "confirmation_frames": max(1, int(self.config.get("confirmation_frames", 3))),
                "calibration_required_for_camera": True,
                "score_meaning": "Pixel difference and area measurements, not calibrated probability",
                "coverage": "Every-item coverage requires external triggers and deadline monitoring",
                "field_validated": False}

    def prepare_worker(self):
        """Warm kernels on their actual inspection thread before capture opens."""
        if self._defect_model is not None:
            started=time.perf_counter()
            self._fast_surface_quality(np.zeros((1280,500,3),dtype=np.uint8))
            quality={'quality_warmup_ms':round((time.perf_counter()-started)*1000,3),
                     'quality_warmup_thread':threading.current_thread().name,
                     'quality_warmup_inspected_observations':0}
            self._quality_initialization=quality
            if self._defect_model.neural is not None:
                self._defect_model.neural.warmup(self._defect_model.pixel_threshold,self._defect_model.area_threshold)
                self._defect_model.neural.initialization_report.update(quality)
                return dict(self._defect_model.neural.initialization_report)
            return quality
        return None

    def reset(self) -> None:
        """Reset event history, preserving the explicitly approved reference."""
        self._generation = getattr(self, "_generation", 0) + 1
        self._candidate_mask: np.ndarray | None = None
        self._candidate_count = 0
        self._clean_count = 0
        self._latched = False
        self._event_number = 0
        self._unit_id: str | None = None

    @staticmethod
    def _quality(frame: np.ndarray | None) -> tuple[np.ndarray | None, dict, str]:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 64):
            return None, {}, "Expected a uint8 BGR image at least 64 pixels on each side."
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        metrics = {"laplacian_variance": round(sharpness, 3),
                   "grayscale_std": round(contrast, 3)}
        return gray, metrics, ""

    @staticmethod
    def _fast_surface_quality(frame: np.ndarray | None) -> tuple[np.ndarray | None, dict, str]:
        """Same integer Laplacian, with native double-precision reductions.

        A uint8 ksize=1 Laplacian is an integer in [-1020, 1020], so int16
        stores exactly the same values as the old float64 intermediate.
        meanStdDev avoids large float64 arrays and NumPy variance temporaries.
        """
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 64):
            return None, {}, "Expected a uint8 BGR image at least 64 pixels on each side."
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_16S)
        _, lap_std = cv2.meanStdDev(laplacian)
        _, gray_std = cv2.meanStdDev(gray)
        sharpness, contrast = float(lap_std[0, 0]) ** 2, float(gray_std[0, 0])
        # Reduction order can differ at machine precision. Resolve an input
        # within 1e-6 of a three-decimal rounding boundary using the old path.
        if any(abs(value * 1000 - (math.floor(value * 1000) + .5)) < .001
               for value in (sharpness, contrast)):
            return FactoryProfile._quality(frame)
        return gray, {"laplacian_variance": round(sharpness, 3),
                      "grayscale_std": round(contrast, 3)}, ""

    def calibrate(self, frame: np.ndarray | None) -> dict[str, Any]:
        """Operator declaration of an approved good reference, never auto-learning."""
        if self._defect_model is not None:
            return {"ok": False, "error": "Surface-model mode is fixed. Restart without defect_model_path to use approved-reference calibration."}
        gray, metrics, error = self._quality(frame)
        if not error and (metrics["grayscale_std"] < 10 or
                          metrics["laplacian_variance"] < float(self.config.get("min_sharpness", 25))):
            error = "Reference lacks contrast or is blurred; supply an approved, focused good item."
        if not error:
            keypoints, descriptors = self._orb.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < 15:
                error = "Reference has too few stable features for supported registration."
        if error:
            self._calibration_error = error
            return {"ok": False, "error": error, "retained_previous_reference": self.reference is not None}
        self.reference = frame.copy()
        self._reference_gray = gray.copy()
        self._reference_keypoints, self._reference_descriptors = keypoints, descriptors
        self._calibration_error = ""
        self.reset()
        return {"ok": True, "width": frame.shape[1], "height": frame.shape[0],
                "reference_source": "operator_approved" if not self.demo else "demo_or_operator_approved"}

    set_reference = calibrate

    def _inconclusive(self, frame, title: str, metrics: dict, guidance: str,
                      status: str = "needs_input") -> InspectionResult:
        self._candidate_count = self._clean_count = 0
        self._candidate_mask = None
        return InspectionResult(status=status, title=title, metrics=metrics, guidance=guidance,
                                action="request_input", annotated_frame=frame.copy() if isinstance(frame, np.ndarray) else None,
                                domain_state={"calibrated": self.reference is not None or self._defect_model is not None,
                                              "demo": self.demo and self._defect_model is None, "field_validated": False})

    def process(self, frame: np.ndarray, context: FrameContext) -> InspectionResult:
        quality = self._fast_surface_quality if self._defect_model is not None and self._defect_model.backend != "sklearn" else self._quality
        gray, metrics, error = quality(frame)
        if error:
            return self._inconclusive(frame, "Unsupported camera input", metrics, error)
        if self._defect_model is not None:
            return self._process_surface(frame, context, metrics)
        if self.reference is None:
            return self._inconclusive(frame, "Calibration required", metrics, self._calibration_error)
        if frame.shape != self.reference.shape:
            return self._inconclusive(frame, "Camera resolution changed", metrics,
                                      "Restore the calibrated resolution or explicitly calibrate a new reference.")
        if metrics["grayscale_std"] < 10 or metrics["laplacian_variance"] < float(self.config.get("min_sharpness", 25)):
            return self._inconclusive(frame, "Image quality prevents inspection", metrics,
                                      "Restore focus, lighting and the approved view; no clear result was issued.")
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None:
            return self._inconclusive(frame, "Registration failed", metrics, "Restore the approved product view.")
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, self._reference_descriptors, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .72 * b.distance]
        metrics["registration_matches"] = len(good)
        if len(good) < 12:
            return self._inconclusive(frame, "Registration failed", metrics, "Too few matching features; restore the approved product view.")
        current = np.float32([keypoints[m.queryIdx].pt for m in good])
        target = np.float32([self._reference_keypoints[m.trainIdx].pt for m in good])
        transform, inliers = cv2.estimateAffinePartial2D(current, target, method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.5)
        if transform is None or inliers is None:
            return self._inconclusive(frame, "Registration failed", metrics, "A stable affine view could not be estimated.")
        scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
        angle = math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))
        ratio = float(inliers.mean())
        metrics.update({"registration_inlier_ratio": round(ratio, 4), "view_rotation_deg": round(angle, 3)})
        h, w = gray.shape
        if (ratio < .5 or not .90 <= scale <= 1.10 or abs(angle) > 10 or
                abs(transform[0, 2]) > w * .15 or abs(transform[1, 2]) > h * .15):
            return self._inconclusive(frame, "Unsupported view change", metrics,
                                      "Return to the calibrated fixture; this baseline supports small affine camera jitter.")
        aligned = cv2.warpAffine(gray, transform, (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(np.full(gray.shape, 255, np.uint8), transform, (w, h), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        valid[:5] = valid[-5:] = 0
        valid[:, :5] = valid[:, -5:] = 0
        difference = cv2.absdiff(cv2.GaussianBlur(aligned, (3, 3), 0),
                                 cv2.GaussianBlur(self._reference_gray, (3, 3), 0))
        threshold = max(1, min(254, int(self.config.get("difference_threshold", 28))))
        mask = np.where((difference > threshold) & (valid > 0), 255, 0).astype(np.uint8)
        for region in self.config.get("ignore_regions", []):
            x, y, rw, rh = map(int, region)
            mask[max(0, y):max(0, y + rh), max(0, x):max(0, x + rw)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = float(self.config.get("minimum_area", 65))
        contours = [c for c in contours if cv2.contourArea(c) >= minimum_area]
        retained = np.zeros_like(mask)
        cv2.drawContours(retained, contours, -1, 255, -1)
        changed = int(np.count_nonzero(retained))
        metrics.update({"changed_pixels": changed, "changed_fraction": round(changed / (w * h), 6),
                        "difference_threshold_0_255": threshold, "candidate_regions": len(contours),
                        "reference_affine_current_to_reference": transform.tolist(),
                        "triggered_unit": context.unit_id is not None})
        if changed > w * h * float(self.config.get("maximum_changed_fraction", .22)):
            return self._inconclusive(frame, "Scene differs too much for this reference", metrics,
                                      "Confirm the product family and lighting; explicitly recalibrate only with an approved good item.", "inconclusive")
        if context.unit_id is not None and context.unit_id != self._unit_id:
            self._candidate_count = self._clean_count = 0
            self._candidate_mask = None
            self._latched = False
        self._unit_id = context.unit_id
        annotated = frame.copy()
        findings = []
        inverse = cv2.invertAffineTransform(transform)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            x, y, rw, rh = cv2.boundingRect(contour)
            corners = np.float32([[[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]]])
            original = cv2.transform(corners, inverse)[0]
            x0 = max(0, min(w - 1, math.floor(float(original[:, 0].min()))))
            y0 = max(0, min(h - 1, math.floor(float(original[:, 1].min()))))
            x1 = max(x0 + 1, min(w, math.ceil(float(original[:, 0].max()))))
            y1 = max(y0 + 1, min(h, math.ceil(float(original[:, 1].max()))))
            bbox = (x0, y0, x1 - x0, y1 - y0)
            findings.append(Finding("visual_deviation", "Visible difference from approved reference", bbox,
                                    method="opencv_affine_reference_difference", severity="warning"))
            cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (35, 65, 245), 2)
        emit = False
        if contours:
            self._clean_count = 0
            same = False
            if self._candidate_mask is not None:
                overlap = np.count_nonzero((retained > 0) & (self._candidate_mask > 0))
                union = np.count_nonzero((retained > 0) | (self._candidate_mask > 0))
                same = union > 0 and overlap / union >= .25
            self._candidate_count = self._candidate_count + 1 if same else 1
            self._candidate_mask = retained
            required = max(1, int(self.config.get("trigger_confirmation_frames" if context.unit_id is not None else "confirmation_frames",
                                                1 if context.unit_id is not None else 3)))
            confirmed = self._candidate_count >= required
            status = "fault" if confirmed else "pending"
            title = "Visual fault requires review" if confirmed else "Verifying a visible difference"
            if confirmed and not self._latched:
                self._latched = emit = True
                self._event_number += 1
            guidance = "Review marked regions against the approved reference; electrical function and root cause remain unknown."
        else:
            self._candidate_count = 0
            self._candidate_mask = None
            self._clean_count += 1
            if self._clean_count >= max(1, int(self.config.get("clear_confirmation_frames", 3))):
                self._latched = False
            status, title = "clear", "No selected visible difference found"
            guidance = "Supported visible checks only; this is not electrical or machine-health certification."
        metrics["confirmation_count"] = self._candidate_count
        state = {"calibrated": True, "demo": self.demo, "field_validated": False,
                 "unit_id": context.unit_id, "continuous_view": context.unit_id is None,
                 "coverage": "untriggered_observation" if context.unit_id is None else "triggered_observation"}
        event_key = (f"factory:{context.session_id}:{context.source_id}:{self._generation}:"
                     f"{context.unit_id if context.unit_id is not None else self._event_number}") if self._latched else None
        cv2.putText(annotated, ("SYNTHETIC DEMO | " if self.demo else "") + title, (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 2, cv2.LINE_AA)
        return InspectionResult(status=status, title=title, findings=findings, metrics=metrics,
                                guidance=guidance, action="human_review" if status == "fault" else "observe",
                                annotated_frame=annotated, event_key=event_key, emit_event=emit,
                                domain_state=state)

    def _process_surface(self, frame, context, metrics):
        model = self._defect_model
        review_only = bool(model.metadata.get('manual_review_only',False))
        if metrics["grayscale_std"] < 5 or metrics["laplacian_variance"] < float(self.config.get("model_min_sharpness", 8)):
            return self._inconclusive(frame, "Surface image quality prevents inspection", metrics,
                                      "Supply a focused, lit grayscale surface close-up; no clear result was issued.")
        # The trained model is explicitly narrow; reject geometry outside its acquisition family.
        aspect = frame.shape[0] / frame.shape[1]
        minimum, maximum = model.metadata.get("supported_height_width_ratio", [.25, 4.0])
        if not minimum <= aspect <= maximum:
            return self._inconclusive(frame, "Surface model view does not match", metrics,
                                      "Use the original portrait surface crop or restart with an approved-reference configuration.", "inconclusive")
        _, candidates, model_metrics = model.predict(frame)
        metrics.update(model_metrics)
        if context.unit_id is not None and context.unit_id != self._unit_id:
            self._candidate_count = self._clean_count = 0
            self._candidate_mask = None
            self._latched = False
        self._unit_id = context.unit_id
        annotated = frame.copy()
        findings = []
        mask = np.zeros(frame.shape[:2], np.uint8)
        for item in candidates:
            x, y, w, h = item["bbox"]
            mask[y:y+h, x:x+w] = 1
            method=('local_neural_surface_segmentation' if model.neural is not None else
                    'two_tree_proposals_learned_region_rf' if model.region_model is not None else
                    'opencv_features_random_forest_surface_pixels')
            findings.append(Finding("surface_review_candidate" if review_only else "surface_defect",
                                    "Surface review candidate" if review_only else "Surface model candidate", item["bbox"],
                                    score=round(item["score"], 5), method=method,
                                    severity="warning"))
            cv2.rectangle(annotated, (x, y), (x+w-1, y+h-1), (35, 65, 245), 2)
        emit = False
        if candidates:
            same = False
            if self._candidate_mask is not None and self._candidate_mask.shape == mask.shape:
                overlap = np.count_nonzero((mask > 0) & (self._candidate_mask > 0))
                union = np.count_nonzero((mask > 0) | (self._candidate_mask > 0))
                same = union > 0 and overlap / union >= .25
            self._candidate_count = self._candidate_count + 1 if same else 1
            self._candidate_mask = mask
            self._clean_count = 0
            required = max(1, int(self.config.get("trigger_confirmation_frames" if context.unit_id else "confirmation_frames",
                                                1 if context.unit_id else 3)))
            confirmed = self._candidate_count >= required
            status = ("review" if review_only else "fault") if confirmed else "pending"
            title = "Surface defect candidate requires review" if confirmed else "Verifying surface defect candidate"
            if confirmed and not self._latched:
                self._latched = emit = True
                self._event_number += 1
        else:
            self._candidate_count = 0
            self._candidate_mask = None
            self._clean_count += 1
            if self._clean_count >= max(1, int(self.config.get("clear_confirmation_frames", 3))):
                self._latched = False
            status, title = ("no_candidate", "No review candidate; unit is not approved") if review_only else ("clear", "No selected surface defect candidate found")
        metrics["confirmation_count"] = self._candidate_count
        metrics["triggered_unit"] = context.unit_id is not None
        metrics['manual_review_only']=review_only
        key = f"factory:{context.session_id}:{context.source_id}:{self._generation}:{context.unit_id or self._event_number}" if self._latched else None
        cv2.putText(annotated, "LOCAL SURFACE MODEL | " + status.upper(), (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
        return InspectionResult(status=status, title=title, findings=findings, metrics=metrics,
                                guidance=("Manual review mode: candidates are neither verified defects nor automatic reject instructions. A missing candidate does not approve a unit. " if review_only else "")+"Review candidates against the actual surface. This model is restricted to its training appearance and does not diagnose machine health.",
                                action="human_review" if review_only or status == "fault" else "observe", annotated_frame=annotated,
                                event_key=key, emit_event=emit,
                                domain_state={"calibrated": True, "demo": False, "field_validated": False,
                                              "model_sha256": model.sha256, "unit_id": context.unit_id,
                                              "manual_review_only":review_only,"automatic_acceptance":False,"automatic_rejection":False,
                                              "physical_unit_disposition":"not_decided",
                                              "completed_unit_semantics":"inspection_observation_only",
                                              "coverage": "triggered_observation" if context.unit_id else "untriggered_observation"})
