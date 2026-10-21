"""Guided, task-selected construction research. No unvalidated category is released.

Image-space joint geometry is an observation, not a grout/material classifier.
Physical measurements require an explicit same-plane marker/intrinsic reference;
even then independent real-installation validation is still outstanding.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from edge_runtime.contracts import Finding, FrameContext, InspectionResult


TASK_CATALOG = {
    "walls_finishes": {
        "title": "Walls and finishes",
        "categories": ["damp_stained_appearance", "mold_appearance", "peeling_paint", "paint_blister", "efflorescence", "crack", "spalling", "paint_run", "patchy_coverage", "broken_sealant", "missing_sealant", "uneven_door_window_reveal"],
        "status": "unavailable_without_validated_model",
        "capture": "Film the selected wall close up, evenly lit; include a dry intact area and avoid moving shadows.",
    },
    "brackets_railings": {
        "title": "Brackets and railings",
        "categories": ["missing_bolt", "missing_nut", "fastener_absence_appearance", "visibly_unseated_connection", "corrosion_appearance"],
        "status": "unavailable_without_validated_model",
        "capture": "Show each steel connection head-on and obliquely; provide expected fastener locations and a clear view of each slot.",
    },
    "tile_installation": {
        "title": "Tile installation",
        "categories": ["joint_width", "joint_too_wide", "joint_too_narrow", "inconsistent_spacing", "joint_alignment", "row_misalignment", "missing_grout", "cracked_tile", "chipped_edge", "broken_corner", "missing_tile", "open_perimeter_joint", "misaligned_skirting"],
        "status": "research_geometry_only",
        "capture": "Show a close, sharp floor section with joints and a printed marker on the same plane. Enter the specified joint width/tolerance.",
    },
}

UNSUPPORTED_CHECKS = ["moisture_percentage", "fastener_torque", "concealed_anchorage", "hollow_tiles", "adhesion", "floor_slope", "lippage"]
_LABELS = {
    "damp_stained_appearance": "Possible damp or stained wall appearance",
    "mold_appearance": "Mold-like surface appearance (publisher Mold)",
    "peeling_paint": "Possible peeling paint",
    "paint_blister": "Possible paint blister",
    "efflorescence": "Possible salt deposits",
    "crack": "Possible surface crack",
    "spalling": "Possible surface material loss",
    "missing_bolt": "Possible missing expected bolt",
    "missing_nut": "Possible missing expected nut",
    "fastener_absence_appearance": "Publisher-labelled missing-bolt-or-nut appearance; expected installation unassessed",
    "visibly_unseated_connection": "Possible visibly loosened bolt/nut connection (publisher Loosen)",
    "corrosion_appearance": "Possible corrosion appearance",
    "joint_width": "Edge spacing research (grout identity unverified)",
}


_CATEGORY_REASONS = {
    "damp_stained_appearance": "MBDD publisher leakage boxes were trained as visible stain appearance; measured accuracy fails. The 47 positive/51 other-defect-negative TEST photos are not independent healthy-wall validation or moisture measurements.",
    "mold_appearance": "Public PHELE Mold labels were trained, but only 3 Mold-positive TEST photos exist, recall is poor, and physical sites are unknown. Appearance is not humidity or fungal diagnosis.",
    "peeling_paint": "PHELE box-labelled photographs were trained; 11 peeling-positive TEST photos show many misses and false alarms, with no independent building/video qualification.",
    "paint_blister": "A public indoor-defects project was identified, but its annotated archive has not been acquired and accepted. No paint-blister model or validated masks are installed.",
    "efflorescence": "A real CC BY 4.0 Batalha limestone photo/mask source was partially audited at one known site; white-joint annotation scope needs review, and no local fit or qualified indoor-wall/video evaluation exists.",
    "crack": "Legacy crack experiments and evaluations are retained; thin cracks, localization and false alarms remain unqualified. No new production release is established.",
    "spalling": "Legacy spalling experiments are retained and remain unqualified on real localization. Material loss is not automatically peeling paint or another finishing fault.",
    "missing_bolt": "SDNET combines missing bolt/nut appearance rather than separate bolt absence. An installation finding also needs an expected layout and positively visible empty seat; no qualified model supplies those prerequisites.",
    "missing_nut": "SDNET combines missing bolt/nut appearance rather than separate nut absence. An expected layout and positively visible empty seat are required; detector silence never proves a missing nut.",
    "fastener_absence_appearance": "SDNET publisher Missing appearance was trained for research. Source labels are incomplete and physical sites are unknown; it does not establish a required missing installation component or pass real-video release.",
    "visibly_unseated_connection": "SDNET publisher Loosen connections were trained for research; incomplete labels, unknown sites and failed release evidence remain. Visible appearance does not measure torque, embedment or anchorage strength.",
    "corrosion_appearance": "Although advertised in metadata, the acquired SDNET annotation taxonomy has no corrosion class. No accepted steel-connection corrosion detector is established in this family.",
    "joint_width": "Only experimental parallel-edge spacing geometry is implemented. It is not a grout classifier; installed-floor data with independently measured widths is missing, so millimetre results and fault decisions are unqualified.",
    "joint_too_wide": "Requires identified grout joints, valid same-plane calibration, entered nominal width/tolerance and independently measured installed-floor validation; those release data are missing.",
    "joint_too_narrow": "Requires identified grout joints, valid same-plane calibration, entered nominal width/tolerance and independently measured installed-floor validation; those release data are missing.",
    "inconsistent_spacing": "Parallel-edge observations do not establish a tile grid. No accepted installed-floor spacing labels or independently measured validation support this decision.",
    "joint_alignment": "No accepted installed-floor joint/grid annotations or project-reference alignment validation exist; reported edge angles alone are not a crooked-installation decision.",
    "row_misalignment": "Requires a verified tile grid and expected row layout. No accepted installed-floor annotations or measured layout validation are available.",
    "missing_grout": "No accepted real installed-joint grout-presence labels are available. Dark joints, shadows, decorative seams and intended open joints must be distinguished before detection can be released.",
    "cracked_tile": "Manufacturing-tile anomalies do not validate cracks in installed floors. Accepted installation data and real decorative-vein/reflection negatives are still missing.",
    "chipped_edge": "No accepted installed-tile edge-damage annotations and independent real-installation evaluation are available.",
    "broken_corner": "No accepted installed-tile corner-fracture annotations and independent real-installation evaluation are available.",
    "missing_tile": "Requires an expected installation layout and visible intended tile position. No accepted missing-installed-tile data/reference evaluation is available.",
    "open_perimeter_joint": "No accepted perimeter-joint labels or approved joint detail are available; an intentionally open movement joint is not automatically a fault.",
    "misaligned_skirting": "No accepted skirting-installation alignment labels, project reference or measured validation are available.",
    "paint_run": "No accepted real paint-run training labels or independent finish/lighting negatives are available; no detector has been trained for this category.",
    "patchy_coverage": "No accepted coating-coverage labels or controlled reference/lighting validation are available; shadows and intended texture cannot be treated as paint faults.",
    "broken_sealant": "No accepted broken-sealant annotations or expected joint-detail validation are available; no detector is trained for this category.",
    "missing_sealant": "No accepted missing-sealant labels or approved expected joint layout are available; no detector is trained for this category.",
    "uneven_door_window_reveal": "No accepted reveal-gap labels, approved reference or independently measured installation validation are available.",
}


def task_catalog():
    """JSON-safe public availability catalog; adding a task never releases it."""
    return {"tasks": [{"id": task_id, "title": task["title"], "guidance": task["capture"],
                       "categories": [{"id": category, "title": _LABELS.get(category, category.replace("_", " ").capitalize()),
                                       "status": "research_only" if task_id == "tile_installation" and category == "joint_width" else "unavailable",
                                       "reason": _CATEGORY_REASONS.get(category, "Independent real-installation validation with at least 50 positive and 50 negative examples has not passed")}
                                      for category in task["categories"]]} for task_id, task in TASK_CATALOG.items()],
            "unsupported_checks": list(UNSUPPORTED_CHECKS), "released_categories": []}


def _finite_number(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a finite number in the supported range")
    return float(value)


def _box(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or (isinstance(value, np.ndarray) and value.ndim != 1):
        raise ValueError(f"{name} must be normalized numeric [x1,y1,x2,y2]")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in value):
        raise ValueError(f"{name} must be normalized numeric [x1,y1,x2,y2]")
    try:
        values = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be normalized [x1,y1,x2,y2]") from error
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f"{name} must be normalized [x1,y1,x2,y2] with positive area")
    return values.tolist()


def _iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-x)*max(0, bottom-y)
    return intersection / max(1, a[2]*a[3]+b[2]*b[3]-intersection)


class GuidedConstructionProfile:
    """Selected task with conservative capture gates and optional research adapter.

    ``detector`` is an optional local adapter implementing infer(BGR)->detections.
    It does not grant validation or release status. Research labels must match the
    task's exact semantic labels; wetspot/bulge are never relabelled as humidity
    measurements or paint blisters.
    """

    def __init__(self, config=None, *, detector=None):
        if config is not None and not isinstance(config, dict):
            raise ValueError("guided construction configuration must be an object")
        self.options = dict(config or {})
        self.task_id = self.options.get("inspection_task", "walls_finishes")
        if not isinstance(self.task_id, str) or self.task_id not in TASK_CATALOG:
            raise ValueError("inspection_task must be walls_finishes, brackets_railings or tile_installation")
        if self.options.get("research_model") is not None and self.task_id != "walls_finishes":
            raise ValueError("Legacy research_model only supports walls_finishes; choose a task-matched research_model_manifest")
        self.roi = _box(self.options["roi"], "roi") if self.options.get("roi") is not None else None
        self.nominal = self.options.get("nominal_joint_mm")
        self.tolerance = self.options.get("tolerance_mm")
        if self.nominal is not None:
            self.nominal = _finite_number(self.nominal, "nominal_joint_mm", .01, 100)
        if self.tolerance is not None:
            self.tolerance = _finite_number(self.tolerance, "tolerance_mm", 0, 50)
            if self.nominal is None:
                raise ValueError("tolerance_mm requires nominal_joint_mm")
        if self.options.get("calibration_reference") is not None and self.options.get("calibration") is not None:
            raise ValueError("Use only calibration_reference (calibration is a legacy alias)")
        self.calibration = self._validate_calibration(self.options.get("calibration_reference", self.options.get("calibration")))
        self.expected = []
        expected = self.options.get("expected_fasteners", [])
        if not isinstance(expected, list):
            raise ValueError("expected_fasteners must be an array of objects")
        for index, item in enumerate(expected):
            if not isinstance(item, dict):
                raise ValueError("expected_fasteners entries must be objects")
            self.expected.append({"id": str(item.get("id", index)), "bbox": _box(item.get("bbox"), "expected_fastener bbox")})
        self.min_sharpness = _finite_number(self.options.get("min_sharpness", 35), "min_sharpness", 1, 10000)
        confirm = self.options.get("confirm_frames", 2)
        if isinstance(confirm, bool) or not isinstance(confirm, int) or not 1 <= confirm <= 30:
            raise ValueError("confirm_frames must be integer 1..30")
        self.confirm_frames = confirm
        self.detector = detector
        if self.options.get("research_model_manifest") is not None:
            if detector is not None or self.options.get("research_model") is not None:
                raise ValueError("Choose only one research model adapter")
            manifest = self.options["research_model_manifest"]
            if not isinstance(manifest, str) or not manifest:
                raise ValueError("research_model_manifest must be a local manifest path")
            from edge_runtime.profiles.construction_task_detector import ConstructionTaskDetector
            self.detector = ConstructionTaskDetector(manifest)
            if self.detector.spec.get("inspection_task") != self.task_id:
                raise ValueError("Research model task does not match inspection_task")
        if detector is None and self.options.get("research_model") is not None:
            from edge_runtime.profiles.construction_model import ConstructionModel
            if not isinstance(self.options["research_model"], dict):
                raise ValueError("research_model must contain explicit ConstructionModel options")
            self.detector = ConstructionModel(self.options["research_model"])
        self.reset()

    @staticmethod
    def _validate_calibration(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("calibration must be an object")
        size = _finite_number(value.get("marker_size_mm"), "marker_size_mm", 5, 1000)
        try:
            camera = np.asarray(value.get("camera_matrix"), dtype=float)
            distortion = np.asarray(value.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("calibration matrices must contain finite numerical values") from error
        if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0 or not np.allclose(camera[2], [0, 0, 1]) or distortion.shape not in {(4,), (5,), (8,), (12,), (14,)} or not np.isfinite(distortion).all():
            raise ValueError("calibration requires finite camera_matrix 3x3 and supported distortion coefficients")
        marker_id = value.get("marker_id", 0)
        if isinstance(marker_id, bool) or not isinstance(marker_id, int) or not 0 <= marker_id < 50:
            raise ValueError("marker_id must be integer 0..49 for DICT_4X4_50")
        if value.get("same_plane") is not True:
            raise ValueError("calibration requires explicit same_plane=true for the marker and inspected joints")
        dimensions = value.get("image_size")
        if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 32 for n in dimensions):
            raise ValueError("calibration image_size must be [analysis_width,analysis_height] for these intrinsics")
        return {"marker_size_mm": size, "marker_id": marker_id, "same_plane": True,
                "camera_matrix": camera.tolist(), "distortion_coefficients": distortion.tolist(), "image_size": list(dimensions)}

    def reset(self):
        self._identity = None
        self._sequence = None
        self._last_time = None
        self._tracks = []
        self._next_id = 1

    def describe(self):
        provenance = self.detector.describe() if self.detector is not None and hasattr(self.detector, "describe") else None
        return {"inspection_task": self.task_id, "title": TASK_CATALOG[self.task_id]["title"],
                "task_catalog": task_catalog(), "roi": self.roi, "calibration_reference": self.calibration,
                "nominal_joint_mm": self.nominal, "tolerance_mm": self.tolerance,
                "expected_fasteners": self.expected, "experimental": True,
                "mode": "guided_construction", "released_categories": [], "code_compliance": "not_determined",
                "model_available": self.detector is not None, "measurement_validation": "not_independently_validated",
                "inspection_kind": "geometry" if self.task_id == "tile_installation" else "model" if self.detector is not None else "guidance",
                "model_provenance": provenance, "model_initialization": getattr(self, "_worker_preparation", None),
                "unsupported_checks": UNSUPPORTED_CHECKS, "confirm_frames": self.confirm_frames,
                "overlay_scope": "Only the inspected frame; image-space geometry and research model candidates"}

    def prepare_worker(self):
        if self.detector is not None and hasattr(self.detector, "prepare_worker"):
            self._worker_preparation = self.detector.prepare_worker()
            return self._worker_preparation
        started = time.perf_counter()
        calls = 0
        if self.detector is not None:
            size = int(getattr(self.detector, "input_size", 320))
            for _ in range(3):
                self.detector.infer(np.full((size, size, 3), 127, np.uint8))
                calls += 1
        self._worker_preparation = {"worker_warmup_calls": calls, "worker_warmup_inspected_observations": 0,
                "worker_warmup_ms": round((time.perf_counter()-started)*1000, 3),
                "scope": "Stateless model initialization only: generated initialization tensors are discarded, never evidence or benchmark input"}
        return self._worker_preparation

    def _reference(self, crop, offset, frame_dimensions):
        """Return metric-plane research image and inverse mapping, or abstain."""
        if self.calibration is None:
            return crop, None, None, "No marker/intrinsic reference: pixel observations only"
        if not hasattr(cv2, "aruco"):
            return crop, None, None, "ArUco support unavailable in installed OpenCV"
        cfg = self.calibration
        if list(frame_dimensions) != cfg["image_size"]:
            return crop, None, None, "Calibration dimensions do not match this analysis frame: pixel observations only"
        camera = np.array(cfg["camera_matrix"], dtype=float)
        camera[0, 2] -= offset[0]
        camera[1, 2] -= offset[1]
        distortion = np.array(cfg["distortion_coefficients"], dtype=float)
        undistorted = cv2.undistort(crop, camera, distortion)
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        corners, ids, _ = detector.detectMarkers(undistorted)
        matches = [] if ids is None else np.flatnonzero(ids.flatten() == cfg["marker_id"])
        if len(matches) != 1:
            return crop, None, None, "Exactly one selected reference marker must be clearly visible"
        quad = corners[int(matches[0])].reshape(4, 2).astype(np.float32)
        if min(np.linalg.norm(quad[(i+1)%4]-quad[i]) for i in range(4)) < 30:
            return crop, None, None, "Reference marker too small: move closer"
        side = cfg["marker_size_mm"]
        world = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(quad, world)
        height, width = crop.shape[:2]
        image_corners = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.]], np.float32)
        denominators = np.column_stack((image_corners, np.ones(4))) @ transform[2]
        if denominators.min()*denominators.max() <= 0 or np.min(np.abs(denominators)) < 1e-7:
            return crop, None, None, "Reference perspective crosses a horizon inside the inspected plane"
        bounds = cv2.perspectiveTransform(image_corners[None], transform)[0]
        if not np.isfinite(bounds).all():
            return crop, None, None, "Unstable reference perspective"
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        extent = high-low
        if np.any(extent < 1) or np.any(extent > side*30):
            return crop, None, None, "Reference plane is too oblique or extrapolated"
        scale = min(4., 1200./max(extent))
        translate = np.array([[scale, 0, -low[0]*scale], [0, scale, -low[1]*scale], [0, 0, 1.]])
        homography = translate @ transform
        rectified = cv2.warpPerspective(undistorted, homography, tuple(np.ceil(extent*scale).astype(int)), borderValue=(127, 127, 127))
        inverse = np.linalg.inv(homography)

        def original_points(points):
            plane = cv2.perspectiveTransform(np.array([points], np.float32), inverse)[0]
            normalized = cv2.perspectiveTransform(plane[None], np.linalg.inv(camera))[0]
            xyz = np.column_stack((normalized, np.ones(len(normalized)))).astype(float)
            projected = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), camera, distortion)[0].reshape(-1, 2)
            return projected + np.array(offset)

        return rectified, scale, original_points, "Experimental marker-plane measurement; independent installation validation outstanding"

    @staticmethod
    def _joint_edges(image):
        """Measure supported visible edge bands; never infer grout or a fault."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Suppress fine surface grain before line voting. Width support is still
        # sampled from the original gray pixels, not this smoothed image.
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
        minimum = max(45, int(min(gray.shape)*.18))
        raw = cv2.HoughLinesP(edges, 1, np.pi/360, threshold=max(25, minimum//2), minLineLength=minimum, maxLineGap=8)
        if raw is None:
            return []
        # Both Nx4 and Nx1x4 bindings occur. Canonicalize endpoints along the
        # positive unit direction too: merely negating a vector changes overlap.
        segments = np.asarray(raw, dtype=float).reshape(-1, 4)[:100]
        p, q = segments[:, :2].copy(), segments[:, 2:].copy()
        vectors = q-p
        lengths = np.linalg.norm(vectors, axis=1)
        nonzero = lengths > 0
        p, q, vectors, lengths = p[nonzero], q[nonzero], vectors[nonzero], lengths[nonzero]
        if len(p) < 2:
            return []
        units = vectors/lengths[:, None]
        reverse = (units[:, 0] < 0) | ((np.abs(units[:, 0]) < .001) & (units[:, 1] < 0))
        p[reverse], q[reverse] = q[reverse].copy(), p[reverse].copy()
        units[reverse] *= -1
        normals = np.column_stack((-units[:, 1], units[:, 0]))
        # Bulk pair geometry avoids thousands of Python/NumPy calls per frame.
        i, j = np.triu_indices(len(p), 1)
        delta_r, delta_s = p[j]-p[i], q[j]-p[i]
        distance_r = np.einsum('ij,ij->i', delta_r, normals[i])
        distance_s = np.einsum('ij,ij->i', delta_s, normals[i])
        separation = np.abs((distance_r+distance_s)/2)
        projection_r = np.einsum('ij,ij->i', delta_r, units[i])
        projection_s = np.einsum('ij,ij->i', delta_s, units[i])
        starts = np.maximum(0, np.minimum(projection_r, projection_s))
        ends = np.minimum(lengths[i], np.maximum(projection_r, projection_s))
        overlap = ends-starts
        valid = ((np.abs(np.einsum('ij,ij->i', units[i], units[j])) >= math.cos(math.radians(1.5)))
                 & (distance_r*distance_s >= 0) & (np.abs(distance_r-distance_s) <= 2)
                 & (separation >= 2.5) & (separation <= min(40, min(gray.shape)*.08))
                 & (overlap >= minimum) & (overlap >= .65*np.minimum(lengths[i], lengths[j])))
        candidates = []
        for pair in np.flatnonzero(valid):
            first, second = i[pair], j[pair]
            unit, origin = units[first], p[first]
            normal = normals[first] * (1 if distance_r[pair] >= 0 else -1)
            positions = np.linspace(starts[pair]+2, ends[pair]-2, 24)
            bases = origin+positions[:, None]*unit
            # Actual line-to-line perpendicular widths along their shared span.
            widths = ((p[second]-bases) @ normals[second]) / np.dot(normal, normals[second])
            probes = np.stack((np.full(24, -2.), widths/2, widths+2), axis=1)
            points = bases[:, None, :]+probes[:, :, None]*normal
            inside = ((points[..., 0] >= 0) & (points[..., 0] < gray.shape[1]-1)
                      & (points[..., 1] >= 0) & (points[..., 1] < gray.shape[0]-1)).all(axis=1)
            samples = cv2.remap(gray, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32), cv2.INTER_LINEAR)
            samples = samples.astype(float)
            left, right = samples[:, 1]-samples[:, 0], samples[:, 1]-samples[:, 2]
            contrast = np.minimum(np.abs(left), np.abs(right))
            supported = inside & (left*right > 0) & (contrast >= 8)
            # A measurable band needs agreement along the view, not two arbitrary
            # Hough lines. This is image evidence, not a calibrated confidence.
            if supported.mean() < .6:
                continue
            values = widths[supported]
            start_point = origin+starts[pair]*unit
            end_point = origin+ends[pair]*unit
            width_start = float(np.dot(p[second]-start_point, normals[second])/np.dot(normal, normals[second]))
            width_end = float(np.dot(p[second]-end_point, normals[second])/np.dot(normal, normals[second]))
            polygon = np.array([start_point, end_point, end_point+width_end*normal, start_point+width_start*normal])
            center = polygon.mean(axis=0)
            candidates.append({"width_px": float(np.median(values)), "polygon": polygon.tolist(), "center": center.tolist(),
                               "width_px_p10": float(np.quantile(values, .1)), "width_px_p90": float(np.quantile(values, .9)),
                               "angle_degrees": float(math.degrees(math.atan2(unit[1], unit[0]))),
                               "edge_overlap_px": float(overlap[pair]), "support_fraction": float(supported.mean()),
                               "contrast_gray": float(np.median(contrast[supported])), "sample_count": int(supported.sum()),
                               "observed_edge_polarity": "dark_band" if float(np.median(left[supported])) < 0 else "bright_band",
                               "width_method": "perpendicular_hough_edge_pair_with_image_band_support"})
        # Prefer sustained visible bands. Suppress duplicate pieces of the same
        # span, rather than depending on OpenCV's arbitrary segment output order.
        observations = []
        for candidate in sorted(candidates, key=lambda item: item['edge_overlap_px']*item['support_fraction'], reverse=True):
            angle = math.radians(candidate['angle_degrees'])
            unit = np.array([math.cos(angle), math.sin(angle)])
            normal = np.array([-unit[1], unit[0]])
            duplicate = False
            for kept in observations:
                if abs(math.sin(angle-math.radians(kept['angle_degrees']))) > math.sin(math.radians(3)):
                    continue
                delta = np.array(candidate['center'])-np.array(kept['center'])
                if abs(np.dot(delta, normal)) > max(3, .5*max(candidate['width_px'], kept['width_px'])):
                    continue
                shared = (candidate['edge_overlap_px']+kept['edge_overlap_px'])/2-abs(np.dot(delta, unit))
                if shared >= .6*min(candidate['edge_overlap_px'], kept['edge_overlap_px']):
                    duplicate = True
                    break
            if not duplicate:
                observations.append(candidate)
                if len(observations) == 12:
                    break
        return observations

    def process(self, frame, context: FrameContext):
        started = time.perf_counter()
        if str(context.source_id).lower().startswith("demo"):
            raise ValueError("Guided construction requires real video, camera or image input")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 32:
            raise ValueError("Guided construction requires uint8 BGR input at least 32 pixels")
        if not math.isfinite(context.captured_monotonic):
            raise ValueError("captured_monotonic must be finite")
        if context.source_time_ms is not None and (not math.isfinite(context.source_time_ms) or context.source_time_ms < 0):
            raise ValueError("source_time_ms must be finite and nonnegative")
        identity = (context.source_id, context.session_id)
        if identity != self._identity or (self._sequence is not None and context.sequence <= self._sequence) or (self._last_time is not None and context.captured_monotonic < self._last_time):
            self.reset()
        self._identity, self._sequence, self._last_time = identity, context.sequence, context.captured_monotonic
        height, width = frame.shape[:2]
        normalized = self.roi or [0, 0, 1, 1]
        left, top, right, bottom = [round(v*s) for v, s in zip(normalized, [width, height, width, height])]
        crop = frame[top:bottom, left:right]
        annotated = frame.copy()
        overlay_scale = max(1., width / 960.)
        observations, findings, reasons = [], [], []
        calibration_state = "not_requested"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.zeros((1, 1), np.uint8)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark_fraction, clipped_fraction = float((gray <= 8).mean()), float((gray >= 248).mean())
        if min(crop.shape[:2]) < 96:
            reasons.append("Selected material ROI is too small; move closer (minimum 96 pixels per side)")
        if sharpness < self.min_sharpness:
            reasons.append("Frame lacks sharp detail; hold still and focus on the inspected surface")
        if dark_fraction > .5 or clipped_fraction > .5:
            reasons.append("Surface is too dark or overexposed; adjust lighting")
        capture_accepted = not reasons
        status = "needs_capture" if reasons else "unavailable"
        inspection_performed, model_inference_ran = False, False
        if not reasons and self.task_id == "tile_installation":
            inspection_performed = True
            image, scale, to_original, calibration_state = self._reference(crop, (left, top), (width, height))
            for edge in self._joint_edges(image):
                polygon = np.array(edge.pop("polygon"), dtype=float)
                edge.pop("center")
                polygon = to_original(polygon) if to_original is not None else polygon+np.array([left, top])
                polygon[:, 0] = np.clip(polygon[:, 0], 0, width-1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, height-1)
                x, y = np.floor(polygon.min(axis=0)).astype(int)
                x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
                measurement = {"value": round(edge["width_px"], 3), "unit": "rectified_px" if scale else "px", "uncertainty": "not_estimated", "validation": "unvalidated_edge_pair_not_grout_classifier"}
                if "width_px_p10" in edge:
                    measurement.update(spatial_p10=round(edge["width_px_p10"]/(scale or 1), 3),
                                       spatial_p90=round(edge["width_px_p90"]/(scale or 1), 3),
                                       range_kind="spatial_line_fit_variation_not_measurement_uncertainty",
                                       support_fraction=edge["support_fraction"], sample_count=edge["sample_count"],
                                       contrast_gray=edge["contrast_gray"], method=edge["width_method"])
                if scale:
                    measurement.update(value=round(edge["width_px"]/scale, 3), unit="mm", validation="experimental_not_independently_validated")
                    if self.nominal is not None:
                        measurement["reference_mm"] = self.nominal
                        measurement["difference_mm"] = round(measurement["value"]-self.nominal, 3)
                        measurement["tolerance_mm"] = self.tolerance
                observations.append({"task_id": self.task_id, "label": "parallel_edge_spacing", "bbox": [int(x), int(y), max(1, int(x2-x)), max(1, int(y2-y))],
                                     "polygon": polygon.round(2).tolist(), "measurements": {"edge_spacing": measurement}, "assessment_status": "research_measurement",
                                     "evidence_timestamp_ms": context.source_time_ms, "source_sequence": context.sequence,
                                     "explanation": f"Visible edge band: {measurement['value']:.1f} {measurement['unit']}; grout identity and installation fault are unassessed", **edge})
                cv2.polylines(annotated, [polygon.astype(np.int32)], True, (220, 170, 40), max(1, round(overlay_scale)))
                center = np.rint(polygon.mean(axis=0)).astype(int)
                text = f"{measurement['value']:.1f}{measurement['unit']} edge band"
                label_scale = .45 * overlay_scale
                thickness = max(1, round(overlay_scale))
                text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, thickness)[0]
                label_position = (int(np.clip(center[0], 1, max(1, width-text_width-1))),
                                  int(np.clip(center[1], text_height+round(30*overlay_scale), height-5)))
                tx, ty = label_position
                cv2.rectangle(annotated, (tx, ty-text_height-3), (tx+text_width+2, min(height-1,ty+4)), (20,20,20), -1)
                cv2.putText(annotated, text, label_position, cv2.FONT_HERSHEY_SIMPLEX, label_scale, (220,170,40), thickness)
            status = "research_measurement" if observations else "unassessed"
            if not observations:
                reasons.append("No sufficiently long parallel edge pair was observed; show the joints close up")
        elif not reasons and self.detector is not None:
            inspection_performed, model_inference_ran = True, True
            allowed = set(TASK_CATALOG[self.task_id]["categories"])
            for original in self.detector.infer(crop):
                if not isinstance(original, dict):
                    continue
                label = original.get("label")
                if not isinstance(label, str) or label not in allowed or original.get("occluded") is True:
                    continue
                try:
                    coords = np.asarray(original["bbox"], dtype=float)
                    score = float(original["score"])
                except (ValueError, TypeError, KeyError):
                    continue
                if coords.shape != (4,) or not np.isfinite(coords).all() or min(coords[2:]) <= 0 or not math.isfinite(score) or not 0 <= score <= 1:
                    continue
                x, y, w, h = coords
                x1, y1, x2, y2 = max(left, int(x+left)), max(top, int(y+top)), min(right, int(x+w+left)), min(bottom, int(y+h+top))
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = (x1, y1, x2-x1, y2-y1)
                if label.startswith("missing_"):
                    expected = [(item, (item["bbox"][0]*width, item["bbox"][1]*height,
                                            (item["bbox"][2]-item["bbox"][0])*width, (item["bbox"][3]-item["bbox"][1])*height)) for item in self.expected]
                    match = next((item for item, target in expected if _iou(target, bbox) >= .2), None)
                    # The adapter must positively classify a fully visible empty seat.
                    if match is None or original.get("seat_visible") is not True:
                        continue
                else:
                    match = None
                observation = {"task_id": self.task_id, "label": label, "bbox": list(bbox), "score": score,
                                     "evidence_timestamp_ms": context.source_time_ms, "source_sequence": context.sequence,
                                     "assessment_status": "research_candidate", "expected_location_id": match["id"] if match else None,
                                     "explanation": _LABELS.get(label, label) + "; research model, not independently validated", "measurements": {},
                                     "localization_kind": "bbox"}
                mask = original.get("mask")
                if isinstance(mask, np.ndarray) and mask.shape == crop.shape[:2] and mask.dtype == np.bool_:
                    # Preserve true model localization; boxes are supplementary.
                    mask = mask.copy()
                    mask[:max(0, y1-top)] = False
                    mask[max(0, y2-top):] = False
                    mask[:, :max(0, x1-left)] = False
                    mask[:, max(0, x2-left):] = False
                    if not mask.any():
                        continue
                    region = annotated[top:bottom, left:right]
                    region[mask] = (.65*region[mask]+.35*np.array([0, 180, 240])).astype(np.uint8)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    observation["mask_area_px"] = int(mask.sum())
                    observation["mask_contours_truncated"] = len(contours) > 32
                    observation["mask_polygons"] = [(cv2.approxPolyDP(contour, .75, True).reshape(-1, 2)+np.array([left, top])).tolist()
                                                    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:32]]
                    observation["localization_kind"] = "segmentation_mask"
                elif mask is not None:
                    observation["mask_status"] = "invalid_native_roi_boolean_mask_not_rendered"
                observations.append(observation)
            status = "research_candidate" if observations else "unassessed"
        elif not reasons:
            reasons.append("No validated local model for this task; public-data acquisition and task-specific evaluation are required")
        emitted = []
        self._tracks = [t for t in self._tracks if context.captured_monotonic-t["time"] <= 1.]
        available = list(self._tracks)
        for observation in observations:
            bbox = tuple(observation["bbox"])
            track = next((t for t in available if t["label"] == observation["label"] and _iou(t["bbox"], bbox) > .3), None)
            if track is None:
                track = {"id": self._next_id, "label": observation["label"], "hits": 0, "emitted": False}
                self._next_id += 1
                self._tracks.append(track)
            else:
                available.remove(track)
            track.update(bbox=bbox, hits=track["hits"]+1, time=context.captured_monotonic)
            observation["track_id"] = track["id"]
            observation["confirmed"] = track["hits"] >= self.confirm_frames
            # Research geometry is never elevated into an installation alarm.
            if observation["assessment_status"] == "research_candidate" and observation["confirmed"] and not track["emitted"]:
                track["emitted"] = True
                emitted.append(track["id"])
            finding = Finding("construction_research_observation", observation["explanation"], bbox,
                              observation.get("score"), "guided_construction_research", "info")
            # Additive contract fields are set if the host runtime provides them.
            for name, value in {"task_id": self.task_id, "evidence_timestamp_ms": context.source_time_ms,
                                "measurements": observation["measurements"], "assessment_status": observation["assessment_status"]}.items():
                if hasattr(finding, name):
                    setattr(finding, name, value)
            findings.append(finding)
            if observation["assessment_status"] == "research_candidate":
                x, y, w, h = bbox
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 180, 240), 2)
                cv2.putText(annotated, observation["label"], (x, max(45, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 240), 1)
        if self.roi is not None:
            cv2.rectangle(annotated, (left, top), (right-1, bottom-1), (180, 160, 90), 1)
        if self.task_id == "tile_installation":
            cv2.rectangle(annotated, (0, 0), (min(width-1, round(470*overlay_scale)), round(29*overlay_scale)), (20,20,20), -1)
            cv2.putText(annotated, "EDGE SPACING RESEARCH - FAULTS UNASSESSED", (round(8*overlay_scale), round(21*overlay_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, .46*overlay_scale, (0,210,255), max(1,round(overlay_scale)))
        else:
            cv2.putText(annotated, "GUIDED RESEARCH - CONDITIONS UNASSESSED", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 210, 255), 1)
        guidance = " ".join(reasons+[TASK_CATALOG[self.task_id]["capture"]])
        return InspectionResult(status=status, title=f"{TASK_CATALOG[self.task_id]['title']}: {status.replace('_', ' ')}",
            findings=findings, metrics={"inspection_task": self.task_id, "observations": len(observations), "sharpness": round(sharpness, 3),
                "inspection_performed": inspection_performed, "model_inference_ran": model_inference_ran,
                "dark_fraction": dark_fraction, "clipped_fraction": clipped_fraction, "processing_ms": round((time.perf_counter()-started)*1000, 3),
                "released_categories": 0, "experimental": True}, guidance=guidance, action="request_review" if emitted else "observe",
            annotated_frame=annotated, emit_event=bool(emitted), event_key=f"construction-guided:{context.session_id}:{','.join(map(str, emitted))}" if emitted else None,
            domain_state={"inspection_task": self.task_id, "assessment_status": status, "evidence_timestamp_ms": context.source_time_ms,
                "source_sequence": context.sequence, "source_frame_index": context.source_frame_index,
                "code_compliance": "not_determined", "quality": {"capture_accepted": capture_accepted, "sharpness": sharpness,
                "dark_fraction": dark_fraction, "clipped_fraction": clipped_fraction, "occlusion": "not_automatically_established"},
                "capture_guidance": guidance, "observations": observations, "review_findings": observations,
                "task_catalog": task_catalog(), "calibration_status": calibration_state, "roi": self.roi,
                "expected_fasteners": self.expected, "measurement_validation": "not_independently_validated", "unsupported_checks": UNSUPPORTED_CHECKS})
