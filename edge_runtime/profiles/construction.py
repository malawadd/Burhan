"""Video-first construction observations, always unassessed for code compliance."""
from __future__ import annotations

import math
import threading
import time
import cv2
import numpy as np

from edge_runtime.contracts import Finding, FrameContext, InspectionResult
from edge_runtime.profiles.construction_model import ConstructionModel


class ConstructionProfile:
    def __init__(self, config=None):
        self.options = dict(config or {})
        self.stage = str(self.options.get("stage", "site_safety"))
        self.show_context_objects = self.options.get("show_context_objects", False)
        if type(self.show_context_objects) is not bool:
            raise ValueError("construction show_context_objects must be boolean")
        self.roi = self.options.get("roi")
        if self.roi is not None:
            try:
                points = np.asarray(self.roi, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError("construction roi must be normalized [x1,y1,x2,y2] with positive area") from error
            if points.shape != (4,) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1) or not (points[0] < points[2] and points[1] < points[3]):
                raise ValueError("construction roi must be normalized [x1,y1,x2,y2] with positive area")
            self.roi = points.tolist()
        self.model = ConstructionModel(self.options)
        self.supported_capabilities = list(self.model.describe().get("supported_capabilities", self.model.spec["capabilities"]))
        self.confirm_frames = int(self.options.get("confirm_frames", 2))
        if not 1 <= self.confirm_frames <= 30:
            raise ValueError("construction confirm_frames must be 1..30")
        self.reset()

    def reset(self):
        self._identity = None
        self._sequence = None
        self._last_time = None
        self._tracks = []
        self._next_id = 1

    def describe(self):
        return {**self.model.describe(), "selected_stage": self.stage, "mode": self.model.mode,
                "model_initialization":getattr(self,"_worker_preparation",None),
                "code_compliance": "not_determined", "confirm_frames": self.confirm_frames,
                "show_context_objects": self.show_context_objects, "overlay_scope": "Issue candidates only; context objects are hidden unless debug is enabled",
                "roi": self.roi, "roi_scope": ("Operator selects target material; material is not classified automatically. Outside ROI is unassessed."
                                               if self.roi is not None else "Full-frame selected detector only; material is not classified automatically."),
                "method": "Pinned local model; explicit observed classes; temporal review events"}

    def prepare_worker(self):
        """Compile lazy inference kernels before the runtime admits source frames.

        Raw model calls are stateless: no tracker, observation, event or camera
        counter is advanced. The initialization tensor never becomes evidence.
        """
        size=int(getattr(self.model,"input_size",640))
        probe=np.full((size,size,3),127,np.uint8)
        started=time.perf_counter()
        for _ in range(3):
            self.model.infer(probe)
        self._worker_preparation={
            "worker_warmup_ms":round((time.perf_counter()-started)*1000,3),
            "worker_warmup_calls":3,"worker_warmup_inspected_observations":0,
            "worker_warmup_thread":threading.current_thread().name,
            "scope":"Stateless inference initialization before capture; outputs discarded, not source/evidence frames"}
        return dict(self._worker_preparation)

    def process(self, frame, context: FrameContext):
        if context.source_id.lower().startswith("demo"):
            raise ValueError("Construction requires real video, camera or image input; simulation is disabled")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 32:
            raise ValueError("construction requires uint8 BGR input at least 32 pixels in each dimension")
        if not math.isfinite(context.captured_monotonic):
            raise ValueError("construction captured_monotonic must be finite")
        identity = (context.source_id, context.session_id)
        if identity != self._identity or (self._sequence is not None and context.sequence <= self._sequence) or (self._last_time is not None and context.captured_monotonic < self._last_time):
            self.reset()
        self._identity, self._sequence, self._last_time = identity, context.sequence, context.captured_monotonic
        height, width = frame.shape[:2]
        if self.roi is None:
            rx, ry, right, bottom = 0, 0, width, height
        else:
            rx, ry, right, bottom = [round(v * size) for v, size in zip(self.roi, [width, height, width, height])]
        rw, rh = right - rx, bottom - ry
        if min(rw, rh) < 32:
            raise ValueError("construction ROI must cover at least 32 input pixels in each dimension")
        detections = []
        for original in self.model.infer(frame[ry:bottom, rx:right]):
            item = dict(original)
            x, y, w, h = item["bbox"]
            item["bbox"] = (x + rx, y + ry, w, h)
            detections.append(item)
        annotated = frame.copy()
        if self.roi is not None:
            cv2.rectangle(annotated, (rx, ry), (right - 1, bottom - 1), (190, 170, 90), 2)
            cv2.putText(annotated, "OPERATOR ROI: outside unassessed", (rx, max(44, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, .48, (190, 170, 90), 1)
        findings, new_ids, observations, review_findings = [], [], [], []
        self._tracks = [t for t in self._tracks if context.captured_monotonic - t["time"] <= 1.0]
        available = list(self._tracks)
        from edge_runtime.profiles.roads import _iou
        for item in detections:
            label, bbox = item["label"], item["bbox"]
            # Only explicitly trained no-PPE channels or a detected surface
            # crack can request review. Person-only boxes never imply a violation.
            review = label in {"NO-Hardhat", "NO-Safety Vest", "crack", "spalling", "rust", "exposed_rebar", "wetspot", "efflorescence", "rockpocket", "cavity", "hollowareas", "weathering", "residual_formwork", "alligator_crack", "washout", "graffiti", "tile_surface_anomaly", "leakage", "abscission", "corrosion", "bulge"}
            track = max((t for t in available if t["label"] == label), key=lambda t: _iou(t["bbox"], bbox), default=None)
            if track is None or _iou(track["bbox"], bbox) < .2:
                track = {"id": self._next_id, "label": label, "hits": 0, "emitted": False}
                self._next_id += 1; self._tracks.append(track)
            elif track in available:
                available.remove(track)
            track.update(bbox=bbox, hits=track["hits"] + 1, time=context.captured_monotonic)
            confirmed = review and track["hits"] >= self.confirm_frames
            if confirmed and not track["emitted"]:
                track["emitted"] = True; new_ids.append(track["id"])
            color = (30, 80, 240) if review else (40, 190, 100)
            names = {"NO-Hardhat": "Possible missing helmet", "NO-Safety Vest": "Possible missing high-visibility vest", "crack": "Possible surface crack", "spalling":"Possible spalled or chipped concrete", "rust":"Possible corrosion staining", "exposed_rebar":"Possible exposed reinforcement", "wetspot":"Possible wet or moisture-marked surface", "efflorescence":"Possible salt deposits", "rockpocket":"Possible rock pocket or honeycombing", "cavity":"Possible surface cavity", "hollowareas":"Inspector-marked hollow-area boundary (chalk)", "weathering":"Possible surface weathering", "residual_formwork":"Possible remaining formwork", "alligator_crack":"Possible alligator cracking", "washout":"Possible concrete surface loss or erosion appearance", "graffiti":"Possible graffiti or surface markings"}
            why = {"NO-Hardhat": "An exposed head may be vulnerable to falling or flying objects where helmet protection is required.",
                   "NO-Safety Vest": "Low visibility can increase exposure to moving plant where high-visibility clothing is required.",
                   "crack": "A visible surface discontinuity may require repair or structural review; appearance alone does not establish severity."}
            for damage_label in names:
                why.setdefault(damage_label,"A model-labelled surface condition may need close inspection; appearance does not determine structural cause, depth or acceptance.")
            checks = {"NO-Hardhat": "Inspect the original frame and confirm that the head is visible and helmet protection is required in this work zone.",
                      "NO-Safety Vest": "Verify clothing visibility and the actual work-zone PPE requirement with the site supervisor.",
                      "crack": "Inspect this location close up; distinguish a crack from a joint, stain or shadow, then measure width with a calibrated tool."}
            for damage_label in names:
                checks.setdefault(damage_label,"Inspect this exact surface location; verify the material and damage type, measure relevant dimensions, and obtain qualified assessment before repair or acceptance.")
            names["spalling"]="Possible surface spalling or material loss"
            checks["spalling"]="Verify the surface material, loose/chipped areas and depth. If concrete is confirmed, inspect cover loss and exposed reinforcement; obtain qualified assessment of the cause."
            checks["exposed_rebar"]="Verify actual exposed steel, cover loss and corrosion condition; obtain measured inspection and repair design."
            checks["wetspot"]="Verify moisture with a suitable instrument and identify a water source; a dark pixel region is not a leak test."
            checks["hollowareas"]="Obtain the original sounding-test record and inspector markings; a detected chalk boundary is not a new measurement of an internal void."
            why["hollowareas"]="The model identifies a boundary marking made after a prior inspection, not hidden material or an internal void."
            checks["washout"]="Inspect and measure the visible surface loss; obtain qualified cause assessment. The model cannot distinguish chemical corrosion, erosion, abrasion or washout mechanism."
            names["tile_surface_anomaly"] = "Possible tile surface texture anomaly"
            why["tile_surface_anomaly"] = "The surface differs from normal manufacturing tile textures; the defect subtype is unclassified."
            checks["tile_surface_anomaly"] = "Inspect this tile close up. This specialist cannot assess installed alignment, lippage, adhesion, humidity or a construction tolerance."
            names.update(leakage="Possible leakage mark or wet-looking surface", abscission="Possible surface material detachment",
                         corrosion="Possible corrosion appearance", bulge="Possible surface bulge appearance")
            why.update(leakage="Visible staining may need investigation; the image does not measure humidity, water content or an active hidden leak.",
                       abscission="Detached-looking material needs close inspection; this class does not establish paint, plaster or concrete substrate.",
                       corrosion="Corrosion-like appearance may require material assessment; an image does not measure section loss or establish a chemical mechanism.",
                       bulge="A raised-looking surface may require inspection; an RGB box does not measure depth, flatness or a paint blister.")
            checks.update(leakage="Inspect this location and use a suitable moisture instrument; identify the actual water source.",
                          abscission="Confirm the coating and substrate, inspect loose material, and measure the damaged area and depth.",
                          corrosion="Confirm the material and corrosion condition, then obtain relevant section-loss or protective-coating measurements.",
                          bulge="Confirm the material and surface geometry with a straightedge or calibrated depth measurement; inspect adhesion where relevant.")
            x, y, w, h = bbox
            if review or self.show_context_objects:
                if "mask" in item:
                    mask = item["mask"]
                    region = annotated[ry:bottom, rx:right]
                    region[mask] = (.6 * region[mask] + .4 * np.array(color)).astype(np.uint8)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                text = f"#{track['id']} {names.get(label, label)} {item['score']:.2f}"
                cv2.putText(annotated, text, (x, max(46, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
                findings.append(Finding("construction_issue_candidate" if review else "construction_context", names.get(label, label), bbox=bbox,
                                        score=round(item["score"], 4), method=f"construction_{self.model.mode}_onnx", severity="warning" if review else "info"))
            if review:
                evidence_basis = (f"Nearest-normal texture distance {item['anomaly_distance']:.3f}; bounded display score {item['score']:.2f} is not defect probability."
                                  if label == "tile_surface_anomaly" else
                                  f"Local model explicitly predicted {label}; confidence {item['score']:.2f}. This is not verified ground truth.")
                review_findings.append({"id": track["id"], "label": names[label], "bbox": list(bbox),
                    "what": names[label], "where": f"Analysis pixels x={x}, y={y}, width={w}, height={h}",
                    "why_it_matters": why[label], "next_check": checks[label],
                    "evidence_basis": evidence_basis,
                    "status": "repeated_candidate" if confirmed else "single_frame_candidate"})
            observations.append({k: v for k, v in item.items() if k != "mask"} | {"track_id": track["id"], "review": review, "confirmed": confirmed})
        banner=(self.model.spec.get("diagnostic_label", "RESEARCH DIAGNOSTIC - FAILED CALIBRATION") if self.model.spec.get("diagnostic_only",False)
                else "CONSTRUCTION OBSERVATIONS - CODE COMPLIANCE UNASSESSED")
        cv2.putText(annotated, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 200, 255), 1)
        review_count = sum(x["review"] for x in observations)
        return InspectionResult(
            status="review" if review_count else "unassessed",
            title=f"Construction: {review_count} visible observations for review" if review_count else "Construction: code compliance remains unassessed",
            findings=findings, metrics={"detections": len(detections), "review_candidates": review_count,
                                       "input_exposure_gain": float(getattr(self.model,"last_input_exposure_gain",1.0)),
                                       "experimental": True, "mode": self.model.mode, "context_objects_hidden": not self.show_context_objects,
                                       "analysis_roi_xywh": [rx, ry, rw, rh],
                                       "mask_rendering": getattr(self.model, "mask_rendering", "masks")},
            action="request_review" if review_count else "observe",
            guidance="Visible candidates need competent inspection. No detection is not proof of compliance. Concealed works, strength, dimensions, tests and other stages are unassessed.",
            annotated_frame=annotated, emit_event=bool(new_ids),
            event_key=f"construction:{context.session_id}:{','.join(map(str,new_ids))}" if new_ids else None,
            domain_state={"selected_stage": self.stage, "supported_capabilities": self.supported_capabilities,
                          "operator_roi": self.roi, "analysis_roi_xywh": [rx, ry, rw, rh],
                          "roi_scope": ("Operator-selected material; no automatic material classification. Outside the ROI is unassessed."
                                        if self.roi is not None else "Full-frame selected detector only; material is not classified automatically."),
                          "code_compliance": "not_determined", "observations": observations, "review_findings": review_findings,
                          "unsupported_checks": "All checks outside the selected visible detector remain unassessed"})
