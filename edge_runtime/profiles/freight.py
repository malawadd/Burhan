"""Local camera/document capture, offline OCR and conservative packet readiness."""
from __future__ import annotations

import hashlib
import threading
import time
import cv2
import numpy as np

from edge_runtime.contracts import Finding, FrameContext, InspectionResult
from edge_runtime.profiles.freight_ocr import LocalOCR, parse_document

PAGE_TYPES = ("delivery_note", "packing_list")
PAGE_LABELS = {"delivery_note": "Delivery note", "packing_list": "Packing list",
               "invoice": "Invoice", "bill_of_lading": "Bill of lading"}


def _ordered(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype(np.float32)
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                     points[np.argmax(sums)], points[np.argmax(diffs)]], np.float32)


def _hash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(gray, (33, 32), interpolation=cv2.INTER_AREA)
    return (reduced[:, 1:] > reduced[:, :-1]).ravel()


class FreightProfile:
    def __init__(self, config: dict | None = None):
        self.config = {"stable_frames": 8, "focus_threshold": 45., "min_page_fraction": .12,
                       "stable_motion_fraction": .012, "duplicate_distance": 50,
                       "ocr_enabled": True, "ocr_async": True, "ocr_min_score": .65,
                       "required_pages": list(PAGE_TYPES), "ocr_retry_frames": 60}
        self.config.update(config or {})
        if not isinstance(self.config["required_pages"], list) or not self.config["required_pages"] or any(k not in PAGE_LABELS for k in self.config["required_pages"]):
            raise ValueError("required_pages must list supported freight document types")
        self._ocr = LocalOCR() if self.config["ocr_enabled"] else None
        self._job = None
        self._closed = False
        self._scope = None
        self._generation = 0
        self.reset()

    def reset(self):
        self._generation += 1
        self.accepted = {}
        self._corners = None
        self._last_page_hash = None
        self._stable = 0
        self._capture_requested = False
        self._ocr_hash = None
        self._ocr_result = None
        self._ocr_error = None
        self._ocr_last_sequence = -100000
        self._ocr_job_ms = None
        self._last_frame = None
        self._last_context = None
        self._source_ended = False
        self._reviewed = set()

    def _read_job(self, job, page):
        started = time.perf_counter()
        try:
            job["result"] = parse_document(self._ocr.read(page))
        except Exception as exc:
            job["error"] = f"Local OCR unavailable: {type(exc).__name__}: {exc}"
        finally:
            job["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
            if job.get("result") is not None:
                job["result"]["ocr_job_ms"] = job["wall_ms"]
            job["done"] = True

    def _collect_job(self, image_hash):
        job = self._job
        if job is None or not job["done"]:
            return
        if not self._closed and job["generation"] == self._generation and job["scope"] == self._scope and int(np.count_nonzero(image_hash != job["hash"])) <= self.config["duplicate_distance"]:
            self._ocr_result = job.get("result")
            self._ocr_error = job.get("error")
            self._ocr_job_ms = job.get("wall_ms")
            self._ocr_hash = job["hash"]
        self._job = None

    def _start_job(self, image_hash, page):
        job = {"hash": image_hash.copy(), "generation": self._generation, "scope": self._scope, "done": False}
        self._job = job
        if self.config["ocr_async"]:
            # One bounded job per active profile. No pending-frame queue, no frame-rate OCR.
            thread = threading.Thread(target=self._read_job, args=(job, page.copy()),
                                      name="freight-local-ocr", daemon=True)
            job["thread"] = thread
            thread.start()
        else:
            self._read_job(job, page)
            self._collect_job(image_hash)

    def source_completion_state(self):
        return {"pending_ocr": self._job is not None and not self._job["done"],
                "source_finished": self._source_ended,
                "last_source_sequence": None if self._last_context is None else self._last_context.sequence}

    def finish_source(self, timeout=3.):
        """Finish OCR against the retained actual final frame, never invent a new capture."""
        self._source_ended = True
        job = self._job
        if job is not None:
            thread = job.get("thread")
            if thread is not None:
                thread.join(max(0., float(timeout)))
                if thread.is_alive():
                    return None
        if self._closed or self._last_frame is None or self._last_context is None:
            return None
        frame, context = self._last_frame, self._last_context
        result = self.process(frame, context)
        result.metrics.update(source_finished=True, final_source_sequence=context.sequence,
                              final_captured_monotonic=context.captured_monotonic)
        return result

    def close(self, timeout=3.):
        """Invalidate observations and wait boundedly; False means native OCR still owns CPU/model."""
        self._closed = True
        self._generation += 1
        job = self._job
        if job is None:
            return True
        thread = job.get("thread")
        if thread is not None:
            thread.join(max(0., float(timeout)))
            return not thread.is_alive()
        return bool(job["done"])

    def request_capture(self):
        self._capture_requested = True

    def describe(self):
        return {"name": "freight", "title": "Freight packet readiness",
                "description": "Offline CPU OCR of camera pages, local image scans and PDFs; English heading and labelled-field checks.",
                "profile_version": "ppocr_v4_english_mobile_document_rules_v2_cpu",
                "supported_templates": list(PAGE_LABELS),
                "model": self._ocr.info if self._ocr else {"available": False, "error": "OCR disabled"},
                "limitations": ["English heading rules are conservative; unknown/mixed pages require review.",
                                "OCR fields are unverified suggestions; no signature/authenticity verification.",
                                "Page presence is not shipment correctness or customs clearance."],
                "controls": ["request_capture", "reset"], "config": dict(self.config)}

    def _state(self, page_type=None):
        required = self.config["required_pages"]
        missing = [key for key in required if key not in self.accepted]
        return {"checklist": {key: {"label": PAGE_LABELS[key], "captured": key in self.accepted} for key in required},
                "captured_pages": list(self.accepted), "missing_pages": missing, "complete": not missing,
                "detected_template": page_type, "stable_frames": self._stable,
                "supported_fictional_templates_only": False,
                "ocr": self._ocr_result, "ocr_error": self._ocr_error, "ocr_job_ms": self._ocr_job_ms,
                "ocr_pending": self._job is not None and not self._job["done"],
                "captured_documents": {key: {k: v for k, v in value.items() if k != "hash"} for key, value in self.accepted.items()}}

    @staticmethod
    def _page(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * .05:
                break
            polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                return _ordered(polygon)
        return None

    @staticmethod
    def _template(page):
        hsv = cv2.cvtColor(cv2.resize(page, (500, 650))[25:125, 350:480], cv2.COLOR_BGR2HSV)
        counts = {"delivery_note": cv2.countNonZero(cv2.inRange(hsv, np.array([95,110,70]), np.array([130,255,255]))),
                  "packing_list": cv2.countNonZero(cv2.inRange(hsv, np.array([40,110,70]), np.array([85,255,255])))}
        best = max(counts, key=counts.get)
        return best if counts[best] >= 350 else None

    def process(self, frame, context):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("FreightProfile expects a BGR image with three channels")
        if self._closed:
            raise RuntimeError("Freight profile is closed")
        scope = (context.source_id, context.session_id)
        if scope != self._scope:
            if self._scope is not None:
                self.reset()
            self._scope = scope
        self._last_frame, self._last_context = frame, context
        h, w = frame.shape[:2]
        digital = context.source_id.startswith(("image:", "images:", "pdf:"))
        demo = context.source_id == "demo"
        # Digital scans already define a page. A camera still needs visible corners.
        corners = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32) if digital else self._page(frame)
        annotated = frame.copy()
        metrics = {"page_detected": corners is not None, "ocr_cached": False, "input_mode": "digital_scan" if digital else "camera_or_video"}
        findings = []
        page_type, event_key, emit = None, None, False
        status, title, action = "needs_evidence", "Find a document", "recapture"
        guidance = "Place a complete page against a contrasting background, with all corners visible."
        if corners is None:
            self._corners, self._stable = None, 0
            self._ocr_result, self._ocr_hash = None, None
        else:
            fraction = abs(cv2.contourArea(corners)) / (w*h)
            x,y,bw,bh = cv2.boundingRect(corners.astype(np.int32))
            cropped = not digital and bool(np.any(corners[:,0] < 6) or np.any(corners[:,1] < 6) or np.any(corners[:,0] > w-7) or np.any(corners[:,1] > h-7))
            # Keep enough printed detail for OCR rather than reducing all pages to 500px.
            if digital:
                page = frame
            else:
                pw = max(np.linalg.norm(corners[1]-corners[0]), np.linalg.norm(corners[2]-corners[3]))
                ph = max(np.linalg.norm(corners[3]-corners[0]), np.linalg.norm(corners[2]-corners[1]))
                out_h = 650 if demo else min(1400, max(650, round(ph)))
                out_w = 500 if demo else min(1600, max(200, round(out_h * pw/max(ph,1))))
                dest = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]],np.float32)
                page = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(corners,dest),(out_w,out_h))
            gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
            focus = float(cv2.Laplacian(gray[18:-18,18:-18], cv2.CV_64F).var()) if min(gray.shape)>40 else 0.
            ink = float(np.mean(gray < 125))
            movement = float(np.mean(np.linalg.norm(corners-self._corners,axis=1))/np.hypot(w,h)) if self._corners is not None and self._corners.shape==corners.shape else None
            acceptable = not cropped and fraction>=self.config["min_page_fraction"] and focus>=self.config["focus_threshold"] and ink>=.008
            image_hash = _hash(page)
            self._collect_job(image_hash)
            same_page = self._ocr_hash is not None and int(np.count_nonzero(image_hash != self._ocr_hash))<=self.config["duplicate_distance"]
            if not same_page:
                self._ocr_result, self._ocr_error = None, None
            same_stability_page = self._last_page_hash is not None and int(np.count_nonzero(image_hash != self._last_page_hash))<=self.config["duplicate_distance"]
            self._stable = self._stable+1 if acceptable and same_stability_page and movement is not None and movement<=self.config["stable_motion_fraction"] else (1 if acceptable else 0)
            self._last_page_hash = image_hash.copy()
            if not acceptable or (movement is not None and movement>self.config["stable_motion_fraction"]):
                self._ocr_result, self._ocr_hash = None, None
            self._corners = corners.copy()
            ready = self._stable>=self.config["stable_frames"] or self._capture_requested
            if demo:
                page_type = self._template(page)
            elif acceptable and ready:
                if not same_page or self._ocr_result is None:
                    if self._ocr is None:
                        self._ocr_error = "Local OCR disabled"
                    elif not self._source_ended and self._job is None and (context.sequence-self._ocr_last_sequence>=self.config["ocr_retry_frames"] or not same_page):
                        self._ocr_last_sequence = context.sequence
                        self._ocr_hash = image_hash.copy()
                        self._start_job(image_hash, page)
                else:
                    metrics["ocr_cached"] = True
                if self._ocr_result and self._ocr_result["mean_score"]>=self.config["ocr_min_score"]:
                    page_type = self._ocr_result["document_type"]
            metrics.update(page_fraction=round(fraction,4),focus=round(focus,2),ink_fraction=round(ink,4),cropped=cropped,stable_frames=self._stable,
                           motion_fraction=None if movement is None else round(movement,5),template=page_type,quality_acceptable=acceptable,
                           ocr_ms=0 if metrics["ocr_cached"] else (self._ocr_result or {}).get("ocr_ms",0),
                           ocr_job_ms=self._ocr_job_ms,
                           ocr_mean_score=(self._ocr_result or {}).get("mean_score"), recognition_method="fictional_color_marker" if demo else "ppocr_cpu_and_heading_rules")
            findings.append(Finding("document",PAGE_LABELS.get(page_type,"Page needs classification"),(x,y,bw,bh),fraction,"digital_page" if digital else "opencv_threshold_quad"))
            if cropped:
                title,guidance="Page cropped","Move back until all four corners are visible."
            elif fraction<self.config["min_page_fraction"]:
                title,guidance="Page too small","Move closer while keeping every corner visible."
            elif focus<self.config["focus_threshold"]:
                title,guidance="Page blurred","Hold still and refocus; fine printed detail is blurred."
            elif ink<.008:
                title,guidance="Insufficient detail","No usable printed detail found. Check exposure."
            elif not ready:
                status,title,action="observing","Hold page steady","observe"
                guidance=f"Hold still: {self._stable}/{self.config['stable_frames']} stable frames."
            elif page_type is None:
                status,title,action="needs_review","OCR page requires review","review"
                guidance=self._ocr_error or ("Text extracted locally; heading is unknown, ambiguous or low confidence. Review the OCR evidence." if self._ocr_result else "No OCR result is available for this page. Hold the page in view until local processing finishes.")
                if self._job is not None and not self._job["done"]:
                    status,title,action="observing","Reading page locally","observe"
                    guidance="Offline OCR is processing this page. Keep it in view until the result appears."
                digest=hashlib.sha256(np.packbits(image_hash).tobytes()).hexdigest()[:16]
                if self._ocr_result and digest not in self._reviewed:
                    self._reviewed.add(digest)
                    if len(self._reviewed)>128:
                        self._reviewed.pop()
                    emit,event_key=True,f"freight:{context.session_id}:review:{digest}"
            else:
                previous=self.accepted.get(page_type)
                duplicate=previous is not None and int(np.count_nonzero(image_hash!=previous["hash"]))<=self.config["duplicate_distance"]
                metrics["duplicate"]=duplicate
                if previous:
                    title="Duplicate page" if duplicate else "Page type already captured"
                    status,action="duplicate","observe"
                    guidance="Already captured. Reset packet before processing another shipment."
                else:
                    digest=hashlib.sha256(np.packbits(image_hash).tobytes()).hexdigest()[:16]
                    self.accepted[page_type]={"hash":image_hash.copy(),"digest":digest,"sequence":context.sequence,
                                              "fields":(self._ocr_result or {}).get("fields",{}),"verified":False}
                    self._capture_requested=False
                    emit,status,action=True,"accepted","capture"
                    event_key=f"freight:{context.session_id}:{page_type}:{digest}"
                    title=PAGE_LABELS[page_type]+" captured for review"
                    missing=self._state()["missing_pages"]
                    guidance="Next show "+PAGE_LABELS[missing[0]]+"." if missing else "Required page types captured. Human review must verify fields and shipment consistency."
            cv2.polylines(annotated,[corners.astype(np.int32)],True,(70,200,80) if acceptable else (20,140,250),3)
        cv2.rectangle(annotated,(0,0),(w,58),(25,30,35),-1)
        cv2.putText(annotated,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        progress=" | ".join(f"{PAGE_LABELS[k]}: {'OK' if k in self.accepted else 'missing'}" for k in self.config["required_pages"])
        cv2.putText(annotated,progress,(12,47),cv2.FONT_HERSHEY_SIMPLEX,.45,(210,220,225),1,cv2.LINE_AA)
        return InspectionResult(status,title,findings,metrics,guidance,action,annotated,event_key,emit,self._state(page_type))

def _form(page_type: str) -> np.ndarray:
    page = np.full((650, 500, 3), 245, np.uint8)
    cv2.rectangle(page, (12, 12), (487, 637), (80, 80, 80), 2)
    color = (220, 85, 30) if page_type == "delivery_note" else (60, 165, 45)
    cv2.rectangle(page, (390, 40), (455, 105), color, -1)
    cv2.putText(page, "DEMO / FICTIONAL", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(page, PAGE_LABELS[page_type].upper(), (28, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2, cv2.LINE_AA)
    lines = (["Reference: DEMO-1042", "Sender: Example Workshop", "Recipient: Fictional Depot", "Items delivered: 12 cartons"]
             if page_type == "delivery_note" else
             ["Packet: DEMO-1042", "Carton | Part | Quantity", "A01    | X11  | 6", "A02    | Y24  | 6"])
    for i, line in enumerate(lines):
        cv2.putText(page, line, (30, 155 + i * 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.line(page, (28, 175 + i * 65), (465, 175 + i * 65), (150, 150, 150), 1)
    if page_type == "packing_list":
        for x in [180, 300, 420]:
            cv2.line(page, (x, 180), (x, 470), (90, 90, 90), 2)
    else:
        cv2.rectangle(page, (30, 440), (455, 545), (100, 100, 100), 2)
        cv2.putText(page, "Received by: DEMO ONLY", (40, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1)
    cv2.putText(page, "Not a real shipping document", (30, 605), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    return page


def demo_frame(index: int, width: int = 960, height: int = 540) -> np.ndarray:
    """Self-authored replay. Index changes rendered pixels only; detection never sees it."""
    phase = index % 240
    page_type = "delivery_note" if phase < 120 else "packing_list"
    form = _form(page_type)
    canvas = np.full((height, width, 3), (48, 42, 36), np.uint8)
    ph = height * 0.87
    pw = ph * 500 / 650
    cx, cy = width * 0.54, height * 0.52
    jitter = 10 * np.sin(index * 0.6) if phase < 20 else 0
    target = np.array([[cx-pw/2+jitter, cy-ph/2], [cx+pw/2+jitter, cy-ph/2+8],
                       [cx+pw/2, cy+ph/2], [cx-pw/2, cy+ph/2-5]], np.float32)
    source = np.array([[0, 0], [499, 0], [499, 649], [0, 649]], np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    rendered = cv2.warpPerspective(form, transform, (width, height))
    mask = cv2.warpPerspective(np.full((650, 500), 255, np.uint8), transform, (width, height))
    canvas[mask > 0] = rendered[mask > 0]
    if phase < 14:
        canvas = cv2.GaussianBlur(canvas, (19, 19), 7)
    return canvas
