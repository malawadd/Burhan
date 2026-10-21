"""Offline OCR. Models must already exist; never resolve or download at inference time."""
from __future__ import annotations

import copy
import hashlib
from importlib.util import find_spec
from pathlib import Path
import re
import time

import numpy as np

MODEL_NAMES = ("ch_PP-OCRv4_det_mobile.onnx", "ch_ppocr_mobile_v2.0_cls_mobile.onnx", "en_PP-OCRv4_rec_mobile.onnx")
MODEL_HASHES = ("d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c", "e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318")


def model_paths():
    spec = find_spec("rapidocr")
    if spec is None or spec.origin is None:
        return []
    root = Path(__file__).resolve().parents[2] / "models" / "freight"
    bundled = Path(spec.origin).parent / "models"
    return [root / MODEL_NAMES[0], bundled / MODEL_NAMES[1], root / MODEL_NAMES[2]]


def model_inventory() -> dict:
    paths = model_paths()
    files = [{"name": path.name, "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in paths if path.is_file()]
    valid = len(files) == len(MODEL_NAMES) and all(x["sha256"] == sha for x, sha in zip(files, MODEL_HASHES))
    return {"available": valid, "models": files,
            "backend": "RapidOCR 3.9.0 / PP-OCRv4 English mobile / ONNX Runtime CPU",
            "error": None if valid else "OCR dependencies/models missing or hash mismatch. Install requirements, then run scripts/fetch_freight_models.py once."}


class LocalOCR:
    def __init__(self, preset="optimized", overrides=None):
        if preset not in ("baseline", "optimized"):
            raise ValueError("OCR preset must be baseline or optimized")
        self.engine = None
        self.preset = preset
        self.overrides = copy.deepcopy(dict(overrides or {}))
        self._params = {
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
            "Rec.rec_batch_num": 6, "Rec.rec_img_shape": [3, 48, 320],
        }
        if self.preset == "optimized":
            self._params.update({"EngineConfig.onnxruntime.intra_op_num_threads": 4,
                                 "EngineConfig.onnxruntime.enable_cpu_mem_arena": True,
                                 "Rec.rec_batch_num": 1})
        self._params.update({key: copy.deepcopy(value) for key, value in self.overrides.items()
                             if not key.startswith("_")})
        self._disable_spinning = self.overrides.get("_disable_spinning", self.preset == "optimized")
        # Published metadata is complete before the worker starts. Never mutate it
        # during initialization/inference: describe/API/event serialization can read
        # this same object concurrently while native OCR creates its sessions.
        self.info = {**model_inventory(), "runtime_settings": {
            "preset": self.preset, "intra_op_threads": self._params["EngineConfig.onnxruntime.intra_op_num_threads"],
            "inter_op_threads": self._params["EngineConfig.onnxruntime.inter_op_num_threads"],
            "cpu_mem_arena": self._params["EngineConfig.onnxruntime.enable_cpu_mem_arena"],
            "disable_spinning": self._disable_spinning,
            "recognition_batch": self._params["Rec.rec_batch_num"],
            "recognition_image_shape": copy.deepcopy(self._params["Rec.rec_img_shape"]),
            "detector_max_side": self.overrides.get("_detector_max_side"),
        }}
        self.session_load_ms = None
        cap = self.overrides.get("_detector_max_side")
        if cap is not None and (not isinstance(cap, int) or not 320 <= cap <= 2000):
            raise ValueError("Detector cap must be an integer from 320 through 2000")

    def read(self, page: np.ndarray) -> dict:
        if not self.info["available"]:
            raise RuntimeError(self.info["error"])
        if self.engine is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import OCRVersion
            paths = model_paths()
            # Explicit local paths prevent model resolver downloads. Hash/size evidence
            # comes from the installed wheel and is exported with every OCR observation.
            params = {
                "Det.model_path": str(paths[0]),
                "Det.ocr_version": OCRVersion.PPOCRV4,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
                "Det.mean": [0.485, 0.456, 0.406],
                "Det.std": [0.229, 0.224, 0.225],
                "Cls.model_path": str(paths[1]),
                "Rec.model_path": str(paths[2]),
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "EngineConfig.onnxruntime.use_cuda": False,
                "EngineConfig.onnxruntime.use_dml": False,
                "Global.log_level": "error",
                "Global.max_side_len": 1600,
                "Det.limit_side_len": 736,
                "Global.text_score": 0.50,
            }
            params.update(copy.deepcopy(self._params))
            disable_spinning = self._disable_spinning
            load_started = time.perf_counter()
            if disable_spinning:
                # RapidOCR 3.9 supports injected ONNX sessions. Its OmegaConf loader
                # cannot accept opaque objects as ordinary params, so inject at its
                # initialization hook rather than silently passing unsupported flags.
                import onnxruntime as ort
                from omegaconf import flag_override
                class ExplicitSessionsOCR(RapidOCR):
                    def _initialize(inner, cfg):
                        with flag_override(cfg, "allow_objects", True):
                            for section, path in zip((cfg.Det, cfg.Cls, cfg.Rec), paths):
                                options = ort.SessionOptions()
                                options.intra_op_num_threads = params["EngineConfig.onnxruntime.intra_op_num_threads"]
                                options.inter_op_num_threads = params["EngineConfig.onnxruntime.inter_op_num_threads"]
                                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                                options.enable_cpu_mem_arena = params["EngineConfig.onnxruntime.enable_cpu_mem_arena"]
                                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                                options.add_session_config_entry("session.inter_op.allow_spinning", "0")
                                section.session = ort.InferenceSession(str(path), sess_options=options,
                                                                       providers=["CPUExecutionProvider"])
                            super()._initialize(cfg)
                self.engine = ExplicitSessionsOCR(params=params)
            else:
                self.engine = RapidOCR(params=params)
            cap = self.overrides.get("_detector_max_side")
            if cap:
                # RapidOCR 3.9's max branch ignores configured side length. Bound only
                # detector tensors; its returned boxes still crop the original page.
                from rapidocr.ch_ppocr_det.utils import DetPreProcess
                mean, std = params["Det.mean"], params["Det.std"]
                self.engine.text_det.get_preprocess = lambda _: DetPreProcess(int(cap), "max", mean, std)
            self.session_load_ms = round((time.perf_counter() - load_started) * 1000, 2)
        started = time.perf_counter()
        result = self.engine(page)
        lines = [{"text": str(text)[:500], "score": round(float(score), 4),
                  "box": np.asarray(box).round(1).tolist()}
                 for text, score, box in zip(() if result.txts is None else result.txts, () if result.scores is None else result.scores,
                                            [] if result.boxes is None else result.boxes)][:250]
        return {"coordinate_space": "rectified_document_pixels", "page_size": [int(page.shape[1]), int(page.shape[0])],
                "lines": lines, "text": "\n".join(x["text"] for x in lines)[:20000],
                "ocr_ms": round((time.perf_counter() - started) * 1000, 2),
                "session_load_ms": self.session_load_ms,
                "ocr_stage_ms": {name: round(float(value) * 1000, 2) for name, value in
                                 zip(("detection", "orientation", "recognition"), result.elapse_list or [])},
                "mean_score": round(float(np.mean([x["score"] for x in lines])), 4) if lines else 0.,
                "provenance": {**self.info, "performance_preset": self.preset, "engine_overrides": self.overrides}}


def parse_document(ocr: dict) -> dict:
    """Conservative English heading rules and verbatim labelled values, never semantic guesses."""
    lines = ocr.get("lines", [])
    text = ocr.get("text", "")
    # Large scanned invoices can have substantial whitespace. Restrict heading rules
    # to the top 40% of image height, rather than relying on detector line ordering.
    top = []
    max_y = max((max(p[1] for p in line.get("box", [[0, 0]])) for line in lines), default=1)
    for line in lines:
        y = min(p[1] for p in line.get("box", [[0, 0]]))
        if y <= max_y * .40:
            top.append(line["text"])
    heading = re.sub(r"[^A-Z0-9 ]", " ", " ".join(top).upper())
    heading = re.sub(r"\s+", " ", heading)
    rules = [("bill_of_lading", r"\bBILL\s+OF\s+LADING\b"),
             ("packing_list", r"\bPACKING\s+(?:LIST|SLIP)\b"),
             ("delivery_note", r"\b(?:DELIVERY\s+(?:NOTE|ORDER)|PROOF\s+OF\s+DELIVERY)\b"),
             ("invoice", r"\b(?:COMMERCIAL\s+)?INVOICE\b")]
    matches = [name for name, pattern in rules if re.search(pattern, heading)]
    # Mixed documents require human separation rather than silently selecting one.
    page_type = matches[0] if len(matches) == 1 else None
    fields = {}
    patterns = {
        "reference": r"(?:invoice\s*(?:(?:no\.?|number|#)|(?=:))|reference|delivery\s*(?:no\.?|number)|BOL\s*(?:no\.?|#))\s*[:#-]?\s*([A-Z0-9][A-Z0-9./-]{2,40})",
        "date": r"\b(?:invoice\s+date|date)\s*[:#-]?\s*((?:\d{1,4}[./-]){2}\d{1,4})",
        "currency": r"\b(USD|EUR|GBP|SAR|AED|CAD|AUD|JPY|INR|IDR)\b",
        "gross_weight": r"\b(?:gross\s+weight)\s*[:#-]?\s*([0-9][0-9., ]*\s*(?:KG|KGS|LB|LBS|TON)?)",
        "net_weight": r"\b(?:net\s+weight)\s*[:#-]?\s*([0-9][0-9., ]*\s*(?:KG|KGS|LB|LBS|TON)?)",
        "packages": r"\b(?:total\s+(?:packages|cartons)|no\.?\s+of\s+(?:packages|cartons))\s*[:#-]?\s*(\d{1,8})",
        "total": r"\b(?:grand\s+total|total\s+amount|amount\s+due|(?:^|\n)\s*total)\s*[:#-]?\s*((?:(?:USD|EUR|GBP|SAR|AED|\$)\s*)?[0-9][0-9.,]*)",
    }
    for name, pattern in patterns.items():
        field_matches = list(re.finditer(pattern, text, re.I))
        if name == "total":
            qualified = [m for m in field_matches if re.search(r"USD|EUR|GBP|SAR|AED|\$", m.group(1), re.I)]
            # Prefer an explicit currency-bearing total over a table-column heading.
            match = (qualified or field_matches)[-1] if field_matches else None
        else:
            match = field_matches[0] if field_matches else None
        if match:
            fields[name] = {"value": match.group(1).strip(), "method": "label_regex",
                            "evidence": match.group(0)[:160], "verified": False}
    return {"document_type": page_type, "heading_matches": matches,
            "coordinate_space": "rectified_document_pixels", "page_size": ocr.get("page_size"),
            "fields": fields, "text": text, "lines": lines,
            "mean_score": ocr.get("mean_score", 0.), "ocr_ms": ocr.get("ocr_ms", 0.),
            "provenance": ocr.get("provenance", {}),
            "ocr_stage_ms": ocr.get("ocr_stage_ms", {}),
            "session_load_ms": ocr.get("session_load_ms"),
            "requires_human_review": True}
