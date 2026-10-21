"""Small, explicitly selected local surface-defect model; no network calls.

The accompanying trainer learns from pixel masks on a single surface family.
Mean tree leaf defect fractions are scores, not calibrated probabilities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


FEATURE_VERSION = "surface_gray_multiscale_v1"


def extract_features(frame: np.ndarray, width: int = 256) -> tuple[np.ndarray, tuple[int, int]]:
    """CPU bounded grid, preserving the input aspect ratio (max height 1024)."""
    height = min(1024, max(64, round(frame.shape[0] * width / frame.shape[1])))
    gray = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    f = gray.astype(np.float32) / 255.0
    near = cv2.GaussianBlur(f, (3, 3), 0)
    far = cv2.GaussianBlur(f, (21, 21), 0)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(f * f, (9, 9), 0) - cv2.GaussianBlur(f, (9, 9), 0) ** 2, 0))
    features = [f, near, far, near - far, std,
                cv2.Sobel(near, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(near, cv2.CV_32F, 0, 1, ksize=3)]
    for size in ((3, 3), (9, 3), (3, 9), (11, 11)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        features.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32) / 255.0)
    return np.ascontiguousarray(np.stack(features, axis=-1).reshape(-1, len(features)), dtype=np.float32), gray.shape


def extract_features_fast(frame: np.ndarray, width: int = 256) -> tuple[np.ndarray, tuple[int, int]]:
    """Same float32 operations/features, avoiding unnecessary array temporaries."""
    height = min(1024, max(64, round(frame.shape[0] * width / frame.shape[1])))
    gray = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    f = gray.astype(np.float32)
    f /= 255.0
    near = cv2.GaussianBlur(f, (3, 3), 0)
    far = cv2.GaussianBlur(f, (21, 21), 0)
    local_mean = cv2.GaussianBlur(f, (9, 9), 0)
    np.square(local_mean, out=local_mean)
    std = cv2.GaussianBlur(f * f, (9, 9), 0)
    np.subtract(std, local_mean, out=std)
    np.maximum(std, 0, out=std)
    np.sqrt(std, out=std)
    features = [f, near, far, near - far, std,
                cv2.Sobel(near, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(near, cv2.CV_32F, 0, 1, ksize=3)]
    for size in ((3, 3), (9, 3), (3, 9), (11, 11)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        blackhat /= 255.0
        features.append(blackhat)
    # Native channel interleaving avoids NumPy's repeated strided writes.
    return cv2.merge(features).reshape(-1, len(features)), gray.shape


class SurfaceDefectModel:
    def __init__(self, path: str | Path, backend: str = "auto"):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError(f"Factory defect model not found: {self.path}")
        if self.path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Factory model exceeds the supported 32 MiB artifact limit")
        metadata_path = self.path.with_suffix(self.path.suffix + ".json")
        if not metadata_path.is_file():
            raise ValueError(f"Factory model metadata missing: {metadata_path}")
        metadata_bytes=metadata_path.read_bytes()
        self.metadata = json.loads(metadata_bytes.decode('utf-8'))
        self.metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest()
        if self.metadata.get("feature_version") not in {FEATURE_VERSION, "surface_region_shape_v1", "surface_mobilenet_early_v1", "surface_mobilenet_seg_v1"}:
            raise ValueError("Unsupported factory model feature version")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.metadata.get("model_sha256") != self.sha256:
            raise ValueError("Factory model checksum differs from its metadata")
        if (('openvino_device' in self.metadata or 'openvino_cpu_threads' in self.metadata)
                and self.metadata.get('feature_version')!='surface_mobilenet_seg_v1'):
            raise ValueError('Explicit device/thread metadata is supported only for fused neural segmentation')
        self.neural=None
        if self.metadata.get('feature_version')=='surface_mobilenet_seg_v1':
            if backend!='auto':
                raise ValueError('Fused factory neural model requires defect_backend=auto; CPU is explicitly selected' if self.metadata.get('openvino_device')=='CPU' else 'Fused factory neural model requires defect_backend=auto; GPU is explicitly required')
            from .factory_encoder import FactorySegmentation
            options={'output_logits':self.metadata.get('output_logits',False),
                     'growth_threshold':self.metadata.get('hysteresis_growth_threshold'),
                     'precision_hint':self.metadata.get('openvino_inference_precision','f16')}
            if 'openvino_disable_winograd_convolution' in self.metadata:
                options['disable_winograd_convolution']=self.metadata['openvino_disable_winograd_convolution']
            if 'openvino_device' in self.metadata:options['device']=self.metadata['openvino_device']
            if 'openvino_cpu_threads' in self.metadata:options['cpu_threads']=self.metadata['openvino_cpu_threads']
            self.neural=FactorySegmentation(self.path,**options)
            self.native=None;self.encoder=None;self.region_model=None
            self.backend=getattr(self.neural,'backend_name','openvino_gpu_neural');self.requested_backend=backend;self.backend_fallback_reason=None
            self.width=getattr(self.neural,'grid_width',256);self.pixel_threshold=float(self.metadata['pixel_threshold'])
            self.area_threshold=int(self.metadata['minimum_component_pixels'])
            if not 0<=self.pixel_threshold<=1 or self.area_threshold<1:raise ValueError('Invalid factory model thresholds')
            return
        # Only use artifacts produced locally by the accompanying trainer. Joblib
        # is pickle based and must never be used for untrusted downloaded models.
        import joblib
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        if self.metadata.get("scikit_learn", sklearn.__version__) != sklearn.__version__:
            raise ValueError("Factory model scikit-learn version does not match the installed version; retrain locally")
        artifact = joblib.load(self.path)
        self.region_model = None
        self.encoder = None
        if isinstance(artifact, dict) and artifact.get("kind") == "surface_region_shape_v1":
            from .factory_regions import REGION_COLUMNS
            self.model = artifact.get("proposal")
            self.region_model = artifact.get("region_classifier")
            if (not isinstance(self.region_model, RandomForestClassifier) or
                    getattr(self.region_model, "n_features_in_", 0) != REGION_COLUMNS or
                    list(self.region_model.classes_) != [0,1]):
                raise ValueError("Factory region classifier is incompatible")
            self.region_model.n_jobs = 1
            self.region_threshold = float(self.metadata["region_threshold"])
            if not 0 <= self.region_threshold <= 1:
                raise ValueError("Invalid factory region score threshold")
        else:
            self.model = artifact
            if self.metadata.get("feature_version") not in {FEATURE_VERSION,"surface_mobilenet_early_v1"}:
                raise ValueError("Factory region feature metadata requires a region model bundle")
        columns = 51 if self.metadata.get("feature_version")=="surface_mobilenet_early_v1" else 11
        if not isinstance(self.model, RandomForestClassifier) or getattr(self.model, "n_features_in_", 0) != columns:
            raise ValueError("Factory model is untrained or has incompatible feature dimensions")
        self.model.n_jobs = 2
        if backend not in {"auto", "native_cascade", "threshold_cascade", "sklearn"}:
            raise ValueError("Factory backend must be auto, native_cascade, threshold_cascade or sklearn")
        self.backend = backend
        self.requested_backend = backend
        self.native = None
        self.backend_fallback_reason = None
        positive = np.flatnonzero(self.model.classes_.astype(int) == 1)
        if len(positive) != 1:
            raise ValueError("Factory model has no unique defect label")
        self._positive = int(positive[0])
        if backend in {"auto", "native_cascade"}:
            from .factory_native import NativeCascade
            try:
                self.native = NativeCascade(self.model, self._positive)
                self.backend = "native_cascade"
            except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
                if backend == "native_cascade":
                    raise ValueError(f"Cannot use requested native cascade: {exc}") from exc
                self.backend, self.backend_fallback_reason = "threshold_cascade", str(exc)
        self.width = int(self.metadata.get("input_width", 256))
        if not 64 <= self.width <= 512:
            raise ValueError("Factory model input width must be 64–512")
        self.pixel_threshold = float(self.metadata["pixel_threshold"])
        self.area_threshold = int(self.metadata["minimum_component_pixels"])
        if not 0 <= self.pixel_threshold <= 1 or self.area_threshold < 1:
            raise ValueError("Invalid factory model thresholds")
        if self.metadata.get("feature_version")=="surface_mobilenet_early_v1":
            from .factory_encoder import FactoryEncoder
            encoder_path=(self.path.parent/self.metadata['encoder_file']).resolve()
            if not encoder_path.is_relative_to(self.path.parent):
                raise ValueError('Factory encoder must reside alongside its head artifact')
            self.encoder=FactoryEncoder(encoder_path,self.metadata['encoder_sha256'])

    def score_features(self, features: np.ndarray) -> tuple[np.ndarray, dict]:
        """Preserve threshold decisions while avoiding provably rejected votes.

        Each tree contributes a value in [0, 1]. A pixel whose accumulated
        value plus *all* remaining maximum votes cannot reach the frozen
        threshold is rejected. Completed candidates retain exact mean votes;
        rejected pixels hold a lower bound, never a fabricated probability.
        No tree order, learned parameter, input resolution or threshold changes.
        """
        trees = self.model.estimators_
        if self.backend == "sklearn":
            return self.model.predict_proba(features)[:, self._positive], {
                "model_tree_pixel_evaluations": int(len(features) * len(trees)),
                "model_score_semantics": "exact_mean_tree_leaf_fraction",
                "model_score_map_exact": True}
        if self.backend == "native_cascade":
            scores, evaluations = self.native.score(features, self.pixel_threshold)
            return scores, {"model_tree_pixel_evaluations": int(evaluations),
                            "model_score_semantics": "exact_candidate_mean_rejected_pixel_lower_bound",
                            "model_score_map_exact": False, "native_code_sha256": self.native.sha256}
        total = np.zeros(len(features), dtype=np.float64)
        active = np.arange(len(features))
        required = self.pixel_threshold * len(trees)
        evaluations = 0
        for index, tree in enumerate(trees):
            if not len(active):
                break
            # Tree traversal itself remains scikit-learn's compiled C code.
            values = tree.predict_proba(features if index == 0 else features[active],
                                        check_input=False)[:, self._positive]
            total[active] += values
            evaluations += len(active)
            remaining = len(trees) - index - 1
            # Conservative floating-point margin prevents boundary rejection.
            active = active[total[active] + remaining + 1e-12 >= required]
        return total / len(trees), {
            "model_tree_pixel_evaluations": int(evaluations),
            "model_score_semantics": "exact_candidate_mean_rejected_pixel_lower_bound",
            "model_score_map_exact": False}

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
        """Return threshold-equivalent map, exact candidate scores, and semantics.

        Cascade maps are lower bounds at rejected pixels. Use ``sklearn`` when
        a complete exact score map (for example pixel-ranking research) is needed.
        """
        if self.neural is not None:
            scores, candidates, metrics = self.neural.predict(frame,self.pixel_threshold,self.area_threshold)
            if isinstance(self.metadata.get('score_semantics'), str):
                metrics['model_score_semantics'] = self.metadata['score_semantics']
            return scores, candidates, metrics
        feature_function = extract_features if self.backend == "sklearn" else extract_features_fast
        features, shape = self.encoder.extract(frame) if self.encoder is not None else feature_function(frame, self.width)
        flat_scores, scoring_metrics = self.score_features(features)
        scores = flat_scores.reshape(shape)
        binary = (scores >= self.pixel_threshold).astype(np.uint8)
        region_scores = None
        if self.region_model is not None:
            from .factory_regions import region_features
            descriptors, regions, stats = region_features(frame, scores, binary)
            count = len(stats)
            region_scores = (self.region_model.predict_proba(descriptors)[:,1]
                             if len(descriptors) else np.empty(0))
        else:
            count, regions, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        candidates = []
        h, w = frame.shape[:2]
        for n in range(1, count):
            x, y, rw, rh, area = [int(v) for v in stats[n]]
            if area < self.area_threshold:
                continue
            if region_scores is not None and region_scores[n-1] < self.region_threshold:
                continue
            left, top = int(x * w / shape[1]), int(y * h / shape[0])
            right = min(w, max(left + 1, int(np.ceil((x + rw) * w / shape[1]))))
            bottom = min(h, max(top + 1, int(np.ceil((y + rh) * h / shape[0]))))
            candidates.append({"bbox": (left, top, right - left, bottom - top), "area": area,
                               "score": float(region_scores[n-1] if region_scores is not None else scores[regions == n].mean())})
        candidates.sort(key=lambda item: item["area"], reverse=True)
        maximum_name = "model_score_max" if scoring_metrics["model_score_map_exact"] else "model_score_max_lower_bound"
        return scores, candidates[:20], {maximum_name: round(float(scores.max()), 5),
                                        "encoder_model_sha256": self.encoder.sha256 if self.encoder is not None else None,
                                        "encoder_execution_devices": self.encoder.execution_devices if self.encoder is not None else None,
                                        "region_model_threshold": self.region_threshold if region_scores is not None else None,
                                        "proposal_regions": count-1,
                                        "candidate_score_meaning": "region_forest_leaf_fraction_not_probability" if region_scores is not None else "pixel_forest_leaf_fraction_not_probability",
                                        "model_backend": self.backend, **scoring_metrics,
                                        "model_vote_threshold": self.pixel_threshold,
                                        "model_grid_width": shape[1], "model_grid_height": shape[0],
                                        "candidate_regions": len(candidates),
                                        "model_minimum_component_pixels": self.area_threshold}
