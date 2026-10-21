"""Optional locally compiled, checksum-checked CPU cascade. Never compiles/downloads."""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


class NativeCascade:
    def __init__(self, model, positive: int):
        root = Path(__file__).resolve().parents[2]
        suffix = ".dll" if sys.platform == "win32" else ".so"
        artifact = root / "models/runtime" / ("factory_cascade" + suffix)
        metadata_path = artifact.with_suffix(artifact.suffix + ".json")
        source = root / "edge_runtime/native/factory_cascade.c"
        if not artifact.is_file() or not metadata_path.is_file():
            raise ValueError("Optional local native cascade has not been built; run scripts/build_factory_native.py")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("abi") != "factory_cascade_v1" or metadata.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Native cascade source/ABI changed; rebuild explicitly")
        self.sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if metadata.get("artifact_sha256") != self.sha256:
            raise ValueError("Native cascade checksum mismatch; rebuild explicitly")
        self._library = ctypes.CDLL(str(artifact))
        self._run = self._library.factory_cascade_v1
        int_array = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        double_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        float_matrix = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
        self._run.argtypes = [float_matrix, ctypes.c_int, ctypes.c_int, int_array, ctypes.c_int,
                             int_array, int_array, int_array, double_array, double_array,
                             ctypes.c_double, double_array, ctypes.POINTER(ctypes.c_longlong)]
        self._run.restype = ctypes.c_int
        roots, left, right, feature, threshold, leaf = [], [], [], [], [], []
        offset = 0
        for estimator in model.estimators_:
            tree = estimator.tree_
            roots.append(offset)
            left.extend(np.where(tree.children_left < 0, -1, tree.children_left + offset))
            right.extend(np.where(tree.children_right < 0, -1, tree.children_right + offset))
            feature.extend(tree.feature)
            threshold.extend(tree.threshold)
            leaf.extend(tree.value[:, 0, positive])
            offset += tree.node_count
        self.roots = np.ascontiguousarray(roots, dtype=np.int32)
        self.left = np.ascontiguousarray(left, dtype=np.int32)
        self.right = np.ascontiguousarray(right, dtype=np.int32)
        self.feature = np.ascontiguousarray(feature, dtype=np.int32)
        self.threshold = np.ascontiguousarray(threshold, dtype=np.float64)
        self.leaf = np.ascontiguousarray(leaf, dtype=np.float64)
        self.columns = int(model.n_features_in_)

    def score(self, features: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
        if (features.dtype != np.float32 or features.ndim != 2 or not features.flags.c_contiguous or
                features.shape[1] != self.columns or not 0 < len(features) < 2 ** 31):
            raise ValueError("Native cascade requires a bounded contiguous float32 feature matrix with matching columns")
        output = np.empty(len(features), dtype=np.float64)
        evaluations = ctypes.c_longlong()
        error = self._run(features, len(features), self.columns, self.roots, len(self.roots),
                          self.left, self.right, self.feature, self.threshold, self.leaf,
                          threshold * len(self.roots), output, ctypes.byref(evaluations))
        if error:
            raise ValueError(f"Native cascade failed with code {error}")
        return output, evaluations.value
