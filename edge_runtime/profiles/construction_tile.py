"""Local normal-only tile texture anomaly specialist, separate from installation.

The output is an uncalibrated texture-distance map, not a semantic crack label.
Asset metadata/hashes pin the exact fitted bank, preprocessing and threshold.
"""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np


class TileSurfaceModel:
    def __init__(self, metadata_path, threads=1):
        import onnxruntime as ort
        self.path = Path(metadata_path)
        self.spec = json.loads(self.path.read_text(encoding='utf-8'))
        self.directory = self.path.parent
        for field in ['encoder', 'bank']:
            asset = self.directory / self.spec[field]['file']
            if hashlib.sha256(asset.read_bytes()).hexdigest() != self.spec[field]['sha256']:
                raise ValueError('Tile specialist asset checksum mismatch: ' + field)
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = threads
        settings.inter_op_num_threads = 1
        settings.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(self.directory / self.spec['encoder']['file']), sess_options=settings,
                                            providers=['CPUExecutionProvider'])
        bank = np.load(self.directory / self.spec['bank']['file'], allow_pickle=False)
        self.mean, self.scale, self.bank = bank['mean'], bank['scale'], bank['bank']
        if self.bank.ndim != 2 or self.bank.shape[1] != 40 or not np.isfinite(self.bank).all():
            raise ValueError('Invalid frozen tile patch bank')
        self.threshold = float(self.spec['threshold'])

    def features(self, frame):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Tile input must be original uint8 BGR')
        image = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        grid = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0]
        if grid.shape != (40, 32, 32) or not np.isfinite(grid).all():
            raise ValueError('Tile encoder shape/finite contract failed')
        return grid

    def predict_maps(self, frame):
        grid = self.features(frame)
        points = (grid.reshape(40, -1).T - self.mean) / self.scale
        # Deterministic exact nearest neighbor to frozen TRAIN-only normal bank.
        square = (points * points).sum(1)[:, None] + (self.bank * self.bank).sum(1)[None] - 2 * points @ self.bank.T
        distances = np.sqrt(np.maximum(square.min(1), 0)).reshape(32, 32)
        smooth = cv2.GaussianBlur(distances, (5, 5), 0)
        return cv2.resize(smooth, frame.shape[1::-1], interpolation=cv2.INTER_LINEAR)

    def infer(self, frame):
        anomaly = self.predict_maps(frame)
        mask = (anomaly > self.threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        results = []
        for region in range(1, count):
            x, y, w, h, area = map(int, stats[region])
            if area < 4:
                continue
            results.append({'label': 'tile_surface_anomaly', 'bbox': [x, y, w, h], 'mask': labels == region,
                            'anomaly_distance': float(anomaly[labels == region].max()),
                            'evidence_basis': 'Distance from frozen normal manufacturing tile texture; defect subtype unclassified'})
        return results

    def describe(self):
        return {'name': 'Tile surface texture anomaly research specialist', 'model_kind': 'normal-only nearest-patch bank',
                'metadata_path': str(self.path), 'metadata_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'classes': ['tile_surface_anomaly'], 'threshold': self.threshold, 'engine': 'ONNX Runtime CPU',
                'input': 'BGR uint8 -> INTER_AREA256 square -> RGB/ImageNet normalization inside encoder',
                'scope': 'Manufacturing tile closeups only; no installed-grid alignment, lippage, adhesion or moisture measurement',
                'score_meaning': 'Uncalibrated nearest-normal texture distance, not probability or physical severity',
                'license': self.spec['license'], 'experimental': True, 'code_compliance': 'not_determined'}
