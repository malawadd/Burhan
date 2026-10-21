"""Experimental author LiteCrackSeg adapter; explicit local assets only."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .construction_preprocessing import camera_exposure

ROOT = Path(__file__).resolve().parents[2]


def decode_crack_map(probability, shape, threshold=.5, minimum_area=4, masks=False):
    """Retain actual 8-connected mask components, without closing/grouping boxes."""
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError('Expected one finite crack probability map')
    if not 0 < threshold < 1 or type(minimum_area) is not int or minimum_area < 1:
        raise ValueError('Invalid crack component threshold or native-pixel area')
    height, width = shape
    native = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (native > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sums = np.bincount(labels.ravel(), weights=native.ravel(), minlength=count)
    result = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < minimum_area:
            continue
        row = {'class_id': 0, 'label': 'crack', 'bbox': (x, y, w, h),
               'score': float(sums[index] / area), 'native_component_pixels': area}
        if masks:
            row['mask'] = labels == index
        result.append(row)
    return sorted(result, key=lambda row: -row['score'])[:100]


class LiteCrackConstructionModel:
    def __init__(self, options, spec):
        self.spec = spec
        self.size = int(spec['input_size'])
        self.path = (ROOT / 'models/construction' / spec['file']).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != spec['sha256'] or spec['classes'] != ['crack']:
            raise ValueError('LiteCrackSeg asset identity/class mismatch')
        self.threshold = float(options.get('confidence', spec.get('confidence', .5)))
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError('Crack confidence must be in (0,1)')
        self.minimum_area = options.get('minimum_component_pixels', 4)
        if type(self.minimum_area) is not int or self.minimum_area < 1:
            raise ValueError('Minimum native component area must be a positive integer')
        self.mask_rendering = options.get('mask_rendering', 'masks')
        self.exposure_policy = options.get('exposure_policy', 'off')
        if not isinstance(self.exposure_policy, str) or self.exposure_policy not in {'off', 'low_light_gain'}:
            raise ValueError('Unknown exposure policy')
        if self.mask_rendering not in {'boxes', 'masks'}:
            raise ValueError('mask_rendering must be boxes or masks')
        selected = options.get('damage_classes', ['crack'])
        if selected != ['crack']:
            raise ValueError('This specialist supports only crack; spalling is unsupported')
        self.engine = options.get('engine', 'onnxruntime')
        self.device = options.get('openvino_device', 'CPU')
        self.threads = options.get('cpu_threads', 1)
        if type(self.threads) is not int or not 1 <= self.threads <= 8:
            raise ValueError('CPU thread count must be an integer from1 through8')
        started = time.perf_counter()
        input_shape, output_shape = [1, 3, self.size, self.size], [1, 1, self.size, self.size]
        if self.engine == 'onnxruntime':
            if self.device != 'CPU':
                raise ValueError('GPU execution requires OpenVINO')
            import onnxruntime as ort
            settings = ort.SessionOptions()
            settings.intra_op_num_threads = self.threads
            settings.inter_op_num_threads = 1
            settings.add_session_config_entry('session.intra_op.allow_spinning', '0')
            self.session = ort.InferenceSession(str(self.path), sess_options=settings, providers=['CPUExecutionProvider'])
            if self.session.get_inputs()[0].shape != input_shape or self.session.get_outputs()[0].shape != output_shape:
                raise ValueError('LiteCrackSeg ONNX shape mismatch')
            self.input_name = self.session.get_inputs()[0].name
            self.version = ort.__version__
            self.execution_devices = ['CPUExecutionProvider']
        elif self.engine == 'openvino':
            import openvino as ov
            core = ov.Core()
            graph = core.read_model(str(self.path))
            if list(graph.input().shape) != input_shape or list(graph.output().shape) != output_shape:
                raise ValueError('LiteCrackSeg OpenVINO shape mismatch')
            config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', 'INFERENCE_PRECISION_HINT': 'f32'}
            if self.device == 'CPU':
                config['INFERENCE_NUM_THREADS'] = self.threads
            self.compiled = core.compile_model(graph, self.device, config)
            self.request = self.compiled.create_infer_request()
            self.version = ov.get_version()
            self.execution_devices = list(self.compiled.get_property('EXECUTION_DEVICES'))
        else:
            raise ValueError('Engine must be onnxruntime or openvino')
        self.load_ms = 1000 * (time.perf_counter() - started)

    def predict_maps(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected actual uint8 BGR camera input')
        corrected, _ = camera_exposure(frame, self.exposure_policy)
        image = cv2.resize(corrected, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        if self.engine == 'onnxruntime':
            probability = self.session.run(None, {self.input_name: tensor})[0]
        else:
            response = self.request.infer({0: tensor})
            probability = response[self.compiled.output()]
        result = np.array(probability[0, 0], dtype=np.float32, copy=True)
        if result.shape != (self.size, self.size) or not np.isfinite(result).all():
            raise ValueError('Invalid crack probability output')
        return result

    def infer(self, frame):
        return decode_crack_map(self.predict_maps(frame), frame.shape[:2], self.threshold,
                                self.minimum_area, self.mask_rendering == 'masks')

    def describe(self):
        return {'name': 'Experimental thin-crack semantic review', 'mode': 'damage',
                'damage_variant': self.spec.get('variant', 'litecrackseg_tut'),
                'model_path': str(self.path), 'sha256': self.sha256, 'model_source': self.spec['source'],
                'classes': ['crack'], 'selected_damage_classes': ['crack'],
                'supported_capabilities': ['surface_crack'], 'engine': self.engine,
                'engine_version': self.version, 'device': self.device, 'execution_devices': self.execution_devices,
                'cpu_threads': self.threads, 'model_load_ms': self.load_ms,
                'input_shape': [1, 3, self.size, self.size], 'confidence_threshold': self.threshold,
                'minimum_component_pixels': self.minimum_area, 'mask_rendering': self.mask_rendering,
                'exposure_policy': self.exposure_policy,
                'preprocessing': 'Full-frame square stretch; BGR uint8 input; RGB float0..255 inside model (actual author train/test preprocessing)',
                'mask_scope': 'Bilinear native-size probability projection, strict cutoff, 8-connected components; no closing or box grouping; no pixel ground truth in ev-CIVIL',
                'score_meaning': 'Mean uncalibrated sigmoid over a retained crack component; not engineering certainty',
                'license_status': self.spec['license'], 'experimental': True,
                'validation_scope': self.spec.get('validation_scope', 'Calibration only; author training overlap unknown'),
                'code_compliance': 'not_determined',
                'limitations': 'Crack appearance only; no spalling, physical width, depth, structural severity or acceptance decision.'}
