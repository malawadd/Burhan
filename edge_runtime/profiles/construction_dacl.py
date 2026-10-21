"""Local multi-label DACL semantic damage adapter; no automatic downloads."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTHOR_CLASSES = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']
CLASSES = ['crack', 'alligator_crack', 'wetspot', 'efflorescence', 'rust', 'rockpocket', 'hollowareas', 'cavity', 'spalling', 'graffiti', 'weathering', 'residual_formwork', 'exposed_rebar', 'bearing', 'expansion_joint', 'drainage', 'protective_equipment', 'joint_tape', 'washout']
DAMAGE_CLASSES = CLASSES[:13] + ['washout']
CAPABILITIES = ['surface_' + name for name in DAMAGE_CLASSES]


def decode_masks(probabilities, shape, selected, threshold=.5, minimum_area=4, mask_rendering='boxes'):
    """Independent sigmoid channels; an overlapping pixel can have many labels."""
    if probabilities.ndim != 3 or probabilities.shape[0] != 19 or not np.isfinite(probabilities).all():
        raise ValueError('DACL output must be 19 finite independent probability maps')
    height, width = shape
    gh, gw = probabilities.shape[1:]
    detections = []
    for class_id, name in enumerate(CLASSES):
        if name not in selected:
            continue
        mask = (probabilities[class_id] > threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum_area) + 1
        if len(regions) > 20:
            regions = regions[np.argsort(-stats[regions, cv2.CC_STAT_AREA], kind='stable')[:20]]
        sums = np.bincount(labels.ravel(), weights=probabilities[class_id].ravel(), minlength=count)
        for region in regions:
            x, y, rw, rh, area = map(int, stats[region])
            # Exact nearest-neighbour grid projection, without stretching labels
            # by an extra bilinear interpolation or inflating their min-area.
            left, top = math.ceil(x * width / gw), math.ceil(y * height / gh)
            right, bottom = min(width, math.ceil((x + rw) * width / gw)), min(height, math.ceil((y + rh) * height / gh))
            if right <= left or bottom <= top:
                continue
            item = {'class_id': class_id, 'label': name, 'bbox': (left, top, right - left, bottom - top),
                    'score': float(sums[region] / area), 'model_grid_component_pixels': area}
            if mask_rendering == 'masks':
                item['mask'] = cv2.resize((labels == region).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            detections.append(item)
    return sorted(detections, key=lambda item: -item['score'])[:100]


class DaclConstructionModel:
    def __init__(self, options, spec):
        self.spec = spec
        self.mode = 'damage'
        self.variant = options.get('damage_variant')
        self.size = int(spec['input_size'])
        self.path = (ROOT / 'models/construction' / spec['file']).resolve()
        if not self.path.is_file():
            raise ValueError('DACL local model is missing; acquire/export it explicitly')
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != spec['sha256']:
            raise ValueError('DACL model checksum mismatch')
        if spec['classes'] != CLASSES:
            raise ValueError('DACL ordered 19-class contract mismatch')
        try:
            self.threshold = float(options.get('confidence', spec.get('confidence', .5)))
        except (TypeError, ValueError) as error:
            raise ValueError('DACL confidence must be numeric') from error
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError('DACL confidence must be in (0,1)')
        self.minimum_area = options.get('minimum_component_pixels', 4)
        if type(self.minimum_area) is not int or not 1 <= self.minimum_area <= self.size * self.size:
            raise ValueError('DACL minimum_component_pixels must be a positive grid-pixel integer')
        selected = options.get('damage_classes', DAMAGE_CLASSES)
        if not isinstance(selected, list) or not selected or any(not isinstance(name, str) or name not in DAMAGE_CLASSES for name in selected):
            raise ValueError('DACL damage_classes must select supported canonical visible-condition names')
        self.selected = list(dict.fromkeys(selected))
        cutoffs = options.get('class_confidence', {} if 'confidence' in options else spec.get('class_confidence', {}))
        if not isinstance(cutoffs, dict) or any(name not in DAMAGE_CLASSES for name in cutoffs):
            raise ValueError('DACL class_confidence must map supported canonical classes to cutoffs')
        self.class_thresholds = {}
        for name, value in cutoffs.items():
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('DACL per-class confidence must be finite and in (0,1)')
            if name in self.selected:
                self.class_thresholds[name] = float(value)
        self.mask_rendering = options.get('mask_rendering', 'boxes')
        if not isinstance(self.mask_rendering, str) or self.mask_rendering not in {'boxes', 'masks'}:
            raise ValueError('DACL mask_rendering must be boxes or masks')
        self.engine = options.get('engine', 'onnxruntime')
        self.device = options.get('openvino_device', 'CPU')
        self.threads = options.get('cpu_threads', 4)
        if type(self.threads) is not int or not 1 <= self.threads <= 8 or not isinstance(self.device, str) or self.device not in {'CPU', 'GPU'}:
            raise ValueError('Invalid DACL device or CPU thread count')
        started = time.perf_counter()
        expected_input, expected_output = [1, 3, self.size, self.size], [1, 19, self.size // 4, self.size // 4]
        if self.engine == 'onnxruntime':
            if self.device != 'CPU':
                raise ValueError('DACL GPU requires OpenVINO')
            import onnxruntime as ort
            settings = ort.SessionOptions()
            settings.intra_op_num_threads = self.threads
            settings.inter_op_num_threads = 1
            settings.add_session_config_entry('session.intra_op.allow_spinning', '0')
            self.session = ort.InferenceSession(str(self.path), sess_options=settings, providers=['CPUExecutionProvider'])
            if self.session.get_inputs()[0].shape != expected_input or self.session.get_outputs()[0].shape != expected_output:
                raise ValueError('DACL ONNX shape mismatch')
            self.input_name = self.session.get_inputs()[0].name
            self.version = ort.__version__
            self.execution_devices = ['CPUExecutionProvider']
        elif self.engine == 'openvino':
            import openvino as ov
            self.core = ov.Core()
            if self.device not in self.core.available_devices:
                raise ValueError('Requested DACL OpenVINO device unavailable')
            model = self.core.read_model(str(self.path))
            if list(model.input().shape) != expected_input or list(model.output().shape) != expected_output:
                raise ValueError('DACL OpenVINO shape mismatch')
            config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', 'INFERENCE_PRECISION_HINT': 'f32'}
            if self.device == 'CPU':
                config['INFERENCE_NUM_THREADS'] = self.threads
            self.compiled = self.core.compile_model(model, self.device, config)
            self.request = self.compiled.create_infer_request()
            self.execution_devices = list(self.compiled.get_property('EXECUTION_DEVICES'))
            self.version = ov.get_version()
        else:
            raise ValueError('DACL engine must be onnxruntime or openvino')
        self.load_ms = (time.perf_counter() - started) * 1000

    def predict_maps(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('DACL requires real uint8 BGR image input')
        image = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        if self.engine == 'onnxruntime':
            logits = self.session.run(None, {self.input_name: tensor})[0]
        else:
            result = self.request.infer({0: tensor})
            logits = result[self.compiled.output()]
        logits = np.array(logits[0], dtype=np.float32, copy=True)
        if not np.isfinite(logits).all():
            raise ValueError('DACL model produced nonfinite logits')
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def infer(self, frame):
        start = time.perf_counter()
        probabilities = self.predict_maps(frame)
        inferred = time.perf_counter()
        if self.class_thresholds:
            result = []
            for name in self.selected:
                result.extend(decode_masks(probabilities, frame.shape[:2], [name],
                                           self.class_thresholds.get(name, self.threshold),
                                           self.minimum_area, self.mask_rendering))
            result = sorted(result, key=lambda item: -item['score'])[:100]
        else:
            result = decode_masks(probabilities, frame.shape[:2], self.selected, self.threshold,
                                  self.minimum_area, self.mask_rendering)
        done = time.perf_counter()
        self.last_timing_ms = {'preprocess_inference_sigmoid': (inferred - start) * 1000,
                               'components_masks': (done - inferred) * 1000, 'adapter': (done - start) * 1000}
        return result

    def describe(self):
        return {'name': 'Construction multi-label visible-condition review', 'mode': self.mode,
                'damage_variant': self.variant, 'model_path': str(self.path), 'sha256': self.sha256,
                'model_source': self.spec['source'], 'classes': CLASSES, 'author_classes': AUTHOR_CLASSES,
                'selected_damage_classes': self.selected, 'supported_capabilities': ['surface_' + name for name in self.selected],
                'engine': self.engine, 'engine_version': self.version, 'device': self.device,
                'execution_devices': self.execution_devices, 'cpu_threads': self.threads,
                'input_shape': [1, 3, self.size, self.size], 'model_load_ms': round(self.load_ms, 2),
                'confidence_threshold': self.threshold, 'minimum_component_pixels': self.minimum_area,
                'class_confidence': dict(self.class_thresholds),
                'preprocessing': 'Stretch resize BGR to fixed square; uint8 NCHW; RGB/ImageNet normalization inside ONNX',
                'mask_rendering': self.mask_rendering,
                'mask_scope': 'Independent sigmoid channels; nearest projected retained components; classes may overlap. No physical dimensions.',
                'class_meanings': {'hollowareas': 'Visible inspector chalk/crayon boundaries after sounding; not internal void detection',
                                   'washout': 'Combined planar concrete surface corrosion/erosion/abrasion appearance; mechanism unclassified'},
                'score_meaning': 'Mean uncalibrated per-class sigmoid over retained component; not engineering certainty',
                'license_status': self.spec.get('license', 'Dataset author CC BY-NC4; separate demo-weight license undeclared; commercial clearance unresolved'),
                'validation_scope': self.spec.get('validation_scope', 'Local per-class evaluation pending; data/model availability does not establish category accuracy'),
                'experimental': True, 'code_compliance': 'not_determined',
                'limitations': 'Visible appearance only; no structural severity, concealed work, material strength, internal delamination or code acceptance.'}
