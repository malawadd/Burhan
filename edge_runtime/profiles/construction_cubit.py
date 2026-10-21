"""Research-only author crack/spall segmentation adapter, not yet registered.

The author order is crack=0, spalling=1. Publisher camera boxes do not supply
pixel-mask ground truth. Masks are actual prototype reconstruction, not GT.
"""
from pathlib import Path
import hashlib
import time

import numpy as np

from edge_runtime.profiles.construction_model import prepare, decode
from edge_runtime.profiles.construction_preprocessing import camera_exposure

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['crack', 'spalling']
CAPABILITIES = ['surface_crack', 'surface_spalling']


class CubitConstructionModel:
    def __init__(self, options, spec):
        self.spec = dict(spec)
        if self.spec['classes'] != CLASSES or self.spec.get('input_size') != 640:
            raise ValueError('Author order crack=0/spalling=1 and static640 are required')
        model_root = (ROOT / 'models/construction').resolve()
        self.path = (model_root / self.spec['file']).resolve()
        if not self.path.is_relative_to(model_root):
            raise ValueError('Candidate model path must remain inside local construction assets')
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != self.spec['sha256']:
            raise ValueError('Frozen author segmentation asset identity differs')
        self.threshold = float(options.get('confidence', .25))
        self.nms_iou = float(options.get('nms_iou', .45))
        self.exposure_policy = options.get('exposure_policy', 'low_light_gain')
        if self.threshold != .25 or self.nms_iou != .45 or self.exposure_policy != 'low_light_gain':
            raise ValueError('This fixed candidate requires .25/.45 and shared native low-light gain')
        self.mask_rendering = options.get('mask_rendering', 'masks')
        if self.mask_rendering not in {'masks', 'boxes'}:
            raise ValueError('Select actual reconstructed masks or detector boxes')
        self.selected_classes = options.get('damage_classes', CLASSES)
        if not isinstance(self.selected_classes, list) or not self.selected_classes or len(set(self.selected_classes)) != len(self.selected_classes) or any(x not in CLASSES for x in self.selected_classes):
            raise ValueError('Select only author crack/spalling classes')
        self.engine = options.get('engine', 'openvino')
        self.device = options.get('openvino_device', 'GPU')
        self.threads = options.get('cpu_threads', 1)
        if type(self.threads) is not int or not 1 <= self.threads <= 8:
            raise ValueError('Invalid CPU thread count')
        started = time.perf_counter()
        if self.engine == 'openvino':
            import openvino as ov
            core = ov.Core()
            if self.device not in {'CPU', 'GPU'} or self.device not in core.available_devices:
                raise ValueError('Unavailable local OpenVINO device')
            model = core.read_model(self.path)
            self._check_contract(list(model.input().shape), [list(x.shape) for x in model.outputs])
            config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32', 'NUM_STREAMS': '1'}
            if self.device == 'CPU':
                config.update(INFERENCE_NUM_THREADS=self.threads, NUM_STREAMS=1)
            self.compiled = core.compile_model(model, self.device, config)
            self.request = self.compiled.create_infer_request()
            self.requested_execution_config = dict(config)
            self.actual_execution_config = {}
            for key in ['EXECUTION_DEVICES', 'INFERENCE_PRECISION_HINT', 'NUM_STREAMS', 'INFERENCE_NUM_THREADS', 'PERFORMANCE_HINT']:
                try:
                    value = self.compiled.get_property(key)
                    self.actual_execution_config[key] = value if isinstance(value, (str, int, float, bool, list, type(None))) else str(value)
                except Exception as error:
                    self.actual_execution_config[key] = {'unavailable': type(error).__name__}
            self.version = ov.get_version()
        elif self.engine == 'onnxruntime':
            import onnxruntime as ort
            if self.device != 'CPU':
                raise ValueError('ONNX Runtime candidate is CPU only')
            settings = ort.SessionOptions()
            settings.intra_op_num_threads = self.threads
            settings.inter_op_num_threads = 1
            settings.add_session_config_entry('session.intra_op.allow_spinning', '0')
            self.session = ort.InferenceSession(str(self.path), sess_options=settings, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self._check_contract(self.session.get_inputs()[0].shape, [x.shape for x in self.session.get_outputs()])
            self.version = ort.__version__
            self.requested_execution_config = {'provider': 'CPUExecutionProvider', 'intra_op_num_threads': self.threads, 'inter_op_num_threads': 1}
            self.actual_execution_config = {'providers': self.session.get_providers(), 'intra_op_num_threads': self.session.get_session_options().intra_op_num_threads,
                                            'inter_op_num_threads': self.session.get_session_options().inter_op_num_threads}
        else:
            raise ValueError('Use local OpenVINO or ONNX Runtime')
        self.load_ms = (time.perf_counter() - started) * 1000

    @staticmethod
    def _check_contract(shape, outputs):
        if shape != [1, 3, 640, 640] or outputs != [[1, 38, 8400], [1, 32, 160, 160]]:
            raise ValueError('Static YOLOv8n-seg author output contract differs')

    def predict_raw(self, frame):
        corrected, self.last_input_exposure_gain = camera_exposure(frame, self.exposure_policy)
        tensor, transform = prepare(corrected, 'crack', size=640)
        if self.engine == 'onnxruntime':
            outputs = self.session.run(None, {self.input_name: tensor})
        else:
            result = self.request.infer({0: tensor})
            outputs = [result[self.compiled.output(i)] for i in range(2)]
        if any(not np.isfinite(x).all() for x in outputs):
            raise RuntimeError('Nonfinite actual author model outputs')
        return outputs, transform

    def infer(self, frame):
        outputs, transform = self.predict_raw(frame)
        rows = decode(outputs, 'crack', transform, frame.shape[:2], .25, .45, self.mask_rendering, classes=CLASSES)
        return [row for row in rows if row['label'] in self.selected_classes]

    def describe(self):
        return {'name': 'Author crack/spall mask research', 'sha256': self.sha256,
                'model_path': str(self.path), 'model_source': self.spec['source'],
                'classes': CLASSES, 'selected_damage_classes': self.selected_classes,
                'supported_capabilities': [CAPABILITIES[CLASSES.index(x)] for x in self.selected_classes],
                'engine': self.engine, 'device': self.device, 'engine_version': self.version,
                'requested_execution_config': self.requested_execution_config,
                'actual_execution_config': self.actual_execution_config,
                'cpu_threads': self.threads, 'model_load_ms': self.load_ms,
                'input_shape': [1, 3, 640, 640], 'preprocessing': 'letterbox RGB/255',
                'exposure_policy': self.exposure_policy, 'confidence_threshold': .25,
                'class_confidence': dict.fromkeys(CLASSES, .25), 'nms_iou': .45,
                'mask_rendering': self.mask_rendering,
                'mask_scope': 'Actual author prototype logits >0 reconstructed at native size and cropped to detector boxes; no publisher pixel-mask GT or physical-width claim.',
                'license_status': self.spec['license'], 'diagnostic_only': True,
                'validation_scope': self.spec.get('validation_scope', 'Not evaluated locally; source training overlap unknown; research only.')}
