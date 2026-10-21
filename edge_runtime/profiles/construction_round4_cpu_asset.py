"""Data-only FAILED CPU research asset; no model loading or defaults."""
VARIANT = 'round4_mixed_cpu_failed'
SPEC = {'capabilities': ['surface_crack', 'surface_spalling'],
 'class_capabilities': {'crack': 'surface_crack', 'spalling': 'surface_spalling'},
 'class_confidence': {'crack': 0.25, 'spalling': 0.25},
 'classes': ['crack', 'spalling'],
 'confidence': 0.25,
 'decoder': 'cubit_yolo_seg',
 'diagnostic_only': True,
 'display_name': 'CPU crack/spall masks · FAILED900 research',
 'experimental': True,
 'exposure_policy': 'low_light_gain',
 'file': 'round4-mixed-v3/crack-spalling-step0900-640.onnx',
 'input_size': 640,
 'license': 'CUBIT-InSeg publisher dataCCBY4; DACL dataCCBYNC4; ev-CIVIL dataCCBY4; Ultralytics codeAGPL3. '
            'Inherited author checkpoint standalone license/training overlap unresolved. Research only.',
 'mask_rendering': 'masks',
 'nms_iou': 0.45,
 'required_execution': {'cpu_threads': 1, 'engine': 'onnxruntime', 'openvino_device': 'CPU'},
 'sha256': '8bf254cc4e1079bc58bd9590f5e57630890f6d4048e2afc76a1ceb51dfaa1d2b',
 'source': 'https://github.com/BenyunZhao/CUBIT-InSeg',
 'validation_scope': 'No checkpoint passed complete122 CPU calibration guards. Exact900 failure diagnostic; '
                     'actual native masks retained. Private outputs unscored; no reliable '
                     'humidity/fastener/structural/SBC assessment or15FPS qualification.',
 'wrapper_sha256': 'fe41777344cd92ae6a2b9cca05db56ab5418392448ebc87076cd51511b45686d'}
