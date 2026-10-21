"""Rebuild ONLY installed official YOLOv8n-seg; strict restored state export.

Running this script constructs a network and executes stride/export forwards.
It is forbidden until the separate candidate protocol and NN slot are approved.
Untrusted author YAML is compared as data, never sent to parse_model.
"""
from pathlib import Path
import copy
from datetime import datetime, timezone
import hashlib
import json
import traceback

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'models/construction/cubit-author-release-v1'
OUT = ROOT / 'output/evaluation/construction-cubit-author-v1'
PROTOCOL = OUT / 'candidate-protocol-before-network.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    import torch
    import ultralytics
    import yaml
    from ultralytics.nn.tasks import SegmentationModel
    protocol = read(PROTOCOL)
    if not protocol.get('parent_approved_network'):
        raise RuntimeError('Explicit reviewed network protocol required')
    for path, digest in protocol['pinned_files'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError('Frozen candidate dependency differs: ' + path)
    if protocol['classes'] != ['crack', 'spalling'] or protocol['class_confidence'] != {'crack': .25, 'spalling': .25}:
        raise RuntimeError('Author class mapping or fixed policy differs')
    schema_path = Path(ultralytics.__file__).parent / 'cfg/models/v8/yolov8-seg.yaml'
    if sha(schema_path) != protocol['installed_trusted_schema_sha256']:
        raise RuntimeError('Installed trusted schema differs')
    trusted = yaml.safe_load(schema_path.read_text(encoding='utf-8'))
    author = read(DIRECTORY / 'author-model-yaml.json')
    allowed_keys = {'nc', 'scales', 'backbone', 'head', 'scale', 'yaml_file', 'ch'}
    if set(author) != allowed_keys or any(author[key] != trusted[key] for key in ['scales', 'backbone', 'head']):
        raise RuntimeError('Author architecture does not exactly match installed trusted structure')
    if author['nc'] != 2 or author['scale'] != 'n' or author['ch'] != 3 or author['yaml_file'] != 'yolov8n-seg.yaml':
        raise RuntimeError('Unexpected author scalar architecture metadata')
    # Only locally installed schema values go to the network parser. Neither
    # author YAML objects nor checkpoint object graph reach the constructor.
    config = copy.deepcopy(trusted)
    config.update(nc=2, scale='n')
    receipt_path = OUT / 'export-execution.json'
    if receipt_path.exists():
        raise RuntimeError('Preserve any prior candidate export attempt')
    receipt = {'state': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'candidate_protocol_sha256': sha(PROTOCOL), 'exporter_sha256': sha(Path(__file__)),
               'trusted_schema_sha256': sha(schema_path), 'author_schema_exact_structure_match': True,
               'network_constructor_executes_dummy_stride_forward': True,
               'policy': 'Installed trusted schema only; strict state loading; no training or class reset.'}
    write(receipt_path, receipt)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        state = torch.load(DIRECTORY / 'author-state-dict.pt', map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or len(state) != 417 or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Plain finite author state contract differs')
        model = SegmentationModel(cfg=config, ch=3, nc=2, verbose=False).float().eval()
        if set(model.state_dict()) != set(state) or any(model.state_dict()[key].shape != value.shape for key, value in state.items()):
            raise RuntimeError('Installed trusted model and author tensor shapes differ')
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(model.state_dict()[key], value.float() if value.is_floating_point() else value) for key, value in state.items()):
            raise RuntimeError('Strict source tensors did not transfer exactly')
        model.names = {0: 'crack', 1: 'spalling'}
        head = model.model[-1]
        if head.nc != 2 or head.nm != 32:
            raise RuntimeError('Author segmentation head dimensions differ')
        head.export = True
        head.format = 'onnx'
        head.dynamic = False
        # Nonconstant deterministic export fixture, never called inspection data.
        sample = torch.linspace(0, 1, 3 * 640 * 640).reshape(1, 3, 640, 640)
        with torch.inference_mode():
            outputs = model(sample)
        if [list(x.shape) for x in outputs] != [[1, 38, 8400], [1, 32, 160, 160]] or any(not torch.isfinite(x).all() for x in outputs):
            raise RuntimeError('Actual author prediction/prototype contract differs')
        target = DIRECTORY / 'author-yolov8n-seg-640.onnx'
        if target.exists():
            raise RuntimeError('Preserve existing export')
        torch.onnx.export(model, sample, str(target), opset_version=17, input_names=['images'],
                          output_names=['predictions', 'prototypes'], dynamo=False)
        import numpy as np
        np.savez_compressed(OUT / 'export-fixture-torch-outputs.npz', images=sample.numpy(),
                            predictions=outputs[0].detach().numpy(), prototypes=outputs[1].detach().numpy())
        import onnxruntime as ort
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 1
        settings.inter_op_num_threads = 1
        session = ort.InferenceSession(str(target), sess_options=settings, providers=['CPUExecutionProvider'])
        converted = session.run(None, {session.get_inputs()[0].name: sample.numpy()})
        np.savez_compressed(OUT / 'export-fixture-ort-cpu-outputs.npz', predictions=converted[0], prototypes=converted[1])
        stats = {'coordinate_max_abs': float(np.max(np.abs(converted[0][:, :4] - outputs[0][:, :4].numpy()))),
                 'class_score_max_abs': float(np.max(np.abs(converted[0][:, 4:6] - outputs[0][:, 4:6].numpy()))),
                 'mask_coefficient_max_abs': float(np.max(np.abs(converted[0][:, 6:] - outputs[0][:, 6:].numpy()))),
                 'prototype_max_abs': float(np.max(np.abs(converted[1] - outputs[1].numpy()))),
                 'all_finite': all(bool(np.isfinite(x).all()) for x in converted)}
        passed = stats['all_finite'] and stats['coordinate_max_abs'] <= 1 and stats['class_score_max_abs'] <= .01 and stats['mask_coefficient_max_abs'] <= .1 and stats['prototype_max_abs'] <= .1
        write(OUT / 'export-fixture-ort-cpu-numeric.json', {'stats': stats, 'pass': passed, 'onnxruntime_version': ort.__version__,
                  'scope': 'Exactly one deterministic numeric fixture, not inspection footage or accuracy evidence.'})
        if not passed:
            raise RuntimeError('First Torch/ORTCPU export fixture failed; preserve all outputs, no retry')
        metadata = {'onnx_sha256': sha(target), 'source_checkpoint_sha256': protocol['source_checkpoint_sha256'],
                    'source_state_sha256': sha(DIRECTORY / 'author-state-dict.pt'),
                    'candidate_protocol_sha256': sha(PROTOCOL), 'classes': ['crack', 'spalling'],
                    'input_size': 640, 'architecture': 'yolov8n-seg', 'decoder': 'cubit_yolo_seg',
                    'confidence': .25, 'class_confidence': {'crack': .25, 'spalling': .25}, 'nms_iou': .45,
                    'exposure_policy': 'low_light_gain', 'source': 'https://benyunzhao.github.io/CUBIT-Dataset/',
                    'license': 'Author standalone checkpoint license undeclared; exact training overlap unknown. Ultralytics code AGPL-3.0. CUBIT photo dataset CC BY4.0 does not independently clear checkpoint or video rights. Research only.',
                    'trusted_reconstruction': True, 'strict_state_transfer': True, 'training_performed': False,
                    'synthetic_fixture_scope': 'Initialization/export numeric check only, never defect footage or accuracy evidence.'}
        write(target.with_suffix('.provenance.json'), metadata)
        receipt.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                       model_sha256=sha(target), metadata_sha256=sha(target.with_suffix('.provenance.json')),
                       torch_fixture_sha256=sha(OUT / 'export-fixture-torch-outputs.npz'),
                       ort_cpu_fixture_sha256=sha(OUT / 'export-fixture-ort-cpu-outputs.npz'),
                       ort_cpu_numeric_report_sha256=sha(OUT / 'export-fixture-ort-cpu-numeric.json'))
        write(receipt_path, receipt)
        print('EXPORTED', sha(target), flush=True)
    except Exception:
        receipt.update(state='failed', failed_utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc())
        write(receipt_path, receipt)
        raise


if __name__ == '__main__':
    main()
