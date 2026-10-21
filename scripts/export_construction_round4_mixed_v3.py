"""Export one fixed joint-training checkpoint, preserving first Torch/ORT numerical results."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def export(step):
    import numpy as np
    import torch
    import onnxruntime as ort
    from train_construction_round4_mixed_v3 import approved, construct, OUT as TRAIN
    from evaluate_construction_round4_mixed_v3 import step_paths
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan, _, plan_hash = approved('fit')
    if plan.get('parent_approved_export_and_calibration') is not True:
        raise ValueError('Explicit reviewed three-checkpoint export/CAL approval required')
    folder, model_path = step_paths(step)
    protocol_path = folder / 'candidate-protocol-before-network.json'
    protocol = json.loads(protocol_path.read_text())
    for relative, digest in protocol['pinned_files'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Pre-export checkpoint protocol source changed:' + relative)
    if protocol['step'] != step or protocol['approved_fit_plan_sha256'] != plan_hash:
        raise ValueError('Exact snapshot/fit identity differs')
    receipt_path = folder / 'export-execution.json'
    if receipt_path.exists() or model_path.exists():
        raise ValueError('Preserve first export/partial artifacts; no silent retry')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'state': 'started', 'step': step, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_protocol_sha256': sha(protocol_path), 'approved_fit_plan_sha256': plan_hash,
        'scope': 'Trusted construction/export initialization, not footage, accuracy or capacity.'}
    write(receipt_path, receipt)
    try:
        model, _ = construct()
        checkpoint = TRAIN / f'step{step:04d}-state.pt'
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if sha(checkpoint) != protocol['checkpoint_sha256'] or len(state) != 417 or any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Actual finite fixed-step state differs')
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(model.state_dict()[key], value) for key, value in state.items()):
            raise ValueError('Exact trained state restoration differs')
        model.eval()
        head = model.model[-1]
        head.export, head.format, head.dynamic = True, 'onnx', False
        sample = torch.linspace(0, 1, 3 * 640 * 640).reshape(1, 3, 640, 640)
        # Critical: no_grad keeps cached anchors ordinary tensors for ONNX tracing.
        with torch.no_grad():
            reference = model(sample)
        if [list(x.shape) for x in reference] != [[1, 38, 8400], [1, 32, 160, 160]] or any(not torch.isfinite(x).all() for x in reference):
            raise ValueError('Exact author-class prediction/prototype contract differs')
        fixture = folder / 'export-fixture-torch-outputs.npz'
        np.savez_compressed(fixture, images=sample.numpy(), predictions=reference[0].numpy(), prototypes=reference[1].numpy())
        receipt.update(strict_trained_state_restore=True, torch_fixture_sha256=sha(fixture))
        write(receipt_path, receipt)  # Preserve first Torch result BEFORE tracing.
        torch.onnx.export(model, sample, str(model_path), opset_version=17, input_names=['images'],
            output_names=['predictions', 'prototypes'], dynamo=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})
        result_path = folder / 'export-fixture-ort-cpu-outputs.npz'
        np.savez_compressed(result_path, predictions=actual[0], prototypes=actual[1])
        errors = {'coordinate_max_abs': float(np.max(np.abs(actual[0][:, :4] - reference[0][:, :4].numpy()))),
            'class_score_max_abs': float(np.max(np.abs(actual[0][:, 4:6] - reference[0][:, 4:6].numpy()))),
            'mask_coefficient_max_abs': float(np.max(np.abs(actual[0][:, 6:] - reference[0][:, 6:].numpy()))),
            'prototype_max_abs': float(np.max(np.abs(actual[1] - reference[1].numpy()))),
            'all_finite': all(np.isfinite(x).all() for x in actual)}
        passed = bool(errors['all_finite'] and errors['coordinate_max_abs'] <= 1 and errors['class_score_max_abs'] <= .01
                      and errors['mask_coefficient_max_abs'] <= .1 and errors['prototype_max_abs'] <= .1)
        errors['all_finite'] = bool(errors['all_finite'])
        numeric_path = folder / 'export-fixture-ort-cpu-numeric.json'
        write(numeric_path, {'stats': errors, 'pass': passed, 'scope': 'Single deterministic initialization fixture, not inspection or accuracy data.'})
        if not passed:
            raise ValueError('First actual Torch/ORTCPU guard failed; stop, no CAL/retry/guard relaxation')
        metadata = {'onnx_sha256': sha(model_path), 'candidate_protocol_sha256': sha(protocol_path), 'step': step,
            'classes': ['crack', 'spalling'], 'input_size': 640, 'architecture': 'yolov8n-seg', 'decoder': 'cubit_yolo_seg',
            'source': 'https://github.com/BenyunZhao/CUBIT-InSeg', 'license': plan['license'],
            'source_state_sha256': plan['source_state_sha256'], 'trained_state_sha256': sha(checkpoint),
            'training_manifest_sha256': plan['training_manifest_sha256'], 'training_plan_sha256': plan_hash,
            'class_confidence': {'crack': .25, 'spalling': .25}, 'confidence': .25, 'nms_iou': .45,
            'exposure_policy': 'low_light_gain', 'training_configuration': plan['configuration'],
            'validation_scope': 'Pending complete122 fixed calibration and actual full-serving parity. No private-video fit or independent-site claim.'}
        write(model_path.with_suffix('.provenance.json'), metadata)
        receipt.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(), model_sha256=sha(model_path),
            metadata_sha256=sha(model_path.with_suffix('.provenance.json')), ort_cpu_fixture_sha256=sha(result_path),
            ort_cpu_numeric_report_sha256=sha(numeric_path))
        write(receipt_path, receipt)
        print('EXPORTED', step, sha(model_path), flush=True)
    except Exception:
        receipt.update(state='failed', failed_utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc())
        write(receipt_path, receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=[300, 600, 900], required=True)
    export(parser.parse_args().step)
