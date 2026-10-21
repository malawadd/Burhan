"""Versioned exact300 restart after v1 metadata-path failure; no weights-only resume."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INPUT_OUT = ROOT / 'output/optimization/construction-pixel-supervision-v1'
OUT = ROOT / 'output/optimization/construction-pixel-supervision-v2'
MODELS = ROOT / 'models/construction/pixel-supervision-v2'
sys.path.insert(0, str(ROOT))
from probe_construction_pixel_supervision import record_protocol, masked_loss, sha, SEED, LIMIT
from construction_pixel_source_paths import source_pin


def proposal():
    path = INPUT_OUT / 'protocol-before-probe-and-fit.json'
    if sha(path) != '1f893e2bdcf025dbb5513a680a72b052729ea66405880900549f76afd24e32e4':
        raise ValueError('Immutable reviewed proposal changed')
    _, protocol, prepared = record_protocol()
    probe_path = INPUT_OUT / 'train-gradient-memory-probe.json'
    if sha(probe_path) != '416a65b137830f73ae058f06694dee0aa4ad2b26071ce05d23533b87f15206e8':
        raise ValueError('Reviewed bounded TRAIN probe changed')
    probe = json.loads(probe_path.read_text())
    if not probe['all_checks_passed'] or probe['protocol_sha256'] != sha(path):
        raise ValueError('Required finite/gradient/memory probe failed')
    return path, protocol, prepared


def canonical_calibration_rows(rows):
    result = [{'source': r['source'], 'image': str(Path(r['image']).relative_to(ROOT)),
               'frame_index': r['frame_index'], 'negative': bool(r['negative']), 'expected': r['expected']} for r in rows]
    counts = {'images': len(result), 'negative_images': sum(r['negative'] for r in result),
              'positive_images': sum(not r['negative'] for r in result),
              'crack_boxes': sum(e['label'] == 'crack' for r in result for e in r['expected']),
              'spall_boxes': sum(e['label'] == 'spalling' for r in result for e in r['expected'])}
    if counts != {'images': 122, 'negative_images': 10, 'positive_images': 112, 'crack_boxes': 125, 'spall_boxes': 38}:
        raise ValueError('Protected122-frame/125-crack/38-spall publisher GT scope changed')
    labels = [e for r in result for e in r['expected']]
    if len({e['id'] for e in labels}) != 163 or any(e['label'] not in {'crack', 'spalling'} for e in labels) or any(bool(r['expected']) == r['negative'] for r in result):
        raise ValueError('Annotation IDs, class scope or positive/negative identities are inconsistent')
    return {'counts': counts, 'ordered_records': result,
            'sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}


def calibration_ground_truth():
    """Canonical source metadata only; never decode a calibration image."""
    base = ROOT / 'datasets/construction/evcivil-round3'
    prior = json.loads((ROOT / 'output/evaluation/construction-round3/litecrackseg-calibration/preregistered-protocol.json').read_text())
    pins = {name: sha(base / name) for name in ['exposures.json', 'background-import.json']}
    if pins['exposures.json'] != prior['source_manifest_sha256'] or pins['background-import.json'] != prior['background_import_sha256']:
        raise ValueError('Publisher GT/background annotation bytes differ from protected receipt')
    rows = []
    for r in json.loads((base / 'exposures.json').read_text()):
        if r['split'] == 'calibration':
            rows.append({'source': r['global_defect_id'], 'image': ROOT / r['image'], 'frame_index': r['original_frame_index'],
                         'negative': False, 'expected': [{'id': f"{r['global_defect_id']}:{r['original_frame_index']}:{i}", **b} for i, b in enumerate(r['boxes'])]})
    for r in json.loads((base / 'background-import.json').read_text())['records']:
        if r['split'] == 'calibration':
            rows.append({'source': r['source_name'], 'image': ROOT / r['derived_image'], 'frame_index': 0,
                         'negative': True, 'expected': []})
    return {**canonical_calibration_rows(rows), 'annotation_json_pins': pins}


def record_execution():
    path, protocol, prepared = proposal()
    scripts = ['scripts/train_construction_pixel_supervision_v2.py',
               'scripts/evaluate_construction_pixel_supervision_v2.py',
               'scripts/probe_construction_pixel_supervision.py', 'scripts/construction_pixel_source_paths.py',
               'scripts/export_litecrackseg_contingency.py',
               'edge_runtime/profiles/construction_lite.py',
               'edge_runtime/profiles/construction_preprocessing.py',
               'scripts/evaluate_construction_round3.py',
               'scripts/evaluate_construction_defect_video.py']
    failed_archive = INPUT_OUT / 'failed100-executed-source-snapshot/snapshot-manifest.json'
    if sha(failed_archive) != '116f8292dcc70cb1853ccd76930f2b71f52493a4e395ce419f9c48b33cf9e414':
        raise ValueError('Failed100 history must remain preserved before a new exact restart')
    source_digest = source_pin(protocol['source_pins'], 'models/construction/thin-contingency/litecrackseg-TUT.pth')
    if source_digest != 'c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497':
        raise ValueError('Export provenance source pin is unresolved before optimizer')
    value = {'schema_version': 2, 'proposal_sha256': sha(path),
             'resolved_source_checkpoint_sha256': source_digest,
             'restart_lineage': {'failed100_archive_sha256': sha(failed_archive), 'prior_failed_execution_receipt_sha256': sha(INPUT_OUT / 'training-execution-receipt-before-optimizer.json'), 'initialization': 'ORIGINAL TUT checkpoint, original seed and fresh AdamW; never resume failed100 weights', 'schedule': 'New exact300 steps/save100200300; identical141 inputs/loss/layers/sampling; no CAL or102 outcomes before restart', 'metadata_fix': 'Resolve relative source pins independent of Windows/POSIX separators BEFORE optimizer'},
             'prepared_manifest_sha256': protocol['prepared_manifest_sha256'],
             'bounded_probe_sha256': sha(INPUT_OUT / 'train-gradient-memory-probe.json'),
             'source_pins': protocol['source_pins'], 'implementation_pins': {p: sha(ROOT / p) for p in scripts},
             'steps': 300, 'save_at': [100, 200, 300], 'batch_size': 1, 'cpu_threads': 1,
             'seed': SEED, 'exposure_policy': 'low_light_gain',
             'loss_scope': protocol['loss'], 'calibration_protocol': protocol['calibration'],
             'calibration_ground_truth_identity': calibration_ground_truth(),
             'execution_guard': 'Recorded before first optimizer. --fit requires explicit launch; no fit is implied by this receipt. Current model/data/source proposal remain unchanged.',
             'checkpoints': {str(s): str(MODELS / f'crack-pixel-step{s}.onnx') for s in [100, 200, 300]},
             'train_role_counts': {r: sum(row['role'] == r for row in prepared['records']) for r in ['lcsd_positive', 'dacl_positive', 'dacl_crack_negative']}}
    receipt = OUT / 'training-execution-receipt-before-optimizer.json'
    if receipt.exists() and json.loads(receipt.read_text()) != value:
        raise ValueError('Training implementation differs from protected execution receipt')
    if not receipt.exists():
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(value, indent=2), encoding='utf-8')
    return receipt, value, protocol, prepared


def validate_execution():
    receipt = OUT / 'training-execution-receipt-before-optimizer.json'
    if not receipt.exists():
        raise ValueError('Record additive reviewed execution receipt before any optimizer')
    original = json.loads(receipt.read_text())
    result = record_execution()
    if original != result[1]:
        raise ValueError('Execution receipt changed')
    return result


def batch(prepared, rng, step):
    import numpy as np
    import torch
    role = ['lcsd_positive', 'dacl_positive', 'dacl_crack_negative'][(step - 1) % 3]
    pool = [r for r in prepared['records'] if r['role'] == role]
    row = pool[int(rng.integers(len(pool)))]
    path = ROOT / row['prepared']
    if sha(path) != row['prepared_sha256']:
        raise ValueError('Prepared TRAIN pair changed')
    with np.load(path) as data:
        image = torch.from_numpy(np.ascontiguousarray(data['bgr'].transpose(2, 0, 1)[None]))
        truth = torch.from_numpy(data['target'][None, None].copy()).float()
        valid = torch.from_numpy(data['valid'][None, None].copy()).float()
    return image, truth, valid, row


def state_digest(values):
    digest = hashlib.sha256()
    for name, tensor in values.items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit(contention_note):
    receipt, execution, protocol, prepared = validate_execution()
    import cv2
    import numpy as np
    import psutil
    import torch
    from export_litecrackseg_contingency import reviewed_model, Export
    if (OUT / 'training-started.json').exists() or any(MODELS.glob('crack-pixel-step*')):
        raise ValueError('Refusing to restart/extend or overwrite this fixed experiment')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    for row in prepared['records']:
        for field in ['image', 'annotation']:
            if sha(ROOT / row[field]) != row[field + '_sha256']:
                raise ValueError('Raw TRAIN source changed before fit')
        if sha(ROOT / row['prepared']) != row['prepared_sha256']:
            raise ValueError('A prepared TRAIN source changed before fit')
    started_note = {'started_utc': datetime.now(timezone.utc).isoformat(), 'execution_receipt_sha256': sha(receipt),
                    'actual_workload_note': contention_note, 'fit_launch': 'Explicit --fit invocation',
                    'scope': 'TRAIN optimizer only; no inference capacity claim',
                    'torch_version': str(torch.__version__), 'opencv_version': cv2.__version__, 'python_version': sys.version,
                    'environment_threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}}
    (OUT / 'training-started.json').write_text(json.dumps(started_note, indent=2), encoding='utf-8')
    model, _, _ = reviewed_model()
    model.eval()
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith('encoder.'))
    encoder = state_digest(model.encoder.state_dict())
    bn = {n: t.clone() for n, t in model.named_buffers() if n.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.0001, weight_decay=.0001)
    process = psutil.Process(); peak = 0; history = []; checkpoints = []; start = time.perf_counter()
    MODELS.mkdir(parents=True, exist_ok=True)
    for step in range(1, 301):
        peak = max(peak, process.memory_info().rss)
        if peak > LIMIT:
            raise MemoryError('Fixed TRAIN continuation exceeded6GiB RSS; no extension/restart')
        image, target, valid, row = batch(prepared, rng, step)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image[:, [2, 1, 0]].float())[0]
        loss = masked_loss(logits, target, valid)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite TRAIN logits/loss')
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError('Nonfinite/missing TRAIN gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid gradient norm')
        optimizer.step()
        if any(p.grad is not None for p in model.encoder.parameters()) or not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
            raise ValueError('Nonfinite state or frozen encoder gradient after update')
        if state_digest(model.encoder.state_dict()) != encoder or not all(torch.equal(t, dict(model.named_buffers())[n]) for n, t in bn.items()):
            raise ValueError('Frozen encoder/BN state changed')
        peak = max(peak, process.memory_info().rss)
        if peak > LIMIT:
            raise MemoryError('Fixed TRAIN continuation exceeded6GiB RSS')
        history.append({'step': step, 'source': row['name'], 'role': row['role'], 'loss': float(loss.detach()),
                        'grad_norm_before_clip': norm, 'elapsed_seconds': time.perf_counter() - start, 'rss_bytes': process.memory_info().rss})
        if step % 10 == 0:
            (OUT / 'training-progress.json').write_text(json.dumps({'execution_receipt_sha256': sha(receipt), 'history': history}, indent=2), encoding='utf-8')
            print(json.dumps(history[-1]), flush=True)
        del logits, loss, image, target, valid, gradients
        if step in [100, 200, 300]:
            path = MODELS / f'crack-pixel-step{step}.onnx'
            torch.save(model.state_dict(), path.with_suffix('.pt'))
            with torch.inference_mode():
                torch.onnx.export(Export(model).eval(), torch.zeros((1, 3, 384, 384), dtype=torch.uint8), str(path),
                                  input_names=['bgr_uint8'], output_names=['crack_probability'], opset_version=17, dynamo=False, do_constant_folding=True)
            import onnx
            onnx.checker.check_model(onnx.load(path))
            reference_row = prepared['records'][0]  # Fixed TRAIN source, never chosen from outcomes.
            if sha(ROOT / reference_row['prepared']) != reference_row['prepared_sha256']:
                raise ValueError('Fixed TRAIN export reference input changed')
            with np.load(ROOT / reference_row['prepared']) as content:
                reference_bgr = content['bgr'].copy()
            with torch.inference_mode():
                reference_tensor = torch.from_numpy(np.ascontiguousarray(reference_bgr.transpose(2, 0, 1)[None]))
                reference_probability = Export(model)(reference_tensor).numpy()
                zero_probability = Export(model)(torch.zeros_like(reference_tensor)).numpy()
            reference_path = path.with_suffix('.train-parity.npz')
            np.savez_compressed(reference_path, bgr=reference_bgr, probability=reference_probability, zero_probability=zero_probability)
            meta = {'file': str(path.relative_to(ROOT / 'models/construction')), 'sha256': sha(path),
                    'source': 'https://github.com/1Kaleb/LiteCrackSeg/tree/082d33111f1a3b9ac1d78e984b06fd3530bf3533',
                    'source_checkpoint_sha256': execution['resolved_source_checkpoint_sha256'],
                    'checkpoint_sha256': sha(path.with_suffix('.pt')), 'classes': ['crack'], 'input_size': 384,
                    'input_shape': [1, 3, 384, 384], 'output_shape': [1, 1, 384, 384],
                    'input': 'uint8BGR -> RGBfloat0..255 inside graph; no /255/ImageNetNormalize',
                    'output': 'One uncalibrated sigmoid crack score map; no spall support',
                    'step': step, 'exposure_policy': 'low_light_gain', 'prepared_manifest_sha256': protocol['prepared_manifest_sha256'],
                    'proposal_sha256': sha(INPUT_OUT / 'protocol-before-probe-and-fit.json'), 'execution_receipt_sha256': sha(receipt),
                    'train_parity_reference': str(reference_path.relative_to(ROOT)), 'train_parity_reference_sha256': sha(reference_path),
                    'train_parity_source': reference_row,
                    'license': protocol['rights'], 'experimental': True,
                    'validation_scope': 'TRAIN-only pixel-supervised continuation; no CAL/heldout input in fitting. CAL thresholds still required. LCSD author test repurposed TRAIN, no LCSD accuracy claim.'}
            path.with_suffix('.provenance.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
            checkpoints.append({'step': step, 'model': str(path), 'sha256': sha(path), 'metadata_sha256': sha(path.with_suffix('.provenance.json')), 'checkpoint_sha256': sha(path.with_suffix('.pt'))})
    report = {'execution_receipt_sha256': sha(receipt), 'proposal_sha256': sha(INPUT_OUT / 'protocol-before-probe-and-fit.json'),
              'history': history, 'checkpoints': checkpoints, 'fixed_steps_completed': 300, 'peak_rss_bytes': peak,
              'elapsed_seconds': time.perf_counter() - start, 'encoder_state_unchanged': True, 'batchnorm_buffers_unchanged': True,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
              'calibration_images_opened': 0, 'heldout_images_opened': 0, 'reserved_lcsd102_pixels_opened': 0,
              'actual_workload_note': contention_note, 'scope': 'Checked TRAIN optimizer/export walltime, not inferenceFPS or accuracy'}
    (OUT / 'training-complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'report_sha256': sha(OUT / 'training-complete.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--record-execution', action='store_true')
    group.add_argument('--fit', action='store_true')
    parser.add_argument('--contention-note', default='Actual launch workload must be supplied by coordinator')
    args = parser.parse_args()
    if args.record_execution:
        path, _, _, _ = record_execution()
        print(json.dumps({'receipt': str(path), 'sha256': sha(path), 'fit_started': False}, indent=2))
    else:
        if args.contention_note == 'Actual launch workload must be supplied by coordinator':
            parser.error('--fit requires a truthful --contention-note; authorization is a separate coordinator decision')
        fit(args.contention_note)
