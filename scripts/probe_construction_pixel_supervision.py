"""Record a protected proposal, then optionally probe TRAIN gradients only. No fit/GPU mode."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/construction-pixel-supervision-v1'
sys.path.insert(0, str(ROOT))
SEED = 20261028
LIMIT = 6 * 1024 ** 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_protocol():
    prepared = OUT / 'prepared-train-manifest.json'
    data = json.loads(prepared.read_text())
    if len(data['records']) != 141:
        raise ValueError('Expected141 unique source-pinned TRAIN pairs')
    if data['preparation_source_sha256'] != sha(ROOT / 'scripts/prepare_construction_pixel_supervision.py') or data['shared_preprocessing_sha256'] != sha(ROOT / 'edge_runtime/profiles/construction_preprocessing.py'):
        raise ValueError('Prepared tensors do not match the current pinned preparation/gain implementation')
    source = ROOT / 'models/construction/thin-contingency'
    pinned = {
        source / 'litecrackseg-TUT.pth': 'c12fb1ce991e479a32e5a26d7648ce9d20a440b9af4fd0411d321aa6a2eed497',
        source / 'lite-source/model/LiteCrackSeg.py': 'b139c4432b8728b84519998ef8b6368fe2e757096cd147b5e88df74e64ce072d',
        source / 'lite-source/model/mobilevit-xx-small/config.json': '6f1b7c93da05058a62efcd10266079bbf1e6b430babd9367c8e20fdb21726f22',
    }
    for path, expected in pinned.items():
        if sha(path) != expected:
            raise ValueError('Original source model changed')
    helper_paths = ['scripts/probe_construction_pixel_supervision.py', 'scripts/prepare_construction_pixel_supervision.py',
                    'scripts/export_litecrackseg_contingency.py', 'edge_runtime/profiles/construction_preprocessing.py',
                    'edge_runtime/profiles/construction_lite.py']
    prior_cal = ROOT / 'output/evaluation/construction-round3/litecrackseg-calibration/preregistered-protocol.json'
    if sha(prior_cal) != 'e740b80e9993236528fd040db1ab49bb04b8ea4eb2d47ebf06548f5dc89007ba':
        raise ValueError('Existing source-only CAL identity receipt changed')
    prior = json.loads(prior_cal.read_text())
    protocol = {
        'name': 'construction-crack-pixel-supervision-v1', 'seed': SEED,
        'status': 'Protected proposal and authorized bounded TRAIN probe only;300-step fit requires separate parent authorization',
        'fixed_fit_steps_if_authorized': 300, 'save_at': [100, 200, 300], 'fit_batch_size': 1,
        'cpu_threads': 1, 'optimizer': {'name': 'AdamW', 'lr': .0001, 'weight_decay': .0001, 'clip_global_grad_norm': 5},
        'trainable': 'All existing non-encoder parameters: bottleneck,btl_adapter,adapters,ECA,up1..5,CABlock1..5,fuse1..5,final. Original MobileViT encoder frozen for entire bounded proposal.',
        'mode': 'Whole model eval with autograd enabled: frozen BN running stats and disabled DropPath/dropout; explicit low-data continuation, not author training recipe',
        'head': 'Original single crack channel unchanged; source742-tensor strict loading; no spall head',
        'loss': 'Per-image valid-pixel BCEWithLogits(pos_weight2) +.5*softDice(smooth1) for GT-positive images only; mean batch. Unknown/255 pixels excluded from BOTH terms. Exactly main raw logit; auxiliary logits are NOT added, matching VERIFIED author total_loss=output_loss. Loss formula itself is this separate experiment, not the author Tversky recipe.',
        'sampling': 'Fixed cyclic LCSD-positive,DACL-positive,DACL-crack-negative roles; uniform source within role via fixed NumPy seed. All141 prepared sources are TRAIN only. No model mining, CAL feedback, crop, geometric augmentation or validation schedule feedback.',
        'preprocessing': 'Original full-native BGR -> shared low_light_gain EXACTLY ONCE -> INTER_LINEAR384square ->uint8BGR tensor. Loss path converts RGB float0..255 exactly like source export; no /255/ImageNet normalization. GT/valid INTER_NEAREST384. Prepared arrays already gained: trainer MUST NOT gain again.',
        'mask_scope': 'LCSD explicitmask==2,255ignore; DACL complete publisher filled Crack/ACrack polygons equality-checked against pinned samples metadata. Other damage not healthy building. No LCSD spall-negative assumption or ev-CIVIL box-filled semantic targets.',
        'prepared_manifest_sha256': sha(prepared), 'source_pins': {str(p.relative_to(ROOT)): digest for p, digest in pinned.items()},
        'implementation_pins': {p: sha(ROOT / p) for p in helper_paths},
        'calibration': {'partition': 'UNCHANGED original construction122 CAL identities, real raw observations',
                        'source_identity_receipt_sha256': sha(prior_cal),
                        'ordered_input_metadata': prior['ordered_image_hashes'],
                        'read_scope': 'Prior protocol metadata only; no CAL/heldout image pixels decoded during preparation/probe',
                        'exposure_policy': 'low_light_gain exactly once from raw native observations',
                        'threshold_grid': [.10, .15, .20, .25, .35, .45, .55, .65, .75, .85],
                        'decoder': 'Existing strictcutoff/bilinearnativesize/8connected/area4/top100meanscore, no closing/grouping/boxinflation',
                        'primary': 'One-to-one publisher-box IoU.5; crack-only annotated scope and true target negatives; no pixel-GT assertion for ev-CIVIL',
                        'guards': {'crack_precision_min': .8, 'negative_image_fpr_max': .1},
                        'per_checkpoint_rank': ['crack_box_recall', 'crack_box_precision', 'higher_threshold'],
                        'global_rank': ['crack_box_recall', 'crack_box_precision', 'lower_negative_image_fpr', 'earlier_checkpoint'],
                        'all_checkpoints_required': [100, 200, 300],
                        'implementation_status': 'Future all-three CAL/freezer enforcement is DECLARED only; not implemented by this probe. No full-fit mode exists.',
                        'no_eligible_point': 'REJECT; preserve failed diagnostic points only, never promote or invent a passing cutoff',
                        'pixel_scope': 'Any TRAIN semantic traces supplementary; unchanged ev-CIVIL primary box proxy cannot be replaced after outcomes'},
        'probe': {'batches': [1, 2], 'threads': [1], 'gradient_steps_per_trial': 3, 'max_gradient_steps_per_trial': 4,
                  'reset': 'Reload original TUT checkpoint and reset all seeds before each trial; discard optimizer/model after trial, no weights saved',
                  'source_sigmoid_parity': 'Fixed first realTRAIN batch: sigmoid(main raw logits) vs actual existing Export wrapper same source weights, atol1e-6/rtol1e-6',
                  'rss_limit_bytes': LIMIT, 'memory_measurement': 'Own process RSS sampled50ms plus stage guards; fail immediately at next guard if limit exceeded',
                  'capacity_scope': 'Contended CPU TRAIN checked optimizer throughput including finite/state guards, NOT inferenceFPS/movingvideo qualification',
                  'state_guards': 'All frozen encoder parameters+buffers SHA identical and all BN buffers equal after each trial; all model state tensors finite after EACH optimizer step; no frozen encoder gradients'},
        'future_export': 'If fit authorized, same existing single-output uint8BGR sigmoid384graph, immutable new artifact/provenance, CPU/export/GPU finite+parity and exact serving decoding before unchangedCAL; no overwrite of TUT baseline',
        'evaluation_reservation': 'LCSD102welllit officialTRAIN proposed as separately reserved OURpixeltest ONLY after preopening leakage/encoding protocol; zero such pixels downloaded or used here; no author test-accuracy claim',
        'rights': 'LiteCrackSeg whole-model license undeclared; experimental only. DACL CC BY-NC4.0; LCSD author-linked Apache2.0 declaration with source-lineage caveat.',
    }
    path = OUT / 'protocol-before-probe-and-fit.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protected proposal/probe protocol changed')
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    return path, protocol, data


def masked_loss(logits, target, valid):
    import torch
    import torch.nn.functional as F
    if logits.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Exact1channel input-resolution logits/target/valid shape required')
    counts = valid.sum((1, 2, 3))
    if bool((counts == 0).any()):
        raise ValueError('Empty valid annotation channel')
    bce = (F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(2.), reduction='none') * valid).sum((1, 2, 3)) / counts
    probability = torch.sigmoid(logits) * valid
    truth = target * valid
    positive = truth.sum((1, 2, 3)) > 0
    dice = 1 - (2 * (probability * truth).sum((1, 2, 3)) + 1) / (probability.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
    return (bce + .5 * dice * positive).mean()


def probe(protocol_path, protocol, data):
    import cv2
    import numpy as np
    import psutil
    import torch
    from export_litecrackseg_contingency import reviewed_model, Export
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def watch():
        while not stop.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    def guard():
        peak[0] = max(peak[0], process.memory_info().rss)
        if peak[0] > LIMIT:
            raise MemoryError('TRAIN probe exceeded6GiB process RSS; no fit authorized')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(protocol_path),
              'execution_scope': 'Concurrent construction CPU4 fit and ordinary host services; curvature fit/CAL jobs have completed. CPU1 checked TRAIN optimizer only, no inferenceFPS claim',
              'trials': [], 'full_fit_started': False, 'gpu_inference_started': False, 'probe_weights_saved': False,
              'environment_threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}}
    roles = ['lcsd_positive', 'dacl_positive', 'dacl_crack_negative']
    pools = {role: [r for r in data['records'] if r['role'] == role] for role in roles}
    if any(not rows for rows in pools.values()):
        raise ValueError('Every real positive/negative source cohort required')
    try:
        for batch_size in protocol['probe']['batches']:
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
            rng = np.random.default_rng(SEED)
            model, _, _ = reviewed_model()
            model.eval()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(not name.startswith('encoder.'))
            def encoder_digest():
                digest = hashlib.sha256()
                for name, tensor in model.encoder.state_dict().items():
                    digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
                return digest.hexdigest()
            original_encoder = encoder_digest()
            original_bn = {name: tensor.clone() for name, tensor in model.named_buffers()
                           if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=.0001, weight_decay=.0001)
            trial = {'batch_size': batch_size, 'cpu_threads': 1, 'steps': [],
                     'trainable_parameters': sum(p.numel() for p in parameters),
                     'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
                     'source_reloaded_before_trial': True}
            for step in range(3):
                chosen = [pools[roles[(step * batch_size + i) % 3]][int(rng.integers(len(pools[roles[(step * batch_size + i) % 3]])))] for i in range(batch_size)]
                arrays = []
                for row in chosen:
                    path = ROOT / row['prepared']
                    if sha(path) != row['prepared_sha256']:
                        raise ValueError('Prepared TRAIN tensor bytes changed')
                    with np.load(path) as content:
                        arrays.append({k: content[k].copy() for k in ['bgr', 'target', 'valid']})
                bgr = torch.from_numpy(np.ascontiguousarray(np.stack([a['bgr'].transpose(2, 0, 1) for a in arrays])))
                target = torch.from_numpy(np.stack([a['target'][None] for a in arrays])).float()
                valid = torch.from_numpy(np.stack([a['valid'][None] for a in arrays])).float()
                rgb = bgr[:, [2, 1, 0]].float()
                optimizer.zero_grad(set_to_none=True)
                guard(); started = time.perf_counter()
                logits = model(rgb)[0]
                if step == 0:
                    with torch.no_grad():
                        actual_source = Export(model)(bgr)
                        expected = torch.sigmoid(logits.detach())
                    error = float((actual_source - expected).abs().max())
                    trial['source_sigmoid_raw_logit_max_error'] = error
                    if not torch.allclose(actual_source, expected, atol=1e-6, rtol=1e-6):
                        raise ValueError('Selected raw-logit path differs from source sigmoid export')
                    del actual_source, expected
                    # Exclude this explicit parity forward from optimizer step timing.
                    started = time.perf_counter()
                    trial['first_step_timing_scope'] = 'Backward+optimizer only; initial forward excluded because source parity interrupted timing'
                loss = masked_loss(logits, target, valid)
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite real TRAIN logits/loss')
                guard(); loss.backward(); guard()
                grads = [p.grad for p in parameters if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise ValueError('Missing/nonfinite real TRAIN gradient')
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5))
                if norm <= 0 or not np.isfinite(norm):
                    raise ValueError('Zero/nonfinite total gradient')
                optimizer.step(); guard()
                if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                    raise ValueError('Nonfinite model state after optimizer update')
                if any(p.grad is not None for p in model.encoder.parameters()):
                    raise ValueError('Frozen encoder received a gradient')
                trial['steps'].append({'step': step + 1, 'sources': [r['name'] for r in chosen],
                                       'roles': [r['role'] for r in chosen], 'loss': float(loss.detach()),
                                       'grad_norm_before_clip': norm, 'nonzero_grad_tensors': sum(bool(g.count_nonzero()) for g in grads),
                                       'timed_seconds': time.perf_counter() - started, 'rss_bytes': process.memory_info().rss,
                                       'logit_range': [float(logits.detach().min()), float(logits.detach().max())]})
                del logits, loss, grads, bgr, target, valid, rgb
            # Only complete step2/3 forward+backward+optimizer timings characterize TRAIN capacity.
            measured = [s['timed_seconds'] for s in trial['steps'][1:]]
            trial['complete_optimizer_step_seconds'] = measured
            trial['contended_train_images_per_second'] = batch_size * len(measured) / sum(measured)
            trial['encoder_state_unchanged'] = encoder_digest() == original_encoder
            trial['batchnorm_buffers_unchanged'] = all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in original_bn.items())
            if not trial['encoder_state_unchanged'] or not trial['batchnorm_buffers_unchanged']:
                raise ValueError('Protected encoder/BN state changed during decoder-only probe')
            report['trials'].append(trial)
            del model, optimizer, parameters
            gc.collect()
        report['all_checks_passed'] = True
    except Exception as error:
        report['all_checks_passed'] = False
        report['failure'] = repr(error)
        raise
    finally:
        stop.set(); watcher.join(timeout=1)
        report['peak_rss_bytes'] = peak[0]
        report['rss_limit_bytes'] = LIMIT
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        (OUT / 'train-gradient-memory-probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-protocol', action='store_true')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.record_protocol == args.probe:
        parser.error('Choose only --record-protocol or --probe; no full-fit mode exists')
    if args.probe and (OUT / 'train-gradient-memory-probe.json').exists():
        raise ValueError('Refusing to replace previous bounded probe')
    path, protocol, data = record_protocol()
    if args.record_protocol:
        print(json.dumps({'protocol_path': str(path), 'sha256': sha(path)}, indent=2))
    else:
        probe(path, protocol, data)


if __name__ == '__main__':
    main()
