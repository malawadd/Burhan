"""Source-pinned joint segmentation adaptation; inert until explicit reviewed plan approval."""
from pathlib import Path
from collections import Counter
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback

for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'output/training/construction-round4-mixed-v1'
DATA = ROOT / 'datasets/construction/round4-pixel-domain/prepared-mixed-v1/training-manifest.json'
SOURCE = ROOT / 'models/construction/cubit-author-release-v1/author-state-dict.pt'
SOURCE_SHA = 'afe2ae068dd6b502ac69baa940c5da8eb7d907a037a907739451589644af2a8c'
CLASSES = ['crack', 'spalling']
CONFIG = {'steps': 900, 'checkpoints': [300, 600, 900], 'image_size': 640, 'batch': 1,
    'cpu_threads': 1, 'freeze': 4, 'optimizer': 'fresh AdamW', 'lr': .0002, 'weight_decay': .0005,
    'seed': 20261020, 'memory_format': 'contiguous', 'mask_ratio': 1,
    'source_cycle': ['mask_crack', 'mask_spalling', 'mask_spalling', 'box_crack', 'box_spalling', 'negative'],
    'augmentation': {'horizontal_flip_probability': .5, 'photo_grayscale_probability': .5},
    'actual_mask_loss': '640 real source coverage; per actual GT ROI half foreground/background-mass-balanced BCE plus half softDice, no dilation/boxfill.',
    'partial_camera_classification': 'Assigned positive GT-class anchors ONLY. Other and unassignedsameclass anchors unknown; actual bbox/DFL retained and maskloss0.',
    'calibration_execution_order': 'Completed fixed snapshots may export/CAL while immutable900-step fit continues; no feedback/retuning/earlystop. All900+all3CAL must complete before global selection.',
    'wall_time_cap': None, 'checkpoint_selection': 'All3 fixed snapshots, same.25/.45; declared calibration-only ranking, never private-video outcomes.'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pools_for(manifest):
    pools = {key: [] for key in set(CONFIG['source_cycle'])}
    for row in manifest['records']:
        if row['split'] != 'train':
            raise ValueError('This trainer may only load TRAIN records')
        if row['supervision_kind'] == 'verified_target_negative':
            pools['negative'].append(row)
        else:
            prefix = 'mask_' if row['supervision_kind'] == 'actual_masks' else 'box_'
            for category in set(row['classes']):
                pools[prefix + CLASSES[category]].append(row)
    if any(not pool for pool in pools.values()):
        raise ValueError('Every preregistered source pool must be populated')
    return pools  # Immutable prepared-manifest/source-acquisition order.


def probe_cases_for(manifest):
    pools = pools_for(manifest)
    selected = [(key, pools[key][0]) for key in ['mask_crack', 'mask_spalling', 'box_crack', 'box_spalling']]
    worst = sorted([r for r in manifest['records'] if r['supervision_kind'] == 'actual_masks'], key=lambda r: (-len(r['classes']), r['id']))[0]
    if worst['id'] not in {r['id'] for _, r in selected}:
        selected.append(('mask_worst_polygon_count', worst))
    return selected


def proposal():
    import ultralytics
    manifest = json.loads(DATA.read_text())
    pools = pools_for(manifest)
    schema = Path(ultralytics.__file__).parent / 'cfg/models/v8/yolov8-seg.yaml'
    sources = [Path(__file__), ROOT / 'scripts/construction_round4_mixed_loss.py',
        ROOT / 'scripts/prepare_construction_round4_mixed_data.py', schema,
        ROOT / 'scripts/export_construction_round4_mixed.py', ROOT / 'scripts/evaluate_construction_round4_mixed.py',
        ROOT / 'scripts/evaluate_cubit_author_candidate_v2.py',
        ROOT / 'edge_runtime/profiles/construction_model.py', ROOT / 'edge_runtime/profiles/construction_cubit.py',
        ROOT / 'edge_runtime/profiles/construction.py', ROOT / 'edge_runtime/profiles/construction_preprocessing.py',
        ROOT / 'datasets/construction/round4-pixel-domain/cubit-inseg/acceptance-v1.json',
        ROOT / 'output/evaluation/construction-cubit-author-v2/candidate-protocol-before-network.json',
        ROOT / 'output/evaluation/construction-cubit-author-v2/calibration-runtime-records.json',
        Path(ultralytics.__file__).parent / 'utils/loss.py', Path(ultralytics.__file__).parent / 'nn/tasks.py',
        Path(ultralytics.__file__).parent / 'nn/modules/head.py', Path(ultralytics.__file__).parent / 'nn/modules/block.py',
        Path(ultralytics.__file__).parent / 'nn/modules/conv.py']
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Original author tensor identity differs')
    OUT.mkdir(parents=True, exist_ok=True)
    value = {'parent_approved_probe': False, 'parent_approved_fit': False, 'parent_approved_export_and_calibration': False, 'configuration': CONFIG,
        'classes': CLASSES, 'source_state': SOURCE.relative_to(ROOT).as_posix(), 'source_state_sha256': SOURCE_SHA,
        'training_manifest_sha256': sha(DATA), 'source_pins': {str(p.resolve()): sha(p) for p in sources},
        'pool_counts': {key: len(rows) for key, rows in pools.items()},
        'probe': 'First TRAIN maskcrack/maskspalling/boxcrack/boxspalling plus maximum actualpolygoncount preparedTRAIN maskview ifdistinct; ≤5 independently source-reset backward checks, discarded weights. No CAL/private pixels.',
        'probe_cases': [{'pool': key, 'id': row['id'], 'sample_sha256': row['sample_sha256'],
                         'actual_gt_instances': len(row['classes']), 'classes': row['classes'],
                         'supervised_classes': row['supervised_classes']} for key, row in probe_cases_for(manifest)],
        'probe_maximum_peak_rss_bytes': 3 * 1024 ** 3,
        'policy': {'confidence': .25, 'nms_iou': .45, 'exposure_policy': 'low_light_gain', 'mask_rendering': 'masks'},
        'minimum_calibration_guards': {'each_class_precision': .8, 'each_class_box_recall': .5, 'geometric_thin_recall': .5,
            'verified_negative_flag_rate_max': .1, 'numeric_parity': True, 'full_serving_parity': True},
        'selection': 'Among checkpoints passing ALL guards maximize minimum(class0recall,class1recall,thinrecall), '
                     'then minimum class precision, then fewer unmatched predictions, then earlier step. '
                     'If none pass, no winner; exact900 remains named final failure diagnostic. No threshold grid.',
        'evaluation_scope': 'Unchanged complete122 CAL before any protected heldout decision; same5 development videos only after freeze. '
                            'Private4videos diagnostic-only after model freeze. No independent-site or source-checkpoint overlap assurance.',
        'license': 'CUBIT-InSeg publisher dataCCBY4; DACL dataCCBYNC4; ev-CIVIL dataCCBY4; Ultralytics codeAGPL3. '
                   'Inherited author checkpoint standalone license/training overlap unresolved. Research only.'}
    target = OUT / 'proposed-plan-before-probe.json'
    if target.exists():
        raise ValueError('Preserve prior proposal')
    write(target, value)
    print('PROPOSAL', sha(target), flush=True)


def approved(stage):
    target = OUT / 'approved-plan-before-probe-and-fit.json'
    plan = json.loads(target.read_text())
    if plan['configuration'] != CONFIG or plan['classes'] != CLASSES or plan.get('parent_approved_' + stage) is not True:
        raise ValueError('Explicit exact reviewed stage approval required')
    if sha(SOURCE) != plan['source_state_sha256'] or sha(DATA) != plan['training_manifest_sha256']:
        raise ValueError('Source state / TRAIN manifest changed')
    for path, digest in plan['source_pins'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Approved implementation/source changed: ' + path)
    manifest = json.loads(DATA.read_text())
    for row in manifest['records']:
        if sha(ROOT / row['sample']) != row['sample_sha256']:
            raise ValueError('Prepared actual TRAIN data changed')
    return plan, manifest, sha(target)


def construct():
    import torch
    import ultralytics
    import yaml
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.cfg import get_cfg
    from construction_round4_mixed_loss import MixedConstructionLoss
    torch.manual_seed(CONFIG['seed'])
    schema_path = Path(ultralytics.__file__).parent / 'cfg/models/v8/yolov8-seg.yaml'
    config = copy.deepcopy(yaml.safe_load(schema_path.read_text()))
    config.update(nc=2, scale='n')
    model = SegmentationModel(cfg=config, ch=3, nc=2, verbose=False).float()
    state = torch.load(SOURCE, map_location='cpu', weights_only=True)
    if len(state) != 417 or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Actual author source tensor contract differs')
    model.load_state_dict(state, strict=True)
    if any(not torch.equal(model.state_dict()[key], value) for key, value in state.items()):
        raise ValueError('Retained author tensor differs before training')
    model.names = {0: 'crack', 1: 'spalling'}
    model.args = get_cfg(overrides={'overlap_mask': False, 'mask_ratio': 1, 'box': 7.5, 'cls': .5, 'dfl': 1.5})
    for name, parameter in model.named_parameters():
        if any(name.startswith(f'model.{i}.') for i in range(CONFIG['freeze'])) or '.dfl.' in name:
            parameter.requires_grad_(False)
    criterion = MixedConstructionLoss(model)
    return model, criterion


def load(row, rng=None):
    import torch
    with np_load(ROOT / row['sample']) as data:
        # Match the serving NumPy float32/255 operation before optional TRAIN
        # augmentation, rather than relying on another backend's reciprocal.
        image = torch.from_numpy(data['image_rgb_u8'].astype('float32') / 255.0)
        classes = torch.from_numpy(data['cls'].copy()).float().reshape(-1, 1)
        boxes = torch.from_numpy(data['bboxes'].copy()).float()
        masks = torch.from_numpy(data['masks'].copy()).float() if 'masks' in data else None
    if rng is not None:
        if row['source_family'] in {'inseg', 'dacl', 'dacl_target_negative'} and rng.random() < .5:
            image = (.299 * image[0] + .587 * image[1] + .114 * image[2]).unsqueeze(0).expand_as(image).contiguous()
        if rng.random() < .5:
            image = image.flip(-1)
            if len(boxes):
                boxes[:, 0] = 1 - boxes[:, 0]
            if masks is not None:
                masks = masks.flip(-1)
    batch = {'img': image.unsqueeze(0), 'cls': classes, 'bboxes': boxes,
        'batch_idx': torch.zeros(len(classes)), 'supervision_kind': row['supervision_kind'],
        'supervised_classes': torch.tensor([row['supervised_classes']], dtype=torch.float32)}
    if masks is not None:
        batch.update(masks=masks, mask_origin=row['mask_origin'])
    return batch


def np_load(path):
    import numpy as np
    return np.load(path, allow_pickle=False)


def mode(model):
    model.train()
    for index in range(CONFIG['freeze']):
        model.model[index].eval()


def frozen_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()
            if any(key.startswith(f'model.{i}.') for i in range(CONFIG['freeze']))}


def probe():
    import torch
    import psutil
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan, manifest, plan_hash = approved('probe')
    target = OUT / 'train-gradient-probe.json'
    if target.exists():
        raise ValueError('Preserve first probe result/partial attempt')
    receipt = {'state': 'started', 'plan_sha256': plan_hash, 'cases': [], 'scope': 'TRAIN-only discarded gradient/numerical checks, not accuracy or inferenceFPS.'}
    write(target, receipt)
    try:
        selected = probe_cases_for(manifest)
        actual_cases = [{'pool': key, 'id': row['id'], 'sample_sha256': row['sample_sha256'],
                         'actual_gt_instances': len(row['classes']), 'classes': row['classes'],
                         'supervised_classes': row['supervised_classes']} for key, row in selected]
        if actual_cases != plan['probe_cases']:
            raise ValueError('Exact predeclared TRAIN probe source cases changed')
        for key, row in selected:
            model, criterion = construct()  # every case starts at exact same source
            before = frozen_state(model)
            mode(model)
            batch = load(row)
            started = time.perf_counter()
            predictions = model(batch['img'])
            vector, _ = criterion(predictions, batch)
            loss = vector.sum()
            score_gradient, box_gradient, proto_gradient, coefficient_gradient = torch.autograd.grad(loss,
                [predictions['scores'], predictions['boxes'], predictions['proto'], predictions['mask_coefficient']], retain_graph=True)
            loss.backward()
            unknown = batch['supervised_classes'][0] == 0
            unknown_score_grad = float(score_gradient[:, unknown].abs().max()) if unknown.any() else 0.0
            ignored = criterion.bce.last_selected_anchors == 0
            unassigned_score_grad = float(score_gradient.permute(0, 2, 1)[ignored].abs().max()) if ignored.any() else 0.0
            mask_gradient = float(proto_gradient.abs().sum() + coefficient_gradient.abs().sum())
            mask_parameter_gradient = float(sum(p.grad.abs().sum() for name, p in model.named_parameters()
                if ('.proto.' in name or '.cv4.' in name) and p.grad is not None))
            actual_box_gradient = float(box_gradient.abs().sum())
            cls_gradient = float(sum(p.grad.abs().sum() for name, p in model.named_parameters() if '.cv3.' in name and p.grad is not None))
            first_unchanged = all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
            finite = bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            expected_mask = mask_gradient > 0 if key.startswith('mask_') else mask_gradient == 0
            memory = psutil.Process().memory_info()
            peak_rss = getattr(memory, 'peak_wset', memory.rss)
            case = {'pool': key, 'record_id': row['id'], 'actual_gt_instances': len(row['classes']), 'loss': float(loss.detach()), 'all_finite': finite,
                'actual_target_assignment': criterion.last_assignment,
                'actual_mask_branch_gradient_l1': mask_gradient, 'detector_classifier_gradient_l1': cls_gradient,
                'mask_proto_cv4_parameter_gradient_l1': mask_parameter_gradient, 'actual_box_distribution_gradient_l1': actual_box_gradient,
                'unknown_class_score_gradient_max_abs': unknown_score_grad, 'first4_and_bn_unchanged': first_unchanged,
                'all_ignored_anchor_score_gradient_max_abs': unassigned_score_grad,
                'seconds_train_forward_backward': time.perf_counter() - started,
                'rss_bytes': memory.rss, 'peak_rss_bytes_process': peak_rss,
                'pass': finite and expected_mask and cls_gradient > 0 and actual_box_gradient > 0 and unknown_score_grad == 0
                        and unassigned_score_grad == 0 and first_unchanged and peak_rss <= plan['probe_maximum_peak_rss_bytes']
                        and (mask_parameter_gradient > 0 if key.startswith('mask_') else mask_parameter_gradient == 0)}
            receipt['cases'].append(case)
            write(target, receipt)
            if not case['pass']:
                raise ValueError('Actual TRAIN gradient/annotation-scope contract failed')
            del model, criterion, predictions, vector, loss, batch
        receipt.update(state='completed', all_pass=True, optimizer_updates_saved=0)
        write(target, receipt)
        print('PROBE', sha(target), flush=True)
    except Exception:
        receipt.update(state='failed', traceback=traceback.format_exc())
        write(target, receipt)
        raise


def fit():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan, manifest, plan_hash = approved('fit')
    probe_path = OUT / 'train-gradient-probe.json'
    result = json.loads(probe_path.read_text())
    if not result.get('all_pass') or sha(probe_path) != plan['approved_train_probe_sha256']:
        raise ValueError('Actual bounded TRAIN gradient probe must pass and be approved')
    receipt_path = OUT / 'fit-execution.json'
    if receipt_path.exists():
        raise ValueError('Preserve prior fixed-fit attempt')
    receipt = {'state': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': plan_hash,
               'completed_steps': 0, 'configuration': CONFIG, 'timing_scope': 'CPU training under ordinary host/project contention; not inferenceFPS.'}
    write(receipt_path, receipt)
    started = time.perf_counter()
    try:
        model, criterion = construct()
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        frozen = frozen_state(model)
        receipt.update(initial_full_state_sha256=state_digest(initial), retained_semantic_channels=CLASSES,
                       all417_source_tensors_exactly_restored=True, source_optimizer_restored=False)
        write(receipt_path, receipt)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        pools, rng = pools_for(manifest), random.Random(CONFIG['seed'])
        actual_counts = Counter()
        records_path = OUT / 'actual-step-records.jsonl'
        with records_path.open('x', encoding='utf8') as trace:
            for step in range(1, CONFIG['steps'] + 1):
                key = CONFIG['source_cycle'][(step - 1) % len(CONFIG['source_cycle'])]
                row = rng.choice(pools[key])
                batch = load(row, rng)
                mode(model)
                optimizer.zero_grad(set_to_none=True)
                lr = CONFIG['lr'] * (.1 + .9 * (1 + math.cos(math.pi * (step - 1) / CONFIG['steps'])) / 2)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                output = model(batch['img'])
                losses, _ = criterion(output, batch)
                loss = losses.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite actual TRAIN loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite actual TRAIN gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                if any(not torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()):
                    raise ValueError('Frozen early source/normalization state changed')
                actual_counts[key] += 1
                trace.write(json.dumps({'step': step, 'pool': key, 'record': row['id'], 'loss': float(loss.detach()),
                    'loss_vector': losses.detach().tolist(), 'lr': lr, 'actual_target_assignment': criterion.last_assignment}) + '\n')
                trace.flush()
                if step in CONFIG['checkpoints']:
                    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    if any(not torch.isfinite(value).all() for value in state.values()):
                        raise ValueError('Nonfinite actual checkpoint state')
                    target = OUT / f'step{step:04d}-state.pt'
                    torch.save(state, target)
                    torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'rng_state': rng.getstate(),
                                'torch_rng_state': torch.get_rng_state()}, OUT / f'step{step:04d}-optimizer.pt')
                    write(target.with_suffix('.json'), {'step': step, 'sha256': sha(target), 'state_digest': state_digest(state),
                        'classes': CLASSES, 'plan_sha256': plan_hash, 'source_counts': dict(actual_counts),
                        'fixed_checkpoint_not_selected_by_accuracy': True})
                if step % 25 == 0:
                    receipt.update(completed_steps=step, actual_source_counts=dict(actual_counts), elapsed_seconds=time.perf_counter() - started)
                    write(receipt_path, receipt)
                    print('STEP', step, 'LOSS', float(loss.detach()), 'ELAPSED', round(time.perf_counter() - started, 2), flush=True)
        receipt.update(state='completed', completed_steps=CONFIG['steps'], completed_utc=datetime.now(timezone.utc).isoformat(),
                       actual_source_counts=dict(actual_counts), elapsed_seconds=time.perf_counter() - started,
                       step_trace_sha256=sha(records_path), first4_state_unchanged=True)
        write(receipt_path, receipt)
    except Exception:
        receipt.update(state='failed', traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        write(receipt_path, receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['proposal', 'probe', 'fit'], required=True)
    args = parser.parse_args()
    {'proposal': proposal, 'probe': probe, 'fit': fit}[args.stage]()
