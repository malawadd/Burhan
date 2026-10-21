"""One fixed thin-structure supervised experiment. All NN needs root release.

Topology is a TRAIN-only auxiliary loss. Serving remains the exact retained
PackedPixelRefiner + MaxPool2d(2), with no skeleton/GT input at inspection.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from factory_round4_supervised import SourceCache, crop_observed, augment, sources_unchanged, sha, read, write, INIT_WEIGHTS, HARD
from factory_stfpm_common import CAL_SHA, rank

OUT = ROOT / 'output/optimization/factory-round5-cldice'
NAME = 'supervised_round5_thin_cldice'
PROPOSAL = OUT / 'proposed-protocol-before-fit.json'
PROTOCOL = OUT / 'protocol-before-fit.json'
DECISION = OUT / 'parent-approved-fit-decision.json'
FIXTURE = OUT / 'TRAIN-only-preparation-tests.json'
R4 = ROOT / 'output/optimization/factory-round4-supervised'
SOURCE = ROOT / 'models/factory/supervised_round4_real_coverage-step900.pt'
SOURCE_SHA = '4b39937f9719ba39c2fb256f29c5cec4e45be2474e312c1b577aec6a58a818d8'
SEED = 20261101
STEPS = 600
SAVE_AT = [200, 400, 600]
CAL_REFERENCE = ROOT / 'output/optimization/factory-round3l9-learned-partitioned/checkpoint200-calibration.json'


def cal_identity(value):
    """Exact historical7b09 serializer: default JSON separators include spaces."""
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protected_cal_rows(split):
    """File identity/ordered metadata only; never decode CAL pixels for fitting."""
    rows = read(CAL_REFERENCE)['calibration_source_identity']['ordered_sources']
    if len(rows) != 80 or sum(row['defective'] for row in rows) != 10 or cal_identity(rows) != CAL_SHA:
        raise ValueError('Exact protected historical CAL80 metadata required')
    base = ROOT / 'datasets/factory/KolektorSDD'
    paths = [image for item in split for image in sorted((base / item).glob('*.jpg'))]
    if [str(image.relative_to(base)) for image in paths] != [row['image'] for row in rows]:
        raise ValueError('Protected CAL order/paths changed')
    for image, row in zip(paths, rows):
        if sha(image) != row['image_sha256'] or sha(image.with_name(image.stem + '_label.bmp')) != row['mask_sha256']:
            raise ValueError('Protected CAL source bytes/implicit native geometry changed')
    return rows


def soft_skeleton(image, iterations=5):
    """Author clDice 2D cross erosion/opening, fixed five iterations."""
    import torch
    from torch.nn import functional as F
    def erode(x):
        return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))
    def opening(x):
        return F.max_pool2d(erode(x), 3, 1, 1)
    skeleton = F.relu(image - opening(image))
    for _ in range(iterations):
        image = erode(image)
        delta = F.relu(image - opening(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def topology_loss(logits, target, valid):
    import torch
    from torch.nn import functional as F
    # Five erosion iterations plus one opening have radius at most seven.
    # Explicit zero outside crop/observed support, then an eight-pixel margin.
    observed = (valid > .5).to(logits.dtype)
    inside = (-F.max_pool2d(-F.pad(observed, (8, 8, 8, 8), value=0), 17, 1) > .5).to(logits.dtype)
    truth = target * observed
    with torch.no_grad():
        true_skeleton = soft_skeleton(truth) * inside
    if not bool(true_skeleton.sum() > 0):
        return logits.sum() * 0
    probability = logits.sigmoid() * observed
    predicted_skeleton = soft_skeleton(probability) * inside
    precision = ((predicted_skeleton * truth).sum() + 1) / (predicted_skeleton.sum() + 1)
    sensitivity = ((true_skeleton * probability).sum() + 1) / (true_skeleton.sum() + 1)
    return 1 - 2 * precision * sensitivity / (precision + sensitivity)


def loss(logits, target, valid):
    import torch
    from torch.nn import functional as F
    if (logits.shape != target.shape or target.shape != valid.shape or not torch.isfinite(logits).all()
        or not torch.isfinite(target).all() or bool((target < 0).any()) or bool((target > 1).any())):
        raise ValueError('Finite paired logits/actual0..1 TRAIN target required')
    observed = valid > .5
    positive, negative = (target > .5) & observed, (target <= .5) & observed
    if not observed.any() or logits.shape != target.shape or target.shape != valid.shape:
        raise ValueError('Actual observed paired TRAIN geometry required')
    pos, neg = F.softplus(-logits)[positive], F.softplus(logits)[negative]
    k = min(len(neg), max(1024, len(neg) // 10))
    value = 2 * (pos.mean() if len(pos) else logits.sum() * 0) + 3 * (neg.topk(k, sorted=False).values.mean() if k else logits.sum() * 0)
    if positive.any():
        probability = logits.sigmoid()
        tp, fp, fn = (probability * positive).sum(), (probability * negative).sum(), ((1 - probability) * positive).sum()
        value = value + 1 - (tp + 1) / (tp + .3 * fp + .7 * fn + 1) + topology_loss(logits, target, valid)
    return value


def positive_top(y, height, source_height, jitter):
    if source_height < 320 or height > 192 or height <= 0 or not -64 <= jitter <= 64 or y < 0 or y + height > source_height:
        raise ValueError('Invalid whole-component native geometry')
    top = int(min(max(y + height / 2 + jitter - 160, 0), source_height - 320))
    if not top <= y or y + height > top + 320:
        raise ValueError('Selected positive component truncated')
    return top


def source_inputs():
    paths = [Path(__file__), ROOT / 'tests/test_factory_round5_cldice.py', ROOT / 'scripts/factory_round4_supervised.py',
             ROOT / 'scripts/evaluate_factory_round4.py', ROOT / 'scripts/factory_stfpm_common.py', ROOT / 'scripts/factory_mask_metrics.py',
             ROOT / 'scripts/train_factory_round3d.py', ROOT / 'scripts/train_factory_round3c.py', ROOT / 'scripts/train_factory_round3.py',
             ROOT / 'scripts/train_factory_round3b.py', ROOT / 'scripts/train_factory_spatial_unet.py', ROOT / 'scripts/factory_training_keys.py',
             ROOT / 'scripts/train_factory_surface.py', ROOT / 'edge_runtime/profiles/factory_encoder.py']
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def prepare():
    if PROPOSAL.exists() or PROTOCOL.exists():
        raise ValueError('Preserve preparation')
    tests = read(FIXTURE)
    if tests['all_checks_passed'] is not True or tests['tests'] != 8 or tests['skipped'] != 0:
        raise ValueError('Actual TRAIN-only pure fixtures required')
    pins = source_inputs()
    if tests['source_pins'] != pins:
        raise ValueError('Stale preparation fixture source')
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Exact fixed latest900 warmstart required, not an original-selected checkpoint')
    rows = read(R4 / 'TRAIN-source-manifest.json')['sources']
    if len(rows) != 3444:
        raise ValueError('Retain exact existing real TRAIN inventory')
    sources_unchanged(rows)
    cache = SourceCache(rows)
    original = [row for row in rows if row['dataset'] == 'SDD1']
    normals = [row for row in original if not row['defective']]
    components = []
    for row in original:
        if not row['defective']:
            continue
        _, target = cache.get(row['id'])
        _, _, stats, _ = cv2.connectedComponentsWithStats(target, connectivity=8)
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if h > 192 or x < 0 or x + w > 512:
                raise ValueError('Selected component cannot fit native320 height with fixed+-64 jitter')
            components.append({'source_id': row['id'], 'component': index, 'bbox_xywh': [int(x), int(y), int(w), int(h)], 'pixels': int(area)})
    if len(components) != 33 or len(normals) != 208:
        raise ValueError('Reconcile projected TRAIN component/normal inventory before fitting')
    from factory_training_keys import hardnegative_keys
    hard = hardnegative_keys(read(HARD)['records'])
    difficult = [row for row in original if hard.get(row['id'][5:])]
    transfer = read(R4 / 'transfer-ledger-before-fit.json')['ledger']
    rng = np.random.default_rng(SEED)
    rng.shuffle(components)
    rng.shuffle(normals)
    groups = {'positive': [], 'hard': [], 'normal': [], 'rehearsal': []}
    for index in range(300):
        component = components[index % len(components)]
        _, truth = cache.get(component['source_id'])
        x, y, w, h = component['bbox_xywh']
        top = positive_top(y, h, len(truth), float(rng.uniform(-64, 64)))
        groups['positive'].append({'source_id': component['source_id'], 'top': top, 'height': 320, 'role': 'whole-component-positive',
                                   'selected_component': component, 'foreground_pixels': int(truth[top:top + 320].sum())})
    for index in range(120):
        row = difficult[int(rng.integers(len(difficult)))]; _, truth = cache.get(row['id'])
        centers = hard[row['id'][5:]]; cy = centers[int(rng.integers(len(centers)))]['y'] + rng.uniform(-64, 64)
        top = int(np.clip(cy - 160, 0, len(truth) - 320))
        groups['hard'].append({'source_id': row['id'], 'top': top, 'height': 320, 'role': 'actual-TRAIN-hard', 'foreground_pixels': int(truth[top:top + 320].sum())})
        normal = normals[index % len(normals)]; _, target = cache.get(normal['id'])
        top = int(rng.integers(len(target) - 320 + 1))
        groups['normal'].append({'source_id': normal['id'], 'top': top, 'height': 320, 'role': 'actual-normal', 'foreground_pixels': 0})
    for index in range(60):
        domain = ['SDD2', 'MTD'][index % 2]
        pool = [tile for tile in transfer if tile['source_id'].startswith(domain + '/')]
        groups['rehearsal'].append(dict(pool[int(rng.integers(len(pool)))], role='existing-real-domain-rehearsal'))
    ledger = [tile for group in groups.values() for tile in group]; rng.shuffle(ledger)
    if len(ledger) != STEPS:
        raise ValueError('Exact600 updates required')
    write(OUT / 'TRAIN-ledger-before-fit.json', {'steps': STEPS, 'groups': {key: len(value) for key, value in groups.items()}, 'components': components, 'ledger': ledger})
    template = read(SOURCE.with_suffix('.onnx.json'))
    fixed_paths = [SOURCE, SOURCE.with_suffix('.onnx'), SOURCE.with_suffix('.onnx.json'), INIT_WEIGHTS, HARD,
                   R4 / 'TRAIN-source-manifest.json', R4 / 'transfer-ledger-before-fit.json', R4 / 'training-complete.json',
                   OUT / 'TRAIN-ledger-before-fit.json', FIXTURE, CAL_REFERENCE]
    plan = {'name': NAME, 'seed': SEED, 'steps': STEPS, 'save_at': SAVE_AT, 'cpu_threads': 1, 'batch_size': 1,
            'warmstart': str(SOURCE.relative_to(ROOT)), 'warmstart_selection': 'Fixed latestRound4fine900, not chosen by original results',
            'architecture': 'Exact PackedPixelRefiner+MaxPool2d2; original input/output preprocessing unchanged',
            'training_geometry': 'Native320x512 full-width patches; whole selected SDD1components fit±64jitter; all actual GT retained in every role',
            'topology': 'TRAIN-only soft-clDice, fixed5cross-erosion/opening iterations/smooth1; validinterior8px; zero term if no observed interior target skeleton. No inference morphology or GT input.',
            'loss': '2positiveBCE+3hardnegativeBCE(top10%,min1024)+FN-Tversky(.3FP/.7FN)+1softclDice; observed pixels only',
            'optimizer': 'FreshAdamW encoder/context1e-5 decoder/refiner5e-5,weightdecay1e-4,clipnorm5; frozenBN/normalizationbuffers',
            'augmentation': 'ExactRound4matchedflips/contrast.6:1.4/brightness±25/10%mild3x3blur; no new augmentation grid',
            'all600_before_numeric_CAL_feedback': True, 'no_early_stop_extension': True,
            'split_items': template['split_items'], 'protected_CAL_source_rows': protected_cal_rows(template['split_items']['calibration']),
            'CAL_source_identity_sha256': CAL_SHA, 'CAL_metadata_preparation_scope': 'Exactorderedpaths/fileSHA/GTfileSHA/historicalnativegeometry only; noCALpixeldecode usedforfit',
            'input_shape': [1, 3, 1280, 512], 'output_shape': [1, 1, 640, 256],
            'raw_high_grid': read(R4 / 'protocol-before-fit.json')['raw_high_grid'], 'area_grid': [1, 2, 4, 8, 16, 32],
            'growth_grid': [None, .05, .1, .25, .5, .7, .8, .9, .95, .98, .99],
            'selection': 'One prospectivezeroCALalarm arm, pixelP>=.65 and10/10meaningfulIoU>=.1 required; existingraw/growthgrid/seedshortlist/rank unchanged. All3outcomes then globalfreeze.',
            'numeric': 'Each ownlearnedTorch3case vsactualdefaultGPUf16 maxerror<=.01; failingnumeric noCAL, all3terminal required',
            'capacity_runtime': 'Root fresh fitted>=15completeprofile, then exposedoriginal;16FPS60s/counts/tails beforeacceptance. Previousspeednotinherited.',
            'fit_time_scope': 'Approx4–8min estimatefromprior320TRAINpace, not measuredcapacity or deadline; fixed600 not earlystopped forlossfeedback',
            'primary_sources': ['https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html', 'https://github.com/jocpae/clDice'],
            'official_SDD2_TEST_and_reserved466_unopened': True, 'no_original_or_CAL_pixels_in_fit': True,
            'source_pins': pins, 'artifact_pins': {str(path.relative_to(ROOT)): sha(path) for path in fixed_paths}}
    write(PROPOSAL, plan)
    print(json.dumps({'proposal_sha256': sha(PROPOSAL), 'roles': {key: len(value) for key, value in groups.items()}, 'components': len(components)}, indent=2))


def verify_plan():
    plan = read(PROTOCOL); decision = read(DECISION)
    if plan != read(PROPOSAL) or decision.get('approved_exact_round5_fit') is not True or decision['proposal_sha256'] != sha(PROPOSAL):
        raise ValueError('Root exact prepared600-step proposal approval required')
    for relative, digest in {**plan['source_pins'], **plan['artifact_pins']}.items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Protected source/artifact changed: ' + relative)
    rows = read(R4 / 'TRAIN-source-manifest.json')['sources']; sources_unchanged(rows)
    if plan['protected_CAL_source_rows'] != protected_cal_rows(plan['split_items']['calibration']) or plan['CAL_source_identity_sha256'] != CAL_SHA:
        raise ValueError('Protected CAL metadata/source identity changed')
    return plan, rows


def train():
    import torch
    from torch import nn
    from train_factory_round3d import PackedPixelRefiner
    from train_factory_spatial_unet import train_mode, export
    if (OUT / 'training-started.json').exists():
        raise ValueError('Preserve partial fit; no silent restart')
    plan, rows = verify_plan(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); random.seed(SEED); rng = np.random.default_rng(SEED + 1)
    cache = SourceCache(rows); model = PackedPixelRefiner(INIT_WEIGHTS)
    model.load_state_dict(torch.load(SOURCE, map_location='cpu', weights_only=True), strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()): raise ValueError('Nonfinite warmstart state')
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    encoder = list(model.encoder.parameters()) + list(model.context.parameters()) + list(model.context_projection.parameters())
    ids = {id(value) for value in encoder}
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': 1e-5}, {'params': [value for value in model.parameters() if id(value) not in ids], 'lr': 5e-5}], weight_decay=1e-4)
    archive = OUT / 'executed-sources-before-optimizer'
    for relative, digest in plan['source_pins'].items():
        path = archive / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((ROOT / relative).read_bytes())
        if sha(path) != digest: raise ValueError('Source archive mismatch')
    write(archive / 'manifest.json', {'protocol_sha256': sha(PROTOCOL), 'source_pins': plan['source_pins']})
    write(OUT / 'training-started.json', {'protocol_sha256': sha(PROTOCOL), 'threads': 1, 'steps': STEPS, 'warmstart_sha256': sha(SOURCE)})
    reference_row = next(row for row in rows if row['dataset'] == 'SDD1')
    raw = cv2.imread(str(ROOT / reference_row['image']))
    inputs = np.stack([cv2.resize(raw, (512, 1280), interpolation=cv2.INTER_AREA), np.zeros((1280, 512, 3), np.uint8), np.full((1280, 512, 3), 255, np.uint8)])
    started = time.perf_counter(); history = []; train_mode(model)
    for step, tile in enumerate(read(OUT / 'TRAIN-ledger-before-fit.json')['ledger'], 1):
        frame, truth, valid = crop_observed(*cache.get(tile['source_id']), tile['top'], 320)
        if int(truth.sum()) != tile['foreground_pixels']: raise ValueError('Actual source foreground changed')
        frame, truth, valid = augment(frame, truth, valid, rng)
        x = torch.from_numpy(frame.transpose(2, 0, 1)[None].copy()); y = torch.from_numpy(truth[None, None].copy()).float(); v = torch.from_numpy(valid[None, None].copy()).bool()
        optimizer.zero_grad(set_to_none=True); value = loss(model(x), y, v)
        if not torch.isfinite(value): raise ValueError('Nonfinite TRAIN loss')
        value.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('Nonfinite gradient')
        nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()): raise ValueError('Nonfinite model state')
        if any(not torch.equal(value, buffers[name]) for name, value in model.named_buffers()): raise ValueError('Frozen buffers changed')
        if step % 50 == 0:
            record = {'step': step, 'loss': float(value.detach()), 'elapsed_seconds': time.perf_counter() - started, 'role': tile['role']}
            history.append(record); print(json.dumps(record), flush=True)
        if step in SAVE_AT:
            path = ROOT / f'models/factory/{NAME}-step{step}.onnx'
            if path.exists() or path.with_suffix('.pt').exists(): raise ValueError('Preserve checkpoint')
            torch.save(model.state_dict(), path.with_suffix('.pt')); torch.save(optimizer.state_dict(), OUT / f'checkpoint{step}-optimizer.pt')
            serving = nn.Sequential(model, nn.MaxPool2d(2)); export(serving, path)
            with torch.inference_mode(): scores = serving(torch.from_numpy(inputs.transpose(0, 3, 1, 2).copy())).sigmoid().numpy()[:, 0].copy()
            reference = OUT / f'checkpoint{step}-torch-reference.npz'
            with reference.open('xb') as stream: np.savez_compressed(stream, inputs_bgr=inputs, scores=scores)
            meta = read(SOURCE.with_suffix('.onnx.json'))
            parent_lineage = {'Round4latest900_state_sha256': SOURCE_SHA, 'Round4latest900_ONNX_sha256': sha(SOURCE.with_suffix('.onnx')),
                              'Round4_protocol_sha256': meta.pop('round4_training_protocol_sha256'), 'earlier': meta.get('pretraining_lineage')}
            for key in ['calibration_selected_point', 'calibration_source_identity_sha256', 'round4_policy_arm', 'round4_normal_FP_budget']: meta.pop(key, None)
            meta.update(model_sha256=sha(path), profile_version=NAME, model_name='Experimental thin-structure supervised defect review',
                        training_protocol=plan, round5_training_protocol_sha256=sha(PROTOCOL), calibration_status='pending_new_weights',
                        threshold_selection='Previous policy placeholder only; newzeroCALalarm selection required',
                        trained_utc=datetime.now(timezone.utc).isoformat(), topology_loss_training_only=True, pretraining_lineage=parent_lineage)
            write(path.with_suffix('.onnx.json'), meta)
            write(OUT / f'checkpoint{step}-training.json', {'step': step, 'protocol_sha256': sha(PROTOCOL), 'model_sha256': sha(path),
                'state_sha256': sha(path.with_suffix('.pt')), 'initial_metadata_sha256': sha(path.with_suffix('.onnx.json')), 'reference_sha256': sha(reference),
                'optimizer_sha256': sha(OUT / f'checkpoint{step}-optimizer.pt'), 'all_state_finite': True, 'frozen_BN_normalization_unchanged': True})
            train_mode(model)
    sources_unchanged(rows)
    write(OUT / 'training-complete.json', {'protocol_sha256': sha(PROTOCOL), 'steps': STEPS, 'saved_all_three': SAVE_AT,
          'elapsed_seconds': time.perf_counter() - started, 'history': history, 'sources_unchanged': True, 'no_CAL_original_feedback': True})


def checkpoint(step):
    done = read(OUT / 'training-complete.json')
    if done['steps'] != STEPS or done['saved_all_three'] != SAVE_AT or done['protocol_sha256'] != sha(PROTOCOL): raise ValueError('All600 required before feedback')
    path = ROOT / f'models/factory/{NAME}-step{step}.onnx'; reference = OUT / f'checkpoint{step}-torch-reference.npz'; receipt = read(OUT / f'checkpoint{step}-training.json')
    if (receipt['protocol_sha256'] != sha(PROTOCOL) or receipt['model_sha256'] != sha(path) or receipt['state_sha256'] != sha(path.with_suffix('.pt'))
        or receipt['reference_sha256'] != sha(reference) or receipt['initial_metadata_sha256'] != sha(path.with_suffix('.onnx.json'))
        or receipt['optimizer_sha256'] != sha(OUT / f'checkpoint{step}-optimizer.pt')
        or receipt['all_state_finite'] is not True or receipt['frozen_BN_normalization_unchanged'] is not True): raise ValueError('Checkpoint/reference/state lineage changed')
    return path, reference, receipt


def numeric_all():
    # Reuse the exact already-executed numerical guard; bind only new artifacts.
    import evaluate_factory_round4 as common
    common.OUT, common.NAME, common.PROTOCOL, common.SAVE_AT = OUT, NAME, PROTOCOL, SAVE_AT
    common.checkpoint = checkpoint
    for step in SAVE_AT: common.numeric(step)


def checked_numeric(step):
    from evaluate_factory_round4 import numeric_passes
    path, reference, training = checkpoint(step); target = OUT / f'checkpoint{step}-numeric.json'; data = read(target)
    if (data['model_sha256'] != sha(path) or data['reference_sha256'] != sha(reference) or data['initial_metadata_sha256'] != sha(path.with_suffix('.onnx.json'))
        or data['training_receipt_sha256'] != sha(OUT / f'checkpoint{step}-training.json') or data['step'] != step or data['protocol_sha256'] != sha(PROTOCOL)
        or data['all_checks_passed'] != numeric_passes(data)): raise ValueError('Numeric lineage/gates changed')
    return data, target


def calibrate_all(plan):
    from evaluate_factory_round4 import numeric_passes, configuration, recipe_passes, raw_candidates, select_round4, validate_search
    from factory_mask_metrics import original_pixel_weights
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    from train_factory_surface import samples
    for step in SAVE_AT: checked_numeric(step)
    base = ROOT / 'datasets/factory/KolektorSDD'
    for step in SAVE_AT:
        n, nr = checked_numeric(step)
        if not numeric_passes(n): continue
        if any((OUT / f'checkpoint{step}-{suffix}').exists() for suffix in ['CAL-started.json', 'calibration.json', 'scorecache.npz', 'terminal.json']): raise ValueError('Preserve CAL attempt')
        sources, rows = [], []
        for image, truth, defective in samples(base, plan['split_items']['calibration']):
            frame = cv2.imread(str(image))
            sources.append({'image': str(image.relative_to(base)), 'defective': bool(defective), 'image_sha256': sha(image), 'mask_sha256': sha(image.with_name(image.stem + '_label.bmp')), 'native_image_shape': list(frame.shape), 'native_mask_shape': list(truth.shape)})
            rows.append({'image': sources[-1]['image'], 'truth': truth, 'frame': frame, 'defective': bool(defective)})
        if len(rows) != 80 or sum(row['defective'] for row in rows) != 10 or cal_identity(sources) != CAL_SHA or sources != plan['protected_CAL_source_rows']: raise ValueError('Exact historical protected CAL80 identity/content required before NN')
        write(OUT / f'checkpoint{step}-CAL-started.json', {'protocol_sha256': sha(PROTOCOL), 'numeric_sha256': sha(nr), 'CAL_identity': CAL_SHA})
        path, _, _ = checkpoint(step); model = FactorySegmentation(path, output_logits=True); properties = configuration(model.compiled)
        if not recipe_passes(properties): raise ValueError('Actual GPU recipe differs')
        for row in rows:
            row['scores'] = model.predict(row.pop('frame'), .05, 4)[0]; row['projection_weights'] = original_pixel_weights(row['truth'], row['scores'].shape)
        for row in sources:
            image = base / row['image']
            if sha(image) != row['image_sha256'] or sha(image.with_name(image.stem + '_label.bmp')) != row['mask_sha256']: raise ValueError('CAL source changed')
        cache = OUT / f'checkpoint{step}-scorecache.npz'
        with cache.open('xb') as stream: np.savez_compressed(stream, scores=np.stack([row['scores'] for row in rows]), sources_json=np.asarray(json.dumps(sources)), model_sha256=np.asarray(sha(path)))
        raw = raw_candidates(rows, plan); result = select_round4(rows, plan, 0, raw); validate_search(raw, result, plan, 0)
        point = result['selected']; eligible = point is not None and point['meaningful_iou_0_1_recall'] == 1 and point['normal_fp'] == 0 and point['foreground_pixel_precision'] >= .65
        report = {'step': step, 'protocol_sha256': sha(PROTOCOL), 'model_sha256': sha(path), 'numeric_sha256': sha(nr), 'scorecache_sha256': sha(cache),
                  'configuration': properties, 'ordered_CAL_sources': sources, 'CAL_identity': CAL_SHA, 'raw_grid': raw, 'zero_alarm_search': result, 'eligible_all10_zero_alarm': eligible,
                  'scope': 'ActualGPU fixedCAL80 development scores; no FPS qualification'}
        if eligible:
            deployed = path.with_name(path.stem + '-zero-alarm.onnx')
            with deployed.open('xb') as stream: stream.write(path.read_bytes())
            meta = read(path.with_suffix('.onnx.json')); meta.update(pixel_threshold=point['high'], minimum_component_pixels=point['area'], hysteresis_growth_threshold=point['growth'],
                    calibration_selected_point=point, calibration_status='round5_CAL80_zero_alarm_all10', calibration_source_identity_sha256=CAL_SHA)
            write(deployed.with_suffix('.onnx.json'), meta)
            report['deployment'] = {'model_path': str(deployed), 'model_sha256': sha(deployed), 'metadata_sha256': sha(deployed.with_suffix('.onnx.json'))}
        target = OUT / f'checkpoint{step}-calibration.json'; write(target, report)
        write(OUT / f'checkpoint{step}-terminal.json', {'step': step, 'status': 'calibration_complete', 'CAL_inputs_opened': 80, 'protocol_sha256': sha(PROTOCOL), 'numeric_sha256': sha(nr), 'calibration_sha256': sha(target)})
        print(json.dumps({'step': step, 'eligible': eligible, 'point': point}, indent=2), flush=True)


def freeze(plan):
    from evaluate_factory_round4 import numeric_passes, recipe_passes, validate_search
    choices, outcomes = [], {}
    for step in SAVE_AT:
        path, _, _ = checkpoint(step); n, nr = checked_numeric(step); terminal = read(OUT / f'checkpoint{step}-terminal.json')
        if terminal['protocol_sha256'] != sha(PROTOCOL) or terminal['numeric_sha256'] != sha(nr): raise ValueError('Terminal lineage changed')
        outcomes[str(step)] = terminal
        if not numeric_passes(n):
            if terminal['status'] != 'numeric_failed_no_CAL' or terminal['CAL_inputs_opened'] != 0 or any((OUT / f'checkpoint{step}-{suffix}').exists() for suffix in ['CAL-started.json', 'calibration.json', 'scorecache.npz']): raise ValueError('Numeric failure must never open CAL')
            continue
        rp = OUT / f'checkpoint{step}-calibration.json'; r = read(rp)
        if (terminal['status'] != 'calibration_complete' or terminal['CAL_inputs_opened'] != 80 or terminal['calibration_sha256'] != sha(rp)
            or r['protocol_sha256'] != sha(PROTOCOL) or r['model_sha256'] != sha(path) or r['numeric_sha256'] != sha(nr)
            or r['scorecache_sha256'] != sha(OUT / f'checkpoint{step}-scorecache.npz') or not recipe_passes(r['configuration']) or cal_identity(r['ordered_CAL_sources']) != CAL_SHA
            or r['ordered_CAL_sources'] != plan['protected_CAL_source_rows']): raise ValueError('Exact completed CAL lineage required')
        validate_search(r['raw_grid'], r['zero_alarm_search'], plan, 0)
        point = r['zero_alarm_search']['selected']; eligible = point is not None and point['meaningful_iou_0_1_recall'] == 1 and point['normal_fp'] == 0 and point['foreground_pixel_precision'] >= .65
        if r['eligible_all10_zero_alarm'] != eligible: raise ValueError('Fixed qualification guard changed')
        if eligible:
            deployment = r['deployment']; model = Path(deployment['model_path']); meta = read(model.with_suffix('.onnx.json'))
            if (sha(model) != sha(path) or deployment['model_sha256'] != sha(model) or deployment['metadata_sha256'] != sha(model.with_suffix('.onnx.json'))
                or meta['calibration_selected_point'] != point or meta['round5_training_protocol_sha256'] != sha(PROTOCOL)
                or (meta['pixel_threshold'], meta['minimum_component_pixels'], meta['hysteresis_growth_threshold']) != (point['high'], point['area'], point['growth'])): raise ValueError('Exact selected artifact/policy changed')
            choices.append({'step': step, 'point': point, **deployment})
    choices.sort(key=lambda row: rank(row['point']), reverse=True)
    write(OUT / 'winner-frozen-before-root-capacity.json', {'protocol_sha256': sha(PROTOCOL), 'all_three_complete': True, 'outcomes': outcomes,
          'winner': choices[0] if choices else None, 'ranked_candidates': choices, 'original_audit_not_authorized_yet': True,
          'required_next_gate': 'Root fresh fitted>=15 then exposedoriginal, sustained16FPS/counts/tails beforeacceptance', 'production_qualified': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'record-protocol', 'train', 'numeric-all3', 'calibrate-all3', 'freeze'], required=True)
    parser.add_argument('--root-released', action='store_true'); args = parser.parse_args(); cv2.setNumThreads(1)
    if args.stage == 'prepare': prepare()
    elif args.stage == 'record-protocol':
        decision = read(DECISION)
        if decision.get('approved_exact_round5_fit') is not True or decision['proposal_sha256'] != sha(PROPOSAL): raise ValueError('Root exact approval required')
        write(PROTOCOL, read(PROPOSAL)); verify_plan()
    else:
        if not args.root_released: raise ValueError('Separate root compute release required')
        protocol, _ = verify_plan()
        if args.stage == 'train': train()
        elif args.stage == 'numeric-all3': numeric_all()
        elif args.stage == 'calibrate-all3': calibrate_all(protocol)
        else: freeze(protocol)
