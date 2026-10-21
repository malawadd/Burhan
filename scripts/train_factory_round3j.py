"""Bounded matched TRAIN curvature augmentation; unchanged frozen-C topology."""
import json
from collections import Counter
import numpy as np
import torch
import train_factory_round3c as base
from train_factory_round3d import PackedPixelRefiner
from factory_curvature_sampling import training_arrays

NAME = 'spatial_round3j_native_curvature'
SOURCE_PT_SHA = 'ff3b5571295e80978b4be873b6cce24b7cb947ae25415959a4121337dfe22112'
SOURCE_ONNX_SHA = 'ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318'
SOURCE_META_SHA = '6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb'
SOURCE_FREEZE_SHA = 'c3c2a6d86f86d625f9f6499c4d7477a25c2b9610b26a870be994a29ac969db40'
COUNTS = Counter()


def training_batch(rows, sdd2, rng, hard, step):
    frame, target, info = training_arrays(rows, sdd2, rng, hard, step)
    COUNTS[info['role']] += 1
    COUNTS['requested_curve'] += int(info['requested_curve'])
    COUNTS['applied_curve'] += int(info['applied_curve'])
    COUNTS['identity_fallback'] += int(info['identity_fallback'] is not None)
    record = {'step': step, 'counts': dict(COUNTS), 'last_sample': info,
              'scope': 'Training augmentation provenance only; not real inspection evidence'}
    with (base.OUT / (NAME + '-augmentation-history.jsonl')).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
    return (torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)[None])),
            torch.from_numpy(np.ascontiguousarray(target[None, None])).float())


def configure():
    source = base.ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.pt'
    onnx = source.with_suffix('.onnx')
    freeze = base.OUT / 'winner-attempt3-phase-max-freeze-before-comparison.json'
    expected = [(source, SOURCE_PT_SHA), (onnx, SOURCE_ONNX_SHA),
                (onnx.with_suffix('.onnx.json'), SOURCE_META_SHA), (freeze, SOURCE_FREEZE_SHA)]
    for path, digest in expected:
        if base.sha(path) != digest:
            raise ValueError(f'Immutable C200 source changed: {path}')
    base.NAME = NAME
    base.ARCHITECTURE = 'Unchanged C200 context/packed-phase refiner plus2x2max-logit serving; TRAIN-only native full-width smooth vertical-remap augmentation'
    base.FullResolutionRefiner = PackedPixelRefiner
    base.SERVING_POOL = 2
    base.WARMSTART_SOURCE = source
    base.WARMSTART_FREEZE = freeze
    base.WARMSTART_ORIGIN = 'Immutable C calibration-selected step200; complete learned context/refiner retained with strict state loading; no losing checkpoint'
    base.SEED, base.STEPS, base.SAVE_AT = 20261027, 600, [200, 400, 600]
    base.training_batch = training_batch
    fixture = base.OUT / (NAME + '-pairing-fixtures.json')
    report = json.loads(fixture.read_text(encoding='utf-8'))
    sampler = base.ROOT / 'scripts/factory_curvature_sampling.py'
    if not report['all_checks_passed'] or report['sampler_sha256'] != base.sha(sampler):
        raise ValueError('Required hardware-free paired-coordinate fixtures absent or stale')
    if report['actual_train_views'] != 239 or report['actual_train_components'] != 33 or report['metadata_sha256'] != SOURCE_META_SHA:
        raise ValueError('Fixture source partition/identity changed')
    for row in report['train_sources']:
        image = base.ROOT / 'datasets/factory/KolektorSDD' / row['image']
        if base.sha(image) != row['image_sha256'] or base.sha(image.with_name(image.stem + '_label.bmp')) != row['mask_sha256']:
            raise ValueError('Actual TRAIN image/mask differs from geometry fixtures')
    original_sdd2 = base.FullResolutionSdd2
    class VerifiedTrainingSdd2(original_sdd2):
        def get(self, index):
            if index not in self.cache:
                record = self.records[index]
                if base.sha(base.SDD2 / record['image']) != record['image_sha256'] or base.sha(base.SDD2 / record['mask']) != record['mask_sha256']:
                    raise ValueError('Actual SDD2 inner-FIT source differs from frozen split')
            return super().get(index)
    base.FullResolutionSdd2 = VerifiedTrainingSdd2
    receipt = base.OUT / 'attempt3-final-calibration-paired-profile-capacity.json'
    base.PROTOCOL_OVERRIDES = {
        'batch_size': 1, 'patch': [640, 512], 'training_output_shape': [1, 1, 640, 512],
        'full_source_supervision_shape': [1, 1, 1280, 512],
        'warmstart_onnx_sha256': SOURCE_ONNX_SHA, 'warmstart_metadata_sha256': SOURCE_META_SHA,
        'warmstart_freeze_sha256': SOURCE_FREEZE_SHA,
        'sampler_sha256': base.sha(sampler), 'wrapper_sha256': base.sha(base.ROOT / 'scripts/train_factory_round3j.py'),
        'pairing_fixtures_sha256': base.sha(fixture),
        'reused_architecture_capacity_receipt': str(receipt), 'reused_architecture_capacity_receipt_sha256': base.sha(receipt),
        'trainer_helper_sha256': base.sha(base.ROOT / 'scripts/train_factory_round3c.py'),
        'source_model_implementation_sha256': base.sha(base.ROOT / 'scripts/train_factory_round3d.py'),
        'loss_implementation_sha256': base.sha(base.ROOT / 'scripts/train_factory_round3b.py'),
        'source_identity_policy': 'All239 SDD1 image/mask hashes verified against pre-fit geometry fixtures; each consumed SDD2 inner-FIT image/mask verified against immutable split hashes on cache miss; official/inner reserved test never consumed',
        'sampling': 'Five-step cycle SDD1 positive/positive/TRAIN-mined hard/unbiased,SDD2 inner-FIT rehearsal alternating positive and actual normal; batch1; no comparison mining',
        'augmentation': 'Whole-patch paired vertical remap probability.5 for ALL roles: source_y=baseY+destination_y+A*sin(2pi*x/P+phi),A in{8,16},P in{128,256,512},phi uniform[0,2pi); imageLINEAR,GTNEAREST; no invented borders. Then existing paired0/180/horizontal flips,contrast.4:1.4,brightness+-30,15%mildblur. Training augmentation only, never new real defects or inspection evidence.',
        'curvature_geometry': 'Native640H512W,full source width,no resize/90deg. Warped integer baseY16..624; selected SDD1 connected-component y-f(x) bounds with1pixel safety margin constrain baseY; if infeasible identity fallback with selected whole component retained. SDD2 retains sampled actual positive pixel,not necessarily full component. Hard/unbiased targets retain ALL actual source occupancy sampled by full remap; no selected-ROI masks.',
        'mask_projection': 'Original real GT to512x1280 INTER_AREA>0 occupancy,then matched nearest whole-patch remap; loss on full640x512; no evaluation GT morphology/warp',
        'train_geometry_audit_sha256': base.sha(base.OUT / 'attempt4-train-boundary-context-audit.json'),
        'hypothesis_scope': 'Exploratory augmentation informed by TRAIN32horizontal/1diagonal orientation audit and exposed-development failures. Orientation does not measure curvature or prove absence of wavy TRAIN defects. No comparison pixels/IDs/ROIs/masks used in fitting/mining/selection.',
        'evaluation_policy': 'ALL calibration/comparison/video inputs remain untouched real observations; augmentation applied only during TRAIN sampling; no generated inspection evidence',
        'capacity_gate': 'Identical serving topology to qualified C phase-max model; architecture receipt reused only. Final new-weight/policy paired project-quiescent capacity/runtime>=15FPS mandatory after ALL fits finish; no fit-time speed claim',
        'pipeline_calibration': 'Fixed600CPU1steps may overlap constructionCPU4 and F2 accuracy; checkpoint GPU calibration only after F2 GPU release/global freeze; all3planned200/400/600 required before globalfreeze; no feedback/schedule changes',
        'calibration_cache_identity': 'Exact modelSHA and ordered80 original CAL image/maskSHA,nativegeometry,class; unchanged grid/normalFP<=1/70,pixelprecision>=.65,weakest-positive rank; no original scoring by training owner',
    }
    original_initialize = base.initialize
    def strict_initialize(expected_source_sha256=None):
        model, loaded, frozen = original_initialize(expected_source_sha256)
        # The generic initializer permits new refinement keys for earlier
        # architectures; this same-architecture continuation requires ALL keys.
        model.load_state_dict(torch.load(loaded, map_location='cpu', weights_only=True), strict=True)
        return model, loaded, frozen
    base.initialize = strict_initialize


if __name__ == '__main__':
    import sys
    configure()
    if '--record-protocol' not in sys.argv:
        if any((base.ROOT / 'models/factory' / f'{NAME}-step{step}.pt').exists() for step in base.SAVE_AT):
            raise ValueError('Refusing to restart/overwrite an existing protected curvature experiment')
        if (base.OUT / (NAME + '-augmentation-history.jsonl')).exists():
            raise ValueError('Refusing to overwrite an existing training history')
    base.main()
