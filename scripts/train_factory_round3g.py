"""Attempt5: native-pixel full-width crops, unchanged phase-max serving graph."""
import cv2
import numpy as np
import torch
import train_factory_round3c as base
from train_factory_round3d import PackedPixelRefiner


def native_width_training_batch(rows, sdd2, rng, hard, step):
    """One 640H x 512W real crop: no patch resize or quarter-turn transpose."""
    slot = (step - 1) % 5
    if slot == 4:
        positive = ((step // 5) % 2) == 1
        pool = sdd2.positive if positive else sdd2.normal
        image, truth, ys, xs, name = sdd2.get(pool[int(rng.integers(len(pool)))])
        role = 0 if positive else 2
    else:
        role = [0, 0, 1, 2][slot]
        pool = ([row for row in rows if len(row[2])] if role == 0 else
                [row for row in rows if hard.get(row[4])] if role == 1 else rows)
        image, truth, ys, xs, name = pool[int(rng.integers(len(pool)))]

    if role == 0:
        point = int(rng.integers(len(ys)))
        cy = float(ys[point]) + rng.uniform(-.3, .3) * 640
    elif role == 1:
        point = hard[name][int(rng.integers(len(hard[name])))]
        cy = point['y'] + rng.uniform(-.125, .125) * 640
    else:
        cy = rng.uniform(320, 960)
    top = int(np.clip(cy - 320, 0, 640))
    frame = image[top:top + 640, :512].copy()
    target = truth[top:top + 640, :512].copy()
    if frame.shape != (640, 512, 3) or target.shape != (640, 512):
        raise ValueError('Native full-width training source geometry changed')

    if rng.random() < .5:
        frame, target = frame[::-1, ::-1], target[::-1, ::-1]
    if rng.random() < .5:
        frame, target = frame[:, ::-1], target[:, ::-1]
    frame = np.clip((frame.astype(np.float32) - 127.5) * rng.uniform(.4, 1.4)
                    + 127.5 + rng.uniform(-30, 30), 0, 255)
    if rng.random() < .15:
        frame = cv2.GaussianBlur(frame, (3, 3), float(rng.uniform(.3, .8)))
    image_tensor = torch.from_numpy(np.ascontiguousarray(frame.astype(np.uint8).transpose(2, 0, 1)[None]))
    target_tensor = torch.from_numpy(np.ascontiguousarray(target[None, None])).float()
    return image_tensor, target_tensor


if __name__ == '__main__':
    base.NAME = 'spatial_round3g_native_width'
    base.ARCHITECTURE = ('Unchanged qualified C phase-supervised packed-pixel refiner with '
                         'half-grid max-logit serving; native full-width 640x512 training crops')
    base.FullResolutionRefiner = PackedPixelRefiner
    base.SERVING_POOL = 2
    base.WARMSTART_SOURCE = base.ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.pt'
    base.WARMSTART_FREEZE = base.OUT / 'winner-attempt3-phase-max-freeze-before-comparison.json'
    base.WARMSTART_ORIGIN = 'Immutable C calibration-selected step200; no losing D checkpoint selection'
    base.SEED = 20261025
    base.STEPS = 900
    base.SAVE_AT = [300, 600, 900]
    base.training_batch = native_width_training_batch
    base.PROTOCOL_OVERRIDES = {
        'batch_size': 1, 'patch': [640, 512],
        'training_output_shape': [1, 1, 640, 512],
        'full_source_supervision_shape': [1, 1, 1280, 512],
        'sampling': ('Five-step cycle: SDD1 positive, positive, TRAIN-mined hard location, '
                     'unbiased real patch, SDD2 inner-FIT rehearsal. Rehearsal alternates '
                     'positive and actual normal. No comparison mining.'),
        'augmentation': ('No crop resize or 90-degree transpose. Matched 0/180-degree turns '
                         'and horizontal flips; contrast .4:1.4, brightness +-30, '
                         '15% mild blur. Real unchanged evaluation inputs.'),
        'mask_projection': ('Original GT to serving-input 512x1280 binary any-overlap occupancy, '
                            'then actual 640x512 vertical slice with no patch resampling. '
                            'Full-resolution patch loss; no evaluation GT dilation.'),
        'positive_crop_geometry': ('Full source width retained. Point-centered y jitter at most '
                                   '192 pixels; all 33 SDD1 TRAIN boxes have height <=120 and '
                                   'width <=452, so a selected component fits within this crop. '
                                   'This is not a guarantee that every SDD2 component fits.'),
        'capacity_gate': ('Accuracy-first original audit after all three calibrations/global freeze '
                          'because unchanged serving graph previously qualified 19.66 calls/s. '
                          'Final new-weight/policy paired project-quiescent capacity and runtime '
                          '>=15 FPS remain mandatory before promotion.'),
        'pipeline_calibration': ('Immutable completed checkpoints may calibrate while fixed '
                                 '900-step CPU1 fit continues; no feedback or schedule changes; '
                                 'all three required before freeze.'),
        'calibration_cache_identity': ('Exact model SHA plus ordered original calibration image/mask '
                                      'SHA256, native geometry and class; same identity required '
                                      'across all three checkpoints.'),
        'train_geometry_audit_sha256': base.sha(base.OUT / 'attempt4-train-boundary-context-audit.json'),
        'train_geometry_rationale': ('TRAIN-only actual 512x1280 target geometry: 33 components, '
                                     'bbox width 32/294/452 and height 16/55/120 min/median/max. '
                                     'All fit 640x512; prior 256-wide crops truncate 20 boxes. '
                                     'Same 327680 batch pixels as C, no causal single-factor claim.'),
    }
    base.main()
