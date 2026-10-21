"""No-NN pairing fixtures, including exact original TRAIN components only."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from factory_curvature_sampling import paired_crop, displacement, training_arrays

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_image(truth):
    image = np.zeros((1280, 512, 3), np.uint8)
    image[:, :, 0] = (np.arange(1280) // 5).astype(np.uint8)[:, None]
    image[:, :, 1] = (np.arange(512) // 2).astype(np.uint8)[None, :]
    image[:, :, 2] = truth * 255
    return image


def check_pair(truth, curve, retain=None, desired=320):
    image = coordinate_image(truth)
    frame, target, info = paired_crop(image, truth, desired, curve=curve, retain=retain)
    f = displacement(*curve) if info['applied_curve'] else np.zeros(512, np.float32)
    yy = np.arange(640, dtype=np.float32)[:, None] + info['top'] + f[None, :]
    xx = np.broadcast_to(np.arange(512, dtype=np.float32), (640, 512)).copy()
    nearest, _ = cv2.convertMaps(xx, yy, cv2.CV_16SC2, nninterpolation=True)
    expected = truth[nearest[:, :, 1], nearest[:, :, 0]]
    assert np.array_equal(target, expected), 'GT does not equal corresponding observed nearest source occupancy'
    assert np.array_equal(frame[:, :, 1], np.broadcast_to((np.arange(512) // 2).astype(np.uint8), (640, 512)))
    assert np.max(np.abs(frame[:, :, 0].astype(float) - yy / 5)) <= 1.3, 'Image coordinate pairing is wrong'
    assert not target.any() or int(frame[:, :, 2][target > 0].min()) >= 120, 'Image/GT warp is mispaired'
    assert 0 <= yy.min() <= yy.max() <= 1279
    if retain is not None:
        selected = np.zeros_like(truth)
        selected[retain] = 1
        actual = selected[nearest[:, :, 1], nearest[:, :, 0]]
        assert int(actual.sum()) == len(retain[0]), 'Selected actual GT was truncated'
    return info


def main():
    cv2.setNumThreads(1)
    synthetic_checks = 0
    truth = np.zeros((1280, 512), np.uint8)
    truth[590:650, 4:20] = 1
    truth[590:650, 440:512] = 1
    for amplitude in (8, 16):
        for period in (128, 256, 512):
            for phase in (0., 1.37, np.pi / 2, np.pi):
                for top in (0, 16, 320, 624, 640):
                    check_pair(truth, (amplitude, period, phase), np.nonzero(truth), top)
                    synthetic_checks += 1
    for top in (0, 16, 320, 624, 640):
        normal = np.zeros_like(truth)
        info = check_pair(normal, (16, 128, 1.37), desired=top)
        assert info['output_positive_pixels'] == 0
        synthetic_checks += 1
    # A boundary positive cannot satisfy the curved margin; declared identity
    # fallback must retain it without invented border samples.
    boundary = np.zeros_like(truth)
    boundary[0, 0] = 1
    info = check_pair(boundary, (16, 512, np.pi / 2), np.nonzero(boundary))
    assert info['identity_fallback'] and not info['applied_curve'] and info['top'] == 0
    synthetic_checks += 1
    # An impossible whole component is rejected rather than silently cropped.
    tall = np.zeros_like(truth)
    tall[0:900, 20] = 1
    try:
        paired_crop(coordinate_image(tall), tall, 320, curve=(16, 128, 0.), retain=np.nonzero(tall))
    except ValueError:
        synthetic_checks += 1
    else:
        raise AssertionError('Impossible identity containment did not fail closed')

    ys, xs = np.nonzero(truth)
    row = (coordinate_image(truth), truth, ys, xs, 'coordinate-pair-fixture')
    normal = (coordinate_image(np.zeros_like(truth)), np.zeros_like(truth), np.array([], int), np.array([], int), 'normal-pair-fixture')
    class FitFixture:
        positive, normal = [0], [1]
        def get(self, index):
            return row if index == 0 else normal
    role_counts, applied_counts = {}, {}
    rng = np.random.default_rng(20261027)
    for step in range(1, 201):
        frame, target, info = training_arrays([row], FitFixture(), rng, {row[4]: [{'y': 620, 'x': 10}]}, step)
        assert frame.shape == (640, 512, 3) and target.shape == (640, 512)
        assert set(np.unique(target)) <= {0, 1}
        if info['role'] == 'normal':
            assert not target.any()
        role_counts[info['role']] = role_counts.get(info['role'], 0) + 1
        applied_counts[info['role']] = applied_counts.get(info['role'], 0) + int(info['applied_curve'])
    assert set(role_counts) == {'positive', 'hard', 'unbiased', 'normal'}
    assert all(0 < applied_counts[role] < role_counts[role] for role in role_counts)

    meta_path = ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.onnx.json'
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    identities, components, actual_checks = [], [], 0
    for item in metadata['split_items']['train']:
        for path in sorted((ROOT / 'datasets/factory/KolektorSDD' / item).glob('*.jpg')):
            mask_path = path.with_name(path.stem + '_label.bmp')
            source = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if source is None:
                raise ValueError('Invalid actual TRAIN mask')
            mask = (cv2.resize((source > 0).astype(np.float32), (512, 1280), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
            count, labels = cv2.connectedComponents(mask, connectivity=8)
            identities.append({'image': str(path.relative_to(ROOT / 'datasets/factory/KolektorSDD')), 'image_sha256': sha(path), 'mask_sha256': sha(mask_path), 'components': count - 1})
            for component in range(1, count):
                retain = np.nonzero(labels == component)
                fallbacks = 0
                for amplitude in (8, 16):
                    for period in (128, 256, 512):
                        for phase in (0., 1.37):
                            info = check_pair(mask, (amplitude, period, phase), retain, float(np.median(retain[0])) - 320)
                            fallbacks += int(info['identity_fallback'] is not None)
                            actual_checks += 1
                components.append({'image': identities[-1]['image'], 'component': component, 'pixels': len(retain[0]), 'checks': 12, 'identity_fallbacks': fallbacks})
    assert len(identities) == 239 and len(components) == 33
    report = {'all_checks_passed': True, 'sampler_sha256': sha(ROOT / 'scripts/factory_curvature_sampling.py'),
              'fixture_script_sha256': sha(Path(__file__)), 'metadata_sha256': sha(meta_path),
              'synthetic_coordinate_checks': synthetic_checks, 'role_batches': 200,
              'role_counts': role_counts, 'applied_curve_counts': applied_counts,
              'actual_train_views': len(identities), 'actual_train_components': len(components),
              'actual_component_pairing_checks': actual_checks, 'components': components,
              'train_sources': identities, 'calibration_images_opened': 0, 'comparison_images_opened': 0,
              'scope': 'Hardware-free paired-coordinate/GT-validity fixtures only. Synthetic coordinate ramps are test fixtures, never artificial defect accuracy/video evidence. Actual source decoding restricted to239 original TRAIN masks; no NN or optimizer.'}
    target = OUT / 'spatial_round3j_native_curvature-pairing-fixtures.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != report:
            raise ValueError('Refusing to overwrite changed fixture receipt')
    else:
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('components', 'train_sources')}, indent=2))


if __name__ == '__main__':
    main()
