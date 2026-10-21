"""Hardware-free crop/GT pairing checks; no model forward or external data."""
import json
from pathlib import Path
import numpy as np
import torch
from train_factory_round3g import native_width_training_batch


def main():
    torch.set_num_threads(1)
    image = np.zeros((1280, 512, 3), np.uint8)
    # Disjoint left/right GT within the same 120-high band tests a widened
    # hard/unbiased crop retaining annotations outside a prior narrow ROI.
    truth = np.zeros((1280, 512), np.uint8)
    truth[580:700, 10:20] = 1
    truth[580:700, 440:452] = 1
    ys, xs = np.nonzero(truth)
    row = (image, truth, ys, xs, 'synthetic-pairing-fixture')
    normal = (image, np.zeros_like(truth), np.array([], dtype=int),
              np.array([], dtype=int), 'synthetic-normal-pairing-fixture')
    class FixtureFit:
        positive = [0]
        normal = [1]
        def get(self, index):
            return row if index == 0 else normal
    hard = {row[4]: [{'x': 15, 'y': 640}]}
    rng = np.random.default_rng(20261025)
    checked = 0
    for step in range(1, 41):
        batch, target = native_width_training_batch([row], FixtureFit(), rng, hard, step)
        assert batch.shape == (1, 3, 640, 512)
        assert target.shape == (1, 1, 640, 512)
        assert batch.dtype == torch.uint8 and target.dtype == torch.float32
        assert set(np.unique(target.numpy())) <= {0., 1.}
        if step % 10 == 0:
            assert target.sum() == 0, 'Real normal rehearsal GT changed'
        elif step % 5 in (1, 2, 3, 0):
            assert int(target.sum()) == int(truth.sum()), 'Positive/hard crop lost GT'
        # Unbiased crops may legitimately truncate in y; they still retain
        # both full-width components for whatever actual source rows remain.
        nonzero = target[0, 0].numpy()
        left, right = int(nonzero[:, :80].sum()), int(nonzero[:, 430:].sum())
        assert min(left, right) * 12 == max(left, right) * 10
        checked += 1
    report = {'fixture_batches': checked, 'passed': True,
              'scope': 'Synthetic coordinate/GT pairing only; no artificial defect accuracy credit or NN forward'}
    path = Path(__file__).resolve().parents[1] / 'output/optimization/factory-round3/attempt5-native-width-pairing-check.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
