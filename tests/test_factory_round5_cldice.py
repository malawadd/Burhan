"""TRAIN-only tiny loss/geometry checks. No model/weights/data/NN constructor."""
import importlib.util
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('round5', ROOT / 'scripts/factory_round5_cldice.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
TORCH = importlib.util.find_spec('torch') is not None


class GeometryTests(unittest.TestCase):
    def test_CAL_identity_is_exact_historical_spaced_serializer(self):
        rows = [{'image': 'kos00/Part0.jpg', 'defective': False, 'native_mask_shape': [1280, 500]}]
        self.assertEqual(module.cal_identity(rows), hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
        self.assertNotEqual(module.cal_identity(rows), hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest())

    def test_whole_component_containment_at_borders_and_jitter_extremes(self):
        for source_height in [320, 640, 1280]:
            for height in [1, 2, 120, 192]:
                for y in [0, (source_height - height) // 2, source_height - height]:
                    for jitter in [-64, 0, 64]:
                        top = module.positive_top(y, height, source_height, jitter)
                        self.assertLessEqual(top, y); self.assertGreaterEqual(top + 320, y + height)
        with self.assertRaises(ValueError): module.positive_top(100, 193, 1280, 64)


@unittest.skipUnless(TORCH, 'Tiny differentiable math requires exportenvTorch')
class ThinLossTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch; torch.set_num_threads(1)
        self.truth = torch.zeros((1, 1, 64, 96)); self.truth[:, :, 32, 16:80] = 1
        self.valid = torch.ones_like(self.truth, dtype=torch.bool)

    def test_missing_thin_trace_has_higher_topology_loss_and_correct_gradient(self):
        t = self.t
        whole = (self.truth * 16 - 8).detach()
        broken = whole.clone(); broken[:, :, 32, 40:52] = -8; broken.requires_grad_()
        good, bad = module.topology_loss(whole, self.truth, self.valid), module.topology_loss(broken, self.truth, self.valid)
        self.assertGreater(float(bad), float(good)); bad.backward()
        self.assertTrue(t.isfinite(broken.grad).all()); self.assertLess(float(broken.grad[:, :, 32, 40:52].sum()), 0)

    def test_false_bridge_is_penalized_on_actual_background(self):
        good = (self.truth * 16 - 8).detach(); bridge = good.clone(); bridge[:, :, 20:32, 48] = 8
        self.assertGreater(float(module.topology_loss(bridge, self.truth, self.valid)), float(module.topology_loss(good, self.truth, self.valid)))

    def test_empty_GT_has_zero_topology_but_background_loss_gradient(self):
        t = self.t; target = t.zeros_like(self.truth); logits = t.zeros_like(target, requires_grad=True)
        self.assertEqual(float(module.topology_loss(logits, target, self.valid)), 0)
        value = module.loss(logits, target, self.valid); value.backward()
        self.assertTrue(t.isfinite(logits.grad).all()); self.assertGreater(float(logits.grad.sum()), 0)

    def test_padding_has_exact_zero_gradient_and_loss_invariance(self):
        t = self.t; valid = self.valid.clone(); valid[:, :, 48:] = False
        target = self.truth.clone(); a = t.zeros_like(target, requires_grad=True)
        b = a.detach().clone(); b[:, :, 48:] = 30; b.requires_grad_()
        x, y = module.loss(a, target, valid), module.loss(b, target, valid)
        self.assertEqual(float(x), float(y)); y.backward()
        self.assertEqual(int(t.count_nonzero(b.grad[:, :, 48:])), 0)

    def test_crop_border_target_is_not_fabricated_as_interior_topology(self):
        t = self.t; target = t.zeros_like(self.truth); target[:, :, 0, 16:80] = 1
        logits = t.zeros_like(target, requires_grad=True)
        self.assertEqual(float(module.topology_loss(logits, target, self.valid)), 0)
        module.loss(logits, target, self.valid).backward()
        self.assertLess(float(logits.grad[:, :, 0, 16:80].sum()), 0)

    def test_bad_target_rejected_without_clamping_or_GT_rewrite(self):
        target = self.truth.clone(); target[:, :, 32, 48] = 1.1
        with self.assertRaises(ValueError): module.loss(self.t.zeros_like(target), target, self.valid)


if __name__ == '__main__': unittest.main()
