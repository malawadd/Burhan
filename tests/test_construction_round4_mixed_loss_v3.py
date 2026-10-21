"""Small loss semantics fixtures; no network, training data or inspection claims."""
import importlib.util
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    from unittest import SkipTest
    raise SkipTest("Training-only numerical fixtures require optional Torch environment")

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/construction_round4_mixed_loss_v3.py'
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location('round4_mixed_loss_fixture', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unknown_class_has_zero_bce_gradient_at_positive_and_background_anchors():
    logits = torch.zeros(1, 8, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 3, 0] = .8
    criterion = module.PartialClassBCE()
    criterion.supervised_classes = torch.tensor([[1., 0.]])
    criterion(logits, target).sum().backward()
    assert torch.count_nonzero(logits.grad[0, :, 1]) == 0
    assert logits.grad[0, 3, 0] < 0
    assert logits.grad[0, 0, 0] > 0


def test_one_pixel_stroke_keeps_meaningful_positive_gradient_without_thickening():
    # Explicit numerical unit fixture, not a generated inspection image/video.
    target = torch.zeros(64, 64)
    target[31] = 1
    original = target.clone()
    logits = torch.zeros_like(target, requires_grad=True)
    module.balanced_true_mask_loss(logits, target).backward()
    assert torch.equal(target, original)
    assert torch.all(logits.grad[31] < 0)
    assert logits.grad[31].abs().sum() > .1
    assert torch.all(logits.grad[0] > 0)
    assert torch.isfinite(logits.grad).all()


def test_partial_camera_unassigned_same_class_is_unknown_not_background():
    logits = torch.zeros(1, 8, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 3, 0] = .8
    criterion = module.PartialClassBCE()
    criterion.supervised_classes = torch.tensor([[1., 0.]])
    criterion.positive_assignments_only = True
    criterion(logits, target).sum().backward()
    assert logits.grad[0, 3, 0] != 0
    assert torch.count_nonzero(logits.grad[0, :3, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 4:, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, :, 1]) == 0


def test_fractional_real_pixel_coverage_is_retained_not_thresholded_away():
    target = torch.zeros(16, 16)
    target[4, 2:14] = .125
    logits = torch.zeros_like(target, requires_grad=True)
    module.balanced_true_mask_loss(logits, target).backward()
    assert torch.all(logits.grad[4, 2:14] < 0)
    assert logits.grad[4, 2:14].abs().sum() > .1


def test_memory_bounded_anchor_views_equal_parent_assigned_mask_selection():
    from ultralytics.utils.loss import v8SegmentationLoss
    torch.manual_seed(7)
    criterion = object.__new__(module.MixedConstructionLoss)
    criterion.overlap = False
    masks = torch.zeros(2, 16, 16)
    masks[0, 3, 1:14] = 1
    masks[1, 8:12, 8:12] = 1
    foreground = torch.tensor([[True, False, True, True]])
    indices = torch.tensor([[0, 0, 1, 0]])
    boxes = torch.tensor([[[0., 0., 16., 16.]] * 4])
    batch_index = torch.zeros(2, 1)
    proto = torch.randn(1, 2, 16, 16, requires_grad=True)
    coefficients = torch.randn(1, 4, 2, requires_grad=True)
    size = torch.tensor([16., 16.])
    bounded = criterion.calculate_segmentation_loss(foreground, masks, indices, boxes, batch_index, proto, coefficients, size)
    reference = v8SegmentationLoss.calculate_segmentation_loss(criterion, foreground, masks, indices, boxes, batch_index, proto, coefficients, size)
    assert torch.allclose(bounded, reference, atol=1e-7)
    bounded_grad = torch.autograd.grad(bounded, [proto, coefficients], retain_graph=True)
    reference_grad = torch.autograd.grad(reference, [proto, coefficients])
    for actual, expected in zip(bounded_grad, reference_grad):
        assert torch.allclose(actual, expected, atol=1e-7)


def test_source_area_one_ulp_is_locally_clamped_without_mutating_mask():
    one = torch.tensor(1., dtype=torch.float32)
    one_ulp_above = torch.nextafter(one, torch.tensor(2.))
    target = torch.tensor([[0., 1.], [.125, 0.]])
    target[0, 1] = one_ulp_above
    original = target.clone()
    logits = torch.zeros_like(target, requires_grad=True)
    actual = module.balanced_true_mask_loss(logits, target)
    expected = module.balanced_true_mask_loss(logits, target.clamp(0, 1))
    assert torch.equal(target, original)
    assert torch.equal(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_genuine_bad_mask_probability_is_rejected_not_silently_clipped():
    target = torch.tensor([[0., 1.001], [0., 0.]])
    try:
        module.balanced_true_mask_loss(torch.zeros_like(target), target)
    except ValueError:
        return
    raise AssertionError('Out-of-tolerance source mask was accepted')


def test_grouped_anchor_losses_and_all_logit_gradients_equal_scalar_fractional_reference():
    torch.manual_seed(11)
    logits = torch.randn(5, 13, 11, requires_grad=True)
    target = torch.zeros(13, 11)
    target[4, 1:10] = .125
    target[5, 2:8] = 1
    target[5, 4] = torch.nextafter(torch.tensor(1.), torch.tensor(2.))
    grouped = module.grouped_true_mask_loss(logits, target)
    scalar = sum(module.balanced_true_mask_loss(one, target) for one in logits)
    assert torch.allclose(grouped, scalar, atol=1e-6, rtol=1e-6)
    actual = torch.autograd.grad(grouped, logits, retain_graph=True)[0]
    reference = torch.autograd.grad(scalar, logits)[0]
    assert torch.allclose(actual, reference, atol=1e-7, rtol=1e-6)
