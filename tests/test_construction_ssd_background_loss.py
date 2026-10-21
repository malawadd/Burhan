"""Regression for the previously silent loss on negative construction photos."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.ssd import SSD

from scripts.construction_ssd_background_loss import background_aware_loss


def inputs(positive):
    anchor = torch.tensor([[0., 0., 8., 8.], [10., 10., 18., 18.],
                           [20., 20., 28., 28.], [30., 30., 38., 38.]])
    target = {"boxes": anchor[:1] if positive else anchor[:0],
              "labels": torch.ones(1 if positive else 0, dtype=torch.long)}
    matches = torch.tensor([0, -1, -1, -1] if positive else [-1, -1, -1, -1])
    logits = torch.tensor([[[.2, .8], [.6, .4], [.3, .7], [.7, .3]]], requires_grad=True)
    outputs = {"bbox_regression": torch.zeros((1, 4, 4), requires_grad=True), "cls_logits": logits}
    stub = SimpleNamespace(neg_to_pos_ratio=3, box_coder=BoxCoder((10., 10., 5., 5.)))
    return stub, [target], outputs, [anchor], [matches]


def test_empty_targets_now_contribute_background_gradient():
    stub, targets, outputs, anchors, matches = inputs(False)
    original = SSD.compute_loss(stub, targets, outputs, anchors, matches)
    original["classification"].backward(retain_graph=True)
    assert float(original["classification"].detach()) == 0
    assert torch.count_nonzero(outputs["cls_logits"].grad) == 0
    outputs["cls_logits"].grad.zero_()
    corrected = background_aware_loss(stub, targets, outputs, anchors, matches)
    corrected["classification"].backward()
    assert corrected["classification"] > 0
    assert torch.count_nonzero(outputs["cls_logits"].grad) > 0
    assert stub.background_loss_diagnostic["background_anchors_by_image"] == [3]


def test_nonempty_single_image_loss_retains_ssd_behavior():
    stub, targets, outputs, anchors, matches = inputs(True)
    original = SSD.compute_loss(stub, targets, outputs, anchors, matches)
    corrected = background_aware_loss(stub, targets, outputs, anchors, matches)
    for key in original:
        torch.testing.assert_close(original[key], corrected[key], rtol=0, atol=0)
