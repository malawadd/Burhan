"""Tiny annotation/loss contracts; no model construction or real accuracy claim."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_construction_pixel_supervision import lcsd_target, rasterize_cracks, prepare_pair


def test_lcsd_class2_is_binary_one_and_ignore_is_not_background():
    target, valid = lcsd_target(np.array([[0, 2, 255]], dtype=np.uint8))
    assert target.tolist() == [[0, 1, 0]]
    assert valid.tolist() == [[1, 1, 0]]
    with pytest.raises(ValueError):
        lcsd_target(np.array([[0, 1, 255]], dtype=np.uint8))


def test_dacl_only_crack_family_is_unioned_without_treating_spall_as_crack():
    annotation = {'image_width': 16, 'image_height': 12, 'polygons': [
        {'label': 'Crack', 'points': [[(.1, .1), (.3, .1), (.3, .3)]]},
        {'label': 'ACrack', 'points': [[(.6, .6), (.8, .6), (.8, .8)]]},
        {'label': 'Spalling', 'points': [[(.9, 0), (1, 0), (1, .1)]]}]}
    target, count = rasterize_cracks(annotation)
    assert count == 2 and target[1, 2] == 1 and target[8, 11] == 1
    assert target[0, 15] == 0


def test_shared_gain_occurs_once_before_resize(monkeypatch):
    import prepare_construction_pixel_supervision as prep
    calls = []
    def gain(frame, policy):
        calls.append((frame.shape, policy))
        return frame + 3, 2.
    monkeypatch.setattr(prep, 'camera_exposure', gain)
    frame, target, valid, value = prepare_pair(np.zeros((4, 8, 3), np.uint8), np.ones((4, 8), np.uint8), np.ones((4, 8), np.uint8))
    assert calls == [((4, 8, 3), 'low_light_gain')]
    assert frame.shape == (384, 384, 3) and frame.min() == frame.max() == 3
    assert target.min() == valid.min() == 1 and value == 2.


def test_ignored_pixel_has_zero_loss_gradient():
    torch = pytest.importorskip('torch')
    from probe_construction_pixel_supervision import masked_loss
    logits = torch.zeros((1, 1, 1, 3), requires_grad=True)
    target = torch.tensor([[[[1., 0., 1.]]]])
    valid = torch.tensor([[[[1., 1., 0.]]]])
    loss = masked_loss(logits, target, valid)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0, 0, 0, 0] != 0
    assert logits.grad[0, 0, 0, 2] == 0
