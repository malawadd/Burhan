"""Decoder geometry fixtures; no neural inference or claimed data accuracy."""
import numpy as np
from edge_runtime.profiles.construction_dacl import decode_masks, DaclConstructionModel


def test_overlapping_damage_channels_are_not_argmax_classes():
    probabilities = np.zeros((19, 4, 4), np.float32)
    probabilities[4, 1:3, 1:3] = .8  # Rust.
    probabilities[12, 1:3, 1:3] = .9  # Exposed reinforcement at the same pixels.
    rows = decode_masks(probabilities, (8, 8), ['rust', 'exposed_rebar'], minimum_area=4, mask_rendering='masks')
    assert {row['label'] for row in rows} == {'rust', 'exposed_rebar'}
    assert all(row['bbox'] == (2, 2, 4, 4) for row in rows)
    assert np.array_equal(rows[0]['mask'], rows[1]['mask'])


def test_boxes_only_preserves_decisions_and_nearest_geometry():
    probabilities = np.zeros((19, 4, 4), np.float32)
    probabilities[2, 1:3, 1:3] = .7
    masks = decode_masks(probabilities, (7, 11), ['wetspot'], minimum_area=4, mask_rendering='masks')
    boxes = decode_masks(probabilities, (7, 11), ['wetspot'], minimum_area=4)
    assert [{key: value for key, value in row.items() if key != 'mask'} for row in masks] == boxes
    ys, xs = np.nonzero(masks[0]['mask'])
    assert boxes[0]['bbox'] == (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def test_other_classes_and_subthreshold_pixels_do_not_become_damage():
    probabilities = np.zeros((19, 4, 4), np.float32)
    probabilities[0, :, :] = .5
    probabilities[13, :, :] = .99  # Bearing component, not damage.
    assert decode_masks(probabilities, (8, 8), ['crack']) == []


def test_minimum_area_is_model_grid_pixels():
    probabilities = np.zeros((19, 4, 4), np.float32)
    probabilities[4, 1, 1:3] = .9
    assert decode_masks(probabilities, (80, 80), ['rust'], minimum_area=3) == []
    assert len(decode_masks(probabilities, (80, 80), ['rust'], minimum_area=2)) == 1


def test_optional_per_class_cutoffs_use_actual_infer_without_changing_uniform():
    probabilities=np.zeros((19,4,4),np.float32)
    probabilities[0,1:3,1:3]=.6
    probabilities[8,1:3,1:3]=.6
    model=object.__new__(DaclConstructionModel)
    model.selected=['crack','spalling'];model.threshold=.5;model.minimum_area=4
    model.mask_rendering='boxes';model.predict_maps=lambda frame:probabilities
    model.class_thresholds={}
    frame=np.zeros((8,8,3),np.uint8)
    assert model.infer(frame)==decode_masks(probabilities,(8,8),model.selected,.5,4,'boxes')
    model.class_thresholds={'crack':.7,'spalling':.4}
    rows=model.infer(frame)
    assert len(rows)==1 and rows[0]['label']=='spalling'
